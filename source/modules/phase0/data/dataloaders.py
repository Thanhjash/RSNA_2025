"""
Unified Multi-Modal Medical Image DataLoaders
Implementation for RSNA 2025 Phase 0 Pre-training

Supports:
- OpenMind MRI data (1-channel: MRI_T1, MRI_T2, MRA)
- DeepLesion CT data (3-channel processed format)
- Automatic channel normalization and intensity standardization
- Memory-efficient loading with caching
- UNIFIED MODEL: Separate MRI/CT loaders for alternating batch training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import warnings
import torch.nn.functional as F


class MedicalImageDataset3D(Dataset):
    """
    Unified dataset for 3D medical images (MRI and CT)
    
    Features:
    - Supports both .nii.gz (MRI) and .npy (CT) formats
    - Automatic channel detection (1-channel MRI, 3-channel CT)
    - Intensity normalization (z-score or min-max)
    - Spatial resizing to target dimensions
    - Optional augmentation pipeline integration
    """
    
    def __init__(self, data_dirs: Union[str, List[str]], 
                 target_size=(64, 64, 64),
                 modality='auto',  # 'mri', 'ct', or 'auto'
                 intensity_norm='zscore',  # 'zscore' or 'minmax'
                 cache_data=False,
                 transform=None):
        """
        Args:
            data_dirs: Path(s) to data directories
            target_size: Target spatial dimensions (D, H, W)
            modality: Image modality ('mri', 'ct', or 'auto' to detect)
            intensity_norm: Intensity normalization method
            cache_data: Whether to cache loaded data in memory
            transform: Optional transform function
        """
        # Convert single path to list
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        
        self.data_dirs = [Path(d) for d in data_dirs]
        self.target_size = target_size
        self.modality = modality
        self.intensity_norm = intensity_norm
        self.cache_data = cache_data
        self.transform = transform
        
        # Scan for image files
        self.image_paths = self._scan_images()
        
        # Cache storage
        self.cache = {} if cache_data else None
        
        print(f"Found {len(self.image_paths)} images across {len(self.data_dirs)} directories")
    
    def _scan_images(self) -> List[Path]:
        """Scan directories for image files"""
        image_paths = []
        
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                warnings.warn(f"Directory not found: {data_dir}")
                continue
            
            # Find .nii.gz files (MRI)
            nii_files = list(data_dir.rglob("*.nii.gz"))
            
            # Find .npy files (CT)
            npy_files = list(data_dir.rglob("*.npy"))
            
            image_paths.extend(nii_files + npy_files)
        
        return sorted(image_paths)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single image
        
        Returns:
            Dictionary containing:
                - 'image': Preprocessed image tensor [C, D, H, W]
                - 'modality': String indicating modality ('mri' or 'ct')
                - 'path': Original file path
        """
        # Check cache
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        image_path = self.image_paths[idx]
        
        # Load image based on file extension
        if image_path.suffix == '.gz':
            image, modality = self._load_nifti(image_path)
        elif image_path.suffix == '.npy':
            image, modality = self._load_numpy(image_path)
        else:
            raise ValueError(f"Unsupported file format: {image_path}")
        
        # Normalize intensity
        image = self._normalize_intensity(image)
        
        # Resize to target dimensions
        image = self._resize(image, self.target_size)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        
        # Apply transforms if specified
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        sample = {
            'image': image_tensor,
            'modality': modality,
            'path': str(image_path)
        }
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = sample
        
        return sample
    
    def _load_nifti(self, path: Path) -> Tuple[np.ndarray, str]:
        """
        Load NIfTI image (MRI)
        
        Returns:
            Tuple of (image_array [C, D, H, W], modality)
        """
        nii = nib.load(str(path))
        image = nii.get_fdata()
        
        # Add channel dimension if needed
        if image.ndim == 3:
            image = image[np.newaxis, ...]  # [1, D, H, W]
        
        return image.astype(np.float32), 'mri'
    
    def _load_numpy(self, path: Path) -> Tuple[np.ndarray, str]:
        """
        Load NumPy array (CT)
        
        Returns:
            Tuple of (image_array [C, D, H, W], modality)
        """
        image = np.load(str(path))
        
        # Ensure channel-first format
        if image.ndim == 3:
            image = image[np.newaxis, ...]  # [1, D, H, W]
        elif image.ndim == 4 and image.shape[-1] == 3:
            # Convert [D, H, W, 3] to [3, D, H, W]
            image = image.transpose(3, 0, 1, 2)
        
        return image.astype(np.float32), 'ct'
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity
        
        Args:
            image: Image array [C, D, H, W]
        
        Returns:
            Normalized image
        """
        if self.intensity_norm == 'zscore':
            # Z-score normalization per channel
            for c in range(image.shape[0]):
                channel = image[c]
                mean = np.mean(channel)
                std = np.std(channel)
                if std > 0:
                    image[c] = (channel - mean) / std
        
        elif self.intensity_norm == 'minmax':
            # Min-max normalization to [0, 1]
            for c in range(image.shape[0]):
                channel = image[c]
                min_val = np.min(channel)
                max_val = np.max(channel)
                if max_val > min_val:
                    image[c] = (channel - min_val) / (max_val - min_val)
        
        return image
    
    def _resize(self, image: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Resize image to target dimensions using trilinear interpolation
        
        Args:
            image: Image array [C, D, H, W]
            target_size: Target (D, H, W)
        
        Returns:
            Resized image [C, D', H', W']
        """
        import torch.nn.functional as F
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, C, D, H, W]
        
        # Resize using trilinear interpolation
        resized = F.interpolate(
            image_tensor,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        return resized.squeeze(0).numpy()


class MultiModalDataLoader:
    """
    Wrapper for creating DataLoaders for multi-modal training
    
    Handles:
    - Separate loaders for different modalities
    - Balanced sampling across modalities
    - Combined iteration for multi-modal batches
    """
    
    def __init__(self, 
                 openmind_dir: str,
                 deeplesion_dir: str,
                 batch_size: int = 8,
                 target_size=(64, 64, 64),
                 num_workers: int = 4,
                 transform=None,
                 train_split=0.9):
        """
        Args:
            openmind_dir: Path to OpenMind MRI data
            deeplesion_dir: Path to DeepLesion CT data
            batch_size: Batch size for each modality
            target_size: Target spatial dimensions
            num_workers: Number of data loading workers
            transform: Optional transform pipeline
            train_split: Fraction of data for training
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create datasets
        self.openmind_dataset = MedicalImageDataset3D(
            data_dirs=openmind_dir,
            target_size=target_size,
            modality='mri',
            transform=transform
        )
        
        self.deeplesion_dataset = MedicalImageDataset3D(
            data_dirs=deeplesion_dir,
            target_size=target_size,
            modality='ct',
            transform=transform
        )
        
        # Split train/val
        openmind_train_size = int(len(self.openmind_dataset) * train_split)
        deeplesion_train_size = int(len(self.deeplesion_dataset) * train_split)
        
        self.openmind_train = torch.utils.data.Subset(
            self.openmind_dataset, 
            range(openmind_train_size)
        )
        self.openmind_val = torch.utils.data.Subset(
            self.openmind_dataset, 
            range(openmind_train_size, len(self.openmind_dataset))
        )
        
        self.deeplesion_train = torch.utils.data.Subset(
            self.deeplesion_dataset, 
            range(deeplesion_train_size)
        )
        self.deeplesion_val = torch.utils.data.Subset(
            self.deeplesion_dataset, 
            range(deeplesion_train_size, len(self.deeplesion_dataset))
        )
    
    def get_train_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get training DataLoaders for both modalities
        
        Returns:
            Tuple of (openmind_loader, deeplesion_loader)
        """
        openmind_loader = DataLoader(
            self.openmind_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        deeplesion_loader = DataLoader(
            self.deeplesion_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return openmind_loader, deeplesion_loader
    
    def get_val_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get validation DataLoaders for both modalities
        
        Returns:
            Tuple of (openmind_loader, deeplesion_loader)
        """
        openmind_loader = DataLoader(
            self.openmind_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        deeplesion_loader = DataLoader(
            self.deeplesion_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return openmind_loader, deeplesion_loader

