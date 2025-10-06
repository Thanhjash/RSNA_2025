"""
Unified Multi-Modal DataLoaders for Phase 0 Pre-training
RSNA 2025 Project

Provides separate MRI (1-channel) and CT (3-channel) dataloaders
for alternating batch training in unified model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F


class MRIDataset(Dataset):
    """
    1-channel MRI dataset (OpenMind: MRI_T1, MRI_T2, MRA)

    Features:
    - Loads .nii.gz files
    - Returns 1-channel tensors
    - Spatial resizing to target dimensions
    - Intensity normalization already applied during preprocessing
    """

    def __init__(self,
                 mri_dirs: List[str],
                 img_size: Tuple[int, int, int] = (64, 64, 64),
                 max_samples: Optional[int] = None):
        """
        Args:
            mri_dirs: List of paths to MRI data directories (MRI_T1, MRI_T2, MRA)
            img_size: Target spatial dimensions (D, H, W)
            max_samples: Maximum samples to load (for subset testing)
        """
        self.img_size = img_size
        self.image_paths = self._scan_images(mri_dirs, max_samples)

        print(f"MRI Dataset: {len(self.image_paths)} images from {len(mri_dirs)} directories")

    def _scan_images(self, mri_dirs: List[str], max_samples: Optional[int]) -> List[Path]:
        """Scan directories for NIfTI files"""
        all_paths = []
        for mri_dir in mri_dirs:
            dir_path = Path(mri_dir)
            if not dir_path.exists():
                print(f"Warning: Directory not found: {mri_dir}")
                continue

            nii_files = sorted(dir_path.glob("*.nii.gz"))
            all_paths.extend(nii_files)

        if max_samples is not None:
            all_paths = all_paths[:max_samples]

        return all_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load single MRI image

        Returns:
            Dictionary with:
                'image': [1, D, H, W] tensor
                'modality': 'MRI'
                'path': file path string
        """
        img_path = self.image_paths[idx]

        try:
            # Load NIfTI file with nibabel
            img_nib = nib.load(str(img_path))
            img_np = img_nib.get_fdata().astype(np.float32)  # [D, H, W] or [H, W, D]

            # Ensure correct orientation (D, H, W)
            if img_np.ndim == 3:
                # Add channel dimension [1, D, H, W]
                img_np = img_np[np.newaxis, ...]
            else:
                raise ValueError(f"Unexpected dimensions: {img_np.shape}")

            # Convert to tensor
            img_tensor = torch.from_numpy(img_np).float()

            # Resize to target size if needed
            if img_tensor.shape[1:] != self.img_size:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),  # [1, 1, D, H, W]
                    size=self.img_size,
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)  # [1, D, H, W]

            return {
                'image': img_tensor,
                'modality': 'MRI',
                'path': str(img_path)
            }

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return zero tensor on error
            return {
                'image': torch.zeros(1, *self.img_size),
                'modality': 'MRI',
                'path': str(img_path)
            }


class CTDataset(Dataset):
    """
    3-channel CT dataset (DeepLesion with brain/blood/bone windowing)

    Features:
    - Loads .nii.gz files with 3 channels
    - Returns 3-channel tensors
    - Spatial resizing to target dimensions
    - CT windowing already applied during preprocessing
    """

    def __init__(self,
                 ct_dirs: List[str],
                 img_size: Tuple[int, int, int] = (64, 64, 64),
                 max_samples: Optional[int] = None):
        """
        Args:
            ct_dirs: List of paths to CT data directories
            img_size: Target spatial dimensions (D, H, W)
            max_samples: Maximum samples to load (for subset testing)
        """
        self.img_size = img_size
        self.image_paths = self._scan_images(ct_dirs, max_samples)

        print(f"CT Dataset: {len(self.image_paths)} images from {len(ct_dirs)} directories")

    def _scan_images(self, ct_dirs: List[str], max_samples: Optional[int]) -> List[Path]:
        """Scan directories for NIfTI files"""
        all_paths = []
        for ct_dir in ct_dirs:
            dir_path = Path(ct_dir)
            if not dir_path.exists():
                print(f"Warning: Directory not found: {ct_dir}")
                continue

            nii_files = sorted(dir_path.glob("*.nii.gz"))
            all_paths.extend(nii_files)

        if max_samples is not None:
            all_paths = all_paths[:max_samples]

        return all_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load single CT image

        Returns:
            Dictionary with:
                'image': [3, D, H, W] tensor
                'modality': 'CT'
                'path': file path string
        """
        img_path = self.image_paths[idx]

        try:
            # Load NIfTI file (already 3-channel from preprocessing)
            img_nib = nib.load(str(img_path))
            img_np = img_nib.get_fdata().astype(np.float32)  # Can be [D, H, W, 3] or [D, H, W, 1, 3]

            # Handle different possible orientations
            if img_np.ndim == 5:
                # Shape is [D, H, W, 1, 3] - squeeze the singleton dimension
                if img_np.shape[3] == 1 and img_np.shape[4] == 3:
                    img_np = img_np.squeeze(3)  # [D, H, W, 3]
                    img_np = img_np.transpose(3, 0, 1, 2)  # [3, D, H, W]
                else:
                    raise ValueError(f"Unexpected 5D shape: {img_np.shape}")
            elif img_np.ndim == 4:
                # Ensure channel-last to channel-first: [D, H, W, 3] -> [3, D, H, W]
                if img_np.shape[-1] == 3:
                    img_np = img_np.transpose(3, 0, 1, 2)
                elif img_np.shape[0] == 3:
                    pass  # Already [3, D, H, W]
                else:
                    raise ValueError(f"Unexpected 4D shape: {img_np.shape}")
            else:
                raise ValueError(f"Expected 4D or 5D array for CT, got: {img_np.shape}")

            # Convert to tensor
            img_tensor = torch.from_numpy(img_np).float()

            # Resize to target size if needed
            if img_tensor.shape[1:] != self.img_size:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),  # [1, 3, D, H, W]
                    size=self.img_size,
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)  # [3, D, H, W]

            return {
                'image': img_tensor,
                'modality': 'CT',
                'path': str(img_path)
            }

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return zero tensor on error
            return {
                'image': torch.zeros(3, *self.img_size),
                'modality': 'CT',
                'path': str(img_path)
            }


def create_unified_dataloaders(
    mri_dirs: List[str],
    ct_dirs: List[str],
    img_size: Tuple[int, int, int] = (64, 64, 64),
    batch_size_mri: int = 8,
    batch_size_ct: int = 4,
    num_workers: int = 4,
    max_samples_mri: Optional[int] = None,
    max_samples_ct: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create separate MRI and CT dataloaders for alternating batch training

    Args:
        mri_dirs: List of MRI data directories
        ct_dirs: List of CT data directories
        img_size: Target spatial dimensions
        batch_size_mri: Batch size for MRI (1-channel)
        batch_size_ct: Batch size for CT (3-channel, needs more memory)
        num_workers: Number of dataloader workers
        max_samples_mri: Limit MRI samples (for subset testing)
        max_samples_ct: Limit CT samples (for subset testing)

    Returns:
        (mri_loader, ct_loader) tuple for alternating batch sampling
    """
    # Create datasets
    mri_dataset = MRIDataset(mri_dirs, img_size, max_samples_mri)
    ct_dataset = CTDataset(ct_dirs, img_size, max_samples_ct)

    # Create dataloaders
    mri_loader = DataLoader(
        mri_dataset,
        batch_size=batch_size_mri,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )

    ct_loader = DataLoader(
        ct_dataset,
        batch_size=batch_size_ct,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f"\nDataLoader Summary:")
    print(f"  MRI: {len(mri_dataset)} samples, batch_size={batch_size_mri}, {len(mri_loader)} batches")
    print(f"  CT:  {len(ct_dataset)} samples, batch_size={batch_size_ct}, {len(ct_loader)} batches")

    return mri_loader, ct_loader
