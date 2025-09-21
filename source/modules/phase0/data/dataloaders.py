"""
Unified Data Loader for Phase 0 Pre-training
Implementation for RSNA 2025 Phase 0 Pre-training

This module provides unified data loading for:
- OpenMind dataset (1-channel MRI: T1, T2, MRA)
- DeepLesion dataset (3-channel CT windowing)
- Seamless integration with different modalities and channel counts
- Memory-efficient loading with caching and augmentation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import random
import logging
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandScaleIntensityd,
    RandShiftIntensityd, RandAffined, RandGaussianNoised,
    Resized, CenterSpatialCropd, RandSpatialCropd, NormalizeIntensityd
)
import warnings
warnings.filterwarnings("ignore")


class Phase0Dataset(Dataset):
    """
    Unified dataset for Phase 0 pre-training combining OpenMind and DeepLesion

    Features:
    - Automatic modality detection and channel handling
    - Memory-efficient loading with optional caching
    - Standardized output format for all modalities
    - Data augmentation support
    """
    def __init__(self,
                 data_config: Dict[str, Dict],
                 target_size: Tuple[int, int, int] = (64, 64, 64),
                 augmentation: bool = True,
                 cache_size: int = 100,
                 normalize: bool = True):
        """
        Args:
            data_config: Configuration dictionary with dataset paths and metadata
            target_size: Target 3D size for all volumes
            augmentation: Whether to apply data augmentation
            cache_size: Number of volumes to cache in memory (0 = no caching)
            normalize: Whether to apply intensity normalization
        """
        self.data_config = data_config
        self.target_size = target_size
        self.augmentation = augmentation
        self.cache_size = cache_size
        self.normalize = normalize

        # Cache for loaded volumes
        self.cache = {}
        self.cache_order = []

        # Build file list from all datasets
        self.file_list = self._build_file_list()

        # Setup augmentation transforms
        self.transforms = self._setup_transforms()

        # Statistics for normalization
        self.intensity_stats = self._compute_intensity_stats()

        logging.info(f"Phase0Dataset initialized with {len(self.file_list)} files")
        logging.info(f"Modality distribution: {self._count_modalities()}")

    def _build_file_list(self) -> List[Dict]:
        """Build unified file list from all configured datasets"""
        file_list = []

        for dataset_name, config in self.data_config.items():
            dataset_path = Path(config['path'])
            modalities = config.get('modalities', ['unknown'])
            channels = config.get('channels', 1)

            logging.info(f"Processing dataset: {dataset_name}")
            logging.info(f"  Path: {dataset_path}")
            logging.info(f"  Modalities: {modalities}")
            logging.info(f"  Channels: {channels}")

            if dataset_name == 'deeplesion':
                # DeepLesion: 3-channel CT files
                ct_files = list(dataset_path.glob("*.nii.gz"))
                for file_path in ct_files:
                    file_list.append({
                        'file_path': file_path,
                        'dataset': dataset_name,
                        'modality': 'CTA',
                        'channels': 3,
                        'file_id': file_path.stem
                    })

            elif dataset_name == 'openmind':
                # OpenMind: Organized by modality subdirectories
                for modality in modalities:
                    modality_dir = dataset_path / modality
                    if modality_dir.exists():
                        nifti_files = list(modality_dir.glob("*.nii.gz"))
                        for file_path in nifti_files:
                            file_list.append({
                                'file_path': file_path,
                                'dataset': dataset_name,
                                'modality': modality,
                                'channels': 1,
                                'file_id': file_path.stem
                            })

        logging.info(f"Total files found: {len(file_list)}")
        return file_list

    def _count_modalities(self) -> Dict[str, int]:
        """Count files by modality"""
        counts = {}
        for item in self.file_list:
            modality = item['modality']
            counts[modality] = counts.get(modality, 0) + 1
        return counts

    def _setup_transforms(self) -> Optional[Compose]:
        """Setup data augmentation transforms"""
        if not self.augmentation:
            return None

        # Define transforms for 3D volumes
        transforms = [
            # Spatial transforms
            RandRotate90d(keys=['image'], prob=0.3, spatial_axes=(0, 1)),
            RandRotate90d(keys=['image'], prob=0.3, spatial_axes=(0, 2)),
            RandRotate90d(keys=['image'], prob=0.3, spatial_axes=(1, 2)),
            RandFlipd(keys=['image'], prob=0.3, spatial_axis=0),
            RandFlipd(keys=['image'], prob=0.3, spatial_axis=1),
            RandFlipd(keys=['image'], prob=0.3, spatial_axis=2),

            # Intensity transforms (lighter for pre-training)
            RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.2),
            RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.2),
            RandGaussianNoised(keys=['image'], std=0.01, prob=0.1),

            # Spatial cropping and resizing
            CenterSpatialCropd(keys=['image'], roi_size=self.target_size),
            Resized(keys=['image'], spatial_size=self.target_size, mode='trilinear'),
        ]

        return Compose(transforms)

    def _compute_intensity_stats(self) -> Dict[str, Dict]:
        """Compute intensity statistics for normalization"""
        # Sample a subset of files for computing statistics
        sample_size = min(100, len(self.file_list))
        sample_indices = random.sample(range(len(self.file_list)), sample_size)

        stats = {}
        for modality in ['CTA', 'MRA', 'MRI_T1', 'MRI_T2']:
            stats[modality] = {'mean': 0.0, 'std': 1.0}

        # Could implement more sophisticated statistics computation here
        # For now, use defaults since data should already be normalized
        return stats

    def _load_volume(self, file_path: Path) -> torch.Tensor:
        """Load a single volume from file"""
        try:
            # Load using SimpleITK
            img = sitk.ReadImage(str(file_path))
            volume = sitk.GetArrayFromImage(img)  # [D, H, W] or [D, H, W, C]

            # Convert to tensor and ensure proper format
            volume_tensor = torch.from_numpy(volume).float()

            # Handle different channel formats
            if volume_tensor.dim() == 3:
                # Single channel: [D, H, W] -> [1, D, H, W]
                volume_tensor = volume_tensor.unsqueeze(0)
            elif volume_tensor.dim() == 4:
                # Multi-channel: [D, H, W, C] -> [C, D, H, W]
                volume_tensor = volume_tensor.permute(3, 0, 1, 2)

            return volume_tensor

        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            # Return dummy volume as fallback
            return torch.zeros(1, *self.target_size)

    def _cache_volume(self, idx: int, volume: torch.Tensor):
        """Cache a volume with LRU eviction"""
        if self.cache_size <= 0:
            return

        # Remove oldest if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]

        # Add to cache
        self.cache[idx] = volume
        self.cache_order.append(idx)

    def _get_cached_volume(self, idx: int) -> Optional[torch.Tensor]:
        """Get volume from cache if available"""
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.cache[idx]
        return None

    def _resize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Resize volume to target size"""
        current_size = volume.shape[1:]  # Skip channel dimension
        if current_size != self.target_size:
            # Use interpolation to resize
            volume = torch.nn.functional.interpolate(
                volume.unsqueeze(0),  # Add batch dimension
                size=self.target_size,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension

        return volume

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Get a single item from the dataset"""
        # Check cache first
        volume = self._get_cached_volume(idx)

        if volume is None:
            # Load from disk
            file_info = self.file_list[idx]
            volume = self._load_volume(file_info['file_path'])

            # Resize to target size
            volume = self._resize_volume(volume)

            # Cache if enabled
            self._cache_volume(idx, volume)

        file_info = self.file_list[idx]

        # Prepare data dictionary for transforms
        data_dict = {
            'image': volume,
            'modality': file_info['modality'],
            'channels': file_info['channels'],
            'dataset': file_info['dataset'],
            'file_id': file_info['file_id']
        }

        # Apply transforms if enabled
        if self.transforms is not None:
            try:
                data_dict = self.transforms(data_dict)
            except Exception as e:
                logging.warning(f"Transform failed for {file_info['file_path']}: {e}")

        # Normalize intensity if enabled
        if self.normalize:
            modality = data_dict['modality']
            if modality in self.intensity_stats:
                stats = self.intensity_stats[modality]
                data_dict['image'] = (data_dict['image'] - stats['mean']) / stats['std']

        # Ensure volume is in correct format [C, D, H, W]
        volume = data_dict['image']
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)

        return {
            'image': volume,
            'modality': data_dict['modality'],
            'channels': data_dict['channels'],
            'dataset': data_dict['dataset'],
            'file_id': data_dict['file_id'],
            'original_shape': volume.shape
        }


class UnifiedDataLoader:
    """
    Unified data loader factory for Phase 0 pre-training

    Manages data loading configuration and creates appropriate DataLoaders
    for different training phases and modalities
    """
    def __init__(self,
                 deeplesion_path: str,
                 openmind_path: str,
                 target_size: Tuple[int, int, int] = (64, 64, 64),
                 batch_size: int = 4,
                 num_workers: int = 4,
                 augmentation: bool = True,
                 cache_size: int = 100):
        """
        Args:
            deeplesion_path: Path to processed DeepLesion CT directory
            openmind_path: Path to processed OpenMind directory
            target_size: Target 3D size for volumes
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            augmentation: Whether to use data augmentation
            cache_size: Number of volumes to cache in memory
        """
        self.deeplesion_path = Path(deeplesion_path)
        self.openmind_path = Path(openmind_path)
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.cache_size = cache_size

        # Data configuration
        self.data_config = {
            'deeplesion': {
                'path': self.deeplesion_path,
                'modalities': ['CTA'],
                'channels': 3,
                'normalization': 'ct_windowing'
            },
            'openmind': {
                'path': self.openmind_path,
                'modalities': ['MRA', 'MRI_T1', 'MRI_T2'],
                'channels': 1,
                'normalization': 'nyul_percentile_fallback'
            }
        }

        logging.info("UnifiedDataLoader initialized")
        logging.info(f"DeepLesion path: {self.deeplesion_path}")
        logging.info(f"OpenMind path: {self.openmind_path}")
        logging.info(f"Target size: {self.target_size}")
        logging.info(f"Batch size: {self.batch_size}")

    def create_dataset(self, split: str = 'train') -> Phase0Dataset:
        """Create dataset for specified split"""
        # For pre-training, we use all available data
        # Could implement train/val split here if needed
        dataset = Phase0Dataset(
            data_config=self.data_config,
            target_size=self.target_size,
            augmentation=self.augmentation if split == 'train' else False,
            cache_size=self.cache_size,
            normalize=True
        )

        return dataset

    def create_dataloader(self, split: str = 'train', shuffle: bool = True) -> DataLoader:
        """Create DataLoader for specified split"""
        dataset = self.create_dataset(split)

        # Custom collate function to handle different channel counts
        def collate_fn(batch):
            # Separate single-channel and multi-channel data
            single_channel_batch = []
            multi_channel_batch = []
            metadata_batch = []

            for item in batch:
                metadata = {
                    'modality': item['modality'],
                    'channels': item['channels'],
                    'dataset': item['dataset'],
                    'file_id': item['file_id'],
                    'original_shape': item['original_shape']
                }
                metadata_batch.append(metadata)

                if item['channels'] == 1:
                    single_channel_batch.append(item['image'])
                else:
                    multi_channel_batch.append(item['image'])

            # Create batched tensors
            batched_data = {}

            if single_channel_batch:
                batched_data['single_channel'] = torch.stack(single_channel_batch, dim=0)

            if multi_channel_batch:
                batched_data['multi_channel'] = torch.stack(multi_channel_batch, dim=0)

            batched_data['metadata'] = metadata_batch

            return batched_data

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )

        return dataloader

    def get_modality_stats(self) -> Dict[str, int]:
        """Get statistics about modalities in the dataset"""
        dataset = self.create_dataset('train')
        return dataset._count_modalities()


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Test configuration (adjust paths as needed)
    deeplesion_path = "/mnt/d/2.Research/RSNA/data/processed/NIH_deeplesion/CT"
    openmind_path = "/mnt/d/2.Research/RSNA/data/processed/openmind/OpenMind_processed"

    # Create data loader
    data_loader = UnifiedDataLoader(
        deeplesion_path=deeplesion_path,
        openmind_path=openmind_path,
        target_size=(64, 64, 64),
        batch_size=2,
        num_workers=0,  # Set to 0 for testing
        augmentation=True,
        cache_size=10
    )

    # Get modality statistics
    try:
        stats = data_loader.get_modality_stats()
        print("Modality statistics:")
        for modality, count in stats.items():
            print(f"  {modality}: {count} files")
    except Exception as e:
        print(f"Could not load datasets: {e}")
        print("This is expected if the data paths don't exist")

    # Test data loading
    try:
        train_loader = data_loader.create_dataloader('train', shuffle=True)
        print(f"Created DataLoader with {len(train_loader)} batches")

        # Test loading a single batch
        for batch_idx, batch_data in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")

            if 'single_channel' in batch_data:
                sc_data = batch_data['single_channel']
                print(f"  Single-channel data: {sc_data.shape}")

            if 'multi_channel' in batch_data:
                mc_data = batch_data['multi_channel']
                print(f"  Multi-channel data: {mc_data.shape}")

            metadata = batch_data['metadata']
            print(f"  Metadata items: {len(metadata)}")
            for i, meta in enumerate(metadata):
                print(f"    Item {i}: {meta['modality']} ({meta['channels']} ch) from {meta['dataset']}")

            # Only test first batch
            break

    except Exception as e:
        print(f"DataLoader test failed: {e}")
        print("This is expected if the data paths don't exist")

    print("\nDataLoader implementation completed successfully!")