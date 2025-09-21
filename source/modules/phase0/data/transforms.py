"""
Data Transforms for Phase 0 Pre-training
Implementation for RSNA 2025 Phase 0 Pre-training

This module provides data augmentation and preprocessing transforms optimized for:
- 3D medical image volumes
- Self-supervised pre-training
- Mixed modality training (CT and MRI)
- Memory-efficient processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import random
import math


class RandomRotation3D:
    """Random 3D rotation transform for medical volumes"""

    def __init__(self, degrees: Union[float, Tuple[float, float]] = 15, prob: float = 0.5):
        """
        Args:
            degrees: Range of degrees for rotation. If float, uses (-degrees, degrees)
            prob: Probability of applying the transform
        """
        self.prob = prob
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Rotated volume [C, D, H, W]
        """
        if random.random() > self.prob:
            return volume

        # Random rotation angle
        angle = random.uniform(*self.degrees)
        angle_rad = math.radians(angle)

        # Create rotation matrix (rotate around z-axis for axial slices)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=volume.dtype, device=volume.device)

        # Create affine transformation matrix
        affine_matrix = torch.zeros(2, 3, 4, dtype=volume.dtype, device=volume.device)
        affine_matrix[0, :3, :3] = rotation_matrix
        affine_matrix[1, :3, :3] = rotation_matrix

        # Apply rotation using affine grid
        grid = F.affine_grid(
            affine_matrix.unsqueeze(0),
            volume.unsqueeze(0).shape,
            align_corners=False
        )

        rotated = F.grid_sample(
            volume.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).squeeze(0)

        return rotated


class RandomFlip3D:
    """Random 3D flipping transform"""

    def __init__(self, axes: List[int] = [0, 1, 2], prob: float = 0.5):
        """
        Args:
            axes: List of axes along which to flip (0=D, 1=H, 2=W)
            prob: Probability of applying flip for each axis
        """
        self.axes = axes
        self.prob = prob

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Flipped volume [C, D, H, W]
        """
        for axis in self.axes:
            if random.random() < self.prob:
                # Flip along specified axis (add 1 to account for channel dimension)
                volume = torch.flip(volume, dims=[axis + 1])

        return volume


class RandomIntensityScale:
    """Random intensity scaling transform"""

    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1), prob: float = 0.5):
        """
        Args:
            scale_range: Range for intensity scaling
            prob: Probability of applying the transform
        """
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Scaled volume [C, D, H, W]
        """
        if random.random() > self.prob:
            return volume

        scale_factor = random.uniform(*self.scale_range)
        return volume * scale_factor


class RandomIntensityShift:
    """Random intensity shift transform"""

    def __init__(self, shift_range: Tuple[float, float] = (-0.1, 0.1), prob: float = 0.5):
        """
        Args:
            shift_range: Range for intensity shifting
            prob: Probability of applying the transform
        """
        self.shift_range = shift_range
        self.prob = prob

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Shifted volume [C, D, H, W]
        """
        if random.random() > self.prob:
            return volume

        shift_value = random.uniform(*self.shift_range)
        return volume + shift_value


class RandomGaussianNoise:
    """Random Gaussian noise transform"""

    def __init__(self, noise_std: float = 0.01, prob: float = 0.3):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise
            prob: Probability of applying the transform
        """
        self.noise_std = noise_std
        self.prob = prob

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Noisy volume [C, D, H, W]
        """
        if random.random() > self.prob:
            return volume

        noise = torch.randn_like(volume) * self.noise_std
        return volume + noise


class RandomCrop3D:
    """Random 3D cropping transform"""

    def __init__(self, output_size: Tuple[int, int, int], prob: float = 1.0):
        """
        Args:
            output_size: Desired output size (D, H, W)
            prob: Probability of applying the transform
        """
        self.output_size = output_size
        self.prob = prob

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Cropped volume [C, D', H', W']
        """
        if random.random() > self.prob:
            return volume

        c, d, h, w = volume.shape
        td, th, tw = self.output_size

        # If input is smaller than output, pad first
        if d < td or h < th or w < tw:
            pad_d = max(0, td - d)
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)

            volume = F.pad(volume, (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2
            ), mode='constant', value=0)
            c, d, h, w = volume.shape

        # Random crop
        start_d = random.randint(0, max(0, d - td))
        start_h = random.randint(0, max(0, h - th))
        start_w = random.randint(0, max(0, w - tw))

        cropped = volume[
            :,
            start_d:start_d + td,
            start_h:start_h + th,
            start_w:start_w + tw
        ]

        return cropped


class CenterCrop3D:
    """Center 3D cropping transform"""

    def __init__(self, output_size: Tuple[int, int, int]):
        """
        Args:
            output_size: Desired output size (D, H, W)
        """
        self.output_size = output_size

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Cropped volume [C, D', H', W']
        """
        c, d, h, w = volume.shape
        td, th, tw = self.output_size

        # If input is smaller than output, pad first
        if d < td or h < th or w < tw:
            pad_d = max(0, td - d)
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)

            volume = F.pad(volume, (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2
            ), mode='constant', value=0)
            c, d, h, w = volume.shape

        # Center crop
        start_d = (d - td) // 2
        start_h = (h - th) // 2
        start_w = (w - tw) // 2

        cropped = volume[
            :,
            start_d:start_d + td,
            start_h:start_h + th,
            start_w:start_w + tw
        ]

        return cropped


class Resize3D:
    """3D resizing transform using trilinear interpolation"""

    def __init__(self, output_size: Tuple[int, int, int], mode: str = 'trilinear'):
        """
        Args:
            output_size: Desired output size (D, H, W)
            mode: Interpolation mode ('trilinear', 'nearest')
        """
        self.output_size = output_size
        self.mode = mode

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Resized volume [C, D', H', W']
        """
        if volume.shape[1:] == self.output_size:
            return volume

        # Add batch dimension for interpolation
        resized = F.interpolate(
            volume.unsqueeze(0),
            size=self.output_size,
            mode=self.mode,
            align_corners=False if self.mode == 'trilinear' else None
        ).squeeze(0)

        return resized


class Normalize3D:
    """3D normalization transform"""

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-8):
        """
        Args:
            mean: Target mean
            std: Target standard deviation
            eps: Small epsilon for numerical stability
        """
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, D, H, W]
        Returns:
            Normalized volume [C, D, H, W]
        """
        # Z-score normalization
        vol_mean = volume.mean()
        vol_std = volume.std() + self.eps

        normalized = (volume - vol_mean) / vol_std
        normalized = normalized * self.std + self.mean

        return normalized


class Compose:
    """Compose multiple transforms"""

    def __init__(self, transforms: List[Callable]):
        """
        Args:
            transforms: List of transform functions
        """
        self.transforms = transforms

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence"""
        for transform in self.transforms:
            volume = transform(volume)
        return volume


class PretrainingTransforms:
    """Factory class for creating pre-training transform pipelines"""

    @staticmethod
    def get_train_transforms(target_size: Tuple[int, int, int] = (64, 64, 64)) -> Compose:
        """Get training transforms for self-supervised pre-training"""
        transforms = [
            # Spatial transforms
            RandomRotation3D(degrees=10, prob=0.3),
            RandomFlip3D(axes=[0, 1, 2], prob=0.3),

            # Intensity transforms (conservative for pre-training)
            RandomIntensityScale(scale_range=(0.95, 1.05), prob=0.2),
            RandomIntensityShift(shift_range=(-0.05, 0.05), prob=0.2),
            RandomGaussianNoise(noise_std=0.005, prob=0.1),

            # Size transforms
            RandomCrop3D(output_size=target_size, prob=0.8),
            Resize3D(output_size=target_size),

            # Normalization
            Normalize3D(mean=0.0, std=1.0)
        ]

        return Compose(transforms)

    @staticmethod
    def get_val_transforms(target_size: Tuple[int, int, int] = (64, 64, 64)) -> Compose:
        """Get validation transforms (no augmentation)"""
        transforms = [
            # Size transforms only
            CenterCrop3D(output_size=target_size),
            Resize3D(output_size=target_size),

            # Normalization
            Normalize3D(mean=0.0, std=1.0)
        ]

        return Compose(transforms)

    @staticmethod
    def get_test_transforms(target_size: Tuple[int, int, int] = (64, 64, 64)) -> Compose:
        """Get test transforms (minimal processing)"""
        transforms = [
            # Size transforms only
            CenterCrop3D(output_size=target_size),
            Resize3D(output_size=target_size),
        ]

        return Compose(transforms)


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test individual transforms
    test_volume = torch.randn(1, 32, 32, 32).to(device)  # Single channel volume
    print(f"Original volume shape: {test_volume.shape}")

    # Test rotation
    rotate_transform = RandomRotation3D(degrees=15, prob=1.0)
    rotated = rotate_transform(test_volume)
    print(f"After rotation: {rotated.shape}")

    # Test flipping
    flip_transform = RandomFlip3D(axes=[0, 1, 2], prob=1.0)
    flipped = flip_transform(test_volume)
    print(f"After flipping: {flipped.shape}")

    # Test intensity transforms
    scale_transform = RandomIntensityScale(scale_range=(0.9, 1.1), prob=1.0)
    scaled = scale_transform(test_volume)
    print(f"Intensity range before: {test_volume.min():.3f} - {test_volume.max():.3f}")
    print(f"Intensity range after scaling: {scaled.min():.3f} - {scaled.max():.3f}")

    # Test noise
    noise_transform = RandomGaussianNoise(noise_std=0.01, prob=1.0)
    noisy = noise_transform(test_volume)
    print(f"Noise added successfully")

    # Test cropping and resizing
    crop_transform = RandomCrop3D(output_size=(16, 16, 16), prob=1.0)
    cropped = crop_transform(test_volume)
    print(f"After cropping: {cropped.shape}")

    resize_transform = Resize3D(output_size=(64, 64, 64))
    resized = resize_transform(cropped)
    print(f"After resizing: {resized.shape}")

    # Test normalization
    norm_transform = Normalize3D(mean=0.0, std=1.0)
    normalized = norm_transform(test_volume)
    print(f"After normalization - mean: {normalized.mean():.6f}, std: {normalized.std():.6f}")

    # Test composed transforms
    train_transforms = PretrainingTransforms.get_train_transforms(target_size=(64, 64, 64))
    transformed = train_transforms(test_volume)
    print(f"After full training pipeline: {transformed.shape}")

    val_transforms = PretrainingTransforms.get_val_transforms(target_size=(64, 64, 64))
    val_transformed = val_transforms(test_volume)
    print(f"After validation pipeline: {val_transformed.shape}")

    # Test with multi-channel volume (CT)
    ct_volume = torch.randn(3, 32, 32, 32).to(device)  # 3-channel CT
    print(f"\nMulti-channel test:")
    print(f"Original CT volume shape: {ct_volume.shape}")

    ct_transformed = train_transforms(ct_volume)
    print(f"After training pipeline: {ct_transformed.shape}")

    print("\nAll transforms tested successfully!")