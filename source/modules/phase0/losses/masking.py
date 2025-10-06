"""
MiM (Mask-in-Mask) Hierarchical Masking Strategy
Implementation for RSNA 2025 Phase 0 Pre-training

Implements:
- Global masking (60% masking ratio on entire volume)
- Local masking (80% masking ratio within global masked regions)
- Coordinate generation for unmasked/masked voxels
- Integration with SparK sparse representation
- Adaptive block sizing to prevent zero unmasked coordinates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List


def get_adaptive_block_size(spatial_dims: Tuple[int, int, int]) -> int:
    """Calculate safe block size based on spatial dimensions to guarantee unmasked regions."""
    min_dim = min(spatial_dims)
    if min_dim <= 2: return 1
    if min_dim <= 4: return 2
    if min_dim <= 8: return 3
    return 4


class MiMHierarchicalMasking:
    """
    Hierarchical masking strategy following MiM (Mask-in-Mask) approach with adaptive block sizing

    Two-level masking:
    1. Global masking: Randomly mask 60% of all voxels
    2. Local masking: Within globally masked regions, further mask 80%

    This creates three voxel categories:
    - Unmasked: 40% of volume (used for encoding)
    - Global masked: 60% * 20% = 12% (easier reconstruction targets)
    - Local masked: 60% * 80% = 48% (harder reconstruction targets)

    FIXED: Adaptive block_size prevents complete masking of small spatial dimensions
    """

    def __init__(self, global_mask_ratio=0.6, local_mask_ratio=0.8,
                 min_unmasked_blocks=2):
        """
        Args:
            global_mask_ratio: Ratio of globally masked voxels (default 0.6)
            local_mask_ratio: Ratio of locally masked voxels within global mask (default 0.8)
            min_unmasked_blocks: Minimum number of blocks to keep unmasked (default 2)
        """
        self.global_mask_ratio = global_mask_ratio
        self.local_mask_ratio = local_mask_ratio
        self.min_unmasked_blocks = min_unmasked_blocks
        
    def generate_masks(self, feature_shape: Tuple[int, int, int, int, int], 
                       device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Generate hierarchical masks for a batch of features
        
        Args:
            feature_shape: Shape [B, C, D, H, W] of feature map
            device: Device to create masks on
        
        Returns:
            Dictionary containing:
                - 'unmasked_coords': [N, 4] coordinates of unmasked voxels (batch_idx, x, y, z)
                - 'global_masked_coords': [M1, 4] coordinates of globally masked voxels
                - 'local_masked_coords': [M2, 4] coordinates of locally masked voxels
                - 'unmasked_mask': [B, D, H, W] boolean mask of unmasked voxels
                - 'global_mask': [B, D, H, W] boolean mask of global masking
                - 'local_mask': [B, D, H, W] boolean mask of local masking
        """
        B, C, D, H, W = feature_shape
        
        # Generate global mask (60% masked)
        global_mask = self._generate_block_mask(
            batch_size=B,
            spatial_shape=(D, H, W),
            mask_ratio=self.global_mask_ratio,
            device=device
        )  # [B, D, H, W], True = masked

        # Generate local mask (80% of globally masked regions)
        local_mask_base = self._generate_block_mask(
            batch_size=B,
            spatial_shape=(D, H, W),
            mask_ratio=self.local_mask_ratio,
            device=device
        )  # [B, D, H, W]
        
        # Apply local mask only to globally masked regions
        local_mask = global_mask & local_mask_base  # [B, D, H, W]
        
        # Unmasked regions (complement of global mask)
        unmasked_mask = ~global_mask  # [B, D, H, W]
        
        # Generate coordinate lists
        unmasked_coords = self._mask_to_coords(unmasked_mask)  # [N, 4]
        
        # Global masked coords (excluding local masked)
        global_masked_only = global_mask & (~local_mask)
        global_masked_coords = self._mask_to_coords(global_masked_only)  # [M1, 4]
        
        local_masked_coords = self._mask_to_coords(local_mask)  # [M2, 4]
        
        return {
            'unmasked_coords': unmasked_coords,
            'global_masked_coords': global_masked_coords,
            'local_masked_coords': local_masked_coords,
            'unmasked_mask': unmasked_mask,
            'global_mask': global_mask,
            'local_mask': local_mask
        }
    
    def _generate_block_mask(self, batch_size: int, spatial_shape: Tuple[int, int, int],
                            mask_ratio: float, device: torch.device) -> torch.Tensor:
        """
        Generate block-wise random mask with adaptive block sizing

        Args:
            batch_size: Number of samples in batch
            spatial_shape: (D, H, W) spatial dimensions
            mask_ratio: Ratio of voxels to mask
            device: Device to create mask on

        Returns:
            Boolean mask tensor [B, D, H, W], True = masked
        """
        D, H, W = spatial_shape

        # FIXED: Adaptive block sizing based on spatial dimensions
        block_size = get_adaptive_block_size(spatial_shape)

        # Calculate number of blocks in each dimension
        num_blocks_d = (D + block_size - 1) // block_size
        num_blocks_h = (H + block_size - 1) // block_size
        num_blocks_w = (W + block_size - 1) // block_size
        total_blocks = num_blocks_d * num_blocks_h * num_blocks_w

        # FIXED: Guarantee minimum unmasked blocks
        max_maskable = max(1, total_blocks - self.min_unmasked_blocks)
        num_masked_blocks = min(int(total_blocks * mask_ratio), max_maskable)
        
        # Generate masks for each sample in batch
        masks = []
        for b in range(batch_size):
            # Random block indices to mask
            block_indices = torch.randperm(total_blocks, device=device)[:num_masked_blocks]
            
            # Convert flat indices to 3D block coordinates
            block_coords_d = block_indices // (num_blocks_h * num_blocks_w)
            block_coords_h = (block_indices % (num_blocks_h * num_blocks_w)) // num_blocks_w
            block_coords_w = block_indices % num_blocks_w
            
            # Create full resolution mask
            mask = torch.zeros(D, H, W, dtype=torch.bool, device=device)
            
            for bd, bh, bw in zip(block_coords_d, block_coords_h, block_coords_w):
                # Calculate voxel ranges for this block
                d_start = bd * block_size
                d_end = min((bd + 1) * block_size, D)
                h_start = bh * block_size
                h_end = min((bh + 1) * block_size, H)
                w_start = bw * block_size
                w_end = min((bw + 1) * block_size, W)
                
                # Mask the block
                mask[d_start:d_end, h_start:h_end, w_start:w_end] = True
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)  # [B, D, H, W]
    
    def _mask_to_coords(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Convert boolean mask to coordinate list
        
        Args:
            mask: Boolean mask [B, D, H, W]
        
        Returns:
            Coordinates [N, 4] where each row is (batch_idx, d, h, w)
        """
        # Get indices where mask is True
        batch_indices, d_indices, h_indices, w_indices = torch.where(mask)
        
        # Stack into coordinate array
        coords = torch.stack([
            batch_indices,
            d_indices,
            h_indices,
            w_indices
        ], dim=1)  # [N, 4]
        
        return coords
    
    def visualize_masks(self, mask_dict: Dict[str, torch.Tensor], 
                       slice_idx: int = None) -> Dict[str, np.ndarray]:
        """
        Visualize masks for debugging (returns 2D slices)
        
        Args:
            mask_dict: Dictionary from generate_masks()
            slice_idx: Slice index to visualize (middle slice if None)
        
        Returns:
            Dictionary of 2D numpy arrays for visualization
        """
        unmasked_mask = mask_dict['unmasked_mask'][0].cpu().numpy()  # First batch
        global_mask = mask_dict['global_mask'][0].cpu().numpy()
        local_mask = mask_dict['local_mask'][0].cpu().numpy()
        
        D = unmasked_mask.shape[0]
        if slice_idx is None:
            slice_idx = D // 2
        
        return {
            'unmasked_slice': unmasked_mask[slice_idx].astype(np.float32),
            'global_mask_slice': global_mask[slice_idx].astype(np.float32),
            'local_mask_slice': local_mask[slice_idx].astype(np.float32),
            'combined_slice': (
                unmasked_mask[slice_idx].astype(np.float32) * 0.3 +
                (global_mask[slice_idx] & ~local_mask[slice_idx]).astype(np.float32) * 0.6 +
                local_mask[slice_idx].astype(np.float32) * 1.0
            )
        }


class MaskingScheduler:
    """
    Dynamic masking ratio scheduler for curriculum learning
    
    Gradually increases masking difficulty during training:
    - Early training: Lower masking ratios (easier)
    - Late training: Higher masking ratios (harder)
    """
    
    def __init__(self, warmup_epochs=10, max_epochs=100,
                 initial_global_ratio=0.4, final_global_ratio=0.6,
                 initial_local_ratio=0.6, final_local_ratio=0.8):
        """
        Args:
            warmup_epochs: Number of epochs for warmup
            max_epochs: Total training epochs
            initial_global_ratio: Starting global mask ratio
            final_global_ratio: Final global mask ratio
            initial_local_ratio: Starting local mask ratio
            final_local_ratio: Final local mask ratio
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.initial_global_ratio = initial_global_ratio
        self.final_global_ratio = final_global_ratio
        self.initial_local_ratio = initial_local_ratio
        self.final_local_ratio = final_local_ratio
    
    def get_mask_ratios(self, current_epoch: int) -> Tuple[float, float]:
        """
        Get current mask ratios based on training progress
        
        Args:
            current_epoch: Current training epoch (0-indexed)
        
        Returns:
            Tuple of (global_mask_ratio, local_mask_ratio)
        """
        if current_epoch < self.warmup_epochs:
            # Linear warmup
            progress = current_epoch / self.warmup_epochs
            global_ratio = self.initial_global_ratio + progress * (self.final_global_ratio - self.initial_global_ratio)
            local_ratio = self.initial_local_ratio + progress * (self.final_local_ratio - self.initial_local_ratio)
        else:
            # Constant at final values
            global_ratio = self.final_global_ratio
            local_ratio = self.final_local_ratio
        
        return global_ratio, local_ratio

