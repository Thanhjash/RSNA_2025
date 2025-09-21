"""
MiM: Mask-in-Mask Hierarchical Masking Strategy
Implementation for RSNA 2025 Phase 0 Pre-training

This module implements the MiM (Mask-in-Mask) hierarchical masking strategy:
- Two-level hierarchical masking (global and local)
- Precise indexing and subset relationships
- Support for 3D medical image volumes
- Integration with SparK sparse modeling framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


def generate_hierarchical_mask(
    tensor_shape: tuple,
    global_mask_ratio: float = 0.6,
    local_mask_ratio: float = 0.8,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate hierarchical Mask-in-Mask pattern with precise indexing

    The MiM strategy creates two levels of masking:
    1. Global mask: Coarse-level masking across the entire volume
    2. Local mask: Fine-level masking within the globally masked regions

    Args:
        tensor_shape: Shape tuple (B, C, D, H, W)
        global_mask_ratio: Coarse-level masking ratio (0.6 = 60% masked globally)
        local_mask_ratio: Fine-level masking ratio within global masked regions
        device: Target device for tensors

    Returns:
        global_mask: Boolean tensor [B, D, H, W] (True = masked globally)
        local_mask: Boolean tensor [B, D, H, W] (True = masked locally, subset of global)
    """
    B, C, D, H, W = tensor_shape
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_voxels = D * H * W

    # Level 1: Global mask generation
    num_global_masked = int(num_voxels * global_mask_ratio)

    global_noise = torch.rand(B, num_voxels, device=device)
    ids_global_shuffle = torch.argsort(global_noise, dim=1)  # [B, num_voxels]
    ids_global_mask = ids_global_shuffle[:, :num_global_masked]  # [B, num_global_masked]

    global_mask_flat = torch.zeros(B, num_voxels, dtype=torch.bool, device=device)
    global_mask_flat.scatter_(1, ids_global_mask, True)
    global_mask = global_mask_flat.view(B, D, H, W)

    # Level 2: Local mask within global masked regions
    num_local_masked = int(num_global_masked * local_mask_ratio)

    local_noise = torch.rand(B, num_global_masked, device=device)
    ids_local_shuffle = torch.argsort(local_noise, dim=1)  # [B, num_global_masked]
    ids_local_subset = ids_local_shuffle[:, :num_local_masked]  # [B, num_local_masked]

    ids_local_mask = torch.gather(ids_global_mask, 1, ids_local_subset)  # [B, num_local_masked]

    local_mask_flat = torch.zeros(B, num_voxels, dtype=torch.bool, device=device)
    local_mask_flat.scatter_(1, ids_local_mask, True)
    local_mask = local_mask_flat.view(B, D, H, W)

    # Verification: local_mask must be subset of global_mask
    assert (local_mask & ~global_mask).sum() == 0, "Local mask extends beyond global mask!"

    return global_mask, local_mask


def generate_block_wise_mask(
    tensor_shape: tuple,
    block_size: int = 8,
    global_mask_ratio: float = 0.6,
    local_mask_ratio: float = 0.8,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate hierarchical mask with block-wise masking for better spatial coherence

    Args:
        tensor_shape: Shape tuple (B, C, D, H, W)
        block_size: Size of cubic blocks for masking
        global_mask_ratio: Ratio of blocks to mask globally
        local_mask_ratio: Ratio of globally masked blocks to mask locally
        device: Target device

    Returns:
        global_mask: Boolean tensor [B, D, H, W]
        local_mask: Boolean tensor [B, D, H, W]
    """
    B, C, D, H, W = tensor_shape
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate number of blocks in each dimension
    D_blocks = D // block_size
    H_blocks = H // block_size
    W_blocks = W // block_size
    total_blocks = D_blocks * H_blocks * W_blocks

    # Generate global block mask
    num_global_blocks = int(total_blocks * global_mask_ratio)
    global_block_masks = []

    for b in range(B):
        # Random block selection
        block_indices = torch.randperm(total_blocks, device=device)[:num_global_blocks]

        # Convert to 3D block coordinates
        block_coords = []
        for idx in block_indices:
            d_idx = idx // (H_blocks * W_blocks)
            h_idx = (idx % (H_blocks * W_blocks)) // W_blocks
            w_idx = idx % W_blocks
            block_coords.append((d_idx, h_idx, w_idx))

        # Create full resolution mask
        block_mask = torch.zeros(D, H, W, dtype=torch.bool, device=device)
        for d_idx, h_idx, w_idx in block_coords:
            d_start, d_end = d_idx * block_size, (d_idx + 1) * block_size
            h_start, h_end = h_idx * block_size, (h_idx + 1) * block_size
            w_start, w_end = w_idx * block_size, (w_idx + 1) * block_size
            block_mask[d_start:d_end, h_start:h_end, w_start:w_end] = True

        global_block_masks.append(block_mask)

    global_mask = torch.stack(global_block_masks, dim=0)

    # Generate local mask within global blocks
    num_local_blocks = int(num_global_blocks * local_mask_ratio)
    local_block_masks = []

    for b in range(B):
        # Select subset of global blocks for local masking
        global_block_indices = torch.randperm(num_global_blocks, device=device)[:num_local_blocks]

        # Create local mask (subset of global)
        local_mask_b = torch.zeros_like(global_mask[b])
        # Implementation simplified - could extend to select subset of global blocks
        # For now, use random selection within global mask
        global_voxels = global_mask[b].nonzero(as_tuple=False)
        if len(global_voxels) > 0:
            num_local_voxels = int(len(global_voxels) * local_mask_ratio)
            local_indices = torch.randperm(len(global_voxels), device=device)[:num_local_voxels]
            local_voxels = global_voxels[local_indices]
            local_mask_b[local_voxels[:, 0], local_voxels[:, 1], local_voxels[:, 2]] = True

        local_block_masks.append(local_mask_b)

    local_mask = torch.stack(local_block_masks, dim=0)

    return global_mask, local_mask


def compute_mim_reconstruction_loss(
    predicted_image: torch.Tensor,
    original_image: torch.Tensor,
    global_mask: torch.Tensor,
    local_mask: torch.Tensor,
    global_weight: float = 1.0,
    local_weight: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculate the reconstruction loss for both levels of the MiM strategy

    Args:
        predicted_image: Reconstructed image [B, C, D, H, W]
        original_image: Original target image [B, C, D, H, W]
        global_mask: Global level mask [B, D, H, W] (True = masked)
        local_mask: Local level mask [B, D, H, W] (True = masked)
        global_weight: Weight for global reconstruction loss
        local_weight: Weight for local reconstruction loss

    Returns:
        Total loss and dictionary of individual loss components
    """
    # Expand masks to match image channels
    if global_mask.dim() == 4:  # [B, D, H, W]
        global_mask = global_mask.unsqueeze(1).expand_as(original_image)
    if local_mask.dim() == 4:  # [B, D, H, W]
        local_mask = local_mask.unsqueeze(1).expand_as(original_image)

    # Global-only mask: regions masked globally but not locally
    global_only_mask = global_mask & ~local_mask

    # Compute loss per voxel
    loss_per_voxel = F.mse_loss(predicted_image, original_image, reduction='none')

    # Global reconstruction loss (on globally masked regions only)
    global_loss = (loss_per_voxel * global_only_mask.float()).sum() / (global_only_mask.sum().float() + 1e-8)

    # Local reconstruction loss (on locally masked regions)
    local_loss = (loss_per_voxel * local_mask.float()).sum() / (local_mask.sum().float() + 1e-8)

    # Combined loss
    total_loss = global_weight * global_loss + local_weight * local_loss

    loss_dict = {
        'global_loss': global_loss,
        'local_loss': local_loss,
        'total_reconstruction_loss': total_loss
    }

    return total_loss, loss_dict


class MiMController(nn.Module):
    """
    Controller for MiM hierarchical masking strategy

    This class manages the complete MiM workflow:
    - Hierarchical mask generation with different strategies
    - Loss computation for multi-level reconstruction
    - Integration with sparse modeling frameworks
    """
    def __init__(self,
                 global_mask_ratio: float = 0.6,
                 local_mask_ratio: float = 0.8,
                 masking_strategy: str = 'random',  # 'random' or 'block'
                 block_size: int = 8,
                 global_weight: float = 1.0,
                 local_weight: float = 2.0):  # Higher weight for local details
        super().__init__()
        self.global_mask_ratio = global_mask_ratio
        self.local_mask_ratio = local_mask_ratio
        self.masking_strategy = masking_strategy
        self.block_size = block_size
        self.global_weight = global_weight
        self.local_weight = local_weight

    def generate_masks(self, tensor_shape: tuple, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate hierarchical masks based on strategy"""
        if self.masking_strategy == 'random':
            return generate_hierarchical_mask(
                tensor_shape=tensor_shape,
                global_mask_ratio=self.global_mask_ratio,
                local_mask_ratio=self.local_mask_ratio,
                device=device
            )
        elif self.masking_strategy == 'block':
            return generate_block_wise_mask(
                tensor_shape=tensor_shape,
                block_size=self.block_size,
                global_mask_ratio=self.global_mask_ratio,
                local_mask_ratio=self.local_mask_ratio,
                device=device
            )
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

    def compute_loss(self,
                     predicted_image: torch.Tensor,
                     original_image: torch.Tensor,
                     global_mask: torch.Tensor,
                     local_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MiM reconstruction loss"""
        return compute_mim_reconstruction_loss(
            predicted_image=predicted_image,
            original_image=original_image,
            global_mask=global_mask,
            local_mask=local_mask,
            global_weight=self.global_weight,
            local_weight=self.local_weight
        )

    def forward(self,
                images: torch.Tensor,
                global_mask: torch.Tensor = None,
                local_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Generate masks and prepare for reconstruction

        Args:
            images: Input images [B, C, D, H, W]
            global_mask: Optional pre-computed global mask
            local_mask: Optional pre-computed local mask

        Returns:
            Dictionary with masks and metadata
        """
        device = images.device

        # Generate masks if not provided
        if global_mask is None or local_mask is None:
            global_mask, local_mask = self.generate_masks(images.shape, device)

        # Compute unmasked regions for sparse encoding
        unmasked_mask = ~global_mask  # Regions that should be processed

        return {
            'global_mask': global_mask,
            'local_mask': local_mask,
            'unmasked_mask': unmasked_mask,
            'total_masked_ratio': (global_mask.sum().float() / global_mask.numel()).item(),
            'local_masked_ratio': (local_mask.sum().float() / local_mask.numel()).item()
        }


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test hierarchical mask generation
    test_shape = (2, 1, 32, 32, 32)  # Batch=2, Channels=1, 32^3 volume
    global_mask, local_mask = generate_hierarchical_mask(
        tensor_shape=test_shape,
        global_mask_ratio=0.6,
        local_mask_ratio=0.8,
        device=device
    )

    print(f"Test shape: {test_shape}")
    print(f"Global mask shape: {global_mask.shape}")
    print(f"Local mask shape: {local_mask.shape}")
    print(f"Global masked ratio: {global_mask.sum().item() / global_mask.numel():.3f}")
    print(f"Local masked ratio: {local_mask.sum().item() / local_mask.numel():.3f}")

    # Verify subset relationship
    subset_check = (local_mask & ~global_mask).sum()
    print(f"Local mask extends beyond global: {subset_check.item()} (should be 0)")

    # Test block-wise masking
    global_block, local_block = generate_block_wise_mask(
        tensor_shape=test_shape,
        block_size=8,
        global_mask_ratio=0.6,
        local_mask_ratio=0.8,
        device=device
    )

    print(f"\nBlock-wise masking:")
    print(f"Global block masked ratio: {global_block.sum().item() / global_block.numel():.3f}")
    print(f"Local block masked ratio: {local_block.sum().item() / local_block.numel():.3f}")

    # Test loss computation
    original = torch.randn(2, 1, 32, 32, 32).to(device)
    predicted = torch.randn(2, 1, 32, 32, 32).to(device)

    total_loss, loss_dict = compute_mim_reconstruction_loss(
        predicted_image=predicted,
        original_image=original,
        global_mask=global_mask,
        local_mask=local_mask
    )

    print(f"\nLoss computation:")
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Global loss: {loss_dict['global_loss'].item():.6f}")
    print(f"Local loss: {loss_dict['local_loss'].item():.6f}")

    # Test MiM controller
    controller = MiMController(
        global_mask_ratio=0.6,
        local_mask_ratio=0.8,
        masking_strategy='random'
    )

    mask_results = controller(original)
    print(f"\nMiM Controller results:")
    for key, value in mask_results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    # Test with multi-channel input (CT)
    ct_shape = (2, 3, 32, 32, 32)  # 3-channel CT
    ct_controller = MiMController(masking_strategy='block', block_size=8)
    ct_images = torch.randn(*ct_shape).to(device)
    ct_results = ct_controller(ct_images)

    print(f"\nMulti-channel CT test:")
    print(f"Input shape: {ct_images.shape}")
    print(f"Unmasked mask shape: {ct_results['unmasked_mask'].shape}")
    print(f"Total masked ratio: {ct_results['total_masked_ratio']:.3f}")