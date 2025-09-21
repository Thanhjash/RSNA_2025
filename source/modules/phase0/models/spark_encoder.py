"""
SparK: Sparse Encoder Framework using MinkowskiEngine
Implementation for RSNA 2025 Phase 0 Pre-training

This module implements the SparK framework for efficient sparse masked modeling:
- Sparse encoder using MinkowskiEngine for memory-efficient processing
- Optimized decoder with dense conversion
- Dense-to-sparse and sparse-to-dense utilities
- Integration with hierarchical masking strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from typing import Tuple, List, Dict, Union
import numpy as np


class SparKEncoder(nn.Module):
    """
    Sparse Encoder using MinkowskiEngine for efficient masked modeling

    Features:
    - Progressive channel expansion through sparse convolutions
    - Memory-efficient processing of only unmasked voxels
    - Intermediate feature extraction for contrastive learning
    - Batch normalization and activation functions
    """
    def __init__(self, in_channels=1, feature_dim=768, num_layers=4, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # Progressive channel expansion: 1 -> 64 -> 128 -> 256 -> 768
        channels = [in_channels, base_channels, base_channels*2, base_channels*4, feature_dim]

        # Sparse convolution layers with progressive channel expansion
        self.sparse_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            stride = 2 if i > 0 else 1  # First layer stride=1, others stride=2 for downsampling
            self.sparse_convs.append(
                ME.MinkowskiConvolution(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=3,
                    stride=stride,
                    dimension=3
                )
            )
            self.norms.append(ME.MinkowskiBatchNorm(channels[i+1]))

        # Activation function
        self.relu = ME.MinkowskiReLU()

    def forward(self, sparse_tensor: ME.SparseTensor) -> Tuple[ME.SparseTensor, List[ME.SparseTensor]]:
        """
        Forward pass through sparse encoder

        Args:
            sparse_tensor: Input sparse tensor with unmasked voxels

        Returns:
            Tuple of (final_features, intermediate_features_list)
        """
        # Store intermediate features for contrastive learning
        intermediate_features = []

        x = sparse_tensor

        # Apply sparse convolution layers
        for i, (conv, norm) in enumerate(zip(self.sparse_convs, self.norms)):
            x = self.relu(norm(conv(x)))

            # Store intermediate features (except the last one, which is returned separately)
            if i < len(self.sparse_convs) - 1:
                intermediate_features.append(x)

        return x, intermediate_features


class SparKDecoder(nn.Module):
    """
    Optimized decoder using MinkowskiEngine's dense conversion

    Features:
    - Simple MLP projection from sparse features to pixel space
    - Efficient dense conversion using MinkowskiEngine's optimized .dense() method
    - Support for multi-channel output (1-channel MRI, 3-channel CT)
    """
    def __init__(self, feature_dim=768, out_channels=1, hidden_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_channels = out_channels

        if hidden_dim is None:
            hidden_dim = feature_dim // 4

        # Simple MLP to project features back to pixel space
        self.reconstruction_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, sparse_features: ME.SparseTensor, original_shape: tuple) -> torch.Tensor:
        """
        Convert sparse features back to dense reconstruction

        Args:
            sparse_features: Output from sparse encoder
            original_shape: Shape of original image tensor (B, C, D, H, W)

        Returns:
            Dense reconstructed tensor [B, C, D, H, W]
        """
        # Project sparse features to output channels
        projected_features = self.reconstruction_head(sparse_features.F)

        # Create new SparseTensor with projected features
        recon_sparse_tensor = ME.SparseTensor(
            features=projected_features,
            coordinate_map_key=sparse_features.coordinate_map_key,
            coordinate_manager=sparse_features.coordinate_manager
        )

        # Convert to dense using optimized .dense() method
        dense_reconstruction = recon_sparse_tensor.dense(shape=original_shape)[0]
        return dense_reconstruction


def generate_random_mask(image_tensor: torch.Tensor, mask_ratio: float = 0.75) -> torch.Tensor:
    """
    Generate random masking pattern for 3D volumes

    Args:
        image_tensor: Input tensor [B, C, D, H, W]
        mask_ratio: Fraction of voxels to mask (0.75 = 75% masked)

    Returns:
        Boolean mask tensor [B, 1, D, H, W] (True = keep, False = mask)
    """
    B, C, D, H, W = image_tensor.shape
    total_voxels = D * H * W
    num_keep = int(total_voxels * (1 - mask_ratio))

    masks = []
    for b in range(B):
        # Generate random indices to keep
        indices = torch.randperm(total_voxels, device=image_tensor.device)[:num_keep]
        mask = torch.zeros(total_voxels, dtype=torch.bool, device=image_tensor.device)
        mask[indices] = True
        mask = mask.view(D, H, W)
        masks.append(mask)

    return torch.stack(masks, dim=0).unsqueeze(1)  # [B, 1, D, H, W]


def dense_to_sparse(image_tensor: torch.Tensor, mask: torch.Tensor) -> ME.SparseTensor:
    """
    Convert dense masked tensor to MinkowskiEngine sparse format

    Args:
        image_tensor: Dense input tensor [B, C, D, H, W]
        mask: Boolean mask [B, C, D, H, W] or [B, 1, D, H, W] (True = keep)

    Returns:
        MinkowskiEngine SparseTensor containing only unmasked voxels
    """
    batch_coords_list = []
    batch_features_list = []

    # Handle both [B, C, D, H, W] and [B, 1, D, H, W] mask formats
    if mask.shape[1] == 1 and image_tensor.shape[1] > 1:
        mask = mask.expand_as(image_tensor)

    for b in range(image_tensor.shape[0]):
        # Get coordinates of unmasked voxels (using first channel if multi-channel)
        unmasked_coords = torch.nonzero(mask[b, 0], as_tuple=False)  # [N, 3]

        if len(unmasked_coords) == 0:
            continue

        # Create batch coordinates [batch_idx, z, y, x]
        batch_coords = torch.cat([
            torch.full((unmasked_coords.shape[0], 1), b, dtype=torch.int32, device=image_tensor.device),
            unmasked_coords.int()
        ], dim=1)  # [N, 4]

        # Extract features at unmasked locations
        features = image_tensor[b, :, mask[b, 0]].T  # [N, C]

        batch_coords_list.append(batch_coords)
        batch_features_list.append(features)

    # Handle empty case
    if len(batch_coords_list) == 0:
        all_coords = torch.zeros((0, 4), dtype=torch.int32, device=image_tensor.device)
        all_features = torch.zeros((0, image_tensor.shape[1]), dtype=torch.float32, device=image_tensor.device)
    else:
        all_coords = torch.cat(batch_coords_list, dim=0)
        all_features = torch.cat(batch_features_list, dim=0)

    # Create sparse tensor
    sparse_tensor = ME.SparseTensor(
        features=all_features.float(),
        coordinates=all_coords,
        device=image_tensor.device
    )

    return sparse_tensor


def compute_spark_loss(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute SparK reconstruction loss only on masked regions

    Args:
        reconstruction: Reconstructed image [B, C, D, H, W]
        target: Original target image [B, C, D, H, W]
        mask: Boolean mask [B, C, D, H, W] (True = keep, False = masked)

    Returns:
        Scalar loss value
    """
    # Loss computed on masked regions (False in mask)
    masked_regions = ~mask  # True where originally masked

    # Compute MSE loss per voxel
    loss_per_voxel = F.mse_loss(reconstruction, target, reduction='none')

    # Apply mask and normalize by number of masked voxels
    masked_loss = (loss_per_voxel * masked_regions.float()).sum() / (masked_regions.sum().float() + 1e-8)
    return masked_loss


class SparKTrainer(nn.Module):
    """
    Complete SparK training module integrating encoder and decoder

    This class manages the complete sparse masked modeling workflow:
    - Random mask generation
    - Dense-to-sparse conversion
    - Sparse encoding
    - Dense reconstruction
    - Loss computation
    """
    def __init__(self, in_channels=1, feature_dim=768, mask_ratio=0.75):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.mask_ratio = mask_ratio

        # Components
        self.encoder = SparKEncoder(in_channels=in_channels, feature_dim=feature_dim)
        self.decoder = SparKDecoder(feature_dim=feature_dim, out_channels=in_channels)

    def forward(self, images: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Complete SparK forward pass

        Args:
            images: Input images [B, C, D, H, W]
            mask: Optional pre-computed mask, if None will generate random mask

        Returns:
            Dictionary containing losses and outputs
        """
        B, C, D, H, W = images.shape
        device = images.device

        # Generate mask if not provided
        if mask is None:
            mask = generate_random_mask(images, mask_ratio=self.mask_ratio)

        # Expand mask to match image channels if needed
        if mask.shape[1] == 1 and C > 1:
            mask = mask.expand_as(images)

        # Convert to sparse representation (only unmasked voxels)
        sparse_input = dense_to_sparse(images, mask)

        # Forward pass through sparse encoder
        encoded_features, intermediate_features = self.encoder(sparse_input)

        # Decode back to dense representation
        reconstruction = self.decoder(encoded_features, images.shape)

        # Compute reconstruction loss on masked regions only
        reconstruction_loss = compute_spark_loss(reconstruction, images, mask)

        return {
            'reconstruction_loss': reconstruction_loss,
            'reconstruction': reconstruction,
            'mask': mask,
            'sparse_features': encoded_features,
            'intermediate_features': intermediate_features
        }


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test dense-to-sparse conversion
    test_image = torch.randn(2, 1, 32, 32, 32).to(device)
    test_mask = generate_random_mask(test_image, mask_ratio=0.75)

    print(f"Original image shape: {test_image.shape}")
    print(f"Mask shape: {test_mask.shape}")
    print(f"Unmasked voxels: {test_mask.sum().item()} / {test_mask.numel()}")

    # Convert to sparse
    sparse_tensor = dense_to_sparse(test_image, test_mask)
    print(f"Sparse tensor features shape: {sparse_tensor.F.shape}")
    print(f"Sparse tensor coordinates shape: {sparse_tensor.C.shape}")

    # Test SparK components
    encoder = SparKEncoder(in_channels=1, feature_dim=256).to(device)
    decoder = SparKDecoder(feature_dim=256, out_channels=1).to(device)

    # Forward pass
    encoded_features, intermediate_features = encoder(sparse_tensor)
    reconstruction = decoder(encoded_features, test_image.shape)

    print(f"Encoded features shape: {encoded_features.F.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")

    # Test complete trainer
    trainer = SparKTrainer(in_channels=1, feature_dim=256, mask_ratio=0.75).to(device)
    results = trainer(test_image)

    print(f"Reconstruction loss: {results['reconstruction_loss'].item():.6f}")

    # Count parameters
    total_params = sum(p.numel() for p in trainer.parameters())
    print(f"Total SparK parameters: {total_params:,}")

    # Test with multi-channel input (DeepLesion CT)
    test_ct = torch.randn(2, 3, 32, 32, 32).to(device)  # 3-channel CT
    ct_trainer = SparKTrainer(in_channels=3, feature_dim=256, mask_ratio=0.75).to(device)
    ct_results = ct_trainer(test_ct)

    print(f"CT reconstruction loss: {ct_results['reconstruction_loss'].item():.6f}")
    print(f"CT reconstruction shape: {ct_results['reconstruction'].shape}")