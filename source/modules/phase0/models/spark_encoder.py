"""
SparK: Sparse Masked Encoder-Decoder using MinkowskiEngine
Implementation for RSNA 2025 Phase 0 Pre-training

This module implements:
- Sparse 3D convolution encoder using MinkowskiEngine for memory efficiency
- Lightweight decoder for masked voxel reconstruction
- Integration with MiM (Mask-in-Mask) hierarchical masking strategy
"""

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from typing import Tuple, List, Optional
import numpy as np


class SparKEncoder(nn.Module):
    """
    Sparse encoder using MinkowskiEngine for efficient 3D processing
    
    Key features:
    - Processes ONLY unmasked voxels (60-80% memory reduction)
    - Hierarchical feature extraction with strided convolutions
    - Batch normalization and ReLU activations
    - Compatible with dense WaveFormer features
    """
    def __init__(self, in_channels=768, base_channels=96, num_stages=4):
        """
        Args:
            in_channels: Input feature dimension (from WaveFormer patch embedding)
            base_channels: Base channel dimension (doubled at each stage)
            num_stages: Number of downsampling stages
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        
        # Input projection to match sparse encoder dimension
        self.input_proj = ME.MinkowskiLinear(in_channels, base_channels)
        
        # Build encoder stages
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for stage_idx in range(num_stages):
            stage = nn.ModuleDict({
                'conv1': ME.MinkowskiConvolution(
                    in_channels=current_channels,
                    out_channels=current_channels * 2,
                    kernel_size=3,
                    stride=1,
                    dimension=3
                ),
                'bn1': ME.MinkowskiBatchNorm(current_channels * 2),
                'conv2': ME.MinkowskiConvolution(
                    in_channels=current_channels * 2,
                    out_channels=current_channels * 2,
                    kernel_size=3,
                    stride=2 if stage_idx < num_stages - 1 else 1,  # Downsample except last stage
                    dimension=3
                ),
                'bn2': ME.MinkowskiBatchNorm(current_channels * 2)
            })
            self.stages.append(stage)
            current_channels = current_channels * 2
        
        self.output_channels = current_channels
        
    def forward(self, dense_features: torch.Tensor, unmasked_coords: torch.Tensor) -> Tuple[ME.SparseTensor, List[ME.SparseTensor]]:
        """
        Forward pass converting dense features to sparse representation
        
        Args:
            dense_features: Dense feature tensor [B, C, D, H, W] from WaveFormer
            unmasked_coords: Coordinates of unmasked voxels [N, 4] (batch_idx, x, y, z)
        
        Returns:
            Tuple of (final_sparse_features, intermediate_sparse_features)
        """
        B, C, D, H, W = dense_features.shape
        
        # Extract features at unmasked coordinates
        # Convert to sparse tensor format
        batch_indices = unmasked_coords[:, 0].long()
        spatial_coords = unmasked_coords[:, 1:].long()
        
        # Gather features from dense tensor
        unmasked_features = dense_features[
            batch_indices,
            :,
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            spatial_coords[:, 2]
        ]  # [N, C]
        
        # Create sparse tensor (ensure coordinates are int32)
        coords_int = unmasked_coords.int()  # Convert to int32 for MinkowskiEngine

        sparse_input = ME.SparseTensor(
            features=unmasked_features,
            coordinates=coords_int,
            device=dense_features.device
        )
        
        # Input projection
        x = self.input_proj(sparse_input)
        
        # Apply encoder stages
        intermediate_features = []
        for stage in self.stages:
            x = stage['conv1'](x)
            x = stage['bn1'](x)
            x = ME.MinkowskiReLU()(x)
            
            x = stage['conv2'](x)
            x = stage['bn2'](x)
            x = ME.MinkowskiReLU()(x)
            
            intermediate_features.append(x)
        
        return x, intermediate_features


class SparKDecoder(nn.Module):
    """
    Lightweight decoder for masked voxel reconstruction
    
    Features:
    - Predicts features for MASKED voxels only
    - Upsampling using transposed convolutions
    - Skip connections from encoder (optional)
    - Final projection to reconstruct WaveFormer feature space
    """
    def __init__(self, encoder_channels=768, base_channels=96, num_stages=4, use_skip_connections=True):
        """
        Args:
            encoder_channels: SparK encoder output channels
            base_channels: Base decoder channels
            num_stages: Number of upsampling stages (matches encoder)
            use_skip_connections: Whether to use encoder skip connections
        """
        super().__init__()
        self.encoder_channels = encoder_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.use_skip_connections = use_skip_connections
        
        # Build decoder stages (reverse of encoder)
        self.stages = nn.ModuleList()
        current_channels = encoder_channels
        
        for stage_idx in range(num_stages):
            # Calculate skip connection channels if enabled
            skip_channels = current_channels if use_skip_connections else 0
            
            stage = nn.ModuleDict({
                'upsample': ME.MinkowskiConvolutionTranspose(
                    in_channels=current_channels + skip_channels,
                    out_channels=current_channels // 2,
                    kernel_size=2,
                    stride=2 if stage_idx < num_stages - 1 else 1,
                    dimension=3
                ),
                'bn1': ME.MinkowskiBatchNorm(current_channels // 2),
                'conv': ME.MinkowskiConvolution(
                    in_channels=current_channels // 2,
                    out_channels=current_channels // 2,
                    kernel_size=3,
                    stride=1,
                    dimension=3
                ),
                'bn2': ME.MinkowskiBatchNorm(current_channels // 2)
            })
            self.stages.append(stage)
            current_channels = current_channels // 2
        
        # Final projection to WaveFormer feature dimension (768)
        self.output_proj = ME.MinkowskiLinear(current_channels, 768)
        
    def forward(self, sparse_features: ME.SparseTensor, 
                encoder_features: Optional[List[ME.SparseTensor]] = None,
                masked_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode sparse features to reconstruct masked voxels
        
        Args:
            sparse_features: Encoded sparse features from SparKEncoder
            encoder_features: Intermediate encoder features for skip connections
            masked_coords: Coordinates of masked voxels [M, 4] (batch_idx, x, y, z)
        
        Returns:
            Reconstructed features for masked voxels [M, 768]
        """
        x = sparse_features
        
        # Apply decoder stages
        for stage_idx, stage in enumerate(self.stages):
            # Add skip connection if enabled
            if self.use_skip_connections and encoder_features is not None:
                skip_idx = len(encoder_features) - 1 - stage_idx
                if skip_idx >= 0:
                    skip_features = encoder_features[skip_idx]
                    # Concatenate along feature dimension
                    x = ME.cat(x, skip_features)
            
            # Upsample and process
            x = stage['upsample'](x)
            x = stage['bn1'](x)
            x = ME.MinkowskiReLU()(x)
            
            x = stage['conv'](x)
            x = stage['bn2'](x)
            x = ME.MinkowskiReLU()(x)
        
        # Project to output dimension
        x = self.output_proj(x)
        
        # Extract features at masked coordinates if specified
        if masked_coords is not None:
            # Convert sparse tensor to dense for gathering
            dense_coords = x.C  # [N, 4]
            dense_features = x.F  # [N, 768]
            
            # Find masked coordinates in decoded output
            # This assumes masked_coords are within the decoded space
            # In practice, you may need coordinate mapping
            masked_features = self._gather_at_coords(dense_features, dense_coords, masked_coords)
            return masked_features
        else:
            return x.F  # Return all features
    
    def _gather_at_coords(self, features: torch.Tensor, coords: torch.Tensor, 
                          target_coords: torch.Tensor) -> torch.Tensor:
        """
        Gather features at target coordinates using nearest neighbor
        
        Args:
            features: Feature tensor [N, C]
            coords: Coordinate tensor [N, 4]
            target_coords: Target coordinates [M, 4]
        
        Returns:
            Gathered features [M, C]
        """
        # Simple implementation: find nearest neighbor for each target coordinate
        # For production, consider using KD-tree or hash-based lookup
        
        device = features.device
        gathered_features = []
        
        for target_coord in target_coords:
            # Find matching coordinate (exact match)
            batch_mask = coords[:, 0] == target_coord[0]
            spatial_match = torch.all(coords[:, 1:] == target_coord[1:], dim=1)
            match_mask = batch_mask & spatial_match
            
            if match_mask.any():
                # Exact match found
                gathered_features.append(features[match_mask][0])
            else:
                # No exact match - use nearest neighbor
                batch_coords = coords[batch_mask]
                batch_features = features[batch_mask]
                
                if len(batch_coords) > 0:
                    distances = torch.norm(
                        batch_coords[:, 1:].float() - target_coord[1:].float(), 
                        dim=1
                    )
                    nearest_idx = torch.argmin(distances)
                    gathered_features.append(batch_features[nearest_idx])
                else:
                    # Fallback: zero features
                    gathered_features.append(torch.zeros(features.shape[1], device=device))
        
        return torch.stack(gathered_features)


class SparKEncoderDecoder(nn.Module):
    """
    Complete SparK encoder-decoder for masked modeling
    
    Integrates encoder and decoder with convenience methods for:
    - Processing unmasked voxels
    - Reconstructing masked voxels
    - Computing reconstruction loss
    """
    def __init__(self, waveformer_dim=768, base_channels=96, num_stages=4, use_skip_connections=True):
        super().__init__()
        
        self.encoder = SparKEncoder(
            in_channels=waveformer_dim,
            base_channels=base_channels,
            num_stages=num_stages
        )
        
        self.decoder = SparKDecoder(
            encoder_channels=self.encoder.output_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            use_skip_connections=use_skip_connections
        )
    
    def forward(self, dense_features: torch.Tensor, unmasked_coords: torch.Tensor,
                masked_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete forward pass: encode unmasked -> decode masked
        
        Args:
            dense_features: Dense WaveFormer features [B, C, D, H, W]
            unmasked_coords: Unmasked voxel coordinates [N, 4]
            masked_coords: Masked voxel coordinates [M, 4]
        
        Returns:
            Tuple of (reconstructed_masked_features [M, 768], encoded_sparse_features)
        """
        # Encode unmasked voxels
        encoded_features, intermediate_features = self.encoder(dense_features, unmasked_coords)
        
        # Decode to reconstruct masked voxels
        reconstructed_masked = self.decoder(
            encoded_features, 
            encoder_features=intermediate_features if self.decoder.use_skip_connections else None,
            masked_coords=masked_coords
        )
        
        return reconstructed_masked, encoded_features

