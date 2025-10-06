"""
Complete WaveFormer + SparK + MiM Pre-trainer
Implementation for RSNA 2025 Phase 0

Integrates:
- WaveFormer3D encoder for wavelet-based feature extraction
- SparK sparse encoder-decoder for masked modeling with MinkowskiEngine
- MiM hierarchical masking strategy
- Cross-level contrastive learning

REQUIREMENTS:
- MinkowskiEngine must be installed (cuda12-installation branch)
- No fallback - pure SparK sparse operations only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from .waveformer import WaveFormer3D
from .spark_encoder import SparKEncoderDecoder
from ..losses.masking import MiMHierarchicalMasking
from ..losses.contrastive import CrossLevelContrastiveLoss


class WaveFormerSparKMiMPretrainer(nn.Module):
    """
    Complete pre-training model combining WaveFormer, SparK, and MiM

    Architecture flow:
    1. Input: 3D medical image [B, C, D, H, W]
    2. WaveFormer encoder: Extract hierarchical features
    3. MiM masking: Generate unmasked/masked coordinates
    4. SparK encoder: Process unmasked features sparsely
    5. SparK decoder: Reconstruct masked features
    6. Loss computation:
       - Reconstruction loss (MSE between predicted and target masked features)
       - Contrastive loss (cross-level alignment)
    """

    def __init__(self, img_size=(64, 64, 64), in_channels=1, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., wavelet='db1',
                 spark_base_channels=96, spark_stages=4,
                 global_mask_ratio=0.6, local_mask_ratio=0.8,
                 contrastive_temperature=0.07, contrastive_weight=0.1):
        """
        Args:
            img_size: Input image size (D, H, W)
            in_channels: Number of input channels (1 for MRI, 3 for CT)
            embed_dim: WaveFormer embedding dimension
            depth: Number of WaveFormer encoder blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            wavelet: Wavelet type for WaveFormer
            spark_base_channels: Base channels for SparK encoder
            spark_stages: Number of SparK encoder stages
            global_mask_ratio: Global masking ratio (MiM)
            local_mask_ratio: Local masking ratio (MiM)
            contrastive_temperature: Temperature for contrastive loss
            contrastive_weight: Weight for contrastive loss
        """
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.contrastive_weight = contrastive_weight

        # WaveFormer encoder
        self.waveformer = WaveFormer3D(
            img_size=img_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            wavelet=wavelet
        )

        # SparK sparse encoder-decoder (requires MinkowskiEngine)
        self.spark = SparKEncoderDecoder(
            waveformer_dim=embed_dim,
            base_channels=spark_base_channels,
            num_stages=spark_stages,
            use_skip_connections=True
        )

        # FIXED: MiM masking with adaptive block sizing
        self.masking = MiMHierarchicalMasking(
            global_mask_ratio=global_mask_ratio,
            local_mask_ratio=local_mask_ratio,
            min_unmasked_blocks=2
        )

        # FIXED: Cross-level contrastive learning with spatial awareness
        # Feature dimensions from WaveFormer intermediate layers
        feature_dims = [embed_dim] * 3  # Three intermediate features
        self.contrastive_loss_fn = CrossLevelContrastiveLoss(
            feature_dims=feature_dims,
            projection_dim=128,
            temperature=contrastive_temperature,
            num_samples=256
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with self-supervised learning

        Args:
            x: Input image [B, C, D, H, W]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = x.device

        # 1. WaveFormer encoding with intermediate features
        waveformer_features, intermediate_features = self.waveformer(x, return_intermediate=True)
        # waveformer_features: [B, embed_dim, D', H', W']

        # 2. Generate MiM masks
        mask_dict = self.masking.generate_masks(
            feature_shape=waveformer_features.shape,
            device=device
        )

        B, C, D, H, W = waveformer_features.shape

        # SparK sparse path with MinkowskiEngine
        unmasked_coords = mask_dict['unmasked_coords']
        masked_coords = torch.cat([
            mask_dict['global_masked_coords'],
            mask_dict['local_masked_coords']
        ], dim=0)

        # 3. SparK encoding and decoding
        reconstructed_masked, encoded_sparse = self.spark(
            waveformer_features,
            unmasked_coords,
            masked_coords
        )

        # 4. Extract ground truth features for masked voxels
        batch_indices = masked_coords[:, 0].long()
        spatial_coords = masked_coords[:, 1:].long()

        target_masked = waveformer_features[
            batch_indices,
            :,
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            spatial_coords[:, 2]
        ]  # [M, embed_dim]

        # 5. Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed_masked, target_masked)

        # FIXED: 6. Spatial contrastive loss with unmasked mask
        # Expand unmasked_mask to [B, 1, D, H, W] for contrastive loss
        unmasked_mask_expanded = mask_dict['unmasked_mask'].unsqueeze(1)  # [B, 1, D, H, W]
        contrastive_loss, contrastive_dict = self.contrastive_loss_fn(
            intermediate_features,
            unmasked_mask=unmasked_mask_expanded
        )

        # 7. Total loss
        total_loss = reconstruction_loss + self.contrastive_weight * contrastive_loss

        # Collect losses
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            **contrastive_dict
        }

        return total_loss, loss_dict

    def get_encoder(self) -> nn.Module:
        """
        Get WaveFormer encoder for downstream tasks

        Returns:
            WaveFormer encoder module
        """
        return self.waveformer


class MultiModalPretrainer(nn.Module):
    """
    Multi-modal pre-trainer supporting both MRI (1-channel) and CT (3-channel)

    Features:
    - Separate input projections for different modalities
    - Shared WaveFormer backbone
    - Modality-specific batch handling
    """

    def __init__(self, img_size=(64, 64, 64), embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., **kwargs):
        """
        Args:
            img_size: Input image size
            embed_dim: Embedding dimension
            depth: Number of encoder blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            **kwargs: Additional arguments for WaveFormerSparKMiMPretrainer
        """
        super().__init__()

        # MRI pretrainer (1-channel)
        self.mri_pretrainer = WaveFormerSparKMiMPretrainer(
            img_size=img_size,
            in_channels=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            **kwargs
        )

        # CT pretrainer (3-channel)
        self.ct_pretrainer = WaveFormerSparKMiMPretrainer(
            img_size=img_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            **kwargs
        )

    def forward(self, x: torch.Tensor, modality: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with modality selection

        Args:
            x: Input image [B, C, D, H, W]
            modality: 'mri' or 'ct'

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if modality == 'mri':
            return self.mri_pretrainer(x)
        elif modality == 'ct':
            return self.ct_pretrainer(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def get_encoder(self, modality: str) -> nn.Module:
        """
        Get encoder for specific modality

        Args:
            modality: 'mri' or 'ct'

        Returns:
            WaveFormer encoder
        """
        if modality == 'mri':
            return self.mri_pretrainer.get_encoder()
        elif modality == 'ct':
            return self.ct_pretrainer.get_encoder()
        else:
            raise ValueError(f"Unknown modality: {modality}")
