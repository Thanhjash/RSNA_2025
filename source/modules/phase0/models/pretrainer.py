"""
Complete Pre-training Integration: WaveFormer + SparK + MiM
Implementation for RSNA 2025 Phase 0 Pre-training

This module integrates all components for self-supervised pre-training:
- WaveFormer backbone with wavelet attention
- SparK sparse encoding framework
- MiM hierarchical masking strategy
- Cross-level contrastive learning
- Unified training interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from typing import Dict, List, Tuple, Optional, Union
import logging

from .waveformer import WaveFormer3D
from .spark_encoder import SparKEncoder, SparKDecoder, dense_to_sparse, compute_spark_loss
from ..losses.masking import MiMController, compute_mim_reconstruction_loss
from ..losses.contrastive import CrossLevelAlignmentLoss, MultiLevelContrastiveLoss


class WaveFormerSparKMiMPretrainer(nn.Module):
    """
    Complete integration of WaveFormer backbone with SparK + MiM pre-training

    This is the main model for Phase 0 self-supervised pre-training that combines:
    1. WaveFormer backbone for efficient 3D feature extraction
    2. SparK framework for sparse masked modeling
    3. MiM hierarchical masking for multi-level reconstruction
    4. Cross-level contrastive learning for semantic consistency

    Key Features:
    - Single forward pass paradigm (no distribution shift)
    - Multi-modal support (1-channel MRI, 3-channel CT)
    - Memory-efficient sparse processing
    - Hierarchical feature learning
    """

    def __init__(self,
                 # Model architecture parameters
                 img_size: Tuple[int, int, int] = (64, 64, 64),
                 in_channels: int = 1,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 wavelet: str = 'db1',

                 # Masking parameters
                 global_mask_ratio: float = 0.6,
                 local_mask_ratio: float = 0.8,
                 masking_strategy: str = 'random',  # 'random' or 'block'

                 # Loss weights
                 reconstruction_weight: float = 1.0,
                 contrastive_weight: float = 0.1,
                 global_loss_weight: float = 1.0,
                 local_loss_weight: float = 2.0,

                 # Contrastive learning parameters
                 projection_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight

        # Core components
        self.waveformer_backbone = WaveFormer3D(
            img_size=img_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            wavelet=wavelet
        )

        # SparK components
        self.spark_encoder = SparKEncoder(
            in_channels=in_channels,
            feature_dim=embed_dim,
            num_layers=4
        )

        self.spark_decoder = SparKDecoder(
            feature_dim=embed_dim,
            out_channels=in_channels
        )

        # MiM masking controller
        self.mim_controller = MiMController(
            global_mask_ratio=global_mask_ratio,
            local_mask_ratio=local_mask_ratio,
            masking_strategy=masking_strategy,
            global_weight=global_loss_weight,
            local_weight=local_loss_weight
        )

        # Cross-level contrastive learning
        # We'll use features from different scales for contrastive learning
        self.contrastive_loss = CrossLevelAlignmentLoss(
            feature_dim=embed_dim,
            projection_dim=projection_dim,
            temperature=temperature
        )

        logging.info(f"WaveFormerSparKMiMPretrainer initialized")
        logging.info(f"  Image size: {img_size}")
        logging.info(f"  Input channels: {in_channels}")
        logging.info(f"  Embed dimension: {embed_dim}")
        logging.info(f"  Depth: {depth}")
        logging.info(f"  Global mask ratio: {global_mask_ratio}")
        logging.info(f"  Local mask ratio: {local_mask_ratio}")

    def forward(self, images: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Complete pre-training forward pass

        Args:
            images: Input images [B, C, D, H, W]
            training: Whether in training mode

        Returns:
            Dictionary containing all losses, reconstructions, and intermediate outputs
        """
        B, C, D, H, W = images.shape
        device = images.device

        # === STEP 1: HIERARCHICAL MASKING (MiM) ===
        mask_results = self.mim_controller(images)
        global_mask = mask_results['global_mask']  # [B, D, H, W]
        local_mask = mask_results['local_mask']    # [B, D, H, W]
        unmasked_mask = mask_results['unmasked_mask']  # [B, D, H, W]

        # Expand masks to match image channels
        unmasked_mask_expanded = unmasked_mask.unsqueeze(1).expand_as(images)  # [B, C, D, H, W]

        # === STEP 2: SPARSE ENCODING (SparK) ===
        # Convert to sparse representation (only unmasked voxels)
        sparse_input = dense_to_sparse(images, unmasked_mask_expanded)

        # Forward pass through sparse encoder
        encoded_features, intermediate_sparse_features = self.spark_encoder(sparse_input)

        # === STEP 3: DENSE BACKBONE PROCESSING ===
        # Also process with WaveFormer backbone for comparison and contrastive learning
        # Use unmasked regions only to avoid distribution shift
        masked_images = images * unmasked_mask_expanded.float()

        # WaveFormer expects specific input size, so we may need to resize
        if images.shape[-3:] != self.img_size:
            masked_images_resized = F.interpolate(
                masked_images,
                size=self.img_size,
                mode='trilinear',
                align_corners=False
            )
        else:
            masked_images_resized = masked_images

        waveformer_features, intermediate_dense_features = self.waveformer_backbone(
            masked_images_resized,
            return_intermediate=True
        )

        # === STEP 4: RECONSTRUCTION ===
        # Decode from sparse features
        reconstruction = self.spark_decoder(encoded_features, images.shape)

        # === STEP 5: LOSS COMPUTATION ===
        losses = {}

        # 5.1 MiM Reconstruction Loss
        mim_loss, mim_loss_dict = self.mim_controller.compute_loss(
            predicted_image=reconstruction,
            original_image=images,
            global_mask=global_mask,
            local_mask=local_mask
        )
        losses.update(mim_loss_dict)

        # 5.2 Cross-level Contrastive Loss
        if len(intermediate_sparse_features) > 0 and len(intermediate_dense_features) > 0:
            # Use features from different processing paths
            contrastive_loss = self.contrastive_loss(
                features_level1=intermediate_sparse_features[-1],  # Final sparse features
                features_level2=intermediate_dense_features[-1],   # Final dense features
                batch_size=B
            )
            losses['contrastive_loss'] = contrastive_loss
        else:
            losses['contrastive_loss'] = torch.tensor(0.0, device=device)

        # 5.3 Additional contrastive loss between different sparse levels
        if len(intermediate_sparse_features) >= 2:
            sparse_contrastive_loss = self.contrastive_loss(
                features_level1=intermediate_sparse_features[0],
                features_level2=intermediate_sparse_features[-1],
                batch_size=B
            )
            losses['sparse_contrastive_loss'] = sparse_contrastive_loss
        else:
            losses['sparse_contrastive_loss'] = torch.tensor(0.0, device=device)

        # 5.4 Total Loss
        total_loss = (
            self.reconstruction_weight * mim_loss +
            self.contrastive_weight * losses['contrastive_loss'] +
            0.05 * losses['sparse_contrastive_loss']  # Small weight for sparse-to-sparse contrastive
        )
        losses['total_loss'] = total_loss

        # === STEP 6: RETURN COMPREHENSIVE RESULTS ===
        return {
            # Losses
            **losses,

            # Reconstructions and predictions
            'reconstruction': reconstruction,
            'masked_input': masked_images,

            # Masks
            'global_mask': global_mask,
            'local_mask': local_mask,
            'unmasked_mask': unmasked_mask,

            # Features for analysis
            'sparse_features': encoded_features,
            'dense_features': waveformer_features,
            'intermediate_sparse_features': intermediate_sparse_features,
            'intermediate_dense_features': intermediate_dense_features,

            # Metadata
            'mask_metadata': {
                'total_masked_ratio': mask_results['total_masked_ratio'],
                'local_masked_ratio': mask_results['local_masked_ratio'],
                'num_sparse_voxels': sparse_input.F.shape[0] if sparse_input.F.shape[0] > 0 else 0
            }
        }

    def get_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the WaveFormer backbone (for downstream tasks)

        Args:
            images: Input images [B, C, D, H, W]

        Returns:
            Backbone features [B, embed_dim, D', H', W']
        """
        # Resize if necessary
        if images.shape[-3:] != self.img_size:
            images = F.interpolate(
                images,
                size=self.img_size,
                mode='trilinear',
                align_corners=False
            )

        features = self.waveformer_backbone(images, return_intermediate=False)
        return features

    def freeze_backbone(self):
        """Freeze WaveFormer backbone parameters (for fine-tuning)"""
        for param in self.waveformer_backbone.parameters():
            param.requires_grad = False
        logging.info("WaveFormer backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze WaveFormer backbone parameters"""
        for param in self.waveformer_backbone.parameters():
            param.requires_grad = True
        logging.info("WaveFormer backbone unfrozen")

    def save_backbone(self, path: str):
        """Save only the WaveFormer backbone for downstream use"""
        torch.save(self.waveformer_backbone.state_dict(), path)
        logging.info(f"WaveFormer backbone saved to {path}")

    def load_backbone(self, path: str):
        """Load WaveFormer backbone weights"""
        self.waveformer_backbone.load_state_dict(torch.load(path))
        logging.info(f"WaveFormer backbone loaded from {path}")


class MultiModalPretrainer(nn.Module):
    """
    Multi-modal pretrainer that handles different input channel counts

    This wrapper allows training with mixed batches containing both
    single-channel MRI and multi-channel CT data
    """

    def __init__(self,
                 single_channel_config: Dict,
                 multi_channel_config: Dict,
                 shared_features: bool = True):
        super().__init__()

        # Create separate pretrainers for different channel counts
        self.single_channel_pretrainer = WaveFormerSparKMiMPretrainer(
            in_channels=1,
            **single_channel_config
        )

        self.multi_channel_pretrainer = WaveFormerSparKMiMPretrainer(
            in_channels=3,
            **multi_channel_config
        )

        self.shared_features = shared_features

        # If sharing features, align the embedding dimensions
        if shared_features:
            assert single_channel_config.get('embed_dim', 768) == multi_channel_config.get('embed_dim', 768), \
                "Embedding dimensions must match for shared features"

    def forward(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass handling mixed modality batches

        Args:
            batch_data: Dictionary containing 'single_channel' and/or 'multi_channel' tensors

        Returns:
            Combined loss dictionary
        """
        total_losses = {}
        outputs = {}

        # Process single-channel data (MRI)
        if 'single_channel' in batch_data:
            sc_data = batch_data['single_channel']
            sc_results = self.single_channel_pretrainer(sc_data)

            # Prefix losses to avoid conflicts
            for key, value in sc_results.items():
                if 'loss' in key.lower():
                    total_losses[f'single_channel_{key}'] = value
                outputs[f'single_channel_{key}'] = value

        # Process multi-channel data (CT)
        if 'multi_channel' in batch_data:
            mc_data = batch_data['multi_channel']
            mc_results = self.multi_channel_pretrainer(mc_data)

            # Prefix losses to avoid conflicts
            for key, value in mc_results.items():
                if 'loss' in key.lower():
                    total_losses[f'multi_channel_{key}'] = value
                outputs[f'multi_channel_{key}'] = value

        # Compute combined total loss
        if total_losses:
            combined_loss = sum(loss for loss in total_losses.values() if 'total_loss' in str(loss))
            if isinstance(combined_loss, torch.Tensor):
                outputs['combined_total_loss'] = combined_loss
            else:
                # If no total losses found, sum all losses
                outputs['combined_total_loss'] = sum(total_losses.values())

        outputs.update(total_losses)
        return outputs


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test single-channel pretrainer (MRI)
    print("Testing single-channel pretrainer...")
    single_pretrainer = WaveFormerSparKMiMPretrainer(
        img_size=(32, 32, 32),  # Smaller for testing
        in_channels=1,
        embed_dim=256,  # Smaller for testing
        depth=3,  # Fewer layers for testing
        num_heads=8,
        global_mask_ratio=0.6,
        local_mask_ratio=0.8
    ).to(device)

    # Test input
    test_mri = torch.randn(2, 1, 32, 32, 32).to(device)
    mri_results = single_pretrainer(test_mri)

    print(f"MRI results keys: {list(mri_results.keys())}")
    print(f"Total loss: {mri_results['total_loss'].item():.6f}")
    print(f"Reconstruction loss: {mri_results['total_reconstruction_loss'].item():.6f}")
    print(f"Contrastive loss: {mri_results['contrastive_loss'].item():.6f}")

    # Test multi-channel pretrainer (CT)
    print("\nTesting multi-channel pretrainer...")
    multi_pretrainer = WaveFormerSparKMiMPretrainer(
        img_size=(32, 32, 32),
        in_channels=3,
        embed_dim=256,
        depth=3,
        num_heads=8,
        global_mask_ratio=0.6,
        local_mask_ratio=0.8
    ).to(device)

    test_ct = torch.randn(2, 3, 32, 32, 32).to(device)
    ct_results = multi_pretrainer(test_ct)

    print(f"CT results keys: {list(ct_results.keys())}")
    print(f"Total loss: {ct_results['total_loss'].item():.6f}")

    # Test multi-modal pretrainer
    print("\nTesting multi-modal pretrainer...")
    multimodal_config = {
        'img_size': (32, 32, 32),
        'embed_dim': 256,
        'depth': 3,
        'num_heads': 8
    }

    multimodal_pretrainer = MultiModalPretrainer(
        single_channel_config=multimodal_config,
        multi_channel_config=multimodal_config,
        shared_features=True
    ).to(device)

    # Test mixed batch
    mixed_batch = {
        'single_channel': test_mri,
        'multi_channel': test_ct
    }

    mixed_results = multimodal_pretrainer(mixed_batch)
    print(f"Mixed results keys: {list(mixed_results.keys())}")
    print(f"Combined total loss: {mixed_results['combined_total_loss'].item():.6f}")

    # Count parameters
    single_params = sum(p.numel() for p in single_pretrainer.parameters())
    multi_params = sum(p.numel() for p in multi_pretrainer.parameters())
    mixed_params = sum(p.numel() for p in multimodal_pretrainer.parameters())

    print(f"\nParameter counts:")
    print(f"Single-channel pretrainer: {single_params:,}")
    print(f"Multi-channel pretrainer: {multi_params:,}")
    print(f"Multi-modal pretrainer: {mixed_params:,}")

    # Test backbone extraction
    backbone_features = single_pretrainer.get_backbone_features(test_mri)
    print(f"Backbone features shape: {backbone_features.shape}")

    print("\nPretrainer implementation completed successfully!")