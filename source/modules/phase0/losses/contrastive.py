"""
Cross-Level Contrastive Learning for WaveFormer
Implementation for RSNA 2025 Phase 0 Pre-training

Implements:
- InfoNCE loss for cross-level feature alignment
- Multi-scale feature extraction and pooling
- Temperature-scaled contrastive learning
- Positive/negative pair construction across hierarchical levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning
    
    Formula:
        L = -log[ exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) ]
    
    where:
        - z_i, z_j are positive pairs (same sample, different levels)
        - z_k are negative pairs (different samples)
        - τ is temperature parameter
        - sim is cosine similarity
    """
    
    def __init__(self, temperature=0.07, normalize=True):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            normalize: Whether to L2-normalize features before computing similarity
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, features_1: torch.Tensor, features_2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss between two feature sets
        
        Args:
            features_1: Features from level 1 [B, D]
            features_2: Features from level 2 [B, D]
        
        Returns:
            InfoNCE loss scalar
        """
        batch_size = features_1.shape[0]
        
        # Normalize features
        if self.normalize:
            features_1 = F.normalize(features_1, dim=1)
            features_2 = F.normalize(features_2, dim=1)
        
        # Compute similarity matrix [B, B]
        # similarity[i, j] = cosine_similarity(features_1[i], features_2[j])
        similarity_matrix = torch.matmul(features_1, features_2.T) / self.temperature
        
        # Positive pairs are on the diagonal
        positive_samples = torch.diag(similarity_matrix)  # [B]
        
        # Compute denominator: sum over all samples (including positive)
        # For sample i: Σ_j exp(sim(z_i, z_j) / τ)
        exp_similarities = torch.exp(similarity_matrix)  # [B, B]
        denominator = exp_similarities.sum(dim=1)  # [B]
        
        # InfoNCE loss: -log(exp(positive) / denominator)
        loss = -torch.log(torch.exp(positive_samples) / denominator)
        
        return loss.mean()


class CrossLevelContrastiveLoss(nn.Module):
    """
    FIXED: Spatially-aware contrastive learning across WaveFormer hierarchical levels

    Strategy:
    - Sample unmasked spatial locations from features
    - Enforce consistency at SAME coordinates across encoder depths
    - Use InfoNCE with spatial sampling (not global pooling)
    - Positive pairs: same (d,h,w) location across levels
    - Negative pairs: different locations
    """

    def __init__(self, feature_dims: List[int], projection_dim=128,
                 temperature=0.1, num_samples=256):
        """
        Args:
            feature_dims: List of feature dimensions from each level (should be same for simplicity)
            projection_dim: Dimension of spatial projection
            temperature: Temperature for InfoNCE loss
            num_samples: Number of spatial locations to sample per batch
        """
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.num_samples = num_samples
        self.num_levels = len(feature_dims)

        # FIXED: Spatial-preserving 1x1x1 convolution projector (not Linear)
        # Use same projector for all levels (assume same feature_dim)
        self.projector = nn.Conv3d(feature_dims[0], projection_dim, kernel_size=1, bias=False)

    def forward(self, intermediate_features: List[torch.Tensor],
                unmasked_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        FIXED: Compute spatially-aware cross-level contrastive loss

        Args:
            intermediate_features: List of feature tensors from different levels
                Each tensor shape: [B, C, D, H, W]
            unmasked_mask: Optional boolean mask [B, 1, D, H, W] for sampling locations
                If None, sample from all locations

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if len(intermediate_features) < 2:
            return torch.tensor(0.0, device=intermediate_features[0].device), {}

        # Use first and last intermediate features (coarse and fine)
        features_coarse = intermediate_features[0]
        features_fine = intermediate_features[-1]

        B, C, D, H, W = features_coarse.shape

        # FIXED: Align fine features to coarse resolution via interpolation
        if features_fine.shape[2:] != features_coarse.shape[2:]:
            features_fine_aligned = F.interpolate(
                features_fine, size=(D, H, W), mode='trilinear', align_corners=False
            )
        else:
            features_fine_aligned = features_fine

        # FIXED: Project features while preserving spatial structure
        proj_coarse = F.normalize(self.projector(features_coarse), p=2, dim=1)  # [B, proj_dim, D, H, W]
        proj_fine = F.normalize(self.projector(features_fine_aligned), p=2, dim=1)

        # Sample spatial locations
        if unmasked_mask is not None:
            # Resize mask to match feature resolution if needed
            if unmasked_mask.shape[2:] != (D, H, W):
                unmasked_mask_resized = F.interpolate(
                    unmasked_mask.float(), size=(D, H, W), mode='nearest'
                ).bool()
            else:
                unmasked_mask_resized = unmasked_mask

            # Get unmasked indices [Total_Unmasked, 4] where 4 = (b, 1, d, h, w)
            unmasked_indices = torch.nonzero(unmasked_mask_resized, as_tuple=False)

            if len(unmasked_indices) == 0:
                return torch.tensor(0.0, device=features_coarse.device), {}

            # Sample subset if too many
            if len(unmasked_indices) > self.num_samples:
                sample_idx = torch.randperm(len(unmasked_indices), device=features_coarse.device)[:self.num_samples]
                unmasked_indices = unmasked_indices[sample_idx]

            b_idx, _, d_idx, h_idx, w_idx = unmasked_indices.unbind(dim=1)
        else:
            # Sample random locations
            num_samples = min(self.num_samples, B * D * H * W)
            sample_coords = torch.randint(0, B * D * H * W, (num_samples,), device=features_coarse.device)
            b_idx = sample_coords // (D * H * W)
            spatial_idx = sample_coords % (D * H * W)
            d_idx = spatial_idx // (H * W)
            h_idx = (spatial_idx % (H * W)) // W
            w_idx = spatial_idx % W

        # FIXED: Gather features at SAME spatial locations from both levels
        queries = proj_coarse[b_idx, :, d_idx, h_idx, w_idx]  # [N, proj_dim]
        keys = proj_fine[b_idx, :, d_idx, h_idx, w_idx]       # [N, proj_dim]

        if len(queries) < 2:
            return torch.tensor(0.0, device=features_coarse.device), {}

        # FIXED: InfoNCE loss - positive pairs are diagonal (same location)
        logits = torch.matmul(queries, keys.T) / self.temperature  # [N, N]
        labels = torch.arange(len(queries), device=features_coarse.device)

        loss = F.cross_entropy(logits, labels)

        loss_dict = {
            'contrastive_spatial': loss.item(),
            'num_sampled_locations': len(queries)
        }

        return loss, loss_dict


class SymmetricContrastiveLoss(nn.Module):
    """
    Symmetric cross-level contrastive loss (bidirectional)
    
    Computes InfoNCE in both directions:
    - Level i -> Level j
    - Level j -> Level i
    
    Helps with feature alignment when levels have different capacities
    """
    
    def __init__(self, feature_dims: List[int], projection_dim=256, temperature=0.07):
        super().__init__()
        self.cross_level_loss = CrossLevelContrastiveLoss(
            feature_dims=feature_dims,
            projection_dim=projection_dim,
            temperature=temperature
        )
    
    def forward(self, intermediate_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute symmetric contrastive loss
        
        Args:
            intermediate_features: List of feature tensors [B, C, D, H, W]
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Forward direction
        loss_forward, dict_forward = self.cross_level_loss(intermediate_features)
        
        # Backward direction (reverse feature list)
        loss_backward, dict_backward = self.cross_level_loss(intermediate_features[::-1])
        
        # Combine losses
        total_loss = (loss_forward + loss_backward) / 2.0
        
        # Merge dictionaries
        loss_dict = {**dict_forward}
        for k, v in dict_backward.items():
            loss_dict[f'{k}_reverse'] = v
        loss_dict['symmetric_total'] = total_loss.item()
        
        return total_loss, loss_dict


class MomentumContrastiveLoss(nn.Module):
    """
    Momentum-based contrastive learning with memory bank
    
    Maintains a queue of negative samples from previous batches
    to increase the number of negative pairs without larger batch sizes
    
    Based on MoCo (Momentum Contrast) approach
    """
    
    def __init__(self, feature_dim=256, queue_size=4096, momentum=0.999, temperature=0.07):
        """
        Args:
            feature_dim: Dimension of feature vectors
            queue_size: Size of negative sample queue
            momentum: Momentum for updating key encoder (0.999 typical)
            temperature: Temperature for InfoNCE loss
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        # Initialize queue with random features
        self.register_buffer('queue', torch.randn(feature_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        
        # Queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        Update queue with new keys
        
        Args:
            keys: New key features [B, feature_dim]
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest features in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute momentum contrastive loss
        
        Args:
            queries: Query features [B, feature_dim]
            keys: Key features [B, feature_dim] (from momentum encoder)
        
        Returns:
            MoCo loss scalar
        """
        # Normalize
        queries = F.normalize(queries, dim=1)
        keys = F.normalize(keys, dim=1)
        
        # Positive logits: [B, 1]
        positive_logits = torch.einsum('nc,nc->n', [queries, keys]).unsqueeze(-1)
        
        # Negative logits: [B, queue_size]
        negative_logits = torch.einsum('nc,ck->nk', [queries, self.queue.clone().detach()])
        
        # Concatenate: [B, 1 + queue_size]
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positive is always first (index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self._dequeue_and_enqueue(keys)
        
        return loss

