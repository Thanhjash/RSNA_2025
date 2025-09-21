"""
Cross-Level Alignment Loss with InfoNCE
Implementation for RSNA 2025 Phase 0 Pre-training

This module implements contrastive learning for cross-scale semantic consistency:
- InfoNCE contrastive loss between different scale features
- Projection heads for contrastive learning
- Support for both dense and sparse feature representations
- Integration with WaveFormer and SparK intermediate features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from typing import Tuple, List, Dict, Union, Optional
import math


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning

    Projects high-dimensional features to a lower-dimensional space
    optimized for contrastive learning with InfoNCE loss
    """
    def __init__(self,
                 input_dim: int,
                 projection_dim: int = 128,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2

        layers = []

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ])

        # Additional hidden layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])

        # Final projection layer
        layers.extend([
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        ])

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, feature_dim] or [N, feature_dim]
        Returns:
            Projected features [B, projection_dim] or [N, projection_dim]
        """
        return self.projection(x)


def info_nce_loss(query: torch.Tensor,
                  positive_key: torch.Tensor,
                  negative_keys: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss implementation

    Args:
        query: Query representations [B, D]
        positive_key: Positive key representations [B, D]
        negative_keys: Negative key representations [B, N, D] or [N, D]
        temperature: Temperature parameter for softmax

    Returns:
        InfoNCE loss scalar
    """
    # L2 normalize all representations
    query = F.normalize(query, p=2, dim=-1)
    positive_key = F.normalize(positive_key, p=2, dim=-1)

    if negative_keys.dim() == 3:  # [B, N, D]
        negative_keys = F.normalize(negative_keys, p=2, dim=-1)
    else:  # [N, D]
        negative_keys = F.normalize(negative_keys, p=2, dim=-1)

    # Compute positive similarities
    l_pos = torch.sum(query * positive_key, dim=-1, keepdim=True)  # [B, 1]

    # Compute negative similarities
    if negative_keys.dim() == 3:  # [B, N, D]
        l_neg = torch.bmm(query.unsqueeze(1), negative_keys.transpose(1, 2)).squeeze(1)  # [B, N]
    else:  # [N, D] - broadcast across batch
        l_neg = torch.mm(query, negative_keys.T)  # [B, N]

    # Combine positive and negative similarities
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature  # [B, 1+N]

    # Labels: positive is always the first (index 0)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)

    # Cross-entropy loss
    return F.cross_entropy(logits, labels)


def extract_sparse_features(sparse_tensor: ME.SparseTensor, batch_size: int) -> torch.Tensor:
    """
    Extract and pool features from sparse tensor for contrastive learning

    Args:
        sparse_tensor: MinkowskiEngine SparseTensor
        batch_size: Number of items in batch

    Returns:
        Pooled features [B, feature_dim]
    """
    device = sparse_tensor.device
    feature_dim = sparse_tensor.F.shape[1]

    # Pool features by batch
    pooled_features = []
    for b in range(batch_size):
        batch_mask = sparse_tensor.C[:, 0] == b
        if batch_mask.sum() > 0:
            # Use mean pooling across spatial locations
            batch_features = sparse_tensor.F[batch_mask].mean(0)
        else:
            # Handle empty case
            batch_features = torch.zeros(feature_dim, device=device)
        pooled_features.append(batch_features)

    return torch.stack(pooled_features, dim=0)


def extract_dense_features(dense_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract and pool features from dense tensor for contrastive learning

    Args:
        dense_tensor: Dense feature tensor [B, C, D, H, W]

    Returns:
        Pooled features [B, C]
    """
    # Global average pooling across spatial dimensions
    return dense_tensor.flatten(2).mean(dim=2)  # [B, C]


class CrossLevelAlignmentLoss(nn.Module):
    """
    InfoNCE contrastive loss for cross-scale semantic consistency

    This loss encourages features from different scales/levels to maintain
    semantic consistency, helping the model learn hierarchical representations
    """
    def __init__(self,
                 feature_dim: int = 768,
                 projection_dim: int = 128,
                 temperature: float = 0.07,
                 num_negatives: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.num_negatives = num_negatives

        # Projection heads for different feature types
        self.level1_projector = ProjectionHead(feature_dim, projection_dim)
        self.level2_projector = ProjectionHead(feature_dim, projection_dim)

    def forward(self,
                features_level1: Union[torch.Tensor, ME.SparseTensor],
                features_level2: Union[torch.Tensor, ME.SparseTensor],
                batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Compute cross-level alignment loss between two feature representations

        Args:
            features_level1: Features from first level/scale
            features_level2: Features from second level/scale
            batch_size: Batch size (required for sparse tensors)

        Returns:
            Contrastive alignment loss
        """
        device = features_level1.device if hasattr(features_level1, 'device') else features_level1.F.device

        # Extract features based on type
        if isinstance(features_level1, ME.SparseTensor):
            if batch_size is None:
                batch_size = features_level1.C[:, 0].max().item() + 1
            feat1_pooled = extract_sparse_features(features_level1, batch_size)
        else:
            feat1_pooled = extract_dense_features(features_level1)
            batch_size = feat1_pooled.shape[0]

        if isinstance(features_level2, ME.SparseTensor):
            feat2_pooled = extract_sparse_features(features_level2, batch_size)
        else:
            feat2_pooled = extract_dense_features(features_level2)

        # Project to contrastive space
        level1_proj = self.level1_projector(feat1_pooled)  # [B, projection_dim]
        level2_proj = self.level2_projector(feat2_pooled)  # [B, projection_dim]

        # Prepare for InfoNCE
        query = level1_proj
        positive_key = level2_proj

        # Use all other samples in the batch as negatives
        # Create negative keys by shifting positive keys
        if batch_size > 1:
            # Create negatives by using all other samples in batch
            negative_indices = []
            for i in range(batch_size):
                # All other samples except current one
                neg_idx = list(range(batch_size))
                neg_idx.remove(i)
                negative_indices.append(neg_idx)

            # Stack negatives for each query
            negative_keys = []
            for i in range(batch_size):
                if len(negative_indices[i]) > 0:
                    neg_keys = level2_proj[negative_indices[i]]  # [batch_size-1, projection_dim]

                    # Sample random negatives if we have too many
                    if len(neg_keys) > self.num_negatives:
                        rand_idx = torch.randperm(len(neg_keys))[:self.num_negatives]
                        neg_keys = neg_keys[rand_idx]

                    negative_keys.append(neg_keys)
                else:
                    # Handle single batch case
                    negative_keys.append(torch.zeros(1, self.projection_dim, device=device))

            # Pad to same length and stack
            max_negatives = max(len(nk) for nk in negative_keys)
            padded_negatives = []
            for neg_keys in negative_keys:
                if len(neg_keys) < max_negatives:
                    # Pad with zeros
                    padding = torch.zeros(max_negatives - len(neg_keys), self.projection_dim, device=device)
                    neg_keys = torch.cat([neg_keys, padding], dim=0)
                padded_negatives.append(neg_keys)

            negative_keys_tensor = torch.stack(padded_negatives, dim=0)  # [B, max_negatives, projection_dim]
        else:
            # Single batch case - create dummy negatives
            negative_keys_tensor = torch.zeros(1, 1, self.projection_dim, device=device)

        # Compute InfoNCE loss
        contrastive_loss = info_nce_loss(query, positive_key, negative_keys_tensor, self.temperature)

        return contrastive_loss


class MultiLevelContrastiveLoss(nn.Module):
    """
    Multi-level contrastive loss for hierarchical feature learning

    Computes contrastive losses between multiple pairs of feature levels
    to encourage consistent hierarchical representations
    """
    def __init__(self,
                 feature_dims: List[int],
                 projection_dim: int = 128,
                 temperature: float = 0.07,
                 level_weights: Optional[List[float]] = None):
        super().__init__()
        self.feature_dims = feature_dims
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.num_levels = len(feature_dims)

        # Default equal weights for all level pairs
        if level_weights is None:
            level_weights = [1.0] * (self.num_levels - 1)
        self.level_weights = level_weights

        # Create alignment losses for adjacent levels
        self.alignment_losses = nn.ModuleList()
        for i in range(self.num_levels - 1):
            loss_module = CrossLevelAlignmentLoss(
                feature_dim=feature_dims[i],  # Use first level's dimension
                projection_dim=projection_dim,
                temperature=temperature
            )
            # Update second projector for different dimension if needed
            if feature_dims[i] != feature_dims[i + 1]:
                loss_module.level2_projector = ProjectionHead(feature_dims[i + 1], projection_dim)

            self.alignment_losses.append(loss_module)

    def forward(self,
                feature_list: List[Union[torch.Tensor, ME.SparseTensor]],
                batch_size: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-level contrastive loss

        Args:
            feature_list: List of features from different levels
            batch_size: Batch size for sparse tensors

        Returns:
            Total loss and dictionary of individual level losses
        """
        if len(feature_list) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature levels, got {len(feature_list)}")

        total_loss = 0.0
        loss_dict = {}

        # Compute contrastive loss between adjacent levels
        for i in range(self.num_levels - 1):
            level_loss = self.alignment_losses[i](
                feature_list[i],
                feature_list[i + 1],
                batch_size
            )

            weighted_loss = self.level_weights[i] * level_loss
            total_loss += weighted_loss

            loss_dict[f'level_{i}_to_{i+1}_loss'] = level_loss
            loss_dict[f'level_{i}_to_{i+1}_weighted'] = weighted_loss

        loss_dict['total_contrastive_loss'] = total_loss
        return total_loss, loss_dict


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test projection head
    proj_head = ProjectionHead(input_dim=768, projection_dim=128).to(device)
    test_features = torch.randn(4, 768).to(device)
    projected = proj_head(test_features)
    print(f"Projection: {test_features.shape} -> {projected.shape}")

    # Test InfoNCE loss
    batch_size = 4
    projection_dim = 128

    query = torch.randn(batch_size, projection_dim).to(device)
    positive = torch.randn(batch_size, projection_dim).to(device)
    negatives = torch.randn(batch_size, 10, projection_dim).to(device)

    loss = info_nce_loss(query, positive, negatives)
    print(f"InfoNCE loss: {loss.item():.6f}")

    # Test cross-level alignment loss
    alignment_loss = CrossLevelAlignmentLoss(feature_dim=256, projection_dim=128).to(device)

    # Test with dense features
    dense_feat1 = torch.randn(2, 256, 8, 8, 8).to(device)
    dense_feat2 = torch.randn(2, 256, 4, 4, 4).to(device)

    dense_loss = alignment_loss(dense_feat1, dense_feat2)
    print(f"Dense cross-level loss: {dense_loss.item():.6f}")

    # Test multi-level contrastive loss
    feature_dims = [128, 256, 512]
    multi_loss = MultiLevelContrastiveLoss(
        feature_dims=feature_dims,
        projection_dim=128,
        level_weights=[1.0, 0.5]
    ).to(device)

    # Create test features of different dimensions
    test_features = [
        torch.randn(2, 128, 16, 16, 16).to(device),
        torch.randn(2, 256, 8, 8, 8).to(device),
        torch.randn(2, 512, 4, 4, 4).to(device)
    ]

    total_loss, loss_breakdown = multi_loss(test_features)
    print(f"Multi-level contrastive loss: {total_loss.item():.6f}")
    for key, value in loss_breakdown.items():
        print(f"  {key}: {value.item():.6f}")

    # Count parameters
    total_params = sum(p.numel() for p in alignment_loss.parameters())
    print(f"Cross-level alignment loss parameters: {total_params:,}")

    multi_params = sum(p.numel() for p in multi_loss.parameters())
    print(f"Multi-level contrastive loss parameters: {multi_params:,}")