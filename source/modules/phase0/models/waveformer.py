"""
WaveFormer: 3D Wavelet-based Transformer for Medical Image Processing
Implementation for RSNA 2025 Phase 0 Pre-training

This module implements the complete WaveFormer architecture with:
- 3D Discrete Wavelet Transform (DWT) operations using ptwt
- Wavelet Attention Encoder Blocks with identity skip connections for high-frequency
- Complete WaveFormer3D backbone for self-supervised pre-training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt  # PyTorch Wavelet Toolbox
from typing import Tuple, Dict, List, Union
import math


def dwt3d_forward(x: torch.Tensor, wavelet: str = 'db1', level: int = 1) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Performs 3D Discrete Wavelet Transform using ptwt library

    Args:
        x (torch.Tensor): Input tensor of shape [B, C, D, H, W]
        wavelet (str): Wavelet type ('db1', 'haar', 'db4', etc.)
        level (int): Decomposition level (typically 1 for WaveFormer)

    Returns:
        Tuple containing (low_freq, high_freq_levels)
        low_freq: Low-frequency component [B, C, D//2, H//2, W//2]
        high_freq_levels: List of high-frequency detail coefficient dicts
    """
    # ptwt expects [B, D, H, W, C] format conversion
    B, C, D, H, W = x.shape
    x_permuted = x.permute(0, 2, 3, 4, 1)  # -> [B, D, H, W, C]
    x_reshaped = x_permuted.reshape(B, D, H, W * C)

    # Apply 3D DWT using ptwt
    coeffs = ptwt.wavedec3(x_reshaped, wavelet=wavelet, level=level, mode='zero')

    # Unpack coefficients
    low_freq_packed = coeffs[0]  # [B, D//2, H//2, W//2*C]
    high_freq_levels_packed = coeffs[1:]  # List of detail coefficient dicts

    # Reshape low-frequency component back to [B, C, D//2, H//2, W//2]
    low_freq_permuted = low_freq_packed.view(B, D//2, H//2, W//2, C)
    low_freq = low_freq_permuted.permute(0, 4, 1, 2, 3)

    # Process high-frequency detail components
    high_freq_levels = []
    for high_freq_dict_packed in high_freq_levels_packed:
        high_freq_dict = {}
        for key, tensor_packed in high_freq_dict_packed.items():
            tensor_permuted = tensor_packed.view(B, D//2, H//2, W//2, C)
            high_freq_dict[key] = tensor_permuted.permute(0, 4, 1, 2, 3)
        high_freq_levels.append(high_freq_dict)

    return low_freq, high_freq_levels


def dwt3d_inverse(low_freq: torch.Tensor, high_freq_levels: List[Dict[str, torch.Tensor]], wavelet: str = 'db1') -> torch.Tensor:
    """
    Performs 3D Inverse DWT using ptwt library

    Args:
        low_freq: Low-frequency component [B, C, D//2, H//2, W//2]
        high_freq_levels: List of high-frequency detail coefficient dicts
        wavelet: Wavelet type

    Returns:
        Reconstructed tensor [B, C, D, H, W]
    """
    B, C, D_half, H_half, W_half = low_freq.shape

    # Convert back to ptwt format
    low_freq_permuted = low_freq.permute(0, 2, 3, 4, 1)
    low_freq_packed = low_freq_permuted.reshape(B, D_half, H_half, W_half * C)

    # Process high-frequency components
    high_freq_levels_packed = []
    for high_freq_dict in high_freq_levels:
        high_freq_dict_packed = {}
        for key, tensor in high_freq_dict.items():
            tensor_permuted = tensor.permute(0, 2, 3, 4, 1)
            high_freq_dict_packed[key] = tensor_permuted.reshape(B, D_half, H_half, W_half * C)
        high_freq_levels_packed.append(high_freq_dict_packed)

    # Reconstruct using ptwt
    coeffs = [low_freq_packed] + high_freq_levels_packed
    reconstructed_packed = ptwt.waverec3(coeffs, wavelet=wavelet, mode='zero')

    # Reshape back to [B, C, D, H, W]
    D, H, W = D_half * 2, H_half * 2, W_half * 2
    reconstructed_permuted = reconstructed_packed.view(B, D, H, W, C)
    reconstructed = reconstructed_permuted.permute(0, 4, 1, 2, 3)

    return reconstructed


class WaveletAttentionEncoderBlock(nn.Module):
    """
    WaveFormer encoder block with wavelet decomposition and attention on low-frequency components

    Key features:
    - Identity skip connections for high-frequency components (preserves fine details)
    - Self-attention only on low-frequency components (computational efficiency)
    - Dual residual connections for stability
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., norm_layer=nn.LayerNorm, wavelet='db1'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.wavelet = wavelet

        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # Multi-head attention for low-frequency components
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )

        # MLP for feature processing
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing WaveFormer specifications

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Output tensor [B, C, D, H, W]
        """
        # First residual connection (attention path)
        shortcut1 = x

        # Pre-normalization (applied on feature dimension)
        x_norm = self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        # Wavelet decomposition
        low_freq, high_freq_levels = dwt3d_forward(x_norm, wavelet=self.wavelet, level=1)

        # Self-attention ONLY on low-frequency component
        B, C, D_half, H_half, W_half = low_freq.shape
        low_freq_tokens = low_freq.flatten(2).transpose(1, 2)  # [B, DHW/8, C]

        attn_output, _ = self.attn(low_freq_tokens, low_freq_tokens, low_freq_tokens)
        attn_low_freq = attn_output.transpose(1, 2).view(B, C, D_half, H_half, W_half)

        # High-frequency components use IDENTITY SKIP CONNECTION
        reconstructed = dwt3d_inverse(attn_low_freq, high_freq_levels, wavelet=self.wavelet)

        # Add first residual connection
        x = shortcut1 + reconstructed

        # Second residual connection (MLP path)
        shortcut2 = x
        x_norm2 = self.norm2(x.permute(0, 2, 3, 4, 1))
        x_mlp = self.mlp(x_norm2)
        x = shortcut2 + x_mlp.permute(0, 4, 1, 2, 3)

        return x


class WaveFormer3D(nn.Module):
    """
    Complete 3D WaveFormer architecture for medical image processing

    Architecture:
    - Patch embedding layer for initial feature extraction
    - Stack of WaveletAttentionEncoderBlocks
    - Layer normalization and feature projection
    - Support for intermediate feature extraction for contrastive learning
    """
    def __init__(self, img_size=(64, 64, 64), in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., wavelet='db1', drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.wavelet = wavelet
        self.depth = depth

        # Patch embedding layer
        # Use stride=16 for 64^3 -> 4^3 patches, adjust based on input size
        patch_size = 16
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Calculate number of patches
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim,
                                                 img_size[0] // patch_size,
                                                 img_size[1] // patch_size,
                                                 img_size[2] // patch_size))

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            WaveletAttentionEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                wavelet=wavelet
            ) for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_intermediate=True) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass returning features and optionally intermediate representations

        Args:
            x: Input tensor [B, C, D, H, W]
            return_intermediate: Whether to return intermediate features for contrastive learning

        Returns:
            If return_intermediate=False: Final features [B, embed_dim, D', H', W']
            If return_intermediate=True: (final_features, intermediate_features_list)
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, D/16, H/16, W/16]

        # Add positional embedding
        x = x + self.pos_embed

        # Store intermediate features for contrastive learning
        intermediate_features = []

        # Apply encoder blocks
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)

            # Store intermediate features at specific layers for cross-level alignment
            if i in [self.depth//3, 2*self.depth//3]:
                intermediate_features.append(x)

        # Final features
        intermediate_features.append(x)

        if return_intermediate:
            return x, intermediate_features
        else:
            return x

    def get_feature_map_size(self, input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate output feature map size given input size"""
        patch_size = 16
        return tuple(s // patch_size for s in input_size)


# Example usage and testing
if __name__ == "__main__":
    # Test WaveFormer components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test basic DWT operations
    test_tensor = torch.randn(2, 64, 8, 8, 8).to(device)
    low_freq, high_freq = dwt3d_forward(test_tensor)
    reconstructed = dwt3d_inverse(low_freq, high_freq)
    print(f"DWT reconstruction error: {torch.mean((test_tensor - reconstructed)**2).item():.6f}")

    # Test WaveFormer block
    block = WaveletAttentionEncoderBlock(dim=64).to(device)
    output = block(test_tensor)
    print(f"WaveFormer block output shape: {output.shape}")

    # Test complete WaveFormer
    model = WaveFormer3D(
        img_size=(64, 64, 64),
        in_chans=1,
        embed_dim=768,
        depth=6,  # Reduced for testing
        num_heads=12
    ).to(device)

    test_input = torch.randn(2, 1, 64, 64, 64).to(device)
    final_features, intermediate_features = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Final features shape: {final_features.shape}")
    print(f"Number of intermediate features: {len(intermediate_features)}")
    for i, feat in enumerate(intermediate_features):
        print(f"  Intermediate {i}: {feat.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")