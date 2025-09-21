"""
WaveFormer Fallback Implementation
Implementation without external dependencies for testing and development

This provides a fallback implementation of WaveFormer that works without ptwt,
using simple convolutions to approximate wavelet operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Union
import math


def simple_dwt3d_forward(x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Simple approximation of 3D DWT using average pooling and high-pass filtering

    Args:
        x: Input tensor [B, C, D, H, W]

    Returns:
        Tuple of (low_freq, high_freq_levels)
    """
    # Low-frequency: simple average pooling
    low_freq = F.avg_pool3d(x, kernel_size=2, stride=2)

    # High-frequency: approximated using difference
    # This is a simplified approximation - not true wavelets
    high_freq_dict = {
        'LLH': F.avg_pool3d(x - F.interpolate(low_freq, size=x.shape[-3:], mode='trilinear', align_corners=False),
                           kernel_size=2, stride=2),
        'LHL': torch.zeros_like(low_freq),
        'LHH': torch.zeros_like(low_freq),
        'HLL': torch.zeros_like(low_freq),
        'HLH': torch.zeros_like(low_freq),
        'HHL': torch.zeros_like(low_freq),
        'HHH': torch.zeros_like(low_freq)
    }

    return low_freq, [high_freq_dict]


def simple_dwt3d_inverse(low_freq: torch.Tensor, high_freq_levels: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Simple approximation of 3D inverse DWT using upsampling

    Args:
        low_freq: Low-frequency component
        high_freq_levels: High-frequency components (ignored in this simple version)

    Returns:
        Reconstructed tensor
    """
    # Simple bilinear upsampling (approximation)
    D, H, W = low_freq.shape[-3:]
    target_size = (D * 2, H * 2, W * 2)

    reconstructed = F.interpolate(
        low_freq,
        size=target_size,
        mode='trilinear',
        align_corners=False
    )

    return reconstructed


class FallbackWaveletAttentionEncoderBlock(nn.Module):
    """
    Fallback implementation of WaveFormer encoder block without external dependencies
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

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
        Forward pass with fallback wavelet operations
        """
        # First residual connection (attention path)
        shortcut1 = x

        # Pre-normalization
        x_norm = self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        # Simplified wavelet decomposition
        low_freq, high_freq_levels = simple_dwt3d_forward(x_norm)

        # Self-attention on low-frequency component
        B, C, D_half, H_half, W_half = low_freq.shape
        low_freq_tokens = low_freq.flatten(2).transpose(1, 2)  # [B, DHW/8, C]

        attn_output, _ = self.attn(low_freq_tokens, low_freq_tokens, low_freq_tokens)
        attn_low_freq = attn_output.transpose(1, 2).view(B, C, D_half, H_half, W_half)

        # Reconstruct using simplified inverse
        reconstructed = simple_dwt3d_inverse(attn_low_freq, high_freq_levels)

        # Add first residual connection
        x = shortcut1 + reconstructed

        # Second residual connection (MLP path)
        shortcut2 = x
        x_norm2 = self.norm2(x.permute(0, 2, 3, 4, 1))
        x_mlp = self.mlp(x_norm2)
        x = shortcut2 + x_mlp.permute(0, 4, 1, 2, 3)

        return x


class FallbackWaveFormer3D(nn.Module):
    """
    Fallback WaveFormer implementation that works without external dependencies
    """
    def __init__(self, img_size=(64, 64, 64), in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding layer
        patch_size = 16
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim,
                                                 img_size[0] // patch_size,
                                                 img_size[1] // patch_size,
                                                 img_size[2] // patch_size))

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            FallbackWaveletAttentionEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        Forward pass with fallback implementation
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, D/16, H/16, W/16]

        # Add positional embedding
        x = x + self.pos_embed

        # Store intermediate features
        intermediate_features = []

        # Apply encoder blocks
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)

            # Store intermediate features at specific layers
            if i in [self.depth//3, 2*self.depth//3]:
                intermediate_features.append(x)

        # Final features
        intermediate_features.append(x)

        if return_intermediate:
            return x, intermediate_features
        else:
            return x


# Update the imports in the main __init__.py to use fallback when needed
def get_waveformer_implementation():
    """Get the appropriate WaveFormer implementation based on available dependencies"""
    try:
        import ptwt
        # If ptwt is available, use the full implementation
        from .waveformer import WaveFormer3D, WaveletAttentionEncoderBlock
        return WaveFormer3D, WaveletAttentionEncoderBlock, True
    except ImportError:
        # Fall back to simplified implementation
        return FallbackWaveFormer3D, FallbackWaveletAttentionEncoderBlock, False


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test fallback wavelet operations
    test_tensor = torch.randn(2, 64, 8, 8, 8).to(device)
    low_freq, high_freq = simple_dwt3d_forward(test_tensor)
    reconstructed = simple_dwt3d_inverse(low_freq, high_freq)

    print(f"Fallback DWT test:")
    print(f"  Input: {test_tensor.shape}")
    print(f"  Low freq: {low_freq.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")
    print(f"  Reconstruction error: {torch.mean((test_tensor - reconstructed)**2).item():.6f}")

    # Test fallback WaveFormer block
    block = FallbackWaveletAttentionEncoderBlock(dim=64).to(device)
    output = block(test_tensor)
    print(f"Fallback WaveFormer block output: {output.shape}")

    # Test complete fallback WaveFormer
    model = FallbackWaveFormer3D(
        img_size=(32, 32, 32),
        in_chans=64,  # Match test tensor
        embed_dim=256,
        depth=3,
        num_heads=8
    ).to(device)

    # Need to adjust input for patch embedding
    test_input = torch.randn(2, 1, 32, 32, 32).to(device)
    final_features, intermediate_features = model(test_input)

    print(f"Fallback WaveFormer test:")
    print(f"  Input: {test_input.shape}")
    print(f"  Final features: {final_features.shape}")
    print(f"  Intermediate features: {len(intermediate_features)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("Fallback implementation tests completed successfully!")