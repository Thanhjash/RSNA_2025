"""
Configuration for Phase 0 Pre-training
RSNA 2025 Project

Provides different configs for:
- Development (small model, fast iteration)
- Production (full model, complete training)
- Multi-modal (MRI + CT combined)
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Phase0Config:
    """Base configuration for Phase 0 pre-training"""

    # Model architecture
    img_size: Tuple[int, int, int] = (64, 64, 64)
    in_channels: int = 1  # 1 for MRI, 3 for CT
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    wavelet: str = 'db1'

    # SparK configuration
    spark_base_channels: int = 96
    spark_stages: int = 4

    # MiM masking
    global_mask_ratio: float = 0.6
    local_mask_ratio: float = 0.8
    mask_block_size: int = 4

    # Contrastive learning
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.1
    projection_dim: int = 256

    # Training
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10

    # Data - UNIFIED MULTI-MODAL
    # MRI directories (1-channel)
    mri_dirs: list = None  # Set in __post_init__
    # CT directories (3-channel)
    ct_dirs: list = None  # Set in __post_init__

    train_split: float = 0.9
    num_workers: int = 4

    # Multi-modal batch sizes (CT smaller due to 3x memory)
    batch_size_mri: int = 8
    batch_size_ct: int = 4

    def __post_init__(self):
        """Initialize directory lists after dataclass creation"""
        if self.mri_dirs is None:
            self.mri_dirs = [
                "/workspace/rsna/data/processed/openmind/OpenMind_processed/MRI_T1",
                "/workspace/rsna/data/processed/openmind/OpenMind_processed/MRI_T2",
                "/workspace/rsna/data/processed/openmind/OpenMind_processed/MRA",
            ]
        if self.ct_dirs is None:
            self.ct_dirs = [
                "/workspace/rsna/data/processed/NIH_deeplesion/CT"
            ]

    # Checkpointing
    checkpoint_dir: str = "./checkpoints/phase0"
    save_frequency: int = 5
    max_keep_checkpoints: int = 3

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True

    # Logging
    log_frequency: int = 10
    validate_frequency: int = 1


@dataclass
class DevConfig(Phase0Config):
    """Development configuration - fast iteration with small model"""

    # Smaller model for quick testing
    img_size: Tuple[int, int, int] = (32, 32, 32)
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    spark_base_channels: int = 32
    spark_stages: int = 2

    # Training
    batch_size_mri: int = 2
    batch_size_ct: int = 2
    num_epochs: int = 10
    learning_rate: float = 1e-3

    # Less frequent validation
    validate_frequency: int = 2


@dataclass
class ProductionConfig(Phase0Config):
    """Production configuration - full model for complete pre-training"""

    # Full model
    img_size: Tuple[int, int, int] = (64, 64, 64)
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12

    # Training
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4

    # Aggressive checkpointing
    save_frequency: int = 5
    max_keep_checkpoints: int = 5


@dataclass
class MultiModalConfig(Phase0Config):
    """Multi-modal configuration for MRI + CT training"""

    # Training on both modalities
    batch_size: int = 8  # Per modality
    num_epochs: int = 100

    # Balanced sampling
    mri_weight: float = 0.5
    ct_weight: float = 0.5

    # Separate learning rates
    mri_lr: float = 1e-4
    ct_lr: float = 1e-4


@dataclass
class KaggleConfig(Phase0Config):
    """Configuration optimized for Kaggle environment"""

    # Kaggle paths
    openmind_dir: str = "/kaggle/input/openmind-mri/processed"
    deeplesion_dir: str = "/kaggle/input/deeplesion-ct/processed"
    checkpoint_dir: str = "/kaggle/working/checkpoints"

    # Kaggle GPU (T4 16GB)
    device: str = "cuda"
    mixed_precision: bool = True
    batch_size: int = 4  # Conservative for 16GB

    # Time-limited training (Kaggle 9-hour limit)
    num_epochs: int = 50
    save_frequency: int = 5

    # Maximize data workers
    num_workers: int = 2  # Kaggle has 4 CPUs


def get_config(config_name: str = 'dev') -> Phase0Config:
    """
    Get configuration by name

    Args:
        config_name: One of 'dev', 'prod', 'multimodal', 'kaggle'

    Returns:
        Configuration object
    """
    configs = {
        'dev': DevConfig(),
        'prod': ProductionConfig(),
        'production': ProductionConfig(),
        'multimodal': MultiModalConfig(),
        'kaggle': KaggleConfig()
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")

    return configs[config_name]
