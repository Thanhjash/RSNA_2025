"""
Configuration for Phase 0 Pre-training
Implementation for RSNA 2025 Phase 0 Pre-training

This module contains all configuration parameters for the self-supervised pre-training pipeline.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Image parameters
    img_size: Tuple[int, int, int] = (64, 64, 64)
    in_channels: int = 1

    # WaveFormer parameters
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    wavelet: str = 'db1'
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    # SparK parameters
    spark_num_layers: int = 4
    spark_base_channels: int = 64

    # Masking parameters
    global_mask_ratio: float = 0.6
    local_mask_ratio: float = 0.8
    masking_strategy: str = 'random'  # 'random' or 'block'
    block_size: int = 8

    # Loss weights
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.1
    global_loss_weight: float = 1.0
    local_loss_weight: float = 2.0

    # Contrastive learning parameters
    projection_dim: int = 128
    temperature: float = 0.07


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    # Data paths
    deeplesion_path: str = "/mnt/d/2.Research/RSNA/data/processed/NIH_deeplesion/CT"
    openmind_path: str = "/mnt/d/2.Research/RSNA/data/processed/openmind/OpenMind_processed"

    # Data loading parameters
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    cache_size: int = 100

    # Augmentation parameters
    augmentation: bool = True
    target_size: Tuple[int, int, int] = (64, 64, 64)
    normalize: bool = True

    # Train/validation split
    validation_split: float = 0.1
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Optimizer parameters
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Scheduler parameters
    scheduler: str = 'cosine'  # 'cosine', 'step', 'exponential', 'none'
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1

    # Mixed precision training
    use_amp: bool = True
    amp_opt_level: str = 'O1'

    # Checkpoint and logging
    checkpoint_interval: int = 5
    log_interval: int = 10
    validation_interval: int = 1
    max_checkpoints: int = 5

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4

    # Multi-GPU training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Experiment metadata
    experiment_name: str = "phase0_pretraining"
    description: str = "WaveFormer + SparK + MiM self-supervised pre-training"
    tags: List[str] = None

    # Output directories
    output_dir: str = "/home/thanhjash/RSNA/experiments"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    tensorboard_dir: str = "tensorboard"

    # Random seeds
    seed: int = 42
    deterministic: bool = True

    # Debugging
    debug: bool = False
    debug_samples: int = 10

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["phase0", "pretraining", "waveformer", "spark", "mim"]


class Phase0Config:
    """Complete configuration for Phase 0 pre-training"""

    def __init__(self,
                 model_config: Optional[ModelConfig] = None,
                 data_config: Optional[DataConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 experiment_config: Optional[ExperimentConfig] = None):

        self.model = model_config or ModelConfig()
        self.data = data_config or DataConfig()
        self.training = training_config or TrainingConfig()
        self.experiment = experiment_config or ExperimentConfig()

        # Ensure consistency between configs
        self._validate_config()

    def _validate_config(self):
        """Validate configuration consistency"""
        # Ensure img_size consistency
        assert self.model.img_size == self.data.target_size, \
            "Model img_size must match data target_size"

        # Validate paths exist
        if not Path(self.data.deeplesion_path).exists():
            print(f"Warning: DeepLesion path does not exist: {self.data.deeplesion_path}")

        if not Path(self.data.openmind_path).exists():
            print(f"Warning: OpenMind path does not exist: {self.data.openmind_path}")

        # Ensure reasonable batch size for memory
        if self.data.batch_size > 8:
            print(f"Warning: Large batch size ({self.data.batch_size}) may cause OOM")

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Phase0Config':
        """Create configuration from dictionary"""
        return cls(
            model_config=ModelConfig(**config_dict.get('model', {})),
            data_config=DataConfig(**config_dict.get('data', {})),
            training_config=TrainingConfig(**config_dict.get('training', {})),
            experiment_config=ExperimentConfig(**config_dict.get('experiment', {}))
        )

    def save(self, path: str):
        """Save configuration to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Phase0Config':
        """Load configuration from JSON file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different scenarios
def get_development_config() -> Phase0Config:
    """Configuration for development/testing with smaller resources"""
    model_config = ModelConfig(
        img_size=(32, 32, 32),  # Smaller size
        embed_dim=256,          # Smaller embedding
        depth=6,                # Fewer layers
        num_heads=8,            # Fewer heads
    )

    data_config = DataConfig(
        batch_size=2,           # Smaller batch
        num_workers=2,          # Fewer workers
        cache_size=20,          # Smaller cache
        target_size=(32, 32, 32)
    )

    training_config = TrainingConfig(
        num_epochs=10,          # Fewer epochs
        checkpoint_interval=2,
        log_interval=1,
        use_amp=False           # Disable AMP for debugging
    )

    experiment_config = ExperimentConfig(
        experiment_name="phase0_dev",
        debug=True,
        debug_samples=5
    )

    return Phase0Config(model_config, data_config, training_config, experiment_config)


def get_production_config() -> Phase0Config:
    """Configuration for full-scale training"""
    model_config = ModelConfig(
        img_size=(64, 64, 64),
        embed_dim=768,
        depth=12,
        num_heads=12,
    )

    data_config = DataConfig(
        batch_size=4,
        num_workers=8,
        cache_size=100,
        target_size=(64, 64, 64)
    )

    training_config = TrainingConfig(
        num_epochs=100,
        learning_rate=1e-4,
        use_amp=True,
        checkpoint_interval=5,
        validation_interval=1
    )

    experiment_config = ExperimentConfig(
        experiment_name="phase0_production",
        debug=False
    )

    return Phase0Config(model_config, data_config, training_config, experiment_config)


def get_multi_modal_config() -> Phase0Config:
    """Configuration for multi-modal training (MRI + CT)"""
    model_config = ModelConfig(
        img_size=(64, 64, 64),
        embed_dim=768,
        depth=12,
        num_heads=12,
        contrastive_weight=0.2,  # Higher contrastive weight for multi-modal
    )

    data_config = DataConfig(
        batch_size=3,           # Mixed batches
        num_workers=6,
        cache_size=150,
        target_size=(64, 64, 64),
        augmentation=True
    )

    training_config = TrainingConfig(
        num_epochs=150,         # Longer training for multi-modal
        learning_rate=8e-5,     # Slightly lower LR
        warmup_epochs=15,
        use_amp=True
    )

    experiment_config = ExperimentConfig(
        experiment_name="phase0_multimodal",
        description="Multi-modal pre-training with MRI and CT data",
        tags=["phase0", "multimodal", "mri", "ct", "pretraining"]
    )

    return Phase0Config(model_config, data_config, training_config, experiment_config)


# Example usage and testing
if __name__ == "__main__":
    # Test default configuration
    print("Testing Phase0Config...")

    default_config = Phase0Config()
    print(f"Default config created successfully")
    print(f"Model embed_dim: {default_config.model.embed_dim}")
    print(f"Data batch_size: {default_config.data.batch_size}")
    print(f"Training epochs: {default_config.training.num_epochs}")

    # Test predefined configurations
    dev_config = get_development_config()
    print(f"Development config - img_size: {dev_config.model.img_size}")

    prod_config = get_production_config()
    print(f"Production config - embed_dim: {prod_config.model.embed_dim}")

    multimodal_config = get_multi_modal_config()
    print(f"Multi-modal config - contrastive_weight: {multimodal_config.model.contrastive_weight}")

    # Test serialization
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.json")

        # Save and load
        default_config.save(config_path)
        loaded_config = Phase0Config.load(config_path)

        print(f"Config serialization test passed")
        print(f"Original embed_dim: {default_config.model.embed_dim}")
        print(f"Loaded embed_dim: {loaded_config.model.embed_dim}")

    print("Configuration module testing completed successfully!")