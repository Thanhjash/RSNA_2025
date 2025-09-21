#!/usr/bin/env python3
"""
Phase 0 Pre-training Script for RSNA 2025
Main training script implementing WaveFormer + SparK + MiM self-supervised pre-training

This script orchestrates the complete pre-training pipeline:
- WaveFormer backbone with wavelet attention
- SparK sparse masked modeling
- MiM hierarchical masking strategy
- Cross-level contrastive learning
- Unified training for OpenMind + DeepLesion datasets

Usage:
    python pretrain.py --config development  # For development/testing
    python pretrain.py --config production   # For full-scale training
    python pretrain.py --config multimodal   # For multi-modal training
    python pretrain.py --config_file custom_config.json  # Custom config
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os

# Add source directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.phase0 import (
    WaveFormerSparKMiMPretrainer,
    MultiModalPretrainer,
    UnifiedDataLoader,
    generate_hierarchical_mask,
    CrossLevelAlignmentLoss
)
from modules.phase0.utils.checkpoint import CheckpointManager
from config.phase0_config import (
    Phase0Config,
    get_development_config,
    get_production_config,
    get_multi_modal_config
)


class Phase0Trainer:
    """
    Complete trainer for Phase 0 self-supervised pre-training

    Manages the entire training process including:
    - Model initialization and configuration
    - Data loading and preprocessing
    - Training loop with mixed precision
    - Validation and metric tracking
    - Checkpoint management
    - Logging and monitoring
    """

    def __init__(self, config: Phase0Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set up experiment directory
        self.experiment_dir = Path(config.experiment.output_dir) / config.experiment.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Set random seeds
        self._set_seeds()

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.data_loader = None
        self.checkpoint_manager = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []

        logging.info(f"Phase0Trainer initialized")
        logging.info(f"Experiment: {config.experiment.experiment_name}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Output directory: {self.experiment_dir}")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.experiment_dir / self.config.experiment.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        # Save configuration
        config_file = self.experiment_dir / "config.json"
        self.config.save(str(config_file))
        logging.info(f"Configuration saved to: {config_file}")

    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config.experiment.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        if self.config.experiment.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logging.info(f"Random seed set to: {seed}")

    def _create_model(self):
        """Create and initialize the model"""
        logging.info("Creating model...")

        # Check if we need multi-modal support
        # This is a simplified check - in practice, you might want more sophisticated logic
        use_multimodal = True  # Enable multi-modal support by default

        if use_multimodal:
            # Multi-modal configuration
            single_channel_config = {
                'img_size': self.config.model.img_size,
                'embed_dim': self.config.model.embed_dim,
                'depth': self.config.model.depth,
                'num_heads': self.config.model.num_heads,
                'wavelet': self.config.model.wavelet,
                'global_mask_ratio': self.config.model.global_mask_ratio,
                'local_mask_ratio': self.config.model.local_mask_ratio,
                'masking_strategy': self.config.model.masking_strategy,
                'reconstruction_weight': self.config.model.reconstruction_weight,
                'contrastive_weight': self.config.model.contrastive_weight,
                'projection_dim': self.config.model.projection_dim,
                'temperature': self.config.model.temperature
            }

            multi_channel_config = single_channel_config.copy()

            self.model = MultiModalPretrainer(
                single_channel_config=single_channel_config,
                multi_channel_config=multi_channel_config,
                shared_features=True
            )
        else:
            # Single modal configuration
            self.model = WaveFormerSparKMiMPretrainer(
                img_size=self.config.model.img_size,
                in_channels=self.config.model.in_channels,
                embed_dim=self.config.model.embed_dim,
                depth=self.config.model.depth,
                num_heads=self.config.model.num_heads,
                wavelet=self.config.model.wavelet,
                global_mask_ratio=self.config.model.global_mask_ratio,
                local_mask_ratio=self.config.model.local_mask_ratio,
                masking_strategy=self.config.model.masking_strategy,
                reconstruction_weight=self.config.model.reconstruction_weight,
                contrastive_weight=self.config.model.contrastive_weight,
                projection_dim=self.config.model.projection_dim,
                temperature=self.config.model.temperature
            )

        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logging.info(f"Model created successfully")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

    def _create_data_loader(self):
        """Create data loaders"""
        logging.info("Creating data loaders...")

        self.data_loader = UnifiedDataLoader(
            deeplesion_path=self.config.data.deeplesion_path,
            openmind_path=self.config.data.openmind_path,
            target_size=self.config.data.target_size,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            augmentation=self.config.data.augmentation,
            cache_size=self.config.data.cache_size
        )

        # Create train and validation loaders
        self.train_loader = self.data_loader.create_dataloader('train', shuffle=True)
        self.val_loader = self.data_loader.create_dataloader('val', shuffle=False)

        # Get dataset statistics
        try:
            stats = self.data_loader.get_modality_stats()
            logging.info("Dataset statistics:")
            for modality, count in stats.items():
                logging.info(f"  {modality}: {count} files")
        except Exception as e:
            logging.warning(f"Could not get dataset statistics: {e}")

        logging.info(f"Train loader: {len(self.train_loader)} batches")
        logging.info(f"Validation loader: {len(self.val_loader)} batches")

    def _create_optimizer(self):
        """Create optimizer and scheduler"""
        logging.info("Creating optimizer...")

        # Create optimizer
        if self.config.training.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        # Create scheduler
        if self.config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.training.scheduler}")

        # Create mixed precision scaler
        if self.config.training.use_amp:
            self.scaler = GradScaler()

        logging.info(f"Optimizer: {self.config.training.optimizer}")
        logging.info(f"Learning rate: {self.config.training.learning_rate}")
        logging.info(f"Scheduler: {self.config.training.scheduler}")
        logging.info(f"Mixed precision: {self.config.training.use_amp}")

    def _create_checkpoint_manager(self):
        """Create checkpoint manager"""
        checkpoint_dir = self.experiment_dir / self.config.experiment.checkpoint_dir

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            save_interval=self.config.training.checkpoint_interval,
            max_checkpoints=self.config.training.max_checkpoints,
            best_metric='val_loss',
            best_mode='min'
        )

        logging.info(f"Checkpoint manager created: {checkpoint_dir}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'train_loss': 0.0,
            'train_reconstruction_loss': 0.0,
            'train_contrastive_loss': 0.0,
            'train_sparse_contrastive_loss': 0.0,
            'samples_processed': 0,
            'batches_processed': 0
        }

        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Forward pass
            if self.config.training.use_amp:
                with autocast():
                    results = self.model(batch_data)
                    loss = results.get('combined_total_loss', results.get('total_loss', 0))
            else:
                results = self.model(batch_data)
                loss = results.get('combined_total_loss', results.get('total_loss', 0))

            # Backward pass
            self.optimizer.zero_grad()

            if self.config.training.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                self.optimizer.step()

            # Update metrics
            batch_size = 0
            if 'single_channel' in batch_data:
                batch_size += batch_data['single_channel'].shape[0]
            if 'multi_channel' in batch_data:
                batch_size += batch_data['multi_channel'].shape[0]

            epoch_metrics['train_loss'] += loss.item() * batch_size
            epoch_metrics['samples_processed'] += batch_size
            epoch_metrics['batches_processed'] += 1

            # Extract individual loss components if available
            for key in ['reconstruction_loss', 'contrastive_loss', 'sparse_contrastive_loss']:
                if key in results:
                    epoch_metrics[f'train_{key}'] += results[key].item() * batch_size

            # Log progress
            if batch_idx % self.config.training.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                    f"Loss={loss.item():.6f}, LR={lr:.2e}"
                )

            # Debug mode: process only a few batches
            if self.config.experiment.debug and batch_idx >= self.config.experiment.debug_samples:
                logging.info("Debug mode: stopping after limited batches")
                break

        # Average metrics
        if epoch_metrics['samples_processed'] > 0:
            for key in epoch_metrics:
                if key.startswith('train_') and key not in ['samples_processed', 'batches_processed']:
                    epoch_metrics[key] /= epoch_metrics['samples_processed']

        epoch_time = time.time() - start_time
        epoch_metrics['epoch_time'] = epoch_time

        logging.info(f"Epoch {epoch} training completed in {epoch_time:.1f}s")
        logging.info(f"  Train loss: {epoch_metrics['train_loss']:.6f}")

        return epoch_metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_reconstruction_loss': 0.0,
            'val_contrastive_loss': 0.0,
            'samples_processed': 0
        }

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                if self.config.training.use_amp:
                    with autocast():
                        results = self.model(batch_data)
                        loss = results.get('combined_total_loss', results.get('total_loss', 0))
                else:
                    results = self.model(batch_data)
                    loss = results.get('combined_total_loss', results.get('total_loss', 0))

                # Update metrics
                batch_size = 0
                if 'single_channel' in batch_data:
                    batch_size += batch_data['single_channel'].shape[0]
                if 'multi_channel' in batch_data:
                    batch_size += batch_data['multi_channel'].shape[0]

                val_metrics['val_loss'] += loss.item() * batch_size
                val_metrics['samples_processed'] += batch_size

                # Extract individual loss components
                for key in ['reconstruction_loss', 'contrastive_loss']:
                    if key in results:
                        val_metrics[f'val_{key}'] += results[key].item() * batch_size

                # Debug mode: process only a few batches
                if self.config.experiment.debug and batch_idx >= self.config.experiment.debug_samples:
                    break

        # Average metrics
        if val_metrics['samples_processed'] > 0:
            for key in val_metrics:
                if key.startswith('val_') and key != 'samples_processed':
                    val_metrics[key] /= val_metrics['samples_processed']

        logging.info(f"Epoch {epoch} validation completed")
        logging.info(f"  Val loss: {val_metrics['val_loss']:.6f}")

        return val_metrics

    def train(self):
        """Main training loop"""
        logging.info("Starting training...")

        # Initialize all components
        self._create_model()
        self._create_data_loader()
        self._create_optimizer()
        self._create_checkpoint_manager()

        # Training loop
        for epoch in range(1, self.config.training.num_epochs + 1):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate epoch
            if epoch % self.config.training.validation_interval == 0:
                val_metrics = self.validate_epoch(epoch)
            else:
                val_metrics = {}

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if self.checkpoint_manager.should_save_checkpoint(epoch):
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=epoch_metrics
                )

            # Save best model
            if val_metrics:
                self.checkpoint_manager.save_best_model(
                    model=self.model,
                    epoch=epoch,
                    metrics=val_metrics
                )

            # Store training history
            self.training_history.append(epoch_metrics)

            # Log epoch summary
            logging.info(f"Epoch {epoch} completed:")
            for key, value in epoch_metrics.items():
                if isinstance(value, float):
                    logging.info(f"  {key}: {value:.6f}")

        # Save final training history
        self.checkpoint_manager.save_training_history()

        logging.info("Training completed!")
        summary = self.checkpoint_manager.get_training_summary()
        logging.info(f"Training summary: {summary}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Phase 0 Pre-training for RSNA 2025")
    parser.add_argument('--config', type=str, choices=['development', 'production', 'multimodal'],
                       default='development', help='Predefined configuration to use')
    parser.add_argument('--config_file', type=str, help='Path to custom configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Load configuration
    if args.config_file:
        config = Phase0Config.load(args.config_file)
    elif args.config == 'development':
        config = get_development_config()
    elif args.config == 'production':
        config = get_production_config()
    elif args.config == 'multimodal':
        config = get_multi_modal_config()
    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Override debug setting
    if args.debug:
        config.experiment.debug = True

    # Create trainer
    trainer = Phase0Trainer(config)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()