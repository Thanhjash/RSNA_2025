"""
Phase 0 Pre-training Script
RSNA 2025 Project

Runs WaveFormer + SparK + MiM pre-training on OpenMind + DeepLesion data
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.phase0.models.pretrainer import WaveFormerSparKMiMPretrainer, MultiModalPretrainer
from modules.phase0.data.dataloaders import MultiModalDataLoader
from modules.phase0.data.transforms import get_training_transforms, get_validation_transforms
from modules.phase0.utils.checkpoint import CheckpointManager
from config.phase0_config import get_config


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    """
    Train for one epoch

    Args:
        model: Pre-training model
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Training device
        epoch: Current epoch
        config: Training configuration

    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if config.mixed_precision:
            with autocast():
                loss, loss_dict = model(images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = model(images)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        if batch_idx % config.log_frequency == 0:
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['reconstruction_loss']:.4f}",
                'contrast': f"{loss_dict['contrastive_loss']:.4f}"
            })

    return total_loss / num_batches


def validate(model, dataloader, device, config):
    """
    Validate model

    Args:
        model: Pre-training model
        dataloader: Validation data loader
        device: Device
        config: Configuration

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_contrast_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)

            if config.mixed_precision:
                with autocast():
                    loss, loss_dict = model(images)
            else:
                loss, loss_dict = model(images)

            total_loss += loss.item()
            total_recon_loss += loss_dict['reconstruction_loss']
            total_contrast_loss += loss_dict['contrastive_loss']
            num_batches += 1

    metrics = {
        'val_loss': total_loss / num_batches,
        'val_recon_loss': total_recon_loss / num_batches,
        'val_contrast_loss': total_contrast_loss / num_batches
    }

    return metrics


def train_single_modality(config):
    """
    Train on single modality (MRI or CT)

    Args:
        config: Training configuration
    """
    print(f"Starting Phase 0 pre-training with config: {config}")

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = WaveFormerSparKMiMPretrainer(
        img_size=config.img_size,
        in_channels=config.in_channels,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        wavelet=config.wavelet,
        spark_base_channels=config.spark_base_channels,
        spark_stages=config.spark_stages,
        global_mask_ratio=config.global_mask_ratio,
        local_mask_ratio=config.local_mask_ratio,
        contrastive_temperature=config.contrastive_temperature,
        contrastive_weight=config.contrastive_weight
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create data loaders
    transforms_train = get_training_transforms(augment=True)
    transforms_val = get_validation_transforms()

    data_module = MultiModalDataLoader(
        openmind_dir=config.openmind_dir,
        deeplesion_dir=config.deeplesion_dir,
        batch_size=config.batch_size,
        target_size=config.img_size,
        num_workers=config.num_workers,
        transform=transforms_train,
        train_split=config.train_split
    )

    # Use OpenMind (MRI) for single modality
    train_loader_mri, train_loader_ct = data_module.get_train_loaders()
    val_loader_mri, val_loader_ct = data_module.get_val_loaders()

    # Choose loader based on in_channels
    train_loader = train_loader_mri if config.in_channels == 1 else train_loader_ct
    val_loader = val_loader_mri if config.in_channels == 1 else val_loader_ct

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.learning_rate * 0.01
    )

    # Gradient scaler for mixed precision
    scaler = GradScaler() if config.mixed_precision else None

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        max_keep=config.max_keep_checkpoints,
        metric_mode='min',
        save_frequency=config.save_frequency
    )

    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, config
        )

        print(f"Training loss: {train_loss:.4f}")

        # Validate
        if (epoch + 1) % config.validate_frequency == 0:
            val_metrics = validate(model, val_loader, device, config)
            print(f"Validation metrics: {val_metrics}")

            # Save checkpoint
            if (epoch + 1) % config.save_frequency == 0:
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics
                )

        # Update learning rate
        scheduler.step()

    print("\nTraining completed!")
    print(f"Best checkpoint: {checkpoint_manager.best_checkpoint}")


def train_multimodal(config):
    """
    Train on both MRI and CT

    Args:
        config: MultiModalConfig
    """
    print(f"Starting multi-modal pre-training")

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # Create multi-modal model
    model = MultiModalPretrainer(
        img_size=config.img_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Data loaders
    data_module = MultiModalDataLoader(
        openmind_dir=config.openmind_dir,
        deeplesion_dir=config.deeplesion_dir,
        batch_size=config.batch_size,
        target_size=config.img_size,
        num_workers=config.num_workers,
        train_split=config.train_split
    )

    train_loader_mri, train_loader_ct = data_module.get_train_loaders()
    val_loader_mri, val_loader_ct = data_module.get_val_loaders()

    # Optimizers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    scaler = GradScaler() if config.mixed_precision else None

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        max_keep=config.max_keep_checkpoints
    )

    # Training loop with alternating batches
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        model.train()
        total_loss = 0.0

        # Interleave MRI and CT batches
        mri_iter = iter(train_loader_mri)
        ct_iter = iter(train_loader_ct)

        num_batches = min(len(train_loader_mri), len(train_loader_ct))

        for _ in tqdm(range(num_batches), desc='Training'):
            # MRI batch
            try:
                mri_batch = next(mri_iter)
                mri_images = mri_batch['image'].to(device)

                optimizer.zero_grad()
                with autocast() if config.mixed_precision else torch.enable_grad():
                    loss_mri, _ = model(mri_images, modality='mri')

                if scaler:
                    scaler.scale(loss_mri).backward()
                else:
                    loss_mri.backward()

                total_loss += loss_mri.item()
            except StopIteration:
                pass

            # CT batch
            try:
                ct_batch = next(ct_iter)
                ct_images = ct_batch['image'].to(device)

                optimizer.zero_grad()
                with autocast() if config.mixed_precision else torch.enable_grad():
                    loss_ct, _ = model(ct_images, modality='ct')

                if scaler:
                    scaler.scale(loss_ct).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_ct.backward()
                    optimizer.step()

                total_loss += loss_ct.item()
            except StopIteration:
                pass

        avg_loss = total_loss / (num_batches * 2)
        print(f"Average loss: {avg_loss:.4f}")

        scheduler.step()

        # Validation and checkpointing
        if (epoch + 1) % config.validate_frequency == 0:
            val_metrics_mri = validate(model, val_loader_mri, device, config)
            val_metrics_ct = validate(model, val_loader_ct, device, config)

            combined_metrics = {
                'val_loss': (val_metrics_mri['val_loss'] + val_metrics_ct['val_loss']) / 2
            }

            if (epoch + 1) % config.save_frequency == 0:
                checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch, combined_metrics
                )

    print("\nMulti-modal training completed!")


def main():
    parser = argparse.ArgumentParser(description='Phase 0 Pre-training')
    parser.add_argument('--config', type=str, default='dev',
                       choices=['dev', 'prod', 'multimodal', 'kaggle'],
                       help='Configuration to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Train
    if args.config == 'multimodal':
        train_multimodal(config)
    else:
        train_single_modality(config)


if __name__ == '__main__':
    main()
