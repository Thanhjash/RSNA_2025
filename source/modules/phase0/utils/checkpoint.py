"""
Checkpoint Management for Phase 0 Pre-training
Implementation for RSNA 2025

Features:
- Model state saving/loading
- Optimizer state persistence
- Training progress tracking
- Best model selection based on validation metrics
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints during training

    Features:
    - Save checkpoints periodically
    - Keep best N checkpoints based on validation metric
    - Resume training from checkpoint
    - Export final model weights
    """

    def __init__(self, checkpoint_dir: str, max_keep: int = 3,
                 metric_mode: str = 'min', save_frequency: int = 1):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep
            metric_mode: 'min' or 'max' for best model selection
            save_frequency: Save checkpoint every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_keep = max_keep
        self.metric_mode = metric_mode
        self.save_frequency = save_frequency

        # Track saved checkpoints
        self.checkpoints = []
        self.best_metric = float('inf') if metric_mode == 'min' else float('-inf')
        self.best_checkpoint = None

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float],
                       extra_state: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Validation metrics
            extra_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if extra_state is not None:
            checkpoint['extra_state'] = extra_state

        # Checkpoint filename
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'

        # Save
        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics
        })

        # Check if best checkpoint
        primary_metric = metrics.get('val_loss', float('inf'))
        is_best = False

        if self.metric_mode == 'min':
            if primary_metric < self.best_metric:
                self.best_metric = primary_metric
                is_best = True
        else:
            if primary_metric > self.best_metric:
                self.best_metric = primary_metric
                is_best = True

        if is_best:
            self.best_checkpoint = checkpoint_path
            # Save as best model
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        # Save metadata
        self._save_metadata()

        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only best N"""
        if len(self.checkpoints) <= self.max_keep:
            return

        # Sort by metric
        metric_key = 'val_loss'
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x['metrics'].get(metric_key, float('inf')),
            reverse=(self.metric_mode == 'max')
        )

        # Keep best N
        to_keep = sorted_checkpoints[:self.max_keep]
        to_remove = sorted_checkpoints[self.max_keep:]

        # Delete files
        for checkpoint in to_remove:
            if checkpoint['path'] != self.best_checkpoint:
                checkpoint['path'].unlink(missing_ok=True)

        self.checkpoints = to_keep

    def _save_metadata(self):
        """Save checkpoint metadata"""
        metadata = {
            'checkpoints': [
                {
                    'path': str(c['path']),
                    'epoch': c['epoch'],
                    'metrics': c['metrics']
                }
                for c in self.checkpoints
            ],
            'best_checkpoint': str(self.best_checkpoint) if self.best_checkpoint else None,
            'best_metric': self.best_metric
        }

        metadata_path = self.checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None,
                       model: Optional[nn.Module] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint (uses best if None)
            model: Model to load state into
            optimizer: Optimizer to load state into
            device: Device to load checkpoint on

        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint

        if checkpoint_path is None:
            raise ValueError("No checkpoint specified and no best checkpoint found")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def load_latest_checkpoint(self, model: Optional[nn.Module] = None,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint for resuming training

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            device: Device to load checkpoint on

        Returns:
            Checkpoint dictionary or None if no checkpoints exist
        """
        # Find all checkpoints
        checkpoint_files = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))

        if not checkpoint_files:
            return None

        # Load latest
        latest_checkpoint = checkpoint_files[-1]
        return self.load_checkpoint(str(latest_checkpoint), model, optimizer, device)


def save_model_for_inference(model: nn.Module, save_path: str):
    """
    Save model weights for inference (without optimizer state)

    Args:
        model: Trained model
        save_path: Path to save weights
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }, save_path)


def load_model_for_inference(model: nn.Module, checkpoint_path: str, device: str = 'cpu'):
    """
    Load model weights for inference

    Args:
        model: Model architecture
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
