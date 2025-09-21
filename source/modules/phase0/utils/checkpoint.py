"""
Checkpoint Management for Phase 0 Pre-training
Implementation for RSNA 2025 Phase 0 Pre-training

This module provides utilities for:
- Model checkpoint saving and loading
- Training state management
- Best model tracking
- Resume functionality
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Any
import shutil
from datetime import datetime


class CheckpointManager:
    """
    Comprehensive checkpoint manager for pre-training

    Features:
    - Automatic checkpoint saving at intervals
    - Best model tracking based on validation metrics
    - Training state persistence (optimizer, scheduler, epoch)
    - Automatic cleanup of old checkpoints
    - Model-only saving for deployment
    """

    def __init__(self,
                 checkpoint_dir: str,
                 save_interval: int = 5,
                 max_checkpoints: int = 5,
                 best_metric: str = 'val_loss',
                 best_mode: str = 'min'):  # 'min' or 'max'
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N epochs
            max_checkpoints: Maximum number of checkpoints to keep
            best_metric: Metric to track for best model
            best_mode: Whether best metric should be minimized or maximized
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.best_mode = best_mode

        # Track checkpoints and best model
        self.checkpoint_files = []
        self.best_metric_value = float('inf') if best_mode == 'min' else float('-inf')
        self.best_epoch = -1

        # Training history
        self.training_history = []

        logging.info(f"CheckpointManager initialized")
        logging.info(f"  Checkpoint directory: {self.checkpoint_dir}")
        logging.info(f"  Save interval: {save_interval} epochs")
        logging.info(f"  Max checkpoints: {max_checkpoints}")
        logging.info(f"  Best metric: {best_metric} ({best_mode})")

    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any],
                       epoch: int,
                       metrics: Dict[str, float],
                       extra_data: Optional[Dict] = None) -> str:
        """
        Save a training checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            metrics: Dictionary of metrics for this epoch
            extra_data: Additional data to save

        Returns:
            Path to saved checkpoint file
        """
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_metric_value': self.best_metric_value,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat(),
            'training_history': self.training_history
        }

        if extra_data:
            checkpoint_data.update(extra_data)

        # Save checkpoint
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename

        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

        # Update checkpoint tracking
        self.checkpoint_files.append(checkpoint_path)
        self.training_history.append({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def save_best_model(self,
                       model: nn.Module,
                       epoch: int,
                       metrics: Dict[str, float]) -> bool:
        """
        Save model if it's the best so far

        Args:
            model: Model to potentially save
            epoch: Current epoch
            metrics: Dictionary of metrics

        Returns:
            True if model was saved as best, False otherwise
        """
        if self.best_metric not in metrics:
            logging.warning(f"Best metric '{self.best_metric}' not found in metrics")
            return False

        current_value = metrics[self.best_metric]
        is_best = False

        if self.best_mode == 'min':
            is_best = current_value < self.best_metric_value
        else:
            is_best = current_value > self.best_metric_value

        if is_best:
            self.best_metric_value = current_value
            self.best_epoch = epoch

            # Save best model
            best_model_path = self.checkpoint_dir / "best_model.pth"
            best_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'best_metric': self.best_metric,
                'best_metric_value': self.best_metric_value,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(best_data, best_model_path)
            logging.info(f"New best model saved: {best_model_path}")
            logging.info(f"  {self.best_metric}: {self.best_metric_value:.6f} (epoch {epoch})")

        return is_best

    def save_model_only(self,
                       model: nn.Module,
                       filename: str = "model_weights.pth") -> str:
        """
        Save only model weights (for deployment/inference)

        Args:
            model: Model to save
            filename: Filename for the saved weights

        Returns:
            Path to saved model file
        """
        model_path = self.checkpoint_dir / filename
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model weights saved: {model_path}")
        return str(model_path)

    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       checkpoint_path: Optional[str] = None,
                       load_best: bool = False) -> Dict[str, Any]:
        """
        Load a checkpoint

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Specific checkpoint to load (if None, loads latest)
            load_best: If True, loads best model instead

        Returns:
            Dictionary containing loaded metadata
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        elif checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._get_latest_checkpoint()

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        logging.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

        # Load model weights
        model.load_state_dict(checkpoint_data['model_state_dict'])

        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint_data:
            if checkpoint_data['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        # Update internal state
        if 'best_metric_value' in checkpoint_data:
            self.best_metric_value = checkpoint_data['best_metric_value']
        if 'best_epoch' in checkpoint_data:
            self.best_epoch = checkpoint_data['best_epoch']
        if 'training_history' in checkpoint_data:
            self.training_history = checkpoint_data['training_history']

        logging.info(f"Checkpoint loaded successfully")
        logging.info(f"  Epoch: {checkpoint_data.get('epoch', 'Unknown')}")
        logging.info(f"  Best {self.best_metric}: {self.best_metric_value:.6f}")

        return checkpoint_data

    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if we should save a checkpoint at this epoch"""
        return epoch % self.save_interval == 0

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoint_files:
            return None

        # Sort by epoch number
        def extract_epoch(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return -1

        latest_checkpoint = max(checkpoint_files, key=extract_epoch)
        return str(latest_checkpoint)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the latest N"""
        if len(self.checkpoint_files) <= self.max_checkpoints:
            return

        # Sort by modification time
        self.checkpoint_files.sort(key=lambda x: x.stat().st_mtime)

        # Remove oldest checkpoints
        to_remove = self.checkpoint_files[:-self.max_checkpoints]
        for checkpoint_path in to_remove:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logging.info(f"Removed old checkpoint: {checkpoint_path}")

        # Update tracking list
        self.checkpoint_files = self.checkpoint_files[-self.max_checkpoints:]

    def save_training_history(self) -> str:
        """Save training history to JSON file"""
        history_path = self.checkpoint_dir / "training_history.json"

        history_data = {
            'training_history': self.training_history,
            'best_metric': self.best_metric,
            'best_metric_value': self.best_metric_value,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.training_history)
        }

        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        logging.info(f"Training history saved: {history_path}")
        return str(history_path)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress"""
        if not self.training_history:
            return {'status': 'No training history'}

        latest_epoch = self.training_history[-1]['epoch']
        total_epochs = len(self.training_history)

        summary = {
            'total_epochs': total_epochs,
            'latest_epoch': latest_epoch,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'best_metric_value': self.best_metric_value,
            'checkpoint_dir': str(self.checkpoint_dir),
            'num_checkpoints': len(self.checkpoint_files)
        }

        return summary


# Utility functions for easy checkpoint management
def save_model_for_deployment(model: nn.Module,
                             save_path: str,
                             metadata: Optional[Dict] = None):
    """
    Save model in deployment-ready format

    Args:
        model: Model to save
        save_path: Path to save the model
        metadata: Optional metadata to include
    """
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }

    if metadata:
        save_data['metadata'] = metadata

    torch.save(save_data, save_path)
    logging.info(f"Deployment model saved: {save_path}")


def load_model_for_inference(model: nn.Module, model_path: str) -> Dict[str, Any]:
    """
    Load model for inference

    Args:
        model: Model instance to load weights into
        model_path: Path to saved model

    Returns:
        Loaded metadata
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded for inference: {model_path}")
    return checkpoint


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import os

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing checkpoint manager in: {temp_dir}")

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            save_interval=2,
            max_checkpoints=3,
            best_metric='val_loss',
            best_mode='min'
        )

        # Create dummy model and optimizer
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())

        # Simulate training with checkpoints
        for epoch in range(1, 8):
            # Simulate training metrics
            train_loss = 1.0 / epoch  # Decreasing loss
            val_loss = 1.2 / epoch + 0.1 * (epoch % 3)  # Mostly decreasing with noise

            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch
            }

            # Save checkpoint if needed
            if checkpoint_manager.should_save_checkpoint(epoch):
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch,
                    metrics=metrics
                )
                print(f"Saved checkpoint for epoch {epoch}")

            # Check and save best model
            is_best = checkpoint_manager.save_best_model(model, epoch, metrics)
            if is_best:
                print(f"New best model at epoch {epoch}")

        # Test loading
        print("\nTesting checkpoint loading...")

        # Load best model
        new_model = nn.Linear(10, 1)
        loaded_data = checkpoint_manager.load_checkpoint(
            model=new_model,
            load_best=True
        )
        print(f"Loaded best model from epoch {loaded_data['epoch']}")

        # Save training history
        history_path = checkpoint_manager.save_training_history()
        print(f"Training history saved to: {history_path}")

        # Get summary
        summary = checkpoint_manager.get_training_summary()
        print(f"Training summary: {summary}")

        # Test deployment save
        deployment_path = os.path.join(temp_dir, "deployment_model.pth")
        save_model_for_deployment(
            model=model,
            save_path=deployment_path,
            metadata={'description': 'Test model'}
        )

        print("Checkpoint manager testing completed successfully!")