"""
MSL Trainer
Main training loop with validation and checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import logging
from tqdm import tqdm

from ..utils.checkpoint_utils import save_checkpoint, load_checkpoint
from ..utils.logging_utils import MetricTracker, TensorBoardLogger

logger = logging.getLogger(__name__)


class MSLTrainer:
    """
    Trainer for MSL model with support for:
    - Training and validation loops
    - Checkpointing
    - TensorBoard logging
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        checkpoint_dir: Path = Path('checkpoints'),
        log_dir: Optional[Path] = None,
        use_amp: bool = False,
        max_grad_norm: float = 1.0,
        log_interval: int = 10
    ):
        """
        Args:
            model: MSL model
            loss_fn: Loss function (MSLRankingLoss)
            optimizer: Optimizer
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for TensorBoard logs
            use_amp: Use automatic mixed precision
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Log every N batches
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logging
        if log_dir is not None:
            self.tb_logger = TensorBoardLogger(log_dir, enabled=True)
        else:
            self.tb_logger = TensorBoardLogger(Path('logs'), enabled=False)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        
        logger.info(f"Trainer initialized on {device}")
        logger.info(f"Mixed precision: {use_amp}")
        logger.info(f"Gradient clipping: {max_grad_norm}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            scheduler: Optional learning rate scheduler
        
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        metrics = MetricTracker()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(features, mask)
                    loss_dict = self.loss_fn(
                        output['video_scores'],
                        output['snippet_scores'],
                        labels
                    )
                    loss = loss_dict['loss']
            else:
                output = self.model(features, mask)
                loss_dict = self.loss_fn(
                    output['video_scores'],
                    output['snippet_scores'],
                    labels
                )
                loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # Step scheduler if per-batch
            if scheduler is not None and hasattr(scheduler, 'step_update'):
                scheduler.step_update(self.global_step)
            
            # Update metrics
            metrics.update(
                loss=loss.item(),
                loss_video=loss_dict['loss_video'].item(),
                loss_snippet=loss_dict['loss_snippet'].item(),
                n_pairs=loss_dict['n_pairs']
            )
            
            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = metrics.get_average('loss')
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # TensorBoard
                self.tb_logger.log_scalar('train/loss', avg_loss, self.global_step)
                self.tb_logger.log_scalar('train/loss_video', metrics.get_average('loss_video'), self.global_step)
                self.tb_logger.log_scalar('train/loss_snippet', metrics.get_average('loss_snippet'), self.global_step)
                
                # Learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.tb_logger.log_scalar('train/lr', current_lr, self.global_step)
            
            self.global_step += 1
        
        # Step scheduler if per-epoch
        if scheduler is not None and not hasattr(scheduler, 'step_update'):
            scheduler.step()
        
        return metrics.get_all_averages()
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of average metrics
        """
        self.model.eval()
        metrics = MetricTracker()
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            # Move to device
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            output = self.model(features, mask)
            loss_dict = self.loss_fn(
                output['video_scores'],
                output['snippet_scores'],
                labels
            )
            
            # Update metrics
            metrics.update(
                loss=loss_dict['loss'].item(),
                loss_video=loss_dict['loss_video'].item(),
                loss_snippet=loss_dict['loss_snippet'].item(),
                n_pairs=loss_dict['n_pairs']
            )
            
            avg_loss = metrics.get_average('loss')
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Log to TensorBoard
        avg_metrics = metrics.get_all_averages()
        self.tb_logger.log_scalar('val/loss', avg_metrics['loss'], epoch)
        self.tb_logger.log_scalar('val/loss_video', avg_metrics['loss_video'], epoch)
        self.tb_logger.log_scalar('val/loss_snippet', avg_metrics['loss_snippet'], epoch)
        
        return avg_metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        is_best: bool = False,
        **kwargs
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            is_best: Whether this is the best model
            **kwargs: Additional items to save
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=val_metrics['loss'],
            metrics={
                'train': train_metrics,
                'val': val_metrics
            },
            checkpoint_path=checkpoint_path,
            is_best=is_best,
            global_step=self.global_step,
            **kwargs
        )
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device
        )
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_every: int = 1,
        validate_every: int = 1,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
        
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        patience_counter = 0
        
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {epochs}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch + 1, epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch, scheduler)
            history['train_loss'].append(train_metrics['loss'])
            
            # Validation
            val_metrics = None
            if epoch % validate_every == 0:
                val_metrics = self.validate(val_loader, epoch)
                history['val_loss'].append(val_metrics['loss'])
                history['epochs'].append(epoch)
                
                # Check for improvement
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    logger.info(f"✓ New best validation loss: {self.best_val_loss:.4f}")
                else:
                    patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch summary
            logger.info("")
            logger.info(f"Epoch {epoch}/{epochs} - {epoch_time:.2f}s")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
                if is_best:
                    logger.info(f"  ★ Best Model")
            logger.info("")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics or {},
                    is_best=is_best if val_metrics else False
                )
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            self.current_epoch = epoch
        
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"TRAINING COMPLETE - Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)
        
        self.tb_logger.close()
        
        return history
