"""
Self-Training for MSL
Progressive sequence length refinement
"""

import torch
import logging
from typing import Dict, List, Optional
from pathlib import Path
from torch.utils.data import DataLoader

from .trainer import MSLTrainer

logger = logging.getLogger(__name__)


class SelfTrainingStage:
    """Configuration for a self-training stage."""
    
    def __init__(
        self,
        name: str,
        sequence_length: int,
        stride: int,
        epochs: int,
        learning_rate: Optional[float] = None
    ):
        """
        Args:
            name: Stage name (e.g., "stage1")
            sequence_length: Number of snippets per sequence
            stride: Stride between sequences
            epochs: Number of epochs for this stage
            learning_rate: Optional learning rate override
        """
        self.name = name
        self.sequence_length = sequence_length
        self.stride = stride
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def __repr__(self):
        return (f"SelfTrainingStage(name={self.name}, "
                f"length={self.sequence_length}, "
                f"stride={self.stride}, "
                f"epochs={self.epochs})")


class SelfTrainer:
    """
    Self-Training Manager for MSL.
    
    Implements progressive refinement:
    - Stage 1: Long sequences (easier anomaly detection)
    - Stage 2: Medium sequences (balanced)
    - Stage 3: Short sequences (fine-grained localization)
    """
    
    def __init__(
        self,
        trainer: MSLTrainer,
        stages: List[SelfTrainingStage],
        checkpoint_dir: Path = Path('checkpoints')
    ):
        """
        Args:
            trainer: MSLTrainer instance
            stages: List of self-training stages
            checkpoint_dir: Directory for stage checkpoints
        """
        self.trainer = trainer
        self.stages = stages
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Self-Training initialized")
        logger.info(f"Number of stages: {len(stages)}")
        for i, stage in enumerate(stages, 1):
            logger.info(f"  Stage {i}: {stage}")
    
    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler_fn = None,
        **train_kwargs
    ) -> Dict[str, List]:
        """
        Run all self-training stages.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            scheduler_fn: Function to create scheduler for each stage
            **train_kwargs: Additional arguments for trainer.train()
        
        Returns:
            Dictionary with training history for all stages
        """
        history = {
            'stages': [],
            'train_loss': [],
            'val_loss': []
        }
        
        logger.info("=" * 80)
        logger.info("STARTING SELF-TRAINING")
        logger.info("=" * 80)
        
        for stage_idx, stage in enumerate(self.stages, 1):
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"STAGE {stage_idx}/{len(self.stages)}: {stage.name}")
            logger.info("=" * 80)
            logger.info(f"Sequence Length: {stage.sequence_length}")
            logger.info(f"Stride: {stage.stride}")
            logger.info(f"Epochs: {stage.epochs}")
            logger.info("=" * 80)
            
            # Update model sequence parameters
            logger.info(f"Updating model sequence parameters...")
            self.trainer.model.update_sequence_params(
                sequence_length=stage.sequence_length,
                stride=stage.stride
            )
            logger.info(f"  Sequence length: {self.trainer.model.sequence_length}")
            logger.info(f"  Stride: {self.trainer.model.msl.stride}")
            
            # Update learning rate if specified
            if stage.learning_rate is not None:
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = stage.learning_rate
                logger.info(f"  Learning rate: {stage.learning_rate}")
            
            # Create scheduler for this stage
            scheduler = None
            if scheduler_fn is not None:
                scheduler = scheduler_fn(self.trainer.optimizer, stage.epochs)
                logger.info(f"  Scheduler: {type(scheduler).__name__}")
            
            # Train for this stage
            stage_history = self.trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=stage.epochs,
                scheduler=scheduler,
                **train_kwargs
            )
            
            # Save stage checkpoint
            stage_checkpoint_path = self.checkpoint_dir / f"stage_{stage.name}_final.pt"
            logger.info(f"Saving stage checkpoint: {stage_checkpoint_path}")
            self.trainer.save_checkpoint(
                epoch=self.trainer.current_epoch,
                train_metrics={'loss': stage_history['train_loss'][-1]},
                val_metrics={'loss': stage_history['val_loss'][-1]},
                is_best=False,
                stage=stage.name,
                sequence_length=stage.sequence_length,
                stride=stage.stride
            )
            
            # Update history
            history['stages'].append(stage.name)
            history['train_loss'].extend(stage_history['train_loss'])
            history['val_loss'].extend(stage_history['val_loss'])
            
            logger.info("")
            logger.info(f"Stage {stage.name} complete")
            logger.info(f"  Final train loss: {stage_history['train_loss'][-1]:.4f}")
            logger.info(f"  Final val loss: {stage_history['val_loss'][-1]:.4f}")
            logger.info("")
        
        logger.info("=" * 80)
        logger.info("SELF-TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total stages completed: {len(self.stages)}")
        logger.info(f"Final model sequence length: {self.trainer.model.sequence_length}")
        logger.info(f"Best validation loss: {self.trainer.best_val_loss:.4f}")
        logger.info("=" * 80)
        
        return history


def create_default_stages() -> List[SelfTrainingStage]:
    """
    Create default self-training stages based on MSL paper.
    
    Returns:
        List of SelfTrainingStage objects
    """
    stages = [
        SelfTrainingStage(
            name="stage1",
            sequence_length=64,
            stride=32,
            epochs=15
        ),
        SelfTrainingStage(
            name="stage2",
            sequence_length=48,
            stride=24,
            epochs=15
        ),
        SelfTrainingStage(
            name="stage3",
            sequence_length=32,
            stride=16,
            epochs=20
        )
    ]
    return stages


def create_stages_from_config(config: dict) -> List[SelfTrainingStage]:
    """
    Create self-training stages from configuration.
    
    Args:
        config: Configuration dictionary with 'self_training' section
    
    Returns:
        List of SelfTrainingStage objects
    """
    if 'self_training' not in config or 'stages' not in config['self_training']:
        logger.warning("No self-training configuration found, using defaults")
        return create_default_stages()
    
    stages = []
    for stage_config in config['self_training']['stages']:
        stage = SelfTrainingStage(
            name=stage_config['name'],
            sequence_length=stage_config['sequence_length'],
            stride=stage_config.get('stride', stage_config['sequence_length'] // 2),
            epochs=stage_config['epochs'],
            learning_rate=stage_config.get('learning_rate')
        )
        stages.append(stage)
    
    return stages
