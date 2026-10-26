"""
Main training script for MSL
Supports both regular training and self-training
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import UCFCrimeFeatureDataset, collate_fn_pad
from src.models import MSLModel
from src.losses import MSLRankingLoss
from src.training import MSLTrainer, SelfTrainer, create_stages_from_config
from src.utils import setup_logging, print_training_header


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict) -> tuple:
    """Create train and validation data loaders."""
    
    dataset_type = config.get('dataset', {}).get('type', 'npy')
    
    if dataset_type == 'h5':
        # H5 format (UCF-Crime / SHT real features)
        from src.datasets import UCFH5Dataset, collate_fn_h5
        
        train_dataset = UCFH5Dataset(
            h5_path=config['dataset']['train']['h5_path'],
            list_path=config['dataset']['train']['list_path'],
            gt_path=config['dataset']['train'].get('gt_path'),
            max_length=config['dataset'].get('max_length', 320),
            mode='train',
            return_video_name=True
        )
        
        val_dataset = UCFH5Dataset(
            h5_path=config['dataset']['test']['h5_path'],
            list_path=config['dataset']['test']['list_path'],
            gt_path=config['dataset']['test'].get('gt_path'),
            max_length=config['dataset'].get('max_length', 320),
            mode='test',
            return_video_name=True
        )
        
        collate_function = collate_fn_h5
        
    else:
        # NPY format (synthetic or legacy)
        from src.datasets import UCFCrimeFeatureDataset, collate_fn_pad
        
        train_dataset = UCFCrimeFeatureDataset(
            feature_dir=config['data']['feature_dir'],
            split=config['data']['train_split'],
            feature_dim=config['data']['feature_dim'],
            max_len=config['data'].get('max_len'),
            normalize=config['data'].get('normalize', False),
            return_video_id=True
        )
        
        val_dataset = UCFCrimeFeatureDataset(
            feature_dir=config['data']['feature_dir'],
            split=config['data']['test_split'],
            feature_dim=config['data']['feature_dim'],
            max_len=config['data'].get('max_len'),
            normalize=config['data'].get('normalize', False),
            return_video_id=True
        )
        
        collate_function = collate_fn_pad
    
    # Get batch size and num_workers
    if 'training' in config:
        batch_size = config['training'].get('batch_size', 16)
        num_workers = config['dataset'].get('num_workers', 4)
        pin_memory = config['dataset'].get('pin_memory', True)
    else:
        batch_size = config['data']['batch_size']
        num_workers = config['data'].get('num_workers', 0)
        pin_memory = config['data'].get('pin_memory', True)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_function,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_function,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def create_model(config: dict) -> torch.nn.Module:
    """Create MSL model from configuration."""
    model = MSLModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ff_dim=config['model'].get('ff_dim', 2048),
        dropout=config['model'].get('dropout', 0.1),
        sequence_length=config['model']['sequence_length'],
        sequence_stride=config['model']['sequence_stride'],
        num_classes=config['model'].get('num_classes', 2),
        use_pos_encoding=config['model'].get('use_pos_encoding', True),
        aggregation=config['model'].get('aggregation', 'mean')
    )
    
    return model


def create_loss_fn(config: dict) -> torch.nn.Module:
    """Create loss function from configuration."""
    loss_fn = MSLRankingLoss(
        margin=config['loss']['margin'],
        weight_video=config['loss']['weight_video'],
        weight_snippet=config['loss']['weight_snippet'],
        top_k_ratio=config.get('self_training', {}).get('top_k_ratio', 0.8)
    )
    
    return loss_fn


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from configuration."""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=config['training'].get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict, epochs: int):
    """Create learning rate scheduler from configuration."""
    scheduler_name = config['training'].get('scheduler', 'none').lower()
    
    if scheduler_name == 'none':
        return None
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=config['training'].get('lr_min', 1e-6)
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('lr_step_size', 10),
            gamma=config['training'].get('lr_gamma', 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def main(args):
    """Main training function."""
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override epochs if provided via command line
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        print(f"Overriding epochs from command line: {args.epochs}")
    
    # Setup logging (handle missing logging section)
    if 'logging' not in config:
        config['logging'] = {}
    
    log_file = Path(config['logging'].get('log_file', 'logs/training.log'))
    logger = setup_logging(
        log_file=log_file,
        level=config['logging'].get('level', 'INFO')
    )
    
    logger.info("=" * 80)
    logger.info("MSL TRAINING")
    logger.info("=" * 80)
    
    # Print configuration
    print_training_header(config)
    
    # Set random seed
    if 'seed' in config['experiment']:
        seed = config['experiment']['seed']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to: {seed}")
    
    # Device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Create data loaders
    logger.info("\nLoading data...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create loss function
    loss_fn = create_loss_fn(config)
    logger.info(f"Loss function: MSLRankingLoss (margin={config['loss']['margin']})")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    
    # Create trainer
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    log_dir = Path(config['logging'].get('tensorboard_dir', 'logs/tensorboard'))
    
    trainer = MSLTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir if config['logging'].get('tensorboard', True) else None,
        use_amp=config['training'].get('use_amp', False),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        log_interval=config['training'].get('log_interval', 10)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Training
    if config.get('self_training', {}).get('enabled', False):
        # Self-training mode
        logger.info("\n" + "=" * 80)
        logger.info("SELF-TRAINING MODE")
        logger.info("=" * 80)
        
        stages = create_stages_from_config(config)
        self_trainer = SelfTrainer(
            trainer=trainer,
            stages=stages,
            checkpoint_dir=checkpoint_dir
        )
        
        # Create scheduler function for each stage
        def scheduler_fn(opt, epochs):
            return create_scheduler(opt, config, epochs)
        
        history = self_trainer.run(
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler_fn=scheduler_fn,
            save_every=config['training'].get('save_every', 1),
            validate_every=config['training'].get('val_interval', 1),
            early_stopping_patience=config['training'].get('early_stopping_patience')
        )
    else:
        # Regular training mode
        logger.info("\n" + "=" * 80)
        logger.info("REGULAR TRAINING MODE")
        logger.info("=" * 80)
        
        epochs = config['training']['epochs']
        scheduler = create_scheduler(optimizer, config, epochs)
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            scheduler=scheduler,
            save_every=config['training'].get('save_every', 1),
            validate_every=config['training'].get('val_interval', 1),
            early_stopping_patience=config['training'].get('early_stopping_patience')
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINISHED")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved in: {checkpoint_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MSL model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config file)'
    )
    
    args = parser.parse_args()
    main(args)
