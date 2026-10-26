"""
Unit tests for training infrastructure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.datasets import UCFCrimeFeatureDataset, collate_fn_pad
from src.models import MSLModel
from src.losses import MSLRankingLoss
from src.training import MSLTrainer, SelfTrainer, SelfTrainingStage

print("=" * 80)
print("RUNNING TRAINING INFRASTRUCTURE TESTS")
print("=" * 80)

tests_passed = 0
tests_failed = 0

def test(name, condition, message=""):
    """Simple test function"""
    global tests_passed, tests_failed
    if condition:
        print(f"✓ {name}")
        tests_passed += 1
    else:
        print(f"✗ {name}: {message}")
        tests_failed += 1

# Check if synthetic data exists
data_dir = Path(__file__).parent.parent / "data" / "synthetic"
if not data_dir.exists():
    print(f"\n❌ Synthetic data not found: {data_dir}")
    print("Run: python scripts/create_synthetic_data.py")
    sys.exit(1)

# Test 1: Create trainer
print("\n1. Testing MSLTrainer initialization...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    loss_fn = MSLRankingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trainer = MSLTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cpu',
        checkpoint_dir=Path('checkpoints_test'),
        use_amp=False
    )
    
    test("Trainer creation", True)
    test("Trainer has model", trainer.model is not None)
    test("Trainer has optimizer", trainer.optimizer is not None)
    test("Trainer device", trainer.device == 'cpu')
except Exception as e:
    test("Trainer initialization", False, str(e))

# Test 2: Create dataset and loader
print("\n2. Testing dataset and loader...")
try:
    dataset = UCFCrimeFeatureDataset(
        feature_dir=data_dir,
        split="train",
        return_video_id=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_pad
    )
    
    test("Dataset created", len(dataset) > 0)
    test("Loader created", len(loader) > 0)
    
    # Get one batch
    batch = next(iter(loader))
    test("Batch has features", 'features' in batch)
    test("Batch has labels", 'labels' in batch)
    test("Batch has mask", 'mask' in batch)
except Exception as e:
    test("Dataset and loader", False, str(e))

# Test 3: Train one batch
print("\n3. Testing single batch training...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    loss_fn = MSLRankingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Get batch
    dataset = UCFCrimeFeatureDataset(feature_dir=data_dir, split="train")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_pad)
    batch = next(iter(loader))
    
    # Forward pass
    model.train()
    output = model(batch['features'], batch['mask'])
    loss_dict = loss_fn(output['video_scores'], output['snippet_scores'], batch['labels'])
    loss = loss_dict['loss']
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    test("Single batch forward", True)
    test("Loss computed", loss.item() >= 0)
    test("Backward pass", True)
    test("Optimizer step", True)
except Exception as e:
    test("Single batch training", False, str(e))

# Test 4: Train one epoch (small subset)
print("\n4. Testing one epoch training...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    loss_fn = MSLRankingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trainer = MSLTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cpu',
        checkpoint_dir=Path('checkpoints_test'),
        log_interval=1000  # Suppress logs
    )
    
    # Small dataset
    dataset = UCFCrimeFeatureDataset(feature_dir=data_dir, split="train")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_pad)
    
    # Train one epoch
    metrics = trainer.train_epoch(loader, epoch=1)
    
    test("Epoch training", 'loss' in metrics)
    test("Loss value", metrics['loss'] >= 0)
    test("Metrics returned", len(metrics) > 0)
except Exception as e:
    test("One epoch training", False, str(e))

# Test 5: Validation
print("\n5. Testing validation...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    loss_fn = MSLRankingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trainer = MSLTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cpu',
        checkpoint_dir=Path('checkpoints_test'),
        log_interval=1000
    )
    
    dataset = UCFCrimeFeatureDataset(feature_dir=data_dir, split="test")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_pad)
    
    metrics = trainer.validate(loader, epoch=1)
    
    test("Validation", 'loss' in metrics)
    test("Val loss value", metrics['loss'] >= 0)
except Exception as e:
    test("Validation", False, str(e))

# Test 6: Checkpoint saving/loading
print("\n6. Testing checkpoint save/load...")
try:
    from src.utils import save_checkpoint, load_checkpoint
    
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters())
    
    checkpoint_path = Path('checkpoints_test/test_checkpoint.pt')
    
    # Save
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        loss=0.5,
        metrics={'auc': 0.8},
        checkpoint_path=checkpoint_path
    )
    test("Checkpoint saved", checkpoint_path.exists())
    
    # Load
    model2 = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    checkpoint = load_checkpoint(checkpoint_path, model2, device='cpu')
    
    test("Checkpoint loaded", 'epoch' in checkpoint)
    test("Checkpoint epoch", checkpoint['epoch'] == 1)
    
    # Clean up
    if checkpoint_path.exists():
        checkpoint_path.unlink()
except Exception as e:
    test("Checkpoint save/load", False, str(e))

# Test 7: Self-training stages
print("\n7. Testing self-training stages...")
try:
    stage1 = SelfTrainingStage(
        name="stage1",
        sequence_length=64,
        stride=32,
        epochs=2
    )
    
    stage2 = SelfTrainingStage(
        name="stage2",
        sequence_length=32,
        stride=16,
        epochs=2
    )
    
    test("Stage 1 created", stage1.sequence_length == 64)
    test("Stage 2 created", stage2.sequence_length == 32)
    test("Stage epochs", stage1.epochs == 2)
except Exception as e:
    test("Self-training stages", False, str(e))

# Test 8: Complete training (2 epochs, small model)
print("\n8. Testing complete training loop (2 epochs)...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=128, num_layers=1, num_heads=4, sequence_length=32)
    loss_fn = MSLRankingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = MSLTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cpu',
        checkpoint_dir=Path('checkpoints_test'),
        log_interval=1000
    )
    
    train_dataset = UCFCrimeFeatureDataset(feature_dir=data_dir, split="train")
    val_dataset = UCFCrimeFeatureDataset(feature_dir=data_dir, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn_pad)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        save_every=10,  # Don't save
        validate_every=1
    )
    
    test("Training completed", len(history['train_loss']) == 2)
    test("Validation ran", len(history['val_loss']) == 2)
    test("Loss decreased or stable", history['train_loss'][1] <= history['train_loss'][0] * 1.1)
except Exception as e:
    test("Complete training loop", False, str(e))

# Clean up test checkpoints
import shutil
test_checkpoint_dir = Path('checkpoints_test')
if test_checkpoint_dir.exists():
    shutil.rmtree(test_checkpoint_dir)

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✓ Passed: {tests_passed}")
print(f"✗ Failed: {tests_failed}")
print(f"Total: {tests_passed + tests_failed}")

if tests_failed == 0:
    print("\n🎉 ALL TRAINING TESTS PASSED!")
    sys.exit(0)
else:
    print(f"\n❌ {tests_failed} test(s) failed")
    sys.exit(1)
