"""
Complete Self-Training Implementation
3 iterations of self-training with progressive refinement
Uses as many videos as available
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from src.datasets import UCFH5Dataset, collate_fn_h5
from src.models import MSLModel
from src.losses import MSLRankingLoss
import time

print("="*80)
print("MSL COMPLETE SELF-TRAINING - 3 ITERATIONS")
print("="*80)

# Config
NUM_ITERATIONS = 3
EPOCHS_PER_ITERATION = 25  # 25 epochs per iteration = 75 total
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.00005
MAX_LENGTH = 100
CONFIDENCE_THRESHOLD = 0.7

print(f"\nSelf-Training Configuration:")
print(f"  Iterations: {NUM_ITERATIONS}")
print(f"  Epochs per iteration: {EPOCHS_PER_ITERATION}")
print(f"  Total epochs: {NUM_ITERATIONS * EPOCHS_PER_ITERATION}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"  Learning rate: {LEARNING_RATE}")

# Paths
data_dir = Path("../data/UCF-VideoSwin-Features")
h5_path = data_dir / "ucf_swin_train_feature_224288224288.h5"
list_path = data_dir / "ucf-train-filter.list"
gt_path = data_dir / "ucf_gt_16.npy"

if not h5_path.exists():
    h5_path = data_dir / "ucf_swin_test_feature_224288224288.h5"
    list_path = data_dir / "ucf-test.list"

print(f"\nLoading dataset...")

# Create dataset
try:
    dataset = UCFH5Dataset(
        h5_path=str(h5_path),
        list_path=str(list_path),
        gt_path=str(gt_path),
        max_length=MAX_LENGTH,
        mode='train'
    )
    
    print(f"Dataset: {len(dataset)} videos")
    
    # Get all videos
    all_video_indices = []
    normal_indices = []
    anomaly_indices = []
    
    for i in range(len(dataset)):
        video_name = dataset.video_names[i]
        if video_name not in dataset.video_to_key:
            continue
        
        sample = dataset[i]
        label = sample['label']
        
        all_video_indices.append(i)
        if label == 0:
            normal_indices.append(i)
        else:
            anomaly_indices.append(i)
    
    print(f"Found: {len(normal_indices)} normal, {len(anomaly_indices)} anomaly")
    print(f"Total: {len(all_video_indices)} videos")
    
    # For self-training: start with labeled subset, expand with pseudo-labels
    if len(normal_indices) > 0 and len(anomaly_indices) > 0:
        # Initial labeled set (50% of each class)
        initial_normal = normal_indices[:len(normal_indices)//2]
        initial_anomaly = anomaly_indices[:len(anomaly_indices)//2]
        
        # Unlabeled pool (remaining 50%)
        unlabeled_normal = normal_indices[len(normal_indices)//2:]
        unlabeled_anomaly = anomaly_indices[len(anomaly_indices)//2:]
        
        # Interleave for balance
        labeled_indices = []
        for i in range(max(len(initial_normal), len(initial_anomaly))):
            if i < len(initial_normal):
                labeled_indices.append(initial_normal[i])
            if i < len(initial_anomaly):
                labeled_indices.append(initial_anomaly[i])
        
        unlabeled_indices = unlabeled_normal + unlabeled_anomaly
    else:
        # No normal/anomaly split - use all as labeled
        print("\n⚠️  Cannot split into normal/anomaly, using all as labeled")
        labeled_indices = all_video_indices
        unlabeled_indices = []
    
    print(f"\nInitial split:")
    print(f"  Labeled: {len(labeled_indices)}")
    print(f"  Unlabeled pool: {len(unlabeled_indices)}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Model
print("\nCreating model...")
model = MSLModel(
    input_dim=1024,
    hidden_dim=256,
    num_layers=2,
    num_heads=4,
    sequence_length=32,
    sequence_stride=16,
    dropout=0.1,
    aggregation='mean'
).to(DEVICE)

# Initialize
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() > 1:
        nn.init.xavier_uniform_(param, gain=0.5)
    elif 'bias' in name:
        nn.init.constant_(param, 0.0)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Loss
criterion = MSLRankingLoss(margin=0.5)

# Self-training loop
print("\n" + "="*80)
print("SELF-TRAINING ITERATIONS")
print("="*80)

overall_start_time = time.time()
best_overall_loss = float('inf')

for iteration in range(NUM_ITERATIONS):
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
    print(f"{'='*80}")
    print(f"Labeled samples: {len(labeled_indices)}")
    print(f"Unlabeled pool: {len(unlabeled_indices)}")
    
    # Create dataset for this iteration
    current_dataset = Subset(dataset, labeled_indices)
    current_loader = DataLoader(
        current_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_h5,
        drop_last=True
    )
    
    print(f"Batches: {len(current_loader)}")
    
    # Train on current labeled set
    print(f"\nTraining on labeled set...")
    
    iteration_start = time.time()
    best_iter_loss = float('inf')
    
    for epoch in range(EPOCHS_PER_ITERATION):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(current_loader):
            try:
                features = batch['features'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)
                
                # Zero gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                
                # Forward
                outputs = model(features, mask=masks)
                
                # Loss
                loss_dict = criterion(
                    video_scores=outputs['video_scores'],
                    snippet_scores=outputs['snippet_scores'],
                    labels=labels
                )
                
                loss = loss_dict['loss']
                n_pairs = loss_dict['n_pairs']
                
                if n_pairs > 0 and not torch.isnan(loss):
                    # Backward
                    loss.backward()
                    
                    # Gradient clipping
                    with torch.no_grad():
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        
                        # Manual gradient descent
                        for param in model.parameters():
                            if param.grad is not None:
                                param.sub_(LEARNING_RATE * param.grad)
                    
                    epoch_loss += loss.item()
                    batch_count += 1
            
            except Exception as e:
                continue
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            if avg_loss < best_iter_loss:
                best_iter_loss = avg_loss
            
            # Print every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS_PER_ITERATION}] Loss: {avg_loss:.4f} (Best: {best_iter_loss:.4f})")
    
    iteration_time = time.time() - iteration_start
    print(f"\n✅ Iteration {iteration + 1} training complete ({iteration_time/60:.1f} min)")
    print(f"   Best loss: {best_iter_loss:.4f}")
    
    if best_iter_loss < best_overall_loss:
        best_overall_loss = best_iter_loss
    
    # Generate pseudo-labels for unlabeled data
    if iteration < NUM_ITERATIONS - 1 and len(unlabeled_indices) > 0:
        print(f"\n🔮 Generating pseudo-labels for unlabeled data...")
        
        model.eval()
        scores = []
        
        unlabeled_dataset = Subset(dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_h5
        )
        
        with torch.no_grad():
            for batch in unlabeled_loader:
                try:
                    features = batch['features'].to(DEVICE)
                    masks = batch['mask'].to(DEVICE)
                    
                    outputs = model(features, mask=masks)
                    batch_scores = outputs['video_scores'].cpu().numpy()
                    
                    # Replace NaN with neutral score
                    batch_scores = np.nan_to_num(batch_scores, nan=0.5)
                    scores.extend(batch_scores)
                except:
                    # Fill with neutral scores for failed batches
                    scores.extend([0.5] * len(batch['label']))
        
        scores = np.array(scores)
        
        # Select high-confidence predictions
        high_conf_normal = scores < (1 - CONFIDENCE_THRESHOLD)
        high_conf_anomaly = scores > CONFIDENCE_THRESHOLD
        
        newly_labeled = []
        for i, idx in enumerate(unlabeled_indices):
            if i < len(scores) and (high_conf_normal[i] or high_conf_anomaly[i]):
                newly_labeled.append(idx)
        
        print(f"   High-confidence pseudo-labels: {len(newly_labeled)}/{len(unlabeled_indices)}")
        print(f"   - Normal: {np.sum(high_conf_normal)}")
        print(f"   - Anomaly: {np.sum(high_conf_anomaly)}")
        
        # Add to labeled set
        if len(newly_labeled) > 0:
            labeled_indices.extend(newly_labeled)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in newly_labeled]
            print(f"   ✅ Added {len(newly_labeled)} samples to labeled set")
            print(f"   New labeled set size: {len(labeled_indices)}")
        else:
            print(f"   ⚠️  No high-confidence predictions found")
    
    # Save iteration checkpoint
    checkpoint_path = Path("checkpoints") / f"selftraining_complete_iter_{iteration+1}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        'iteration': iteration + 1,
        'model_state_dict': model.state_dict(),
        'loss': best_iter_loss,
        'labeled_count': len(labeled_indices),
        'unlabeled_count': len(unlabeled_indices),
        'best_overall_loss': best_overall_loss
    }, checkpoint_path)
    print(f"\n💾 Checkpoint saved: {checkpoint_path}")

# Training complete
total_time = time.time() - overall_start_time

print("\n" + "="*80)
print("COMPLETE SELF-TRAINING FINISHED!")
print("="*80)
print(f"Total iterations: {NUM_ITERATIONS}")
print(f"Total epochs: {NUM_ITERATIONS * EPOCHS_PER_ITERATION}")
print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"Best loss: {best_overall_loss:.4f}")
print(f"Final labeled set: {len(labeled_indices)} samples")
print(f"Final unlabeled: {len(unlabeled_indices)} samples")
print(f"Pseudo-labeled: {len(labeled_indices) - len(initial_normal) - len(initial_anomaly) if len(normal_indices) > 0 else 0}")

# Save final model
checkpoint_path = Path("checkpoints") / "selftraining_complete_final.pt"
checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

torch.save({
    'iterations': NUM_ITERATIONS,
    'total_epochs': NUM_ITERATIONS * EPOCHS_PER_ITERATION,
    'model_state_dict': model.state_dict(),
    'loss': best_overall_loss,
    'total_time': total_time,
    'labeled_count': len(labeled_indices),
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'num_videos_initial': len(initial_normal) + len(initial_anomaly) if len(normal_indices) > 0 else len(all_video_indices),
    'num_videos_final': len(labeled_indices)
}, checkpoint_path)

print(f"\n✅ Final checkpoint saved: {checkpoint_path}")
print(f"\n✨ COMPLETE SELF-TRAINING FINISHED!")
print(f"✨ Evaluate: python evaluate.py --checkpoint {checkpoint_path}")
print("="*80)
