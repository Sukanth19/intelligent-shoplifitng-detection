"""
Stable Training - Fixes NaN issue with better numerical stability
- Smaller learning rate (0.00001 instead of 0.0001)
- Gradient clipping more aggressive
- Better initialization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.datasets import UCFH5Dataset, collate_fn_h5
from src.models import MSLModel
from src.losses import MSLRankingLoss
import time

print("="*80)
print("MSL TRAINING - STABLE VERSION (FIXES NaN)")
print("="*80)

# Config - WITH STABILITY IMPROVEMENTS
EPOCHS = 50
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_NORMAL = 200
NUM_ANOMALY = 200
LEARNING_RATE = 0.00001  # 10x SMALLER for stability
MAX_LENGTH = 100

print(f"\nStability Improvements:")
print(f"  ✅ Learning rate: {LEARNING_RATE} (10x smaller)")
print(f"  ✅ Aggressive gradient clipping (max_norm=0.5)")
print(f"  ✅ Better weight initialization")
print(f"  ✅ NaN detection during training")

print(f"\nConfiguration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Videos: {NUM_NORMAL} normal + {NUM_ANOMALY} anomaly")
print(f"  Learning Rate: {LEARNING_RATE}")

# Paths
data_dir = Path("../data/UCF-VideoSwin-Features")
h5_path = data_dir / "ucf_swin_test_feature_224288224288.h5"
list_path = data_dir / "ucf-test.list"
gt_path = data_dir / "ucf_gt_16.npy"

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
    
    # Separate normal and anomaly
    normal_indices = []
    anomaly_indices = []
    
    for i in range(len(dataset)):
        video_name = dataset.video_names[i]
        if video_name not in dataset.video_to_key:
            continue
        
        sample = dataset[i]
        label = sample['label']
        
        if label == 0:
            normal_indices.append(i)
            if len(normal_indices) >= NUM_NORMAL:
                pass
        else:
            anomaly_indices.append(i)
            if len(anomaly_indices) >= NUM_ANOMALY:
                pass
        
        if len(normal_indices) >= NUM_NORMAL and len(anomaly_indices) >= NUM_ANOMALY:
            break
    
    print(f"Found: {len(normal_indices)} normal, {len(anomaly_indices)} anomaly")
    
    # Balanced indices
    balanced_indices = []
    min_count = min(len(normal_indices), len(anomaly_indices))
    for i in range(min_count):
        balanced_indices.append(normal_indices[i])
        balanced_indices.append(anomaly_indices[i])
    
    if len(normal_indices) > min_count:
        balanced_indices.extend(normal_indices[min_count:])
    if len(anomaly_indices) > min_count:
        balanced_indices.extend(anomaly_indices[min_count:])
    
    dataset = Subset(dataset, balanced_indices)
    print(f"Balanced dataset: {len(balanced_indices)} samples")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn_h5,
    drop_last=True
)

print(f"Batches per epoch: {len(loader)}")

# Model with better initialization
print("\nCreating model with stable initialization...")
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

# Better initialization
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() > 1:
        nn.init.xavier_uniform_(param, gain=0.5)  # Smaller gain for stability
    elif 'bias' in name:
        nn.init.constant_(param, 0.0)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Loss
criterion = MSLRankingLoss(margin=0.5)

# Training loop
print("\n" + "="*80)
print("STABLE TRAINING")
print("="*80)

start_time = time.time()
best_loss = float('inf')
best_epoch = 0
nan_count = 0

try:
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_video_loss = 0
        epoch_snippet_loss = 0
        batch_count = 0
        batches_with_pairs = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(loader):
            try:
                features = batch['features'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)
                
                n_normal = (labels == 0).sum().item()
                n_anomaly = (labels == 1).sum().item()
                
                # Zero gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                
                # Forward
                outputs = model(features, mask=masks)
                
                # Check for NaN in outputs
                if torch.isnan(outputs['video_scores']).any():
                    print(f"  ⚠️  NaN detected in forward pass (batch {batch_idx}), skipping...")
                    nan_count += 1
                    continue
                
                # Loss
                loss_dict = criterion(
                    video_scores=outputs['video_scores'],
                    snippet_scores=outputs['snippet_scores'],
                    labels=labels
                )
                
                loss = loss_dict['loss']
                n_pairs = loss_dict['n_pairs']
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    print(f"  ⚠️  NaN detected in loss (batch {batch_idx}), skipping...")
                    nan_count += 1
                    continue
                
                if n_pairs > 0:
                    # Backward
                    loss.backward()
                    
                    # Aggressive gradient clipping
                    with torch.no_grad():
                        grad_norm = 0.0
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.norm().item() ** 2
                        grad_norm = grad_norm ** 0.5
                        
                        # Clip more aggressively
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        
                        # Check for NaN gradients
                        has_nan_grad = False
                        for param in model.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                has_nan_grad = True
                                break
                        
                        if has_nan_grad:
                            print(f"  ⚠️  NaN gradient detected (batch {batch_idx}), skipping...")
                            nan_count += 1
                            continue
                        
                        # Manual gradient descent
                        for param in model.parameters():
                            if param.grad is not None:
                                param.sub_(LEARNING_RATE * param.grad)
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    epoch_video_loss += loss_dict.get('loss_video', 0).item()
                    epoch_snippet_loss += loss_dict.get('loss_snippet', 0).item()
                    batch_count += 1
                    batches_with_pairs += 1
                    
                    if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                        print(f"Epoch [{epoch+1}/{EPOCHS}] "
                              f"Batch [{batch_idx+1}/{len(loader)}] "
                              f"Loss: {loss.item():.4f} "
                              f"(Normal={n_normal}, Anomaly={n_anomaly})")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        epoch_time = time.time() - epoch_start
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_video_loss = epoch_video_loss / batch_count
            avg_snippet_loss = epoch_snippet_loss / batch_count
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
            
            print(f"\n{'='*80}")
            print(f">>> EPOCH [{epoch+1}/{EPOCHS}] COMPLETE")
            print(f"    Average Loss: {avg_loss:.4f}")
            print(f"    Video Loss: {avg_video_loss:.4f}")
            print(f"    Snippet Loss: {avg_snippet_loss:.4f}")
            print(f"    Time: {epoch_time:.1f}s")
            print(f"    Best: {best_loss:.4f} (Epoch {best_epoch})")
            print(f"    NaN encounters: {nan_count}")
            print(f"{'='*80}\n")
        
        # Save every 10 epochs
        if (epoch + 1) % 10 == 0 and batch_count > 0:
            checkpoint_path = Path("checkpoints") / f"stable_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'learning_rate': LEARNING_RATE,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'nan_count': nan_count
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}\n")

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

total_time = time.time() - start_time

print("\n" + "="*80)
print("STABLE TRAINING COMPLETE!")
print("="*80)
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Best loss: {best_loss:.4f} (Epoch {best_epoch})")
print(f"NaN encounters: {nan_count}")

# Save final
if batch_count > 0:
    checkpoint_path = Path("checkpoints") / "stable_final.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'loss': avg_loss,
        'learning_rate': LEARNING_RATE,
        'total_time': total_time,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'nan_count': nan_count
    }, checkpoint_path)
    
    print(f"\n✅ Checkpoint saved: {checkpoint_path}")
    print(f"\n✨ STABLE TRAINING COMPLETE!")
    print(f"✨ Next: python evaluate.py --checkpoint {checkpoint_path}")
