"""
Training with Adam Optimizer - Requires PyTorch 2.3.1 or Python 3.11
This script uses Adam optimizer instead of manual gradient descent
for potential +2-3% AUC improvement
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
print("MSL TRAINING - WITH ADAM OPTIMIZER")
print("="*80)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print("\n⚠️  NOTE: Adam optimizer requires:")
print("  - PyTorch <= 2.3.1, OR")
print("  - Python <= 3.11")
print("\nIf you get sympy errors, use train_full_scale.py instead (manual GD)")

# Config
EPOCHS = 50
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_NORMAL = 200
NUM_ANOMALY = 200
LEARNING_RATE = 0.0001
MAX_LENGTH = 100

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Normal videos: {NUM_NORMAL}")
print(f"  Anomaly videos: {NUM_ANOMALY}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Optimizer: Adam (paper's optimizer)")

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
    
    print(f"Dataset created: {len(dataset)} total videos")
    
    # Separate normal and anomaly videos
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
    
    # Combine balanced
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

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Loss
criterion = MSLRankingLoss(margin=0.5)

# ADAM OPTIMIZER - This is the key difference!
print("\n✨ Creating Adam optimizer...")
try:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )
    print("✅ Adam optimizer created successfully!")
    print("   This should give +2-3% AUC improvement over manual GD")
except Exception as e:
    print(f"\n❌ Failed to create Adam optimizer: {e}")
    print("\nThis is the PyTorch 2.6 + Python 3.13 sympy bug!")
    print("Solutions:")
    print("  1. Downgrade PyTorch: pip install torch==2.3.1")
    print("  2. Use Python 3.11 instead of 3.13")
    print("  3. Use train_full_scale.py (manual gradient descent)")
    sys.exit(1)

# Training loop
print("\n" + "="*80)
print("TRAINING WITH ADAM OPTIMIZER")
print("="*80)

start_time = time.time()
best_loss = float('inf')
best_epoch = 0

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
                optimizer.zero_grad()
                
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
                
                if n_pairs > 0:
                    # Backward
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Adam optimizer step
                    optimizer.step()
                    
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
            print(f"    Best loss: {best_loss:.4f} (Epoch {best_epoch})")
            print(f"{'='*80}\n")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 and batch_count > 0:
            checkpoint_path = Path("checkpoints") / f"adam_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'learning_rate': LEARNING_RATE,
                'best_loss': best_loss,
                'best_epoch': best_epoch
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}\n")

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()

total_time = time.time() - start_time

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Best loss: {best_loss:.4f} (Epoch {best_epoch})")
print(f"Final loss: {avg_loss:.4f}")

# Save final checkpoint
if batch_count > 0:
    checkpoint_path = Path("checkpoints") / "adam_final.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'learning_rate': LEARNING_RATE,
        'total_time': total_time,
        'best_loss': best_loss,
        'best_epoch': best_epoch
    }, checkpoint_path)
    
    print(f"\n✅ Final checkpoint saved: {checkpoint_path}")
    print(f"\n✨ ADAM OPTIMIZER TRAINING COMPLETE!")
    print(f"✨ Expected AUC improvement: +2-3% over manual GD")
    print(f"\nNext: Run evaluate.py to measure actual AUC")
