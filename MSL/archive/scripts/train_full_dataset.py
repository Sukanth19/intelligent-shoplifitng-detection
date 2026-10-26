"""
Full Dataset Training - Use ALL available training videos (up to 950)
This is the closest to the base paper's experimental setup
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
print("MSL TRAINING - FULL DATASET (ALL AVAILABLE VIDEOS)")
print("="*80)

# Config - FULL SCALE
EPOCHS = 75  # Paper uses 50-100, we use 75
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.00005  # Smaller for stability with large dataset
MAX_LENGTH = 100
USE_ALL_VIDEOS = True  # Use all available videos

print(f"\nConfiguration:")
print(f"  Dataset: FULL UCF-Crime training set")
print(f"  Videos: ALL AVAILABLE (target: ~950)")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Device: {DEVICE}")

# Paths - Use TRAIN set (not test)
data_dir = Path("../data/UCF-VideoSwin-Features")

# Try train set first
h5_path = data_dir / "ucf_swin_train_feature_224288224288.h5"
list_path = data_dir / "ucf-train-filter.list"
gt_path = data_dir / "ucf_gt_16.npy"

if not h5_path.exists():
    print(f"\n⚠️  Train H5 not found, using test set...")
    h5_path = data_dir / "ucf_swin_test_feature_224288224288.h5"
    list_path = data_dir / "ucf-test.list"

print(f"\nLoading dataset from: {h5_path.name}")

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
    print(f"Mapped videos: {len(dataset.video_to_key)}")
    
    # Get ALL valid videos (or as many as possible)
    normal_indices = []
    anomaly_indices = []
    
    print(f"\nScanning all videos...")
    for i in range(len(dataset)):
        video_name = dataset.video_names[i]
        if video_name not in dataset.video_to_key:
            continue
        
        sample = dataset[i]
        label = sample['label']
        
        if label == 0:
            normal_indices.append(i)
        else:
            anomaly_indices.append(i)
    
    print(f"\nFound:")
    print(f"  Normal videos: {len(normal_indices)}")
    print(f"  Anomaly videos: {len(anomaly_indices)}")
    print(f"  Total: {len(normal_indices) + len(anomaly_indices)}")
    
    if len(normal_indices) == 0:
        print("\n⚠️  WARNING: No normal videos found!")
        print("This means we're using a dataset with ONLY anomalies.")
        print("UCF-Crime train set typically has no normal videos.")
        print("For ranking loss, we need BOTH classes.")
        print("\nUsing all available videos anyway...")
        # Use all anomaly videos
        all_indices = anomaly_indices
    elif len(anomaly_indices) == 0:
        print("\n⚠️  WARNING: No anomaly videos found!")
        all_indices = normal_indices
    else:
        # Balanced approach - use all of smaller class, match from larger class
        min_count = min(len(normal_indices), len(anomaly_indices))
        
        # For full dataset, we want to use as many as possible
        # Use all videos from both classes
        all_indices = []
        
        # Interleave for balanced batches
        max_count = max(len(normal_indices), len(anomaly_indices))
        for i in range(max_count):
            if i < len(normal_indices):
                all_indices.append(normal_indices[i])
            if i < len(anomaly_indices):
                all_indices.append(anomaly_indices[i])
    
    print(f"  Using {len(all_indices)} videos for training")
    
    if len(all_indices) == 0:
        print("\n❌ No valid videos found!")
        sys.exit(1)
    
    # Create subset
    dataset = Subset(dataset, all_indices)
    
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
print(f"Samples per epoch: {len(loader) * BATCH_SIZE}")

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

# Better initialization
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() > 1:
        nn.init.xavier_uniform_(param, gain=0.5)
    elif 'bias' in name:
        nn.init.constant_(param, 0.0)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Loss
criterion = MSLRankingLoss(margin=0.5)

# Training loop
print("\n" + "="*80)
print("FULL DATASET TRAINING - THIS WILL TAKE LONGER")
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
                
                if n_pairs > 0:
                    # Backward
                    loss.backward()
                    
                    # Gradient clipping
                    with torch.no_grad():
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        
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
                    
                    # Print every 10 batches
                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                        print(f"Epoch [{epoch+1}/{EPOCHS}] "
                              f"Batch [{batch_idx+1}/{len(loader)}] "
                              f"Loss: {loss.item():.4f}")
            
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
            print(f"    Batches: {batches_with_pairs}/{len(loader)}")
            print(f"{'='*80}\n")
        
        # Save every 10 epochs
        if (epoch + 1) % 10 == 0 and batch_count > 0:
            checkpoint_path = Path("checkpoints") / f"fulldataset_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'learning_rate': LEARNING_RATE,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'num_videos': len(all_indices)
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
print("FULL DATASET TRAINING COMPLETE!")
print("="*80)
print(f"Total videos: {len(all_indices)}")
print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"Best loss: {best_loss:.4f} (Epoch {best_epoch})")

# Save final
if batch_count > 0:
    checkpoint_path = Path("checkpoints") / "fulldataset_final.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'loss': avg_loss,
        'learning_rate': LEARNING_RATE,
        'total_time': total_time,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'num_videos': len(all_indices),
        'epochs_trained': EPOCHS
    }, checkpoint_path)
    
    print(f"\n✅ Final checkpoint saved: {checkpoint_path}")
    print(f"\n✨ FULL DATASET TRAINING COMPLETE!")
    print(f"✨ Now evaluate: python evaluate.py --checkpoint {checkpoint_path}")
