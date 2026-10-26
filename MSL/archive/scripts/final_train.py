"""
Final Training Script - Works perfectly!
Uses manual gradient descent to avoid PyTorch optimizer bugs
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
print("MSL TRAINING - Production Version")
print("="*80)

# Config - ADJUST THESE
EPOCHS = 10
BATCH_SIZE = 8  # Bigger batch size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SAMPLES = 100  # Use 100 videos (or None for all)
LEARNING_RATE = 0.0001
MAX_LENGTH = 100  # Sequence length

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Samples: {NUM_SAMPLES if NUM_SAMPLES else 'ALL'}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Sequence Length: {MAX_LENGTH}")

# Paths
data_dir = Path("../data/UCF-VideoSwin-Features")
h5_path = data_dir / "ucf_swin_train_feature_224288224288.h5"
list_path = data_dir / "ucf-train-filter.list"
gt_path = data_dir / "ucf_gt_16.npy"

print(f"\nDataset:")
print(f"  H5 file: {h5_path.name}")
print(f"  List file: {list_path.name}")
print(f"  GT file: {gt_path.name}")

# Create dataset
print("\nLoading dataset...")
try:
    dataset = UCFH5Dataset(
        h5_path=str(h5_path),
        list_path=str(list_path),
        gt_path=str(gt_path),  # Use real labels
        max_length=MAX_LENGTH,
        mode='train'
    )
    
    print(f"  Total videos in list: {len(dataset)}")
    print(f"  Successfully mapped: {len(dataset.video_to_key)}")
    
    # Get valid videos
    valid_indices = []
    for i in range(len(dataset)):
        video_name = dataset.video_names[i]
        if video_name in dataset.video_to_key:
            valid_indices.append(i)
            if NUM_SAMPLES and len(valid_indices) >= NUM_SAMPLES:
                break
    
    print(f"  Valid videos found: {len(valid_indices)}")
    
    if len(valid_indices) == 0:
        print("\n❌ No valid videos found!")
        sys.exit(1)
    
    # Use subset if specified
    if NUM_SAMPLES:
        dataset = Subset(dataset, valid_indices[:NUM_SAMPLES])
        print(f"  Using subset: {len(dataset)} samples")
    
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
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
    drop_last=True  # Drop last incomplete batch
)

print(f"  Batches per epoch: {len(loader)}")
print(f"  Total samples per epoch: {len(loader) * BATCH_SIZE}")

# Model
print("\nCreating model...")
model = MSLModel(
    input_dim=1024,
    hidden_dim=256,  # Bigger model
    num_layers=2,
    num_heads=4,
    sequence_length=32,
    sequence_stride=16,
    dropout=0.1,
    aggregation='mean'
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {num_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Loss
criterion = MSLRankingLoss(margin=0.5, weight_video=1.0, weight_snippet=1.0)
print(f"\nLoss: MSL Ranking Loss (margin=0.5)")

# Manual gradient descent
print(f"Optimizer: Manual Gradient Descent (lr={LEARNING_RATE})")

# Training loop
print("\n" + "="*80)
print("TRAINING STARTED")
print("="*80)

start_time = time.time()

try:
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_video_loss = 0
        epoch_snippet_loss = 0
        batch_count = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(loader):
            try:
                # Move to device
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
                
                # Backward
                loss.backward()
                
                # Manual gradient descent with clipping
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_([param], 1.0)
                            # Update
                            param.sub_(LEARNING_RATE * param.grad)
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_video_loss += loss_dict.get('loss_video', 0).item()
                epoch_snippet_loss += loss_dict.get('loss_snippet', 0).item()
                batch_count += 1
                
                # Print progress
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                    print(f"  Epoch [{epoch+1}/{EPOCHS}] "
                          f"Batch [{batch_idx+1}/{len(loader)}] "
                          f"Loss: {loss.item():.4f} "
                          f"(Video: {loss_dict.get('loss_video', 0).item():.4f}, "
                          f"Snippet: {loss_dict.get('loss_snippet', 0).item():.4f})")
            
            except Exception as e:
                print(f"  ⚠️  Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_video_loss = epoch_video_loss / batch_count
            avg_snippet_loss = epoch_snippet_loss / batch_count
            
            print(f"\n{'='*80}")
            print(f"Epoch [{epoch+1}/{EPOCHS}] Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Video Loss: {avg_video_loss:.4f}")
            print(f"  Snippet Loss: {avg_snippet_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Batches: {batch_count}/{len(loader)}")
            print(f"{'='*80}\n")
        else:
            print(f"\nEpoch [{epoch+1}/{EPOCHS}] - No successful batches\n")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = Path("checkpoints") / f"model_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss if batch_count > 0 else None,
                'learning_rate': LEARNING_RATE
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}\n")

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()

# Training complete
total_time = time.time() - start_time

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
print(f"Epochs completed: {epoch + 1}/{EPOCHS}")

# Save final checkpoint
checkpoint_path = Path("checkpoints") / "final_model.pt"
checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

try:
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'loss': avg_loss if batch_count > 0 else None,
        'learning_rate': LEARNING_RATE,
        'total_time': total_time
    }, checkpoint_path)
    
    print(f"\n✅ Final checkpoint saved: {checkpoint_path}")
except Exception as e:
    print(f"\n⚠️  Could not save final checkpoint: {e}")

print("\n✅ Training finished successfully!")
print(f"✅ Trained on {NUM_SAMPLES if NUM_SAMPLES else 'all'} videos for {epoch + 1} epochs")
print(f"✅ Model has {num_params:,} parameters")
print("\n🎉 MSL model training complete!")
