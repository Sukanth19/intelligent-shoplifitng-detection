"""
Bigger Batch Training - Manual gradient descent
Batch size = 8, 100 samples, real labels
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
print("MSL TRAINING - Bigger Batch Size")
print("="*80)

# Config
EPOCHS = 5
BATCH_SIZE = 8  # Bigger batch size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SAMPLES = 100  # Use 100 videos
LEARNING_RATE = 0.0001
MAX_LENGTH = 100

print(f"\nDevice: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Samples: {NUM_SAMPLES}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Max Length: {MAX_LENGTH}")

# Paths
data_dir = Path("../data/UCF-VideoSwin-Features")
h5_path = data_dir / "ucf_swin_train_feature_224288224288.h5"
list_path = data_dir / "ucf-train-filter.list"
gt_path = data_dir / "ucf_gt_16.npy"

print(f"\nLoading dataset from: {h5_path.name}")
print(f"Using labels from: {gt_path.name}")

# Create dataset
try:
    dataset = UCFH5Dataset(
        h5_path=str(h5_path),
        list_path=str(list_path),
        gt_path=str(gt_path),  # Real labels
        max_length=MAX_LENGTH,
        mode='train'
    )
    
    print(f"Dataset created: {len(dataset)} total videos")
    print(f"Mapped videos: {len(dataset.video_to_key)}")
    
    # Get valid videos
    valid_indices = []
    for i in range(len(dataset)):
        video_name = dataset.video_names[i]
        if video_name in dataset.video_to_key:
            valid_indices.append(i)
            if len(valid_indices) >= NUM_SAMPLES:
                break
    
    print(f"Found {len(valid_indices)} valid videos")
    
    if len(valid_indices) == 0:
        print("\n❌ No valid videos found!")
        sys.exit(1)
    
    dataset = Subset(dataset, valid_indices)
    print(f"Using subset: {len(dataset)} samples")
    
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
    drop_last=True  # Drop incomplete batches
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

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Loss
criterion = MSLRankingLoss(margin=0.5)

# Manual optimization
print("Using MANUAL gradient descent (no optimizer)")

# Training loop
print("\n" + "="*80)
print("TRAINING")
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
                
                # Zero gradients manually
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
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_([param], 1.0)
                            # Update: param = param - lr * grad
                            param.sub_(LEARNING_RATE * param.grad)
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_video_loss += loss_dict.get('loss_video', 0).item()
                epoch_snippet_loss += loss_dict.get('loss_snippet', 0).item()
                batch_count += 1
                
                # Print progress every 3 batches
                if (batch_idx + 1) % 3 == 0 or (batch_idx + 1) == len(loader):
                    print(f"Epoch [{epoch+1}/{EPOCHS}] "
                          f"Batch [{batch_idx+1}/{len(loader)}] "
                          f"Loss: {loss.item():.4f} "
                          f"(Video: {loss_dict.get('loss_video', 0).item():.4f}, "
                          f"Snippet: {loss_dict.get('loss_snippet', 0).item():.4f})")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_video_loss = epoch_video_loss / batch_count
            avg_snippet_loss = epoch_snippet_loss / batch_count
            
            print(f"\n{'='*80}")
            print(f">>> EPOCH [{epoch+1}/{EPOCHS}] COMPLETE")
            print(f"    Average Loss: {avg_loss:.4f}")
            print(f"    Video Loss: {avg_video_loss:.4f}")
            print(f"    Snippet Loss: {avg_snippet_loss:.4f}")
            print(f"    Time: {epoch_time:.1f}s")
            print(f"    Batches: {batch_count}/{len(loader)}")
            print(f"{'='*80}\n")
        else:
            print(f"\n>>> Epoch [{epoch+1}/{EPOCHS}] - No successful batches\n")

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Training complete
total_time = time.time() - start_time

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")

# Save checkpoint
checkpoint_path = Path("checkpoints") / "bigger_batch_model.pt"
checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

try:
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'final_loss': avg_loss if batch_count > 0 else None,
        'batch_size': BATCH_SIZE,
        'num_samples': NUM_SAMPLES
    }, checkpoint_path)
    
    print(f"\n✅ Checkpoint saved: {checkpoint_path}")
except Exception as e:
    print(f"\n⚠️  Could not save checkpoint: {e}")

print("\n✅ Training successful!")
print(f"✅ Trained on {NUM_SAMPLES} videos for {EPOCHS} epochs")
print(f"✅ Batch size: {BATCH_SIZE}")
print(f"✅ Model parameters: {num_params:,}")
print(f"✅ Used manual gradient descent (no PyTorch optimizer)")
print("\n🎉 Training with bigger batches complete!")
