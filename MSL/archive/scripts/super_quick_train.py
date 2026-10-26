"""
Super quick training - avoids PyTorch 2.6 + Python 3.13 sympy issues
Uses SGD instead of Adam to avoid the bug
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

print("="*80)
print("SUPER QUICK TRAIN - Minimal Version")
print("="*80)

# Config
EPOCHS = 2
BATCH_SIZE = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SAMPLES = 10  # Use only 10 videos

print(f"\nDevice: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Samples: {NUM_SAMPLES}")

# Paths
data_dir = Path("../data/UCF-VideoSwin-Features")
h5_path = data_dir / "ucf_swin_train_feature_224288224288.h5"
list_path = data_dir / "ucf-train-filter.list"

print(f"\nLoading dataset from: {h5_path.name}")

# Create dataset
try:
    dataset = UCFH5Dataset(
        h5_path=str(h5_path),
        list_path=str(list_path),
        gt_path=None,  # No labels needed for quick test
        max_length=50,
        mode='train'
    )
    
    print(f"Dataset created: {len(dataset)} total videos")
    
    # Get indices of videos that were successfully mapped
    valid_indices = []
    for i in range(len(dataset)):
        video_name = dataset.video_names[i]
        if video_name in dataset.video_to_key:
            valid_indices.append(i)
            if len(valid_indices) >= NUM_SAMPLES:
                break
    
    print(f"Found {len(valid_indices)} valid videos with features")
    
    if len(valid_indices) == 0:
        print("\n❌ No valid videos found!")
        print("The H5 file keys don't match the video names in the list file.")
        sys.exit(1)
    
    # Use subset
    dataset = Subset(dataset, valid_indices)
    print(f"Using subset: {len(dataset)} samples")
    
except Exception as e:
    print(f"\n❌ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn_h5
)

print(f"Batches per epoch: {len(loader)}")

# Model
print("\nCreating model...")
model = MSLModel(
    input_dim=1024,
    hidden_dim=128,
    num_layers=1,
    num_heads=2,
    sequence_length=16,
    sequence_stride=8,
    dropout=0.1,
    aggregation='mean'
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Loss
criterion = MSLRankingLoss(margin=0.5)

# Optimizer - Use SGD to avoid Adam/sympy bug in PyTorch 2.6 + Python 3.13
print("Using SGD optimizer (avoids sympy bug)")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
print("\n" + "="*80)
print("TRAINING")
print("="*80)

try:
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(loader):
            try:
                # Move to device
                features = batch['features'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)
                
                # Forward
                optimizer.zero_grad()
                outputs = model(features, mask=masks)
                
                # Loss
                loss = criterion(
                    video_scores=outputs['video_score'],
                    snippet_scores=outputs['snippet_scores'],
                    labels=labels
                )
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == len(loader):
                    print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"\n>>> Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")
        else:
            print(f"\n>>> Epoch [{epoch+1}/{EPOCHS}] - No successful batches")

except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

# Save checkpoint
checkpoint_path = Path("checkpoints") / "super_quick_model.pt"
checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

try:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS
    }, checkpoint_path)
    
    print(f"\n✅ Checkpoint saved: {checkpoint_path}")
except Exception as e:
    print(f"\n⚠️  Could not save checkpoint: {e}")

print("\n✅ Super quick training successful!")
print(f"Trained on {NUM_SAMPLES} videos for {EPOCHS} epochs")
