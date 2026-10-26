"""
Quick training script - minimal version for fast testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader
from src.datasets import UCFH5Dataset, collate_fn_h5
from src.models import MSLModel
from src.losses import MSLRankingLoss

print("="*80)
print("QUICK TRAIN - Fast Testing Mode")
print("="*80)

# Config
EPOCHS = 2
BATCH_SIZE = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nDevice: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")

# Paths
data_dir = Path("../data/UCF-VideoSwin-Features")
h5_path = data_dir / "ucf_swin_train_feature_224288224288.h5"
list_path = data_dir / "ucf-train-filter.list"
gt_path = data_dir / "ucf_gt_16.npy"

print(f"\nLoading dataset from: {h5_path}")

# Create dataset (small subset)
dataset = UCFH5Dataset(
    h5_path=str(h5_path),
    list_path=str(list_path),
    gt_path=str(gt_path),
    max_length=50,  # Short sequences
    mode='train'
)

print(f"Dataset size: {len(dataset)}")

# Use only first 10 samples for speed
from torch.utils.data import Subset
dataset = Subset(dataset, range(min(10, len(dataset))))
print(f"Using subset: {len(dataset)} samples")

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn_h5
)

print(f"Batches: {len(loader)}")

# Model
print("\nCreating model...")
model = MSLModel(
    input_dim=1024,
    hidden_dim=128,  # Small
    num_layers=1,  # Minimal
    num_heads=2,  # Minimal
    sequence_length=16,
    sequence_stride=8,
    dropout=0.1,
    aggregation='mean'
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Loss & Optimizer
criterion = MSLRankingLoss(margin=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\n" + "="*80)
print("TRAINING")
print("="*80)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        features = batch['features'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        masks = batch['mask'].to(DEVICE)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(features, mask=masks)
        
        # Loss (simplified)
        loss = criterion(
            video_scores=outputs['video_score'],
            snippet_scores=outputs['snippet_scores'],
            labels=labels
        )
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

# Save checkpoint
checkpoint_path = Path("checkpoints") / "quick_model.pt"
checkpoint_path.parent.mkdir(exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': EPOCHS
}, checkpoint_path)

print(f"\nCheckpoint saved: {checkpoint_path}")
print("\n✅ Quick training successful!")
