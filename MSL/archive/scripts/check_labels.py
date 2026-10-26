"""
Check what labels are being loaded from the dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from src.datasets import UCFH5Dataset, collate_fn_h5

print("="*80)
print("CHECKING DATASET LABELS")
print("="*80)

# Paths
data_dir = Path("../data/UCF-VideoSwin-Features")
h5_path = data_dir / "ucf_swin_train_feature_224288224288.h5"
list_path = data_dir / "ucf-train-filter.list"
gt_path = data_dir / "ucf_gt_16.npy"

print(f"\nLoading GT file: {gt_path}")

# Load ground truth directly
if gt_path.exists():
    gt_data = np.load(str(gt_path), allow_pickle=True)
    print(f"GT data type: {type(gt_data)}")
    print(f"GT data shape: {gt_data.shape if hasattr(gt_data, 'shape') else 'N/A'}")
    print(f"GT data dtype: {gt_data.dtype if hasattr(gt_data, 'dtype') else 'N/A'}")
    
    # Try to convert to dict if it's a structured array
    if isinstance(gt_data, np.ndarray) and gt_data.dtype == object and len(gt_data) == 1:
        gt_data = gt_data.item()
    
    print(f"GT keys: {list(gt_data.keys())[:10] if isinstance(gt_data, dict) else 'Not a dict'}")
    
    if isinstance(gt_data, dict):
        # Sample some entries
        print(f"\nTotal GT entries: {len(gt_data)}")
        print("\nFirst 10 entries:")
        for i, (key, value) in enumerate(list(gt_data.items())[:10]):
            print(f"  {i+1}. {key}: {value}")
        
        # Check label distribution
        labels = list(gt_data.values())
        unique_labels = set(labels)
        print(f"\nUnique labels: {unique_labels}")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  Label {label}: {count} videos ({count/len(labels)*100:.1f}%)")
else:
    print(f"GT file not found!")

# Now check what the dataset loads
print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

dataset = UCFH5Dataset(
    h5_path=str(h5_path),
    list_path=str(list_path),
    gt_path=str(gt_path),
    max_length=100,
    mode='train'
)

print(f"\nDataset size: {len(dataset)}")
print(f"Mapped videos: {len(dataset.video_to_key)}")

# Get valid videos
valid_indices = []
for i in range(len(dataset)):
    video_name = dataset.video_names[i]
    if video_name in dataset.video_to_key:
        valid_indices.append(i)
        if len(valid_indices) >= 20:  # Check first 20
            break

print(f"Valid videos found: {len(valid_indices)}")

# Check labels for these videos
print("\nLabels for first 20 valid videos:")
for idx in valid_indices:
    video_name = dataset.video_names[idx]
    sample = dataset[idx]
    label = sample['label']
    print(f"  {idx}: {video_name} -> label={label}")

# Load a batch
print("\n" + "="*80)
print("CHECKING BATCH")
print("="*80)

subset = Subset(dataset, valid_indices[:16])
loader = DataLoader(
    subset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn_h5
)

for batch_idx, batch in enumerate(loader):
    print(f"\nBatch {batch_idx+1}:")
    labels = batch['label']
    print(f"  Labels: {labels.tolist()}")
    print(f"  Unique: {torch.unique(labels).tolist()}")
    print(f"  Normal (0): {(labels == 0).sum().item()}")
    print(f"  Anomaly (1): {(labels == 1).sum().item()}")
    
    if batch_idx >= 1:  # Check 2 batches
        break

print("\n" + "="*80)
print("LABEL CHECK COMPLETE")
print("="*80)
