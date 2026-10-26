"""
Check train vs test split for normal/anomaly distribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets import UCFH5Dataset

print("="*80)
print("CHECKING DATASET SPLITS")
print("="*80)

data_dir = Path("../data/UCF-VideoSwin-Features")

# Check TRAIN set
print("\n📊 TRAIN SET:")
print("-" * 80)

train_dataset = UCFH5Dataset(
    h5_path=str(data_dir / "ucf_swin_train_feature_224288224288.h5"),
    list_path=str(data_dir / "ucf-train-filter.list"),
    gt_path=str(data_dir / "ucf_gt_16.npy"),
    max_length=100,
    mode='train'
)

print(f"Total videos: {len(train_dataset)}")
print(f"Mapped videos: {len(train_dataset.video_to_key)}")

normal_count = 0
anomaly_count = 0
normal_examples = []
anomaly_examples = []

for i in range(min(len(train_dataset), 500)):  # Check first 500
    video_name = train_dataset.video_names[i]
    if video_name not in train_dataset.video_to_key:
        continue
    
    sample = train_dataset[i]
    label = sample['label']
    
    if label == 0:
        normal_count += 1
        if len(normal_examples) < 5:
            normal_examples.append(video_name)
    else:
        anomaly_count += 1
        if len(anomaly_examples) < 5:
            anomaly_examples.append(video_name)

print(f"\nLabel distribution (first 500 checked):")
print(f"  Normal (0): {normal_count}")
print(f"  Anomaly (1): {anomaly_count}")

if normal_examples:
    print(f"\nNormal video examples:")
    for ex in normal_examples:
        print(f"  - {ex}")

if anomaly_examples:
    print(f"\nAnomaly video examples:")
    for ex in anomaly_examples:
        print(f"  - {ex}")

# Check TEST set
print("\n" + "="*80)
print("\n📊 TEST SET:")
print("-" * 80)

test_dataset = UCFH5Dataset(
    h5_path=str(data_dir / "ucf_swin_test_feature_224288224288.h5"),
    list_path=str(data_dir / "ucf-test.list"),
    gt_path=str(data_dir / "ucf_gt_16.npy"),
    max_length=100,
    mode='test'
)

print(f"Total videos: {len(test_dataset)}")
print(f"Mapped videos: {len(test_dataset.video_to_key)}")

normal_count = 0
anomaly_count = 0
normal_examples = []
anomaly_examples = []

for i in range(min(len(test_dataset), 500)):
    video_name = test_dataset.video_names[i]
    if video_name not in test_dataset.video_to_key:
        continue
    
    sample = test_dataset[i]
    label = sample['label']
    
    if label == 0:
        normal_count += 1
        if len(normal_examples) < 5:
            normal_examples.append(video_name)
    else:
        anomaly_count += 1
        if len(anomaly_examples) < 5:
            anomaly_examples.append(video_name)

print(f"\nLabel distribution (first 500 checked):")
print(f"  Normal (0): {normal_count}")
print(f"  Anomaly (1): {anomaly_count}")

if normal_examples:
    print(f"\nNormal video examples:")
    for ex in normal_examples:
        print(f"  - {ex}")

if anomaly_examples:
    print(f"\nAnomaly video examples:")
    for ex in anomaly_examples:
        print(f"  - {ex}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n💡 UCF-Crime dataset uses weakly-supervised learning:")
print("   - Training set: ONLY anomaly videos (for learning anomaly patterns)")
print("   - Test set: Both normal and anomaly (for evaluation)")
print("\n📝 This means:")
print("   - Cannot use MSL ranking loss with train set alone (no normal videos)")
print("   - Need to use DIFFERENT approach:")
print("     Option 1: Use BCE loss instead of ranking loss")
print("     Option 2: Mix train+test for balanced training (not standard)")
print("     Option 3: Use test set only for balanced training")
