"""
Create Synthetic Data for Development
Generates mock VideoSwin features for testing the pipeline
"""

import numpy as np
from pathlib import Path
import json

print("=" * 80)
print("CREATING SYNTHETIC DATA FOR DEVELOPMENT")
print("=" * 80)

# Configuration based on typical VideoSwin output
# VideoSwin base produces 1024-dimensional features
FEATURE_DIM = 1024
NUM_TRAIN_NORMAL = 10
NUM_TRAIN_ANOMALY = 10
NUM_TEST_NORMAL = 5
NUM_TEST_ANOMALY = 5

# Create directory structure
root = Path(__file__).parent.parent
data_dir = root / "data" / "synthetic"
train_dir = data_dir / "train"
test_dir = data_dir / "test"
label_dir = data_dir / "labels"

for d in [train_dir, test_dir, label_dir]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\n1. Creating directory structure:")
print(f"   {data_dir}")

# Generate training data
print(f"\n2. Generating training features:")
train_labels = []

# Normal videos
for i in range(NUM_TRAIN_NORMAL):
    # Variable length: 50-200 temporal snippets
    T = np.random.randint(50, 200)
    features = np.random.randn(T, FEATURE_DIM).astype(np.float32)
    
    # Normalize features
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    filename = f"Normal_{i:03d}.npy"
    np.save(train_dir / filename, features)
    train_labels.append({
        'filename': filename,
        'label': 0,  # Normal
        'category': 'Normal',
        'num_snippets': T
    })
    print(f"   ✓ {filename}: shape={features.shape}")

# Anomaly videos (including Shoplifting)
anomaly_categories = ['Shoplifting', 'Robbery', 'Stealing', 'Burglary']
for i in range(NUM_TRAIN_ANOMALY):
    T = np.random.randint(50, 200)
    features = np.random.randn(T, FEATURE_DIM).astype(np.float32)
    
    # Add some pattern to anomaly features (higher variance in certain segments)
    anomaly_start = T // 3
    anomaly_end = 2 * T // 3
    features[anomaly_start:anomaly_end] *= 1.5  # Simulated anomaly
    
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    category = anomaly_categories[i % len(anomaly_categories)]
    filename = f"{category}_{i:03d}.npy"
    np.save(train_dir / filename, features)
    train_labels.append({
        'filename': filename,
        'label': 1,  # Anomaly
        'category': category,
        'num_snippets': T
    })
    print(f"   ✓ {filename}: shape={features.shape}")

# Generate test data
print(f"\n3. Generating test features:")
test_labels = []

for i in range(NUM_TEST_NORMAL):
    T = np.random.randint(50, 200)
    features = np.random.randn(T, FEATURE_DIM).astype(np.float32)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    filename = f"Normal_{i:03d}.npy"
    np.save(test_dir / filename, features)
    test_labels.append({
        'filename': filename,
        'label': 0,
        'category': 'Normal',
        'num_snippets': T
    })
    print(f"   ✓ {filename}: shape={features.shape}")

for i in range(NUM_TEST_ANOMALY):
    T = np.random.randint(50, 200)
    features = np.random.randn(T, FEATURE_DIM).astype(np.float32)
    
    anomaly_start = T // 3
    anomaly_end = 2 * T // 3
    features[anomaly_start:anomaly_end] *= 1.5
    
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    category = anomaly_categories[i % len(anomaly_categories)]
    filename = f"{category}_{i:03d}.npy"
    np.save(test_dir / filename, features)
    test_labels.append({
        'filename': filename,
        'label': 1,
        'category': category,
        'num_snippets': T
    })
    print(f"   ✓ {filename}: shape={features.shape}")

# Save labels
print(f"\n4. Saving labels:")
with open(label_dir / "train_labels.json", 'w') as f:
    json.dump(train_labels, f, indent=2)
print(f"   ✓ train_labels.json ({len(train_labels)} videos)")

with open(label_dir / "test_labels.json", 'w') as f:
    json.dump(test_labels, f, indent=2)
print(f"   ✓ test_labels.json ({len(test_labels)} videos)")

# Create README
readme_content = """# Synthetic Data for Development

This is **synthetic/mock data** for development and testing.

## Structure
- `train/`: Training features (20 videos: 10 normal, 10 anomaly)
- `test/`: Test features (10 videos: 5 normal, 5 anomaly)
- `labels/`: JSON files with video metadata

## Feature Format
- File format: `.npy` (NumPy arrays)
- Shape: `(T, D)` where T varies (50-200), D=1024
- Dtype: `float32`
- Normalized: L2 norm = 1 per snippet

## Labels
- 0: Normal
- 1: Anomaly (Shoplifting, Robbery, Stealing, Burglary)

## Note
This data simulates VideoSwin features for pipeline development.
Replace with real MSL-VideoSwin-Feature data for actual experiments.
"""

with open(data_dir / "README.md", 'w') as f:
    f.write(readme_content)

print(f"\n5. Summary:")
print(f"   Location: {data_dir}")
print(f"   Train: {len(train_labels)} videos")
print(f"   Test: {len(test_labels)} videos")
print(f"   Feature dim: {FEATURE_DIM}")
print(f"   Format: .npy (T × {FEATURE_DIM})")

print("\n" + "=" * 80)
print("SYNTHETIC DATA CREATED")
print("=" * 80)
print("\n⚠️  This is MOCK DATA for development only!")
print("Replace with real MSL-VideoSwin-Feature data before training.")
print(f"\nNext: Run feature inspection on synthetic data")
print(f"  python scripts/inspect_features.py")
print("=" * 80)
