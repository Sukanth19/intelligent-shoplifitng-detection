"""
Feature Inspection Script
Examines the MSL-VideoSwin feature files to determine format and structure
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("MSL FEATURE INSPECTION")
print("=" * 80)

# This script needs to be updated with the actual feature paths
# Once user provides the location of MSL-VideoSwin-Feature data

# Try multiple possible paths
possible_paths = [
    Path(__file__).parent.parent / "data" / "MSL-Features",
    Path(__file__).parent.parent / "data" / "synthetic",  # Development data
    Path(__file__).parent.parent / "data" / "MSL-VideoSwin-Feature",
]

FEATURE_PATH = None
for path in possible_paths:
    if path.exists():
        FEATURE_PATH = path
        print(f"\n✓ Using feature path: {FEATURE_PATH}")
        break

if FEATURE_PATH is None:
    print(f"\n❌ ERROR: No feature path found")
    print(f"\nSearched locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print(f"\nPlease:")
    print(f"  1. Place real MSL-VideoSwin-Feature data in data/MSL-Features/")
    print(f"  2. Or create synthetic data: python scripts/create_synthetic_data.py")
    sys.exit(1)

print(f"\n1. Feature Path: {FEATURE_PATH}")
print(f"   Exists: ✓")

# List contents
print(f"\n2. Directory Structure:")
try:
    for item in sorted(FEATURE_PATH.iterdir())[:20]:
        if item.is_dir():
            sub_count = len(list(item.iterdir())) if item.is_dir() else 0
            print(f"   📁 {item.name}/ ({sub_count} items)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   📄 {item.name} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"   Error listing directory: {e}")
    sys.exit(1)

# Look for train/test splits
train_path = FEATURE_PATH / "train"
test_path = FEATURE_PATH / "test"
label_path = FEATURE_PATH / "labels"

print(f"\n3. Expected Directories:")
print(f"   train/: {'✓ Found' if train_path.exists() else '✗ Not found'}")
print(f"   test/: {'✓ Found' if test_path.exists() else '✗ Not found'}")
print(f"   labels/: {'✓ Found' if label_path.exists() else '✗ Not found'}")

# Try to find and inspect a sample feature file
print(f"\n4. Sample Feature File Inspection:")

def find_feature_files(directory, extensions=['.npy', '.npz', '.pkl', '.pt', '.pth', '.mat']):
    """Find feature files with common extensions"""
    if not directory.exists():
        return []
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return files

# Search for feature files
feature_files = []
for search_dir in [train_path, test_path, FEATURE_PATH]:
    if search_dir.exists():
        feature_files = find_feature_files(search_dir)
        if feature_files:
            print(f"   Found {len(feature_files)} feature files in {search_dir.name}/")
            break

if not feature_files:
    print(f"   ✗ No feature files found (.npy, .npz, .pkl, .pt, .mat)")
    print(f"   Please verify the feature directory structure")
    sys.exit(1)

# Inspect first few feature files
print(f"\n5. Loading Sample Features:")
for i, feature_file in enumerate(feature_files[:3]):
    print(f"\n   File {i+1}: {feature_file.name}")
    print(f"   Path: {feature_file.relative_to(FEATURE_PATH)}")
    print(f"   Size: {feature_file.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # Try different loading methods based on extension
        if feature_file.suffix == '.npy':
            data = np.load(feature_file)
            print(f"   Format: NumPy array (.npy)")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Value range: [{data.min():.4f}, {data.max():.4f}]")
            
            # Interpret dimensions
            if len(data.shape) == 2:
                T, D = data.shape
                print(f"   Interpretation: T={T} temporal snippets, D={D} feature dimension")
            elif len(data.shape) == 3:
                print(f"   Interpretation: 3D tensor (possibly multiple views or sequences)")
            
        elif feature_file.suffix == '.npz':
            data = np.load(feature_file)
            print(f"   Format: NumPy archive (.npz)")
            print(f"   Keys: {list(data.keys())}")
            for key in data.keys():
                arr = data[key]
                print(f"   - {key}: shape={arr.shape}, dtype={arr.dtype}")
                
        elif feature_file.suffix in ['.pt', '.pth']:
            import torch
            data = torch.load(feature_file, map_location='cpu')
            print(f"   Format: PyTorch tensor (.pt)")
            if isinstance(data, dict):
                print(f"   Keys: {list(data.keys())}")
                for key, val in data.items():
                    if torch.is_tensor(val):
                        print(f"   - {key}: shape={val.shape}, dtype={val.dtype}")
            elif torch.is_tensor(data):
                print(f"   Shape: {data.shape}")
                print(f"   Dtype: {data.dtype}")
                
        elif feature_file.suffix == '.pkl':
            import pickle
            with open(feature_file, 'rb') as f:
                data = pickle.load(f)
            print(f"   Format: Pickle (.pkl)")
            print(f"   Type: {type(data)}")
            if isinstance(data, np.ndarray):
                print(f"   Shape: {data.shape}")
            elif isinstance(data, dict):
                print(f"   Keys: {list(data.keys())}")
                
    except Exception as e:
        print(f"   ✗ Error loading: {e}")

# Look for label files
print(f"\n6. Label Files:")
if label_path.exists():
    label_files = find_feature_files(label_path, ['.npy', '.txt', '.json', '.csv'])
    if label_files:
        print(f"   Found {len(label_files)} label files")
        for label_file in label_files[:3]:
            print(f"   - {label_file.name}")
            try:
                if label_file.suffix == '.npy':
                    labels = np.load(label_file)
                    print(f"     Shape: {labels.shape}, Dtype: {labels.dtype}")
                    print(f"     Unique values: {np.unique(labels)}")
                elif label_file.suffix == '.txt':
                    with open(label_file, 'r') as f:
                        lines = f.readlines()[:10]
                    print(f"     Lines: {len(lines)} (showing first few)")
                    for line in lines[:3]:
                        print(f"     {line.strip()}")
            except Exception as e:
                print(f"     Error: {e}")
    else:
        print(f"   ✗ No label files found")
else:
    print(f"   ✗ Label directory not found")

# Generate documentation
print(f"\n7. Generating Documentation:")
doc_path = Path(__file__).parent.parent / "docs" / "DATA_FORMAT.md"
try:
    with open(doc_path, 'w') as f:
        f.write("# MSL-VideoSwin Feature Data Format\n\n")
        f.write("## Overview\n\n")
        f.write(f"Feature Path: `{FEATURE_PATH}`\n\n")
        f.write("## Structure\n\n")
        f.write("```\n")
        f.write(f"{FEATURE_PATH.name}/\n")
        if train_path.exists():
            f.write(f"├── train/ ({'✓' if train_path.exists() else '✗'})\n")
        if test_path.exists():
            f.write(f"├── test/ ({'✓' if test_path.exists() else '✗'})\n")
        if label_path.exists():
            f.write(f"└── labels/ ({'✓' if label_path.exists() else '✗'})\n")
        f.write("```\n\n")
        f.write("## Feature Format\n\n")
        f.write("**To be completed after inspection**\n\n")
        f.write("- File format: [.npy / .npz / .pt / .pkl]\n")
        f.write("- Feature shape: [T × D]\n")
        f.write("- Temporal snippets (T): [number]\n")
        f.write("- Feature dimension (D): [number]\n")
        f.write("- Data type: [dtype]\n\n")
        f.write("## Label Format\n\n")
        f.write("**To be completed after inspection**\n\n")
        f.write("- Video-level labels: [yes/no]\n")
        f.write("- Snippet-level labels: [yes/no]\n")
        f.write("- Label values: [0=normal, 1=anomaly]\n\n")
        f.write("## Notes\n\n")
        f.write("This file was generated by `scripts/inspect_features.py`\n")
        f.write("Update with actual findings after inspection.\n")
    
    print(f"   ✓ Created: {doc_path}")
except Exception as e:
    print(f"   ✗ Error creating documentation: {e}")

print(f"\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
print(f"\nNext steps:")
print(f"1. Review the output above")
print(f"2. Update docs/DATA_FORMAT.md with detailed findings")
print(f"3. Create dataset loader in src/datasets/ucf_crime_features.py")
print(f"4. Proceed to Phase 3: Feature Dataset Loader")
print("=" * 80)
