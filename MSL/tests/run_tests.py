"""
Simple test runner (works without pytest)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import UCFCrimeFeatureDataset, collate_fn_pad, collate_fn_list
import torch

print("=" * 80)
print("RUNNING DATASET TESTS")
print("=" * 80)

# Check synthetic data
data_dir = Path(__file__).parent.parent / "data" / "synthetic"
if not data_dir.exists():
    print(f"\n❌ Synthetic data not found: {data_dir}")
    print("Run: python scripts/create_synthetic_data.py")
    sys.exit(1)

tests_passed = 0
tests_failed = 0

def test(name, condition, message=""):
    """Simple test function"""
    global tests_passed, tests_failed
    if condition:
        print(f"✓ {name}")
        tests_passed += 1
    else:
        print(f"✗ {name}: {message}")
        tests_failed += 1

# Test 1: Dataset initialization
print("\n1. Testing dataset initialization...")
try:
    dataset = UCFCrimeFeatureDataset(feature_dir=data_dir, split="train")
    test("Dataset init", len(dataset) > 0)
    test("Feature dim", dataset.feature_dim == 1024)
    test("Split name", dataset.split == "train")
except Exception as e:
    test("Dataset init", False, str(e))

# Test 2: Data loading
print("\n2. Testing data loading...")
try:
    features, label, video_id = dataset[0]
    test("Feature type", isinstance(features, torch.Tensor))
    test("Feature shape", features.ndim == 2 and features.shape[1] == 1024)
    test("Label type", isinstance(label, int))
    test("Label value", label in [0, 1])
    test("Video ID type", isinstance(video_id, str))
except Exception as e:
    test("Data loading", False, str(e))

# Test 3: Statistics
print("\n3. Testing statistics...")
try:
    stats = dataset.get_statistics()
    test("Stats total", stats['total'] == len(dataset))
    test("Stats normal+anomaly", stats['normal'] + stats['anomaly'] == stats['total'])
    test("Categories", 'categories' in stats)
except Exception as e:
    test("Statistics", False, str(e))

# Test 4: Test split
print("\n4. Testing test split...")
try:
    test_dataset = UCFCrimeFeatureDataset(feature_dir=data_dir, split="test")
    test("Test split load", len(test_dataset) > 0)
    test("Different sizes", len(test_dataset) != len(dataset))
except Exception as e:
    test("Test split", False, str(e))

# Test 5: Collate function
print("\n5. Testing collate function...")
try:
    batch = [
        dataset[0],
        dataset[1],
        dataset[2]
    ]
    result = collate_fn_pad(batch)
    test("Collate keys", all(k in result for k in ['features', 'labels', 'lengths', 'mask', 'video_ids']))
    test("Batch features shape", result['features'].ndim == 3)
    test("Batch labels shape", len(result['labels']) == 3)
    test("Mask validity", result['mask'].dtype == torch.bool)
except Exception as e:
    test("Collate function", False, str(e))

# Test 6: DataLoader
print("\n6. Testing DataLoader integration...")
try:
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_pad)
    batch = next(iter(loader))
    test("DataLoader batch", isinstance(batch, dict))
    test("Batch size", batch['features'].shape[0] <= 4)
    test("Feature dim preserved", batch['features'].shape[2] == 1024)
except Exception as e:
    test("DataLoader", False, str(e))

# Test 7: Max length truncation
print("\n7. Testing max_len truncation...")
try:
    dataset_truncated = UCFCrimeFeatureDataset(
        feature_dir=data_dir,
        split="train",
        max_len=50,
        return_video_id=False
    )
    for i in range(min(5, len(dataset_truncated))):
        features, label = dataset_truncated[i]
        if not (features.shape[0] <= 50):
            test("Max len truncation", False, f"Sequence {i} has length {features.shape[0]}")
            break
    else:
        test("Max len truncation", True)
except Exception as e:
    test("Max len truncation", False, str(e))

# Test 8: Normalization
print("\n8. Testing normalization...")
try:
    dataset_norm = UCFCrimeFeatureDataset(
        feature_dir=data_dir,
        split="train",
        normalize=True,
        return_video_id=False
    )
    features, _ = dataset_norm[0]
    norms = torch.norm(features, p=2, dim=1)
    normalized = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    test("L2 normalization", normalized)
except Exception as e:
    test("Normalization", False, str(e))

# Test 9: Video info
print("\n9. Testing video info...")
try:
    info = dataset.get_video_info(0)
    required_keys = ['video_id', 'label', 'category', 'feature_path']
    test("Video info keys", all(k in info for k in required_keys))
except Exception as e:
    test("Video info", False, str(e))

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✓ Passed: {tests_passed}")
print(f"✗ Failed: {tests_failed}")
print(f"Total: {tests_passed + tests_failed}")

if tests_failed == 0:
    print("\n🎉 ALL TESTS PASSED!")
    sys.exit(0)
else:
    print(f"\n❌ {tests_failed} test(s) failed")
    sys.exit(1)
