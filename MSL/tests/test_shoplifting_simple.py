"""
Simple test for shoplifting evaluation
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import UCFCrimeFeatureDataset


def test_shoplifting_dataset():
    """Test loading shoplifting-specific dataset."""
    print("Testing shoplifting dataset loading...")
    
    feature_dir = Path('data/synthetic_shoplifting')
    
    if not feature_dir.exists():
        print("  ⚠ Shoplifting dataset not found. Run:")
        print("    python scripts/create_shoplifting_dataset.py")
        return False
    
    # Load dataset
    dataset = UCFCrimeFeatureDataset(feature_dir=feature_dir, split='test')
    print(f"  Loaded {len(dataset)} test videos")
    
    # Check categories
    categories = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        video_name = sample[0]['video_name'].lower() if isinstance(sample, tuple) else sample['video_name'].lower()
        
        if 'normal' in video_name:
            cat = 'Normal'
        elif 'shoplifting' in video_name or 'stealing' in video_name:
            cat = 'Shoplifting'
        else:
            cat = 'Other'
        
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"  Categories found:")
    for cat, count in categories.items():
        print(f"    {cat}: {count}")
    
    assert 'Normal' in categories, "No normal videos found"
    assert 'Shoplifting' in categories, "No shoplifting videos found"
    
    print("  ✓ Shoplifting dataset works")
    return True


def test_video_filtering():
    """Test video filtering logic."""
    print("\nTesting video filtering...")
    
    # Simulate video names
    video_names = [
        'Normal_001.npy',
        'Normal_002.npy',
        'Shoplifting_001.npy',
        'Shoplifting_002.npy',
        'Robbery_001.npy',
        'Stealing_001.npy',
        'Burglary_001.npy'
    ]
    
    shoplifting_keywords = ['shoplifting', 'stealing', 'theft']
    
    normal_count = 0
    shoplifting_count = 0
    other_count = 0
    
    for name in video_names:
        name_lower = name.lower()
        
        if 'normal' in name_lower:
            normal_count += 1
        elif any(kw in name_lower for kw in shoplifting_keywords):
            shoplifting_count += 1
        else:
            other_count += 1
    
    print(f"  Normal: {normal_count}")
    print(f"  Shoplifting: {shoplifting_count}")
    print(f"  Other: {other_count}")
    
    assert normal_count == 2
    assert shoplifting_count == 3  # Shoplifting_* and Stealing_*
    assert other_count == 2  # Robbery and Burglary
    
    print("  ✓ Video filtering works")
    return True


def main():
    print("=" * 80)
    print("TESTING SHOPLIFTING EVALUATION COMPONENTS")
    print("=" * 80)
    
    print("\n1. Video Filtering Logic")
    test_video_filtering()
    
    print("\n2. Shoplifting Dataset")
    dataset_ok = test_shoplifting_dataset()
    
    print("\n" + "=" * 80)
    if dataset_ok:
        print("✓ ALL SHOPLIFTING TESTS PASSED")
        print("\nYou can now run:")
        print("  python scripts/train.py --config configs/default.yaml")
        print("  python scripts/evaluate_shoplifting.py \\")
        print("    --checkpoint checkpoints/best_model.pt \\")
        print("    --config configs/default.yaml \\")
        print("    --feature-dir data/synthetic_shoplifting \\")
        print("    --split test")
    else:
        print("⚠ CREATE SHOPLIFTING DATASET FIRST")
        print("  python scripts/create_shoplifting_dataset.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
