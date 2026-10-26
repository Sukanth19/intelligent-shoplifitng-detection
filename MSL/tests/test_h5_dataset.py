"""
Test H5 Dataset Loader

Tests the UCFH5Dataset and SHTH5Dataset classes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from src.datasets import UCFH5Dataset, SHTH5Dataset, collate_fn_h5


def test_ucf_h5_dataset():
    """Test UCF-Crime H5 dataset loader."""
    print("\n" + "="*80)
    print("Testing UCF-Crime H5 Dataset Loader")
    print("="*80)
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "UCF-VideoSwin-Features"
    
    if not data_dir.exists():
        print(f"⚠️  UCF data directory not found: {data_dir}")
        print("Skipping UCF-Crime tests")
        return False
    
    try:
        # Create dataset
        dataset = UCFH5Dataset(
            h5_path=str(data_dir / "ucf_swin_test_feature_224288224288.h5"),
            list_path=str(data_dir / "ucf-test.list"),
            gt_path=str(data_dir / "ucf_gt_16.npy"),
            max_length=320,
            mode='test'
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Dataset size: {len(dataset)}")
        
        # Test __getitem__
        sample = dataset[0]
        print(f"\n✅ Sample 0 loaded:")
        print(f"   Features shape: {sample['features'].shape}")
        print(f"   Label: {sample['label'].item()}")
        print(f"   Length: {sample['length'].item()}")
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   Video name: {sample.get('video_name', 'N/A')}")
        
        # Validate shapes
        assert sample['features'].shape == (320, 1024), "Features shape mismatch"
        assert sample['mask'].shape == (320,), "Mask shape mismatch"
        assert sample['label'].dtype == torch.long, "Label dtype mismatch"
        
        # Test batching
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn_h5
        )
        
        batch = next(iter(loader))
        print(f"\n✅ Batch loaded:")
        print(f"   Features shape: {batch['features'].shape}")
        print(f"   Labels shape: {batch['label'].shape}")
        print(f"   Lengths: {batch['length'].tolist()}")
        print(f"   Masks shape: {batch['mask'].shape}")
        print(f"   Number of videos: {len(batch['video_name'])}")
        
        # Validate batch shapes
        assert batch['features'].shape == (4, 320, 1024), "Batch features shape mismatch"
        assert batch['label'].shape == (4,), "Batch labels shape mismatch"
        
        dataset.close()
        print(f"\n✅ All UCF-Crime H5 tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sht_h5_dataset():
    """Test Shanghai Tech H5 dataset loader."""
    print("\n" + "="*80)
    print("Testing Shanghai Tech H5 Dataset Loader")
    print("="*80)
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "SHT-VideoSwin-Features"
    
    if not data_dir.exists():
        print(f"⚠️  SHT data directory not found: {data_dir}")
        print("Skipping SHT tests")
        return False
    
    try:
        # Create dataset
        dataset = SHTH5Dataset(
            h5_path=str(data_dir / "sh_swin_test_feature_256320224.h5"),
            list_path=str(data_dir / "sh-test.list"),
            gt_path=str(data_dir / "sh_gt_16.npy"),
            max_length=256,
            mode='test'
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Dataset size: {len(dataset)}")
        
        # Test __getitem__
        sample = dataset[0]
        print(f"\n✅ Sample 0 loaded:")
        print(f"   Features shape: {sample['features'].shape}")
        print(f"   Label: {sample['label'].item()}")
        print(f"   Length: {sample['length'].item()}")
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   Video name: {sample.get('video_name', 'N/A')}")
        
        # Validate shapes
        assert sample['features'].shape == (256, 1024), "Features shape mismatch"
        assert sample['mask'].shape == (256,), "Mask shape mismatch"
        assert sample['label'].dtype == torch.long, "Label dtype mismatch"
        
        # Test batching
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn_h5
        )
        
        batch = next(iter(loader))
        print(f"\n✅ Batch loaded:")
        print(f"   Features shape: {batch['features'].shape}")
        print(f"   Labels shape: {batch['label'].shape}")
        print(f"   Lengths: {batch['length'].tolist()}")
        print(f"   Masks shape: {batch['mask'].shape}")
        print(f"   Number of videos: {len(batch['video_name'])}")
        
        # Validate batch shapes
        assert batch['features'].shape == (4, 256, 1024), "Batch features shape mismatch"
        assert batch['label'].shape == (4,), "Batch labels shape mismatch"
        
        dataset.close()
        print(f"\n✅ All SHT H5 tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_length_handling():
    """Test variable length video handling."""
    print("\n" + "="*80)
    print("Testing Variable Length Handling")
    print("="*80)
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "SHT-VideoSwin-Features"
    
    if not data_dir.exists():
        print(f"⚠️  Data directory not found, skipping test")
        return False
    
    try:
        # Create dataset with small max_length to force truncation
        dataset = SHTH5Dataset(
            h5_path=str(data_dir / "sh_swin_test_feature_256320224.h5"),
            list_path=str(data_dir / "sh-test.list"),
            gt_path=str(data_dir / "sh_gt_16.npy"),
            max_length=50,  # Small to test truncation
            mode='test'
        )
        
        # Test multiple samples
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            assert sample['features'].shape[0] == 50, f"Sample {i} not padded/truncated correctly"
            assert sample['mask'].sum() <= 50, f"Sample {i} mask error"
            print(f"✅ Sample {i}: length={sample['length'].item()}, valid positions={sample['mask'].sum().item()}")
        
        dataset.close()
        print(f"\n✅ Variable length handling test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("H5 DATASET LOADER TESTS")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("UCF-Crime H5", test_ucf_h5_dataset()))
    results.append(("SHT H5", test_sht_h5_dataset()))
    results.append(("Variable Length", test_variable_length_handling()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All H5 dataset tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)
