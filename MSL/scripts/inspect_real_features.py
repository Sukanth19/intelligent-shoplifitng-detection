"""
Inspect Real UCF-Crime and SHT VideoSwin Features

This script inspects the actual H5 feature files downloaded from the MSL authors.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def inspect_h5_file(h5_path: Path):
    """Inspect H5 file structure and contents."""
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed. Install with: pip install h5py")
        return
    
    print(f"\n{'='*80}")
    print(f"Inspecting: {h5_path.name}")
    print(f"{'='*80}\n")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"Keys in file: {list(f.keys())}\n")
        
        for key in f.keys():
            data = f[key]
            print(f"Key: '{key}'")
            print(f"  Type: {type(data)}")
            
            if isinstance(data, h5py.Dataset):
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  Size: {data.nbytes / (1024**2):.2f} MB")
                
                # Sample first few items if small
                if len(data.shape) == 1 and data.shape[0] < 10:
                    print(f"  Data: {data[:]}")
                elif len(data.shape) == 1:
                    print(f"  First 5: {data[:5]}")
                    print(f"  Last 5: {data[-5:]}")
                elif len(data.shape) == 2:
                    print(f"  First row shape: {data[0].shape}")
                    print(f"  Sample stats - Min: {np.min(data[0]):.3f}, Max: {np.max(data[0]):.3f}, Mean: {np.mean(data[0]):.3f}")
            
            elif isinstance(data, h5py.Group):
                print(f"  Group with keys: {list(data.keys())}")
            
            print()


def inspect_list_file(list_path: Path):
    """Inspect .list file contents."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {list_path.name}")
    print(f"{'='*80}\n")
    
    with open(list_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    print(f"\nFirst 10 entries:")
    for line in lines[:10]:
        print(f"  {line.strip()}")
    
    print(f"\nLast 10 entries:")
    for line in lines[-10:]:
        print(f"  {line.strip()}")
    print()


def inspect_gt_file(gt_path: Path):
    """Inspect ground truth .npy file."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {gt_path.name}")
    print(f"{'='*80}\n")
    
    gt = np.load(gt_path, allow_pickle=True)
    
    print(f"Type: {type(gt)}")
    print(f"Shape: {gt.shape}")
    print(f"Dtype: {gt.dtype}")
    
    if len(gt.shape) == 1 and gt.shape[0] < 20:
        print(f"Data: {gt}")
    elif len(gt.shape) == 1:
        print(f"First 10: {gt[:10]}")
        print(f"Last 10: {gt[-10:]}")
    else:
        print(f"Sample:")
        print(gt[:5])
    
    # Try to understand structure
    if gt.dtype == object:
        print(f"\nObject array - inspecting first element:")
        print(f"  Type: {type(gt[0])}")
        if hasattr(gt[0], 'shape'):
            print(f"  Shape: {gt[0].shape}")
        print(f"  Content: {gt[0]}")
    
    print()


def main():
    # Paths
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    ucf_dir = data_dir / "UCF-VideoSwin-Features"
    sht_dir = data_dir / "SHT-VideoSwin-Features"
    
    print("\n" + "="*80)
    print("REAL FEATURE DATA INSPECTION")
    print("="*80)
    
    # Check if directories exist
    if not ucf_dir.exists():
        print(f"\nWARNING: UCF directory not found: {ucf_dir}")
    else:
        print(f"\n✅ UCF directory found: {ucf_dir}")
    
    if not sht_dir.exists():
        print(f"\nWARNING: SHT directory not found: {sht_dir}")
    else:
        print(f"\n✅ SHT directory found: {sht_dir}")
    
    # Inspect UCF-Crime features
    if ucf_dir.exists():
        print("\n" + "="*80)
        print("UCF-CRIME FEATURES")
        print("="*80)
        
        # H5 files
        ucf_train_h5 = ucf_dir / "ucf_swin_train_feature_224288224288.h5"
        ucf_test_h5 = ucf_dir / "ucf_swin_test_feature_224288224288.h5"
        
        if ucf_train_h5.exists():
            inspect_h5_file(ucf_train_h5)
        
        if ucf_test_h5.exists():
            inspect_h5_file(ucf_test_h5)
        
        # List files
        ucf_train_list = ucf_dir / "ucf-train-filter.list"
        ucf_test_list = ucf_dir / "ucf-test.list"
        
        if ucf_train_list.exists():
            inspect_list_file(ucf_train_list)
        
        if ucf_test_list.exists():
            inspect_list_file(ucf_test_list)
        
        # Ground truth
        ucf_gt = ucf_dir / "ucf_gt_16.npy"
        if ucf_gt.exists():
            inspect_gt_file(ucf_gt)
    
    # Inspect SHT features
    if sht_dir.exists():
        print("\n" + "="*80)
        print("SHT (Shanghai Tech) FEATURES")
        print("="*80)
        
        # H5 files
        sht_train_h5 = sht_dir / "sh_swin_train_feature_256320224.h5"
        sht_test_h5 = sht_dir / "sh_swin_test_feature_256320224.h5"
        
        if sht_train_h5.exists():
            inspect_h5_file(sht_train_h5)
        
        if sht_test_h5.exists():
            inspect_h5_file(sht_test_h5)
        
        # List files
        sht_train_list = sht_dir / "sh-train.list"
        sht_test_list = sht_dir / "sh-test.list"
        
        if sht_train_list.exists():
            inspect_list_file(sht_train_list)
        
        if sht_test_list.exists():
            inspect_list_file(sht_test_list)
        
        # Ground truth
        sht_gt = sht_dir / "sh_gt_16.npy"
        if sht_gt.exists():
            inspect_gt_file(sht_gt)
    
    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Update dataset loader to read H5 format")
    print("2. Understand label format from .list and .npy files")
    print("3. Create UCF/SHT dataset classes")
    print("4. Test training on real data")
    print()


if __name__ == "__main__":
    main()
