"""
Environment Check Script
Verifies Python, PyTorch, CUDA, and locates datasets
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("ENVIRONMENT CHECK")
print("=" * 80)

# Python version
print(f"\n1. Python Version:")
print(f"   {sys.version}")
print(f"   Executable: {sys.executable}")

# PyTorch
print(f"\n2. PyTorch:")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Current Device: {torch.cuda.current_device()}")
    else:
        print(f"   CUDA Version: N/A")
        print(f"   Note: Will use CPU")
except ImportError:
    print(f"   ERROR: PyTorch not installed")
    sys.exit(1)

# Other key packages
print(f"\n3. Other Packages:")
packages = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'tensorboard', 'tqdm', 'yaml']
for pkg_name in packages:
    try:
        if pkg_name == 'yaml':
            import yaml as pkg
        else:
            pkg = __import__(pkg_name)
        version = getattr(pkg, '__version__', 'unknown')
        print(f"   {pkg_name}: {version}")
    except ImportError:
        print(f"   {pkg_name}: NOT INSTALLED")

# Workspace
print(f"\n4. Workspace:")
workspace = Path(__file__).parent.parent
print(f"   Root: {workspace}")
print(f"   Exists: {workspace.exists()}")

# Look for common dataset locations
print(f"\n5. Searching for UCF-Crime Dataset:")
search_paths = [
    workspace / "data" / "UCF-Crime",
    workspace.parent / "UCF-Crime",
    workspace.parent / "data" / "UCF-Crime",
    Path("C:/data/UCF-Crime"),
    Path("D:/data/UCF-Crime"),
]
found_dataset = None
for search_path in search_paths:
    print(f"   Checking: {search_path}")
    if search_path.exists():
        print(f"   ✓ FOUND!")
        found_dataset = search_path
        # List contents
        try:
            contents = list(search_path.iterdir())[:10]  # First 10 items
            print(f"     Contents (first 10):")
            for item in contents:
                print(f"       - {item.name}")
        except Exception as e:
            print(f"     Could not list contents: {e}")
        break
else:
    print(f"   ✗ Not found in searched locations")

# Look for MSL features
print(f"\n6. Searching for MSL-VideoSwin Features:")
search_paths = [
    workspace / "data" / "MSL-Features",
    workspace / "data" / "videoswin",
    workspace.parent / "MSL-VideoSwin-Feature",
    workspace.parent / "data" / "MSL-Features",
    Path("C:/data/MSL-Features"),
    Path("D:/data/MSL-Features"),
]
found_features = None
for search_path in search_paths:
    print(f"   Checking: {search_path}")
    if search_path.exists():
        print(f"   ✓ FOUND!")
        found_features = search_path
        # List contents
        try:
            contents = list(search_path.iterdir())[:10]
            print(f"     Contents (first 10):")
            for item in contents:
                print(f"       - {item.name}")
        except Exception as e:
            print(f"     Could not list contents: {e}")
        break
else:
    print(f"   ✗ Not found in searched locations")

# Summary
print(f"\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Python: ✓ {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print(f"PyTorch: ✓ {torch.__version__}")
print(f"CUDA: {'✓ Available' if torch.cuda.is_available() else '✗ Not available (will use CPU)'}")
print(f"UCF-Crime: {'✓ Found at ' + str(found_dataset) if found_dataset else '✗ Not found'}")
print(f"MSL Features: {'✓ Found at ' + str(found_features) if found_features else '✗ Not found'}")

if not found_dataset or not found_features:
    print(f"\n⚠ ATTENTION: Please provide the correct paths to:")
    if not found_dataset:
        print(f"   - UCF-Crime dataset")
    if not found_features:
        print(f"   - MSL-VideoSwin features")
    print(f"\nYou can either:")
    print(f"   1. Place them in: {workspace / 'data'}")
    print(f"   2. Update the search paths in this script")
    print(f"   3. Create symbolic links")

print("=" * 80)
