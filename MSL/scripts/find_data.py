"""
Data Discovery Script
Searches common locations for UCF-Crime and MSL features
"""

import os
from pathlib import Path

print("=" * 80)
print("SEARCHING FOR DATA")
print("=" * 80)

# Search locations
drives = ['C:', 'D:', 'E:']
common_names = ['UCF-Crime', 'UCF_Crime', 'ucf-crime', 'ucf_crime', 'UCFCrime']
feature_names = ['MSL-VideoSwin-Feature', 'MSL-Features', 'MSL_Features', 'videoswin', 'features']

def search_directory(root, target_names, max_depth=4):
    """Search for directories matching target names"""
    found = []
    try:
        for dirpath, dirnames, _ in os.walk(root):
            depth = dirpath.replace(root, '').count(os.sep)
            if depth > max_depth:
                dirnames[:] = []  # Don't recurse deeper
                continue
            
            for dirname in dirnames:
                if any(target in dirname for target in target_names):
                    full_path = Path(dirpath) / dirname
                    found.append(full_path)
                    print(f"   ✓ Found: {full_path}")
    except PermissionError:
        pass
    except Exception as e:
        pass
    return found

print("\n1. Searching for UCF-Crime dataset...")
ucf_found = []
search_roots = [
    Path.home() / 'Downloads',
    Path.home() / 'Desktop',
    Path.home() / 'Documents',
    Path('C:/data'),
    Path('D:/data'),
    Path('C:/datasets'),
    Path('D:/datasets'),
]

for root in search_roots:
    if root.exists():
        print(f"   Searching: {root}")
        found = search_directory(str(root), common_names, max_depth=3)
        ucf_found.extend(found)

print(f"\n   Total UCF-Crime candidates: {len(ucf_found)}")

print("\n2. Searching for MSL-VideoSwin features...")
msl_found = []
for root in search_roots:
    if root.exists():
        print(f"   Searching: {root}")
        found = search_directory(str(root), feature_names, max_depth=3)
        msl_found.extend(found)

print(f"\n   Total MSL feature candidates: {len(msl_found)}")

# Summary
print("\n" + "=" * 80)
print("SEARCH RESULTS")
print("=" * 80)

if ucf_found:
    print("\nUCF-Crime candidates:")
    for i, path in enumerate(ucf_found, 1):
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024**3)
        count = len(list(path.rglob('*.*')))
        print(f"  {i}. {path}")
        print(f"     Files: ~{count}, Size: ~{size:.1f} GB")
else:
    print("\n✗ UCF-Crime not found")

if msl_found:
    print("\nMSL Feature candidates:")
    for i, path in enumerate(msl_found, 1):
        count = len(list(path.rglob('*.*')))
        print(f"  {i}. {path}")
        print(f"     Files: ~{count}")
else:
    print("\n✗ MSL Features not found")

print("\n" + "=" * 80)
print("Please provide the correct paths or place data in:")
print(f"  {Path(__file__).parent.parent / 'data'}")
print("=" * 80)
