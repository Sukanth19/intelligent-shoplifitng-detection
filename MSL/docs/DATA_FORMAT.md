# MSL-VideoSwin Feature Data Format

**Generated from**: Feature inspection on 2024  
**Data Source**: Synthetic development data (to be replaced with real MSL-VideoSwin features)

## Overview

Feature Path: `C:\code\projects\shopliftdetection\MSL\data\synthetic`  
**Status**: Using synthetic data for development

## Directory Structure

```
synthetic/ (or MSL-Features/)
├── train/          # Training features
│   ├── Normal_000.npy
│   ├── Normal_001.npy
│   ├── ...
│   ├── Shoplifting_000.npy
│   ├── Robbery_001.npy
│   └── ...
├── test/           # Test features
│   ├── Normal_000.npy
│   ├── Shoplifting_000.npy
│   └── ...
├── labels/         # Label files
│   ├── train_labels.json
│   └── test_labels.json
└── README.md
```

## Feature File Format

### File Format
- **Extension**: `.npy` (NumPy binary format)
- **Loading**: `np.load(filename)`
- **Dtype**: `float32`

### Tensor Shape
- **Format**: `(T, D)` - 2D array
- **T**: Number of temporal snippets (variable per video)
  - Range observed: 60-200 snippets
  - Typical: ~100-150 snippets
- **D**: Feature dimension = **1024**
  - This matches VideoSwin-Base output dimension

### Feature Properties
- **Normalization**: L2-normalized per snippet (||f|| ≈ 1)
- **Value range**: Typically [-0.15, 0.15]
- **No NaN/Inf**: Clean features

### Example
```python
import numpy as np

# Load feature
features = np.load('train/Shoplifting_000.npy')
# Shape: (105, 1024)
# - 105 temporal snippets
# - 1024-dim VideoSwin features per snippet
```

## Label Format

### File Format
- **Extension**: `.json`
- **Structure**: List of dictionaries

### Label Schema
```json
[
  {
    "filename": "Normal_000.npy",
    "label": 0,
    "category": "Normal",
    "num_snippets": 126
  },
  {
    "filename": "Shoplifting_000.npy",
    "label": 1,
    "category": "Shoplifting",
    "num_snippets": 105
  }
]
```

### Label Fields
- **filename** (str): Feature filename (without path)
- **label** (int): Binary label
  - `0`: Normal
  - `1`: Anomaly
- **category** (str): Anomaly category
  - Normal videos: `"Normal"`
  - Anomaly videos: `"Shoplifting"`, `"Robbery"`, `"Stealing"`, `"Burglary"`, etc.
- **num_snippets** (int): Number of temporal snippets (T)

### Label Types
- **Video-level only**: Labels indicate whether entire video is normal/anomalous
- **No snippet-level annotations**: Weakly supervised setting
- **No temporal annotations**: Exact anomaly timing unknown

## Dataset Statistics

### Synthetic Data (Development)
| Split | Normal | Anomaly | Total |
|-------|--------|---------|-------|
| Train | 10     | 10      | 20    |
| Test  | 5      | 5       | 10    |

### Real UCF-Crime (Expected)
| Split | Normal | Anomaly | Total |
|-------|--------|---------|-------|
| Train | ~800   | ~800    | ~1600 |
| Test  | ~100   | ~100    | ~200  |

## Anomaly Categories in UCF-Crime

1. Abuse
2. Arrest
3. Arson
4. Assault
5. Burglary
6. Explosion
7. Fighting
8. RoadAccidents
9. Robbery
10. **Shoplifting** ← Target category
11. Shooting
12. Stealing
13. Vandalism

## Filename Conventions

### Observed Pattern
```
{Category}_{ID}.npy
```

Examples:
- `Normal_000.npy`, `Normal_001.npy`
- `Shoplifting_000.npy`, `Shoplifting_001.npy`
- `Robbery_001.npy`

### Notes
- IDs are zero-padded (3 digits)
- Category name from UCF-Crime taxonomy
- No subdirectories within train/test

## VideoSwin Feature Extraction Details

Based on MSL authors' configuration:

### Model Configuration
```python
model:
  backbone: VideoSwin-Base
    patch_size: (2, 4, 4)
    drop_path_rate: 0.3
  test_cfg:
    max_testing_views: 4
```

### Preprocessing Pipeline
```python
1. Resize: scale=(-1, 224)
2. CenterCrop: crop_size=(288, 224)
3. Normalize:
   - mean=[123.675, 116.28, 103.53]
   - std=[58.395, 57.12, 57.375]
4. FormatShape: NCTHW
```

### Feature Extraction
- **Input**: Video frames (NCTHW format)
- **Model**: VideoSwin-Base pretrained on Kinetics-400
- **Output**: 1024-dim features per temporal snippet
- **Temporal**: Features extracted at ~1 snippet per N frames

## Implementation Notes

### Variable-Length Handling
- Videos have different durations → different T values
- Dataset loader must support:
  - Padding shorter sequences
  - Masking padded positions
  - Batching with variable lengths

### Memory Considerations
- Single feature file: ~0.3-0.7 MB
- Full dataset: ~500 MB - 2 GB (estimated)
- Can load entire dataset into RAM if needed

### Data Loading Strategy
```python
# Recommended approach
1. Load feature: features = np.load(path)  # (T, 1024)
2. Load label: label = labels_dict[filename]  # 0 or 1
3. Return: (features, label, video_id)
```

## Migration to Real Data

When real MSL-VideoSwin-Feature data is available:

1. **Download** from: [Google Drive Link](https://drive.google.com/drive/folders/12G8HSj7sFK60QHHnGFriFqL4Ld94YgUy)

2. **Place in**: `data/MSL-Features/` (or update config)

3. **Verify structure** matches:
   ```
   MSL-Features/
   ├── train/
   ├── test/
   └── labels/ (or similar)
   ```

4. **Re-run inspection**:
   ```bash
   python scripts/inspect_features.py
   ```

5. **Update this file** with actual findings

6. **Dataset loader** should work without changes if format is similar

## Known Assumptions

### Implementation Assumption 1
**Assumption**: Feature files are named `{Category}_{ID}.npy`  
**Source**: Synthetic data generation  
**Impact**: Filename parsing in dataset loader  
**Risk**: Low - easy to adapt if different

### Implementation Assumption 2
**Assumption**: Labels are in separate JSON files  
**Source**: Synthetic data generation  
**Impact**: Label loading logic  
**Risk**: Medium - real data might use different format (txt, csv, mat)  
**Mitigation**: Inspect real data and adapt loader

### Implementation Assumption 3
**Assumption**: Feature dimension is 1024 (VideoSwin-Base)  
**Source**: VideoSwin-Base architecture  
**Impact**: Model input dimension  
**Risk**: Low - well documented in VideoSwin paper

### Implementation Assumption 4
**Assumption**: Features are L2-normalized  
**Source**: Common practice in video anomaly detection  
**Impact**: Feature preprocessing  
**Risk**: Medium - verify with real data  
**Mitigation**: Check feature statistics and adjust if needed

## References

- **MSL Paper**: https://ojs.aaai.org/index.php/AAAI/article/view/20028
- **MSL Repository**: https://github.com/LiShuo1001/MSL
- **VideoSwin**: https://github.com/SwinTransformer/Video-Swin-Transformer
- **UCF-Crime**: http://www.crcv.ucf.edu/projects/real-world/

---

**Status**: ✅ Format documented for synthetic data  
**Next**: Implement feature dataset loader (`src/datasets/ucf_crime_features.py`)  
**Note**: Update this document after inspecting real MSL-VideoSwin features
