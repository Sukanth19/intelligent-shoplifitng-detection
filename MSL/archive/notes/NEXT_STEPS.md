# Next Steps - Phase 11 and Beyond

**Date**: 2026-09-18  
**Current Phase**: 10/15 Complete (67%)  
**Next Phase**: Phase 11 - Real Data Integration & Optimization

---

## 🎯 Immediate Priority: Get System Running on Real Data

### Current Situation

✅ **What's Complete:**
- Complete MSL implementation (~7,400+ lines of code)
- All core modules working (model, loss, training, inference, evaluation)
- 75+ tests passing
- VideoSwin integration architecture ready
- Synthetic data testing working

✅ **What's Available:**
- **Real UCF-Crime features**: `c:\code\projects\shopliftdetection\data\UCF-VideoSwin-Features\`
  - `ucf_swin_train_feature_224288224288.h5`
  - `ucf_swin_test_feature_224288224288.h5`
  - `ucf-train-filter.list`
  - `ucf-test.list`
  - `ucf_gt_16.npy`

- **Real SHT features**: `c:\code\projects\shopliftdetection\data\SHT-VideoSwin-Features\`
  - `sh_swin_train_feature_256320224.h5`
  - `sh_swin_test_feature_256320224.h5`
  - `sh-train.list`
  - `sh-test.list`
  - `sh_gt_16.npy`

❌ **What's Missing:**
- H5 data format loader (currently only supports .npy)
- Understanding of label format in .list and .npy files
- Configuration for real data paths
- Testing on real data

---

## 📋 Action Plan

### Step 1: Install Missing Dependencies ⚡ CRITICAL

```bash
# From MSL directory
cd c:\code\projects\shopliftdetection\MSL

# Install h5py for reading H5 files
pip install h5py

# Install other missing dependencies
pip install scikit-learn einops opencv-python seaborn

# Or install everything at once
pip install -r requirements.txt

# Verify installation
python -c "import h5py; import sklearn; import einops; print('All dependencies installed!')"
```

### Step 2: Inspect Real Data Format

```bash
# Run the inspection script to understand H5 file structure
python scripts/inspect_real_features.py
```

**This will reveal:**
- How features are stored in H5 files
- Feature dimensions (T, D)
- Label format
- Video name mapping
- Train/test split information

### Step 3: Create H5 Dataset Loader

**File to create:** `src/datasets/ucf_crime_h5.py`

**Requirements:**
- Read features from H5 files
- Parse .list files for video names
- Parse .npy ground truth files for labels
- Handle both UCF-Crime and SHT datasets
- Compatible with existing training pipeline

**Example structure:**
```python
class UCFCrimeH5Dataset(Dataset):
    def __init__(self, h5_path, list_path, gt_path, ...):
        self.features = h5py.File(h5_path, 'r')
        self.video_names = self._load_list(list_path)
        self.labels = np.load(gt_path)
    
    def __getitem__(self, idx):
        # Load features from H5
        # Return (features, label, video_name)
        pass
```

### Step 4: Update Configuration

**File:** `configs/ucf_crime.yaml`

```yaml
# Create new config for UCF-Crime real data
dataset:
  name: "UCF-Crime"
  type: "h5"
  train:
    h5_path: "../data/UCF-VideoSwin-Features/ucf_swin_train_feature_224288224288.h5"
    list_path: "../data/UCF-VideoSwin-Features/ucf-train-filter.list"
    gt_path: "../data/UCF-VideoSwin-Features/ucf_gt_16.npy"
  test:
    h5_path: "../data/UCF-VideoSwin-Features/ucf_swin_test_feature_224288224288.h5"
    list_path: "../data/UCF-VideoSwin-Features/ucf-test.list"
    gt_path: "../data/UCF-VideoSwin-Features/ucf_gt_16.npy"

model:
  input_dim: 1024  # VideoSwin feature dim
  # ... rest of model config
```

### Step 5: Test Data Loading

```bash
# Test H5 dataset loader
python -c "
from src.datasets.ucf_crime_h5 import UCFCrimeH5Dataset
dataset = UCFCrimeH5Dataset(...)
print(f'Dataset size: {len(dataset)}')
features, label, name = dataset[0]
print(f'Feature shape: {features.shape}')
print(f'Label: {label}')
print(f'Video name: {name}')
"
```

### Step 6: Train on Real Data

```bash
# Train MSL on UCF-Crime features
python scripts/train.py --config configs/ucf_crime.yaml --epochs 50

# This should:
# - Load real UCF-Crime features
# - Train MSL model
# - Save checkpoints
# - Log to TensorBoard
```

### Step 7: Evaluate on Real Data

```bash
# Comprehensive evaluation
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/ucf_crime.yaml \
  --output results/ucf_crime_evaluation

# Shoplifting-specific evaluation
python scripts/evaluate_shoplifting.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/ucf_crime.yaml \
  --output results/shoplifting_evaluation
```

---

## 🔧 Detailed Tasks

### Task 1.1: Install h5py

**Priority**: CRITICAL ⚡  
**Time**: 2 minutes

```bash
pip install h5py
```

### Task 1.2: Inspect H5 Files

**Priority**: HIGH  
**Time**: 5 minutes  
**Script**: `scripts/inspect_real_features.py` (already created)

```bash
python scripts/inspect_real_features.py
```

**Expected output:**
- H5 file structure (keys, shapes, dtypes)
- Feature dimensions
- Number of videos
- Label format
- Video name format

### Task 1.3: Understand Label Format

**Priority**: HIGH  
**Time**: 10 minutes

**Files to analyze:**
- `ucf-train-filter.list` - List of training video names
- `ucf-test.list` - List of test video names
- `ucf_gt_16.npy` - Ground truth labels

**Questions to answer:**
- Are labels video-level (normal/abnormal) or snippet-level?
- How are labels mapped to videos?
- What does "gt_16" mean? (probably 16-snippet temporal segments)
- Are there category labels (Shoplifting, Robbery, etc.)?

### Task 1.4: Create H5 Dataset Loader

**Priority**: HIGH  
**Time**: 30-60 minutes  
**File**: `src/datasets/ucf_crime_h5.py`

**Implementation requirements:**
1. Read from H5 files efficiently
2. Support both train and test splits
3. Handle variable-length sequences
4. Parse video names from .list files
5. Load labels from .npy ground truth
6. Return (features, label, metadata) tuples
7. Compatible with DataLoader
8. Handle missing or corrupt data gracefully

**Code structure:**
```python
import h5py
import numpy as np
from torch.utils.data import Dataset

class UCFCrimeH5Dataset(Dataset):
    """
    UCF-Crime dataset loader for H5 feature files.
    
    Expected file structure:
    - h5_path: H5 file with features
    - list_path: Text file with video names
    - gt_path: NPY file with ground truth labels
    """
    
    def __init__(
        self,
        h5_path: str,
        list_path: str,
        gt_path: str = None,
        max_length: int = 320,
        mode: str = 'train'
    ):
        self.h5_file = h5py.File(h5_path, 'r')
        self.video_names = self._load_list(list_path)
        self.labels = self._load_labels(gt_path) if gt_path else None
        self.max_length = max_length
        self.mode = mode
    
    def _load_list(self, list_path):
        """Load video names from .list file."""
        with open(list_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _load_labels(self, gt_path):
        """Load ground truth labels from .npy file."""
        return np.load(gt_path, allow_pickle=True)
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        # Get video name
        video_name = self.video_names[idx]
        
        # Load features from H5
        # (implementation depends on H5 structure)
        features = self._load_features(video_name)
        
        # Get label
        label = self.labels[idx] if self.labels is not None else 0
        
        # Return as dict
        return {
            'features': features,
            'label': label,
            'video_name': video_name,
            'length': len(features)
        }
```

### Task 1.5: Create Test for H5 Loader

**Priority**: MEDIUM  
**Time**: 15 minutes  
**File**: `tests/test_h5_dataset.py`

```python
def test_h5_dataset_loading():
    """Test H5 dataset can load and return correct format."""
    dataset = UCFCrimeH5Dataset(...)
    
    # Test length
    assert len(dataset) > 0
    
    # Test __getitem__
    sample = dataset[0]
    assert 'features' in sample
    assert 'label' in sample
    assert 'video_name' in sample
    
    # Test feature shape
    features = sample['features']
    assert len(features.shape) == 2  # (T, D)
    assert features.shape[1] == 1024  # VideoSwin feature dim
```

### Task 1.6: Update Training Script

**Priority**: MEDIUM  
**Time**: 15 minutes

**Modify** `scripts/train.py` to support H5 dataset:

```python
# Add H5 dataset option
if config['dataset']['type'] == 'h5':
    from src.datasets.ucf_crime_h5 import UCFCrimeH5Dataset
    train_dataset = UCFCrimeH5Dataset(...)
    test_dataset = UCFCrimeH5Dataset(...)
elif config['dataset']['type'] == 'npy':
    from src.datasets.ucf_crime_features import UCFCrimeFeatureDataset
    train_dataset = UCFCrimeFeatureDataset(...)
    test_dataset = UCFCrimeFeatureDataset(...)
```

### Task 1.7: Create UCF-Crime Config

**Priority**: MEDIUM  
**Time**: 10 minutes  
**File**: `configs/ucf_crime.yaml`

Copy from `configs/default.yaml` and update paths:

```yaml
# UCF-Crime Real Data Configuration
experiment:
  name: "ucf_crime_msl"
  output_dir: "results/ucf_crime"

dataset:
  name: "UCF-Crime"
  type: "h5"  # Use H5 loader
  train:
    h5_path: "../data/UCF-VideoSwin-Features/ucf_swin_train_feature_224288224288.h5"
    list_path: "../data/UCF-VideoSwin-Features/ucf-train-filter.list"
    gt_path: "../data/UCF-VideoSwin-Features/ucf_gt_16.npy"
  test:
    h5_path: "../data/UCF-VideoSwin-Features/ucf_swin_test_feature_224288224288.h5"
    list_path: "../data/UCF-VideoSwin-Features/ucf-test.list"
    gt_path: "../data/UCF-VideoSwin-Features/ucf_gt_16.npy"
  max_length: 320
  num_workers: 4

model:
  input_dim: 1024
  hidden_dim: 512
  num_layers: 3
  num_heads: 8
  sequence_length: 32
  sequence_stride: 16
  dropout: 0.1

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  # ... rest of training config
```

### Task 1.8: Train and Validate

**Priority**: HIGH  
**Time**: 2-4 hours (training time)

```bash
# Step 1: Small test run (1 epoch)
python scripts/train.py \
  --config configs/ucf_crime.yaml \
  --epochs 1

# Step 2: If successful, full training
python scripts/train.py \
  --config configs/ucf_crime.yaml \
  --epochs 50

# Step 3: Monitor with TensorBoard
tensorboard --logdir logs/
```

### Task 1.9: Evaluate Results

**Priority**: HIGH  
**Time**: 30 minutes

```bash
# Full evaluation
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/ucf_crime.yaml \
  --output results/ucf_crime_evaluation

# Check results
cat results/ucf_crime_evaluation/metrics.json
```

**Expected metrics (from paper):**
- UCF-Crime AUC: ~85% (paper reports this range)
- Shoplifting detection accuracy
- ROC curves
- PR curves

---

## 🎯 Phase 11 Tasks (After Real Data Works)

### Task 2.1: Feature Caching System

**Purpose**: Speed up repeated data loading

**Implementation:**
- Cache frequently accessed features
- LRU eviction policy
- Configurable cache size
- Thread-safe for DataLoader workers

### Task 2.2: Sliding Window Inference

**Purpose**: Process long videos incrementally

**Implementation:**
- Process video in overlapping windows
- Merge predictions smoothly
- Handle boundary cases
- Reduce memory usage

### Task 2.3: Batch Processing Optimization

**Purpose**: Maximize GPU utilization

**Implementation:**
- Dynamic batching by sequence length
- Efficient padding strategies
- Minimize data transfer overhead

### Task 2.4: Real-time Monitoring

**Purpose**: Live anomaly detection

**Implementation:**
- Streaming video input
- Incremental feature extraction
- Online prediction
- Alert system

---

## 📊 Success Criteria

### Phase 11 Complete When:

✅ **Data Integration**
- [ ] H5 dataset loader implemented and tested
- [ ] Successfully loads UCF-Crime features
- [ ] Successfully loads SHT features
- [ ] Labels correctly parsed and mapped

✅ **Training**
- [ ] Model trains on real UCF-Crime data
- [ ] Achieves reasonable loss convergence
- [ ] Checkpoints saved correctly
- [ ] TensorBoard logging works

✅ **Evaluation**
- [ ] Evaluation runs on test set
- [ ] Metrics computed correctly
- [ ] Results comparable to paper (AUC ~82-85%)
- [ ] ROC/PR curves generated

✅ **Shoplifting**
- [ ] Shoplifting videos correctly identified
- [ ] Shoplifting-specific metrics computed
- [ ] Temporal anomaly scores visualized

---

## 🚨 Potential Issues & Solutions

### Issue 1: H5 File Structure Unknown

**Problem**: Don't know how features are organized in H5  
**Solution**: Run `inspect_real_features.py` first

### Issue 2: Label Format Mismatch

**Problem**: Labels might not match our assumptions  
**Solution**: Carefully inspect ground truth files, adapt loader

### Issue 3: Memory Issues with Large H5 Files

**Problem**: H5 files might be very large  
**Solution**: Use H5 file object, load on-demand, don't load entire file

### Issue 4: Different Feature Dimensions

**Problem**: Features might not be 1024-dim  
**Solution**: Check actual dim, update model config

### Issue 5: Missing Video Names

**Problem**: Video name mapping might be unclear  
**Solution**: Cross-reference .list files with H5 keys

---

## 📚 Documentation to Update

After Phase 11 complete:

1. **DATA_FORMAT.md**: Document H5 file structure
2. **PROGRESS.md**: Update to Phase 11 complete
3. **CURRENT_STATUS.md**: Update status
4. **README.md**: Add real data usage examples
5. **PHASE11_COMPLETE.md**: Create completion document

---

## 🎉 Expected Outcome

After completing these steps:

✅ **Working System**
- Train on real UCF-Crime features
- Evaluate on real test set
- Generate predictions for shoplifting videos
- Visualize anomaly scores
- Compare with paper results

✅ **Validation**
- Verify MSL implementation correctness
- Validate against paper's reported performance
- Ensure shoplifting detection works

✅ **Ready for Next Phase**
- Phase 12: Raw video testing
- Phase 13: Visualization/UI
- Phase 14: Optional extensions
- Phase 15: Final integration

---

## ⏱️ Time Estimate

**Optimistic**: 4-6 hours  
**Realistic**: 8-12 hours  
**Conservative**: 1-2 days

**Breakdown:**
- Install dependencies: 10 min
- Inspect data: 30 min
- Create H5 loader: 1-2 hours
- Testing: 1 hour
- Training: 2-4 hours
- Evaluation: 1 hour
- Documentation: 1-2 hours

---

## 🚀 Let's Get Started!

**First command to run:**

```bash
cd c:\code\projects\shopliftdetection\MSL
pip install h5py scikit-learn einops
python scripts/inspect_real_features.py
```

**This will reveal the data structure and guide the next implementation steps!**

---

**Current Status**: Ready to proceed to Phase 11  
**Next Action**: Install h5py and inspect real data  
**Goal**: Get MSL training on real UCF-Crime features  
**Timeline**: Complete Phase 11 within 1-2 days

