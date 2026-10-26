# Phase 7 Complete - Score Correction & Inference Pipeline

## ✅ Completed

### 1. Score Correction Module (`src/inference/score_correction.py`)

**Already implemented with multiple correction strategies:**
- ✅ Linear blending (default)
- ✅ Multiplicative correction
- ✅ Threshold-based correction
- ✅ No correction mode

**Additional features:**
- ✅ `apply_temporal_smoothing()` - 3 methods (Gaussian, Uniform, Median)
- ✅ `detect_anomaly_intervals()` - Temporal anomaly localization
- ✅ Configurable parameters (alpha, threshold, window size)

**Code:** 321 lines with comprehensive documentation

### 2. MSL Predictor (`src/inference/predictor.py`)

**Complete inference interface:**
- ✅ Single video prediction
- ✅ Batch prediction
- ✅ Variable-length video handling
- ✅ Score correction integration
- ✅ Temporal smoothing integration
- ✅ Anomaly interval detection
- ✅ Formatted output generation
- ✅ Prediction saving to files

**Key Features:**
```python
predictor = MSLPredictor(
    model=model,
    device='cuda',
    score_correction_method='linear',
    score_correction_alpha=0.5,
    temporal_smoothing=True,
    smoothing_window=5,
    anomaly_threshold=0.5
)

result = predictor.predict(features)
# Returns: video_score, snippet_scores, corrected_scores,
#          smoothed_scores, anomaly_intervals, prediction
```

**Code:** 358 lines

### 3. Inference Script (`scripts/inference.py`)

**Complete CLI application:**
```bash
# Single video inference
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/default.yaml \
  --feature path/to/features.npy \
  --output results/inference

# Batch inference
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/default.yaml \
  --feature-dir data/synthetic \
  --split test \
  --output results/inference
```

**Features:**
- ✅ Single video inference
- ✅ Batch inference on dataset
- ✅ Automatic metric calculation (accuracy, precision, recall, F1, AUC)
- ✅ Multiple file format support (.npy, .pt, .npz)
- ✅ GPU/CPU device selection
- ✅ Comprehensive logging
- ✅ Multiple output formats (text, JSON, NPZ)

**Output files:**
- Text predictions (human-readable)
- Score arrays (NPZ format)
- Metrics (JSON)
- Detailed report

**Code:** 368 lines

### 4. Checkpoint Loading

**Utility function:**
```python
predictor = load_predictor_from_checkpoint(
    checkpoint_path=Path('checkpoints/best_model.pt'),
    model_class=MSLModel,
    model_config=config['model'],
    device='cuda',
    score_correction_method='linear',
    temporal_smoothing=True
)
```

**Handles:**
- ✅ Model state dict loading
- ✅ Automatic device placement
- ✅ Predictor configuration
- ✅ Checkpoint metadata logging

### 5. Testing (`tests/test_inference_simple.py`)

**Comprehensive test coverage:**
- Score correction (linear, multiplicative, threshold) ✓
- Temporal smoothing (Gaussian, uniform, median) ✓
- Interval detection (single, multiple, merging) ✓
- Predictor (single, batch, variable-length) ✓
- Formatted output generation ✓

**All 18 tests passing! ✅**

## 📊 Inference Pipeline Architecture

### Complete Flow

```
Trained Model + Checkpoint
        ↓
Load Predictor
        ↓
Input Features (T, D) or (B, T, D)
        ↓
MSL Model Forward Pass
  ├─ Video-level scores
  └─ Snippet-level scores
        ↓
Score Correction
  └─ Linear/Multiplicative/Threshold
        ↓
Temporal Smoothing (Optional)
  └─ Gaussian/Uniform/Median
        ↓
Anomaly Interval Detection
  ├─ Threshold-based detection
  ├─ Minimum length filtering
  └─ Interval merging
        ↓
Output
  ├─ Video anomaly score
  ├─ Binary prediction
  ├─ Snippet-level scores
  ├─ Corrected scores
  ├─ Smoothed scores
  └─ Anomaly intervals [(start, end), ...]
```

### Score Correction Methods

#### 1. Linear Blending (Default)
```
s'_t = α * p_video + (1 - α) * s_t

where:
- s'_t: Corrected snippet score at time t
- p_video: Video-level anomaly probability
- s_t: Original snippet score
- α: Blending factor (default: 0.5)
```

**Use case:** Balanced correction, moderate video-level influence

#### 2. Multiplicative Correction
```
s'_t = p_video * s_t
```

**Use case:** Strong suppression when video score is low

#### 3. Threshold-based Correction
```
s'_t = s_t  if p_video >= threshold
       0    otherwise
```

**Use case:** Binary video-level gating, hard suppression

### Temporal Smoothing

**Purpose:** Reduce temporal noise in anomaly scores

**Methods:**
1. **Gaussian**: Weighted smoothing with Gaussian kernel
2. **Uniform**: Simple moving average
3. **Median**: Robust to outliers

**Parameters:**
- `window_size`: Size of smoothing window (default: 5)
- Automatically handles boundary conditions

### Anomaly Interval Detection

**Algorithm:**
1. Threshold snippet scores
2. Find contiguous anomalous regions
3. Filter by minimum length
4. Merge intervals separated by small gaps

**Parameters:**
- `threshold`: Score threshold (default: 0.5)
- `min_length`: Minimum interval length (default: 3 snippets)
- `merge_gap`: Maximum gap to merge (default: 2 snippets)

**Output:** List of `(start, end)` tuples in snippet indices

## 🔧 Usage Examples

### Example 1: Single Video Inference

```python
from src.models import MSLModel
from src.inference import load_predictor_from_checkpoint
import torch

# Load predictor
predictor = load_predictor_from_checkpoint(
    checkpoint_path='checkpoints/best_model.pt',
    model_class=MSLModel,
    model_config={
        'input_dim': 1024,
        'hidden_dim': 512,
        'num_layers': 3,
        'num_heads': 8
    },
    device='cuda',
    score_correction_method='linear',
    score_correction_alpha=0.5,
    temporal_smoothing=True,
    smoothing_window=5,
    anomaly_threshold=0.5
)

# Load features
features = torch.load('video_features.pt')  # (T, D)

# Predict
result = predictor.predict(features)

print(f"Video Score: {result['video_score']:.4f}")
print(f"Prediction: {'ANOMALY' if result['prediction'] == 1 else 'NORMAL'}")
print(f"Anomaly Intervals: {result['anomaly_intervals']}")

# Formatted output
formatted = predictor.format_prediction(result, video_name='video.mp4', fps=30.0)
print(formatted)
```

### Example 2: Batch Inference

```python
# Batch features
batch_features = [
    torch.randn(50, 1024),  # Video 1
    torch.randn(60, 1024),  # Video 2
    torch.randn(45, 1024)   # Video 3
]

# Predict batch
results = predictor.predict_batch(batch_features)

# Process results
for i, result in enumerate(results):
    print(f"Video {i}: score={result['video_score']:.4f}, pred={result['prediction']}")
```

### Example 3: CLI Inference

```bash
# Single video
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/default.yaml \
  --feature data/synthetic/test/Shoplifting_000.npy \
  --output results/inference \
  --device cuda

# Batch on test set
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/default.yaml \
  --feature-dir data/synthetic \
  --split test \
  --output results/inference \
  --device cuda

# CPU inference
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/default.yaml \
  --feature data/synthetic/test/Normal_000.npy \
  --output results/inference \
  --device cpu
```

### Example 4: Custom Configuration

```yaml
# Add to configs/default.yaml

score_correction:
  method: 'linear'  # or 'multiplicative', 'threshold', 'none'
  alpha: 0.5        # for linear method
  threshold: 0.5    # for threshold method

temporal_smoothing:
  enabled: true
  window_size: 5
  method: 'gaussian'  # or 'uniform', 'median'

anomaly_threshold: 0.5
fps: 30.0
```

## 📈 Output Formats

### Text Output (Human-Readable)

```
Video: Shoplifting_000.npy
Video Anomaly Score: 0.8735
Prediction: ANOMALY

Detected Anomaly Intervals: 2
  1. Snippets 15-28 (0.5s - 0.9s, duration: 0.4s)
  2. Snippets 45-52 (1.5s - 1.7s, duration: 0.2s)
```

### NPZ Output (Score Arrays)

```python
data = np.load('results/inference/Shoplifting_000_scores.npz')
# Arrays:
# - video_score: scalar
# - snippet_scores: (T,)
# - corrected_scores: (T,)
# - smoothed_scores: (T,)
# - prediction: 0 or 1
```

### JSON Metrics (Batch Evaluation)

```json
{
  "accuracy": 0.8500,
  "precision": 0.8235,
  "recall": 0.8750,
  "f1": 0.8485,
  "auc": 0.9123,
  "num_videos": 10,
  "num_anomalies": 4,
  "num_predicted_anomalies": 4
}
```

## 📝 Implementation Notes

### Score Correction Design

**From MSL paper:**
> "Uses video-level anomaly probability to stabilize/correct snippet-level scores"

**Implementation assumption:**
The paper does not specify the exact correction formula. We implemented three plausible strategies:

1. **Linear blending** - Balances video and snippet signals
2. **Multiplicative** - Strong video-level gating
3. **Threshold** - Binary video-level decision

All three are configurable. Linear blending with α=0.5 is the default.

**Documented in:** `docs/IMPLEMENTATION_NOTES.md`

### Temporal Smoothing

**Not explicitly mentioned in MSL paper.**

This is a **project extension** to reduce temporal noise common in video anomaly detection.

Optional and configurable - can be disabled.

### Anomaly Interval Detection

**Implementation assumption:**
The paper focuses on snippet-level scoring but doesn't detail interval extraction.

Our implementation:
- Threshold-based detection
- Minimum length filtering (avoid false positives)
- Gap-based merging (avoid fragmentation)

All parameters are configurable.

### Device Handling

**Automatic GPU/CPU selection:**
- Checks CUDA availability
- Warns if CUDA requested but unavailable
- Automatically moves tensors to correct device
- Converts output to CPU for saving

## 🚀 Integration Points

### With Training System

```python
# After training
from src.training import MSLTrainer

trainer = MSLTrainer(...)
history = trainer.train(train_loader, val_loader, epochs=50)

# Load trained model for inference
predictor = load_predictor_from_checkpoint(
    checkpoint_path='checkpoints/best_model.pt',
    model_class=MSLModel,
    model_config=config['model'],
    device='cuda'
)
```

### With Evaluation System

```python
# Inference script automatically calculates:
# - Accuracy
# - Precision
# - Recall
# - F1 Score
# - AUC

# Saves to: results/inference/test_metrics.json
```

### With Visualization (Next Phase)

```python
# Inference outputs can be directly visualized:
# - Temporal anomaly score plots
# - Interval highlighting
# - Video-level predictions
```

## 🎯 Ready for Next Phase

Phase 7 is complete! The inference pipeline is fully functional and tested.

### What Works Now

✅ **Load trained model**
```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pt ...
```

✅ **Predict on single video**
- Video-level anomaly score
- Snippet-level scores
- Score correction
- Temporal smoothing
- Anomaly interval detection

✅ **Batch evaluation**
- Process entire test set
- Calculate metrics (accuracy, precision, recall, F1, AUC)
- Save predictions and reports

✅ **Multiple output formats**
- Human-readable text
- NumPy arrays (NPZ)
- JSON metrics
- Detailed reports

✅ **Flexible configuration**
- Correction methods
- Smoothing options
- Threshold tuning
- Device selection

### Next Steps - Phase 8

Now ready for:

1. **Evaluation Module** - Comprehensive evaluation with ROC curves, confusion matrices
2. **Visualization** - Plot temporal scores, intervals, video predictions
3. **Shoplifting-specific experiments** - Focus on Shoplifting class
4. **VideoSwin integration** - Raw video feature extraction
5. **Real-time inference** - Optimizations for deployment

---

**Status**: ✅ Phase 7 Complete  
**Inference Pipeline**: Fully implemented and tested  
**Score Correction**: 3 methods available  
**Temporal Smoothing**: 3 methods available  
**Interval Detection**: Configurable and robust  
**CLI Tool**: Complete with metrics and multiple outputs  
**Tests**: 18/18 passing ✅

**Ready**: For evaluation module and visualization! 🎉
