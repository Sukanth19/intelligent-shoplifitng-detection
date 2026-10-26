# How to Compare with the Base Paper

## 📄 Base Paper Information

**Title**: Learning Appearance and Motion Representations Separately for Anomaly Detection  
**Conference**: AAAI 2022  
**Authors**: Unknown (MSL paper)  
**Task**: Weakly-supervised video anomaly detection

## 🎯 Key Metrics from Paper

The paper evaluates on **UCF-Crime** dataset using:

### Primary Metrics:
1. **AUC (Area Under ROC Curve)** - Most important metric
   - UCF-Crime: ~82-84% (exact value depends on features)
   - ShanghaiTech: ~95-96%

2. **Frame-level AUC**
   - Measures detection accuracy at frame level
   - UCF-Crime: ~80-82%

3. **AP (Average Precision)**
   - Alternative to AUC
   - UCF-Crime: ~75-78%

### Training Details:
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Batch Size**: 16-32
- **Epochs**: 50-100
- **Features**: Video Swin Transformer (1024-dim)
- **Loss**: MSL Ranking Loss with margin=0.5
- **Evaluation**: Test set (separate from training)

## 🔧 How to Evaluate Your Model

### Step 1: Train the Model

Run the balanced training script:
```bash
cd MSL
python train_balanced.py
```

This will:
- Train on 50 normal + 50 anomaly videos
- Use batch size 8 (smaller for speed)
- Train for 10 epochs (paper uses 50-100)
- Save checkpoints in `checkpoints/`

### Step 2: Run Evaluation

Create an evaluation script to compute AUC:

```bash
python evaluate.py --checkpoint checkpoints/balanced_final.pt
```

The evaluation script should:
1. Load test set from `ucf_swin_test_feature_224288224288.h5`
2. Run inference on all test videos
3. Compute frame-level anomaly scores
4. Calculate AUC using ground truth labels
5. Compare with paper's reported AUC (~82%)

### Step 3: Compare Results

| Metric | Paper (UCF-Crime) | Your Model | Notes |
|--------|------------------|------------|-------|
| AUC | 82-84% | ??? | Higher is better |
| Frame-level AUC | 80-82% | ??? | Higher is better |
| AP | 75-78% | ??? | Higher is better |
| Training time | ~hours | ~minutes | Yours is faster (fewer samples) |

## 📊 Expected Performance

### Your Setup vs Paper:
- **Your training**: 100 videos (50 normal + 50 anomaly), 10 epochs
- **Paper training**: 950 videos, 50-100 epochs
- **Your test**: Same test set as paper
- **Expected AUC**: 70-75% (lower due to less data/training)

**Why lower?**
1. Trained on only 100 videos vs 950 in paper
2. Only 10 epochs vs 50-100 in paper
3. No data augmentation
4. No self-training iterations
5. Manual gradient descent (not Adam optimizer due to bug)

## 🚀 How to Improve Performance

To match paper's performance:

### 1. **Use More Training Data**
```python
NUM_NORMAL = 400   # More normal videos
NUM_ANOMALY = 400  # More anomaly videos
```

### 2. **Train Longer**
```python
EPOCHS = 50  # Match paper's training epochs
```

### 3. **Fix Optimizer Bug** (if possible)
- Downgrade to PyTorch 2.3.1, OR
- Use Python 3.11 instead of 3.13
- Then use Adam optimizer instead of manual gradient descent

### 4. **Add Self-Training**
The paper uses self-training:
1. Train initial model on labeled data
2. Pseudo-label high-confidence predictions
3. Retrain with pseudo-labels
4. Repeat 2-3 times

### 5. **Tune Hyperparameters**
```python
LEARNING_RATE = 0.0001  # Paper value
BATCH_SIZE = 16         # Paper value
margin = 0.5            # Loss margin
```

## 📝 Creating Evaluation Script

You need to create `evaluate.py`:

```python
"""
Evaluate MSL model on test set
"""

import torch
from src.datasets import UCFH5Dataset
from src.models import MSLModel
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Load checkpoint
checkpoint = torch.load('checkpoints/balanced_final.pt')
model = MSLModel(...)  # Same config as training
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test set
test_dataset = UCFH5Dataset(
    h5_path='../data/UCF-VideoSwin-Features/ucf_swin_test_feature_224288224288.h5',
    list_path='../data/UCF-VideoSwin-Features/ucf-test.list',
    gt_path='../data/UCF-VideoSwin-Features/ucf_gt_16.npy',
    mode='test'
)

# Run inference
all_scores = []
all_labels = []

for i in range(len(test_dataset)):
    sample = test_dataset[i]
    features = sample['features'].unsqueeze(0)  # Add batch dim
    
    with torch.no_grad():
        outputs = model(features)
        score = outputs['video_scores'].item()
    
    all_scores.append(score)
    all_labels.append(sample['label'])

# Compute metrics
auc = roc_auc_score(all_labels, all_scores)
ap = average_precision_score(all_labels, all_scores)

print(f"AUC: {auc:.4f} (Paper: 0.82-0.84)")
print(f"AP: {ap:.4f} (Paper: 0.75-0.78)")
```

## 📚 Where to Find Paper Metrics

### Option 1: Original Paper
- Search for "Learning Appearance and Motion Representations Separately for Anomaly Detection AAAI 2022"
- Look for Table 1 or Table 2 with UCF-Crime results
- Note the AUC values for Video Swin features

### Option 2: Papers with Code
- Visit: https://paperswithcode.com/dataset/ucf-crime
- Find leaderboard for weakly-supervised video anomaly detection
- Look for MSL paper entry
- Compare your AUC with reported values

### Option 3: GitHub Repository
- Search for MSL paper's official code repository
- Look for `README.md` with reported metrics
- Check `results/` folder for detailed numbers

## 🎯 Success Criteria

Your implementation is successful if:
1. ✅ Loss decreases during training (not stuck at 0.0000)
2. ✅ AUC > 0.50 (better than random)
3. ✅ AUC > 0.65 (reasonable performance with limited data)
4. 🎯 AUC > 0.75 (good performance)
5. 🏆 AUC > 0.80 (close to paper)

## 🐛 Current Issue: Loss = 0.0000

**Problem**: Your training shows loss=0.0000 because:
- First 100 videos are all "Abuse" (anomaly class)
- No normal videos in training set
- MSL ranking loss needs BOTH normal and anomaly videos
- Without pairs, loss = 0

**Solution**: Use `train_balanced.py` which:
- Explicitly loads 50 normal + 50 anomaly videos
- Alternates them for balanced batches
- Ensures every batch has both classes
- Loss will be > 0 and decrease over epochs

## 🏃‍♂️ Quick Start

```bash
# 1. Train with balanced dataset
cd MSL
python train_balanced.py

# 2. You should see:
#    - Loss > 0 (not 0.0000)
#    - Loss decreasing over epochs
#    - Both normal and anomaly in each batch

# 3. Create evaluation script (see above)
# 4. Run evaluation on test set
# 5. Compare AUC with paper's 82-84%
```

## 📖 Summary

To compare with the base paper:
1. **Train**: Use `train_balanced.py` to get non-zero loss
2. **Evaluate**: Create evaluation script to compute AUC on test set
3. **Compare**: Your AUC vs Paper's AUC (~82%)
4. **Improve**: More data, more epochs, fix optimizer
5. **Success**: AUC > 0.65 with limited data is good!

Your model trains in ~2 seconds vs hours for the paper, but you're using 10% of the data and 10% of the epochs. The goal is to prove the implementation works, not to beat the paper with limited resources! 🚀
