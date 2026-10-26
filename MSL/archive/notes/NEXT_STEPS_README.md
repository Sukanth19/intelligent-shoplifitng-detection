# Next Steps - Complete Guide

This guide explains how to complete the remaining steps to match the base paper's performance.

---

## 📋 Overview

You currently have a working proof-of-concept. To match the paper's 82-84% AUC, complete these steps:

1. ✅ **Evaluate AUC** - Measure current performance
2. ✅ **Full-scale Training** - 50 epochs, 400 videos, batch 16
3. ✅ **Fix Optimizer** - Use Adam instead of manual GD (+2-3% AUC)
4. ✅ **Self-training** - Iterative pseudo-labeling (+3% AUC)

---

## 🚀 Quick Start - Run Everything

### Option 1: Automated Pipeline (Easiest)

Run the complete pipeline that does everything automatically:

```bash
cd MSL
python run_complete_pipeline.py
```

This will:
- Train full-scale model (50 epochs, 400 videos)
- Evaluate AUC
- (Optional) Train with Adam
- (Optional) Run self-training
- Generate all results

**Time**: ~1-2 hours total

---

### Option 2: Step-by-Step (Recommended)

Run each step individually for better control:

#### Step 1: Full-Scale Training

Train with 50 epochs, 400 videos, batch size 16:

```bash
python train_full_scale.py
```

**Time**: 30-60 minutes  
**Output**: `checkpoints/fullscale_final.pt`  
**Expected**: Loss ~0.7-0.8

---

#### Step 2: Evaluate AUC

Measure performance on test set:

```bash
python evaluate.py --checkpoint checkpoints/fullscale_final.pt
```

**Time**: 1-2 minutes  
**Output**: 
- `results/evaluation_results.txt` - Metrics
- `results/roc_curve.png` - ROC curve plot
- `results/scores.npz` - Detailed scores

**Expected AUC**: 75-80%

---

#### Step 3: Train with Adam (Optional)

Use Adam optimizer for +2-3% AUC improvement:

```bash
python train_with_adam.py
```

**Requirements**:
- PyTorch 2.3.1 or lower, OR
- Python 3.11 or lower

**If you get sympy error**:
```bash
# Downgrade PyTorch
pip install torch==2.3.1

# OR use Python 3.11
```

**Time**: 30-60 minutes  
**Output**: `checkpoints/adam_final.pt`  
**Expected AUC**: 77-83%

Then evaluate:
```bash
python evaluate.py --checkpoint checkpoints/adam_final.pt
```

---

#### Step 4: Self-Training (Optional)

Iterative pseudo-labeling for +3% AUC:

```bash
python train_self_training.py
```

**Time**: 30-45 minutes  
**Output**: `checkpoints/selftraining_final.pt`  
**Expected AUC**: 80-83%

Then evaluate:
```bash
python evaluate.py --checkpoint checkpoints/selftraining_final.pt
```

---

## 📊 Understanding Results

### Evaluation Metrics

After running `evaluate.py`, you'll see:

```
AUC: 0.7823 (78.23%)
AP: 0.7456 (74.56%)
```

**What does this mean?**

- **AUC (Area Under ROC Curve)**: Overall model performance
  - 0.50 = Random guessing
  - 0.75-0.80 = Good performance
  - 0.80-0.85 = Excellent (paper's range)
  - 1.00 = Perfect

- **AP (Average Precision)**: Precision-recall trade-off
  - Paper reports 75-78%
  - Should be close to AUC

### ROC Curve

The ROC curve (`results/roc_curve.png`) shows:
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Diagonal line: Random performance
- Your curve: Should be above diagonal
- Area under curve: Your AUC

---

## 📈 Expected Performance Progression

| Step | Configuration | Expected AUC | Time |
|------|--------------|--------------|------|
| **Proof of Concept** | 10 epochs, 100 videos, batch 8 | 65-70% | 13 sec |
| **Full-Scale** | 50 epochs, 400 videos, batch 16 | 75-80% | 30-60 min |
| **+ Adam Optimizer** | Same + Adam | 77-83% | 30-60 min |
| **+ Self-Training** | Same + pseudo-labels | 80-83% | 30-45 min |
| **Paper** | 75 epochs, 950 videos, Adam | 82-84% | 2-3 hours |

---

## 🎯 Goal: Match Paper's 82-84%

### Current Gap Analysis

If you get **78% AUC** after full-scale training:

**Gap**: 82-84% (paper) - 78% (yours) = **4-6%**

**How to close the gap**:

1. **Adam optimizer**: +2-3% → 80-81% AUC ✅
2. **Self-training**: +3% → 81-83% AUC ✅
3. **More data**: Use all 950 videos → 82-84% AUC

With steps 1+2, you should reach **80-83% AUC** (matching paper!)

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'sympy.functions'"

**Cause**: PyTorch 2.6 + Python 3.13 bug

**Solutions**:
1. Downgrade PyTorch: `pip install torch==2.3.1`
2. Use Python 3.11 instead of 3.13
3. Skip Adam training, use manual GD (works fine)

### Problem: "Checkpoint not found"

**Cause**: Training didn't complete or checkpoint path wrong

**Solution**:
```bash
# Check available checkpoints
ls checkpoints/

# Use correct checkpoint path
python evaluate.py --checkpoint checkpoints/your_checkpoint.pt
```

### Problem: AUC is very low (<0.60)

**Cause**: Training didn't converge or data issue

**Solution**:
1. Check training loss - should decrease
2. Train longer (more epochs)
3. Check balanced batches (need normal + anomaly)

### Problem: Training is too slow

**Cause**: Large batch size or dataset

**Solution**:
1. Reduce epochs: 50 → 20
2. Reduce videos: 400 → 200
3. Use GPU (should be automatic if available)

---

## 📂 File Structure

After completing all steps:

```
MSL/
├── train_balanced.py           # Proof of concept (done)
├── train_full_scale.py         # Full-scale training
├── train_with_adam.py          # Adam optimizer training
├── train_self_training.py      # Self-training loop
├── evaluate.py                 # Evaluation script
├── run_complete_pipeline.py    # Run everything
│
├── checkpoints/
│   ├── balanced_final.pt       # Proof of concept model
│   ├── fullscale_final.pt      # Full-scale model
│   ├── adam_final.pt           # Adam model
│   └── selftraining_final.pt   # Self-trained model
│
├── results/
│   ├── evaluation_results.txt  # AUC metrics
│   ├── roc_curve.png           # ROC curve plot
│   └── scores.npz              # Detailed scores
│
└── [source code files...]
```

---

## 💡 Tips

### For Fast Experimentation

Use smaller configuration:
```python
EPOCHS = 20  # Instead of 50
NUM_NORMAL = 100  # Instead of 200
NUM_ANOMALY = 100  # Instead of 200
```

### For Maximum Performance

1. Use all available data (950 videos)
2. Train for 75-100 epochs
3. Use Adam optimizer
4. Run 3 self-training iterations
5. Tune hyperparameters (learning rate, margin)

### For Quick Results

Just run step 1 (full-scale) and step 2 (evaluate):
```bash
python train_full_scale.py
python evaluate.py --checkpoint checkpoints/fullscale_final.pt
```

This gives you 75-80% AUC in ~1 hour.

---

## 🎓 What You'll Learn

By completing these steps, you'll understand:

1. **Evaluation metrics**: AUC, AP, ROC curves
2. **Optimizer impact**: Adam vs manual gradient descent
3. **Self-training**: Pseudo-labeling for semi-supervised learning
4. **Performance tuning**: How to improve model accuracy

---

## ✅ Success Criteria

You've successfully matched the paper when:

- ✅ AUC ≥ 80% (within 2-4% of paper)
- ✅ Training loss decreases steadily
- ✅ ROC curve is well above diagonal
- ✅ AP close to AUC (within 2-3%)

**Paper's metrics**:
- AUC: 82-84%
- AP: 75-78%

**Your target**:
- AUC: 80-83%
- AP: 73-78%

---

## 📞 Need Help?

If you encounter issues:

1. Check training logs for errors
2. Verify checkpoint exists: `ls checkpoints/`
3. Check GPU usage: `nvidia-smi`
4. Review loss values - should decrease
5. Verify balanced batches in output

---

## 🎉 Summary

**Quick path** (1 hour):
```bash
python train_full_scale.py
python evaluate.py --checkpoint checkpoints/fullscale_final.pt
```
Expected: 75-80% AUC

**Complete path** (2-3 hours):
```bash
python run_complete_pipeline.py
```
Expected: 80-83% AUC (matches paper!)

**Start now**: Choose your path and run the commands! 🚀
