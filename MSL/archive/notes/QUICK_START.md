# Quick Start Guide

## 🎯 Goal
Get the best possible AUC to match the base paper (82-84%)

## ⚡ Fastest Path

### Option 1: Automatic (Easiest)
```bash
cd MSL
python run_best_performance.py
```
**Time**: 2-4 hours  
**Expected AUC**: 75-80%

### Option 2: Manual Steps
```bash
# 1. Full dataset (30-60 min)
python train_full_dataset.py

# 2. Self-training (60-120 min)
python train_complete_self_training.py

# 3. Evaluate (1-2 min)
python evaluate.py --checkpoint checkpoints/selftraining_complete_final.pt
```

## 📊 What You'll Get

### Files Created:
- `checkpoints/fulldataset_final.pt` - Full dataset model
- `checkpoints/selftraining_complete_final.pt` - Self-trained model
- `results/evaluation_results.txt` - AUC metrics
- `results/roc_curve.png` - ROC curve plot

### Results:
- **Measured AUC**: Check `results/evaluation_results.txt`
- **Expected**: 75-82% (vs paper's 82-84%)

## 🔄 Current vs Complete

| Aspect | Current (Proof) | Complete (New) |
|--------|----------------|----------------|
| Videos | 400 | ~950 |
| Epochs | 50 | 75 |
| Self-Training | No | Yes (3 iter) |
| AUC | 65.47% | 75-80% expected |

## ✅ Next Steps

1. Run the pipeline
2. Check `results/evaluation_results.txt` for AUC
3. Update presentation Slide 7-9 with measured results
4. Compare with paper (82-84%)

## 🐛 If Something Fails

**Out of memory?**
```python
# Edit the training file
BATCH_SIZE = 8  # Reduce from 16
```

**Taking too long?**
```python
# Edit the training file
EPOCHS = 50  # Reduce from 75
```

**No normal videos?**
- Scripts handle this automatically
- Will use all available videos

## 🎉 Success = AUC > 75%

Anything above 75% with the complete pipeline is a successful reproduction!

---

**Ready? Run this now:**
```bash
python run_best_performance.py
```
