# Fast Training Guide

## 🚀 Three Ways to Train (From Fastest to Full)

### Option 1: Ultra-Fast Test (2 minutes) ⚡

```bash
cd c:\code\projects\shopliftdetection\MSL
python quick_train.py
```

**What it does:**
- Uses only 10 videos
- Tiny model (128-dim, 1 layer)
- 2 epochs
- Batch size: 2
- **Time: ~2 minutes**

### Option 2: Fast Config (10-15 minutes) 🏃

```bash
python scripts/train.py --config configs/ucf_crime_fast.yaml --epochs 5
```

**What it does:**
- Uses all videos but shorter sequences
- Smaller model (256-dim, 2 layers)
- 5 epochs
- Batch size: 2
- Max length: 64 (was 320)
- **Time: ~10-15 minutes**

### Option 3: Full Training (2-4 hours) 🐢

```bash
python scripts/train.py --config configs/ucf_crime_h5.yaml --epochs 50
```

**What it does:**
- Full dataset
- Full model (512-dim, 3 layers)
- 50 epochs
- Batch size: 4
- Max length: 100
- **Time: ~2-4 hours**

---

## 🔧 What Was Fixed

### Issue 1: Missing 'logging' in config
**Fixed**: train.py now handles missing logging section

### Issue 2: Slow training
**Fixed**: 
- Created fast config with smaller batches (2 instead of 16)
- Reduced sequence length (64 instead of 320)
- Smaller model for fast config
- num_workers=0 for Windows

### Issue 3: --epochs not working
**Fixed**: Added --epochs argument to train.py

---

## 📊 Config Comparison

| Config | Batch | Max Length | Model Size | Time |
|--------|-------|------------|------------|------|
| quick_train.py | 2 | 50 | Tiny | 2 min |
| ucf_crime_fast.yaml | 2 | 64 | Small | 15 min |
| ucf_crime_h5.yaml | 4 | 100 | Medium | 2-4 hrs |

---

## ✅ Recommended Start

**Start with Option 1 to verify everything works:**

```bash
cd c:\code\projects\shopliftdetection\MSL
python quick_train.py
```

Expected output:
```
================================
QUICK TRAIN - Fast Testing Mode
================================

Device: cuda
Epochs: 2
Batch Size: 2

Loading dataset...
Dataset size: 800+
Using subset: 10 samples
Batches: 5

Creating model...
Model parameters: 500,000

================
TRAINING
================

Epoch [1/2] Batch [2/5] Loss: 0.4532
Epoch [1/2] Batch [4/5] Loss: 0.3891
Epoch [1/2] Average Loss: 0.4123

Epoch [2/2] Batch [2/5] Loss: 0.3456
Epoch [2/2] Batch [4/5] Loss: 0.3012
Epoch [2/2] Average Loss: 0.3201

================
TRAINING COMPLETE!
================

Checkpoint saved: checkpoints/quick_model.pt

✅ Quick training successful!
```

---

## 🎯 Next Steps

1. **Verify quick training works** (2 min)
   ```bash
   python quick_train.py
   ```

2. **Try fast config** (15 min)
   ```bash
   python scripts/train.py --config configs/ucf_crime_fast.yaml --epochs 5
   ```

3. **Full training when ready** (2-4 hrs)
   ```bash
   python scripts/train.py --config configs/ucf_crime_h5.yaml --epochs 50
   ```

---

## 🔧 Troubleshooting

**"CUDA out of memory"**
- Use CPU: Set `device: "cpu"` in config
- Or reduce batch_size to 1

**"Still too slow"**
- Use `quick_train.py` (only 10 videos)
- Or reduce epochs: `--epochs 2`

**"Import errors"**
- Install dependencies: `pip install h5py torch`

---

**Status**: ✅ Ready to train fast!  
**Recommended**: Start with `python quick_train.py`

