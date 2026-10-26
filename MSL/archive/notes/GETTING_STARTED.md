# Getting Started with MSL Shoplifting Detection

## 🎯 Quick Start Guide

### Current Status: Phase 1 Complete ✅

The project structure, documentation, and utilities are ready. You need to provide data paths to continue.

---

## 📍 You Are Here

```
[✅ Phase 1: Setup] → [⏸️ Phase 2: Data Inspection] → Phase 3+ (MSL Implementation)
```

---

## 🚦 What To Do Now

### Option 1: Quick Setup (Recommended)

```bash
# Run the interactive setup script
python setup.py
```

This will:
- Verify your environment
- Optionally install dependencies
- Check for data
- Run feature inspection if data is found

### Option 2: Manual Setup

#### Step 1: Provide Data Locations

You mentioned having downloaded:
- ✓ UCF-Crime dataset
- ✓ MSL-VideoSwin-Feature data

**Please provide their locations by either:**

**A) Creating symbolic links (Windows Admin CMD):**
```cmd
mklink /D "C:\code\projects\shopliftdetection\MSL\data\UCF-Crime" "YOUR_ACTUAL_UCF_CRIME_PATH"
mklink /D "C:\code\projects\shopliftdetection\MSL\data\MSL-Features" "YOUR_ACTUAL_FEATURES_PATH"
```

**B) Moving/copying data:**
```cmd
# Move or copy your data to:
C:\code\projects\shopliftdetection\MSL\data\UCF-Crime\
C:\code\projects\shopliftdetection\MSL\data\MSL-Features\
```

**C) Updating the script:**
- Edit `scripts\inspect_features.py`
- Change `FEATURE_PATH` to your actual path

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install scikit-learn einops timm opencv-python decord
```

#### Step 3: Inspect Features

```bash
python scripts\inspect_features.py
```

This generates `docs\DATA_FORMAT.md` with feature specifications.

#### Step 4: Verify Environment

```bash
python scripts\check_environment.py
```

Should show all ✓ marks.

---

## 📚 Important Documents

| Document | Purpose |
|----------|---------|
| `README.md` | Complete project overview and usage |
| `PHASE1_COMPLETE.md` | Current status and what's needed next |
| `docs/PROGRESS.md` | Phase-by-phase development tracking |
| `docs/IMPLEMENTATION_NOTES.md` | Research fidelity and assumptions |
| `docs/ENVIRONMENT_STATUS.md` | System configuration details |

---

## 🗂️ Project Structure

```
MSL/
├── 📄 README.md                    # Main documentation
├── 📄 GETTING_STARTED.md           # This file
├── 📄 PHASE1_COMPLETE.md           # Status report
├── 📄 setup.py                     # Interactive setup
├── 📄 requirements.txt             # Dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 📁 docs/                        # Documentation
│   ├── ENVIRONMENT_STATUS.md       # System info
│   ├── IMPLEMENTATION_NOTES.md     # Research fidelity
│   ├── PROGRESS.md                 # Phase tracking
│   └── DATA_FORMAT.md              # (generated after inspection)
│
├── 📁 scripts/                     # Utility scripts
│   ├── check_environment.py        # ✅ Environment check
│   ├── inspect_features.py         # ⏸️ Feature inspection (needs data)
│   ├── train.py                    # 🔜 Training script (Phase 6)
│   ├── evaluate.py                 # 🔜 Evaluation script (Phase 9)
│   └── inference.py                # 🔜 Inference script (Phase 12)
│
├── 📁 src/                         # Source code
│   ├── datasets/                   # 🔜 Data loaders (Phase 3)
│   ├── models/                     # 🔜 MSL architecture (Phase 4)
│   ├── losses/                     # 🔜 MSL ranking loss (Phase 5)
│   ├── training/                   # 🔜 Training logic (Phase 6-7)
│   ├── inference/                  # 🔜 Inference & correction (Phase 8)
│   ├── evaluation/                 # 🔜 Metrics (Phase 9)
│   └── utils/                      # 🔜 Utilities
│
├── 📁 data/                        # ⚠️ Data (YOU PROVIDE)
│   ├── UCF-Crime/                  # ❌ Not found - provide path
│   ├── MSL-Features/               # ❌ Not found - provide path
│   └── videoswin/                  # 🔜 VideoSwin checkpoint (Phase 11)
│
├── 📁 configs/                     # 🔜 Configuration files (Phase 3)
├── 📁 checkpoints/                 # 🔜 Saved models (Phase 6+)
├── 📁 results/                     # 🔜 Outputs (Phase 9+)
├── 📁 logs/                        # 🔜 Training logs (Phase 6+)
└── 📁 tests/                       # 🔜 Unit tests (ongoing)
```

Legend:
- ✅ Complete and working
- ⏸️ Ready but needs input
- 🔜 To be implemented
- ❌ Missing/blocked

---

## 🔍 What's Been Done (Phase 1)

✅ **Environment Verification**
- Python 3.13.2
- PyTorch 2.6.0+cu126
- CUDA 12.6
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU

✅ **Project Structure**
- Complete directory hierarchy
- Python package structure with `__init__.py` files
- Git configuration (`.gitignore`)

✅ **Documentation**
- Comprehensive README
- Implementation notes with research fidelity guidelines
- Progress tracking system
- Environment status report

✅ **Utilities**
- Environment check script (working)
- Feature inspection script (ready)
- Interactive setup script

✅ **Dependencies**
- `requirements.txt` with all packages
- PyTorch, NumPy, Pandas, etc. already installed
- Missing: scikit-learn, einops, timm (easy to install)

---

## ⚠️ What's Blocking Progress

**Need from you:**
1. Path to UCF-Crime dataset
2. Path to MSL-VideoSwin-Feature data

**Example:**
```
UCF-Crime: D:\datasets\UCF-Crime
MSL-Features: D:\datasets\MSL-VideoSwin-Feature
```

**Once provided → automatic progression to:**
- Phase 2: Feature inspection
- Phase 3: Dataset loader
- Phase 4: MSL model
- ... (see PROGRESS.md for full roadmap)

---

## 🎓 Research Implementation Notes

### What We're Building
**Based on AAAI 2022 Paper**: "Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection"

### Key Components (To Be Implemented)
1. **Multi-Sequence Learning**: Process multiple temporal sequences
2. **Transformer Architecture**: Encode temporal dependencies
3. **MSL Ranking Loss**: Hinge-based loss for anomaly/normal separation
4. **Self-Training**: Progressive sequence length refinement
5. **Score Correction**: Video-level probability → snippet scores

### What's NOT from the Paper
- YOLO/DeepSORT (optional project extension)
- Shoplifting-specific classification beyond anomaly detection
- Production UI

All documented in `docs/IMPLEMENTATION_NOTES.md`

---

## 🆘 Need Help?

### Common Issues

**Q: Can't find Python/pip?**
```bash
# Check installation
python --version
pip --version
```

**Q: CUDA not available?**
- Will fall back to CPU (slower but functional)
- Verify with: `python scripts\check_environment.py`

**Q: Data not found?**
- Update paths in `scripts\inspect_features.py`
- Or use symbolic links (see above)
- Or move data to `data/` directory

**Q: Import errors?**
```bash
pip install -r requirements.txt
```

### Support Resources
1. Check `docs/` for detailed information
2. Review `docs/PROGRESS.md` for current phase
3. See `docs/IMPLEMENTATION_NOTES.md` for methodology

---

## ✉️ Next Communication

Please provide:
```
UCF-Crime path: <your path here>
MSL-Features path: <your path here>
```

Then I will:
1. Update paths in scripts
2. Run feature inspection
3. Generate DATA_FORMAT.md
4. Proceed to Phase 3: Dataset Loader

---

## 🎯 End Goal

```
Video (MP4) → VideoSwin → Features → MSL → Anomaly Score → "Shoplifting Detected"
                                     ↓
                              Temporal Analysis
                                     ↓
                            Suspicious Intervals
```

**We're ready to build this. Just need the data paths!**

---

**Status**: ✅ Phase 1 Complete | ⏸️ Awaiting Data Paths | 🚀 Ready to Continue
