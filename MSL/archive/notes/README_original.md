# Intelligent Shoplifting Detection Using Deep Learning and Computer Vision

**Implementation of Multi-Sequence Learning (MSL) for Video Anomaly Detection**

This project implements the MSL method for weakly supervised video anomaly detection, with a focus on shoplifting detection in retail environments.

## 📄 Research Basis

**Paper**: "Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection"  
**Authors**: Shuo Li, Fang Liu, Licheng Jiao  
**Conference**: AAAI 2022, Vol. 36, No. 2, pp. 1395–1403  
**DOI**: [10.1609/aaai.v36i2.20028](https://ojs.aaai.org/index.php/AAAI/article/view/20028)  
**Official Repository**: https://github.com/LiShuo1001/MSL

> **Note**: The official MSL repository provides VideoSwin features and extraction configs but not the complete MSL training implementation. This project implements the MSL methodology based on the AAAI 2022 paper.

## 🎯 Project Overview

This system detects anomalous behavior (particularly shoplifting) in surveillance videos using a multi-stage pipeline:

```
Video → VideoSwin Features → Multi-Sequence Learning → Transformer
                                         ↓
                                 Self-Training
                                         ↓
                            Video-level Probability
                                         ↓
                           Snippet-level Anomaly Scores
                                         ↓
                              Score Correction
                                         ↓
                          Final Anomaly Detection
```

## 🖥️ System Requirements

**Verified Configuration:**
- Python 3.13.2
- PyTorch 2.6.0+cu126
- CUDA 12.6
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (or similar)
- OS: Windows

**Minimum Requirements:**
- Python 3.10+
- PyTorch 2.0+
- 8GB+ GPU VRAM (recommended)
- CPU-only mode supported but slower

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MSL
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other requirements
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python scripts\check_environment.py
```

## 📁 Data Setup

### Required Data

1. **UCF-Crime Dataset**
   - Download from: http://www.crcv.ucf.edu/projects/real-world/
   - Contains 1900 videos (13 anomaly categories including Shoplifting)

2. **MSL-VideoSwin Features** (Recommended for initial training)
   - Download from: [Google Drive](https://drive.google.com/drive/folders/12G8HSj7sFK60QHHnGFriFqL4Ld94YgUy?usp=sharing)
   - Pre-extracted VideoSwin features for UCF-Crime

3. **VideoSwin Checkpoint** (For raw video processing)
   - Model: `swin_base_patch244_window877_kinetics400_1k.pth`
   - Download from: [GitHub Release](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth)
   - Place in: `data/videoswin/`

### Data Directory Structure
```
MSL/
├── data/
│   ├── UCF-Crime/          # Raw UCF-Crime videos
│   ├── MSL-Features/       # Pre-extracted VideoSwin features
│   │   ├── train/
│   │   ├── test/
│   │   └── labels/
│   └── videoswin/
│       └── swin_base_patch244_window877_kinetics400_1k.pth
```

**⚠️ ACTION REQUIRED**: After downloading, update the data paths in:
- `scripts/check_environment.py` (to verify)
- `configs/default.yaml` (for training)

## 🚀 Quick Start

### Step 1: Inspect Data
```bash
python scripts/inspect_features.py
```
This will document the feature format in `docs/DATA_FORMAT.md`

### Step 2: Train MSL Model
```bash
python scripts/train.py --config configs/default.yaml
```

### Step 3: Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Step 4: Inference on New Video
```bash
python scripts/inference.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pt
```

## 📊 Project Structure

```
MSL/
├── configs/              # Configuration files
├── data/                 # Datasets (see Data Setup)
├── docs/                 # Documentation
│   ├── ENVIRONMENT_STATUS.md
│   ├── IMPLEMENTATION_NOTES.md
│   ├── PROGRESS.md
│   └── DATA_FORMAT.md (generated)
├── src/
│   ├── datasets/        # Data loaders
│   ├── models/          # MSL model architecture
│   ├── losses/          # MSL ranking loss
│   ├── training/        # Training loops and self-training
│   ├── inference/       # Inference and score correction
│   ├── evaluation/      # Metrics and evaluation
│   └── utils/           # Utilities
├── scripts/
│   ├── check_environment.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── checkpoints/         # Model checkpoints
├── results/             # Evaluation results
│   ├── plots/
│   ├── metrics/
│   └── predictions/
├── logs/                # Training logs
├── tests/               # Unit tests
├── requirements.txt
└── README.md
```

## 🔬 Development Phases

- [x] **Phase 1**: Environment setup and documentation
- [ ] **Phase 2**: Feature inspection and data format documentation
- [ ] **Phase 3**: Feature dataset loader
- [ ] **Phase 4**: MSL model architecture
- [ ] **Phase 5**: MSL ranking loss
- [ ] **Phase 6**: Baseline training
- [ ] **Phase 7**: Self-training mechanism
- [ ] **Phase 8**: Score correction
- [ ] **Phase 9**: Evaluation pipeline
- [ ] **Phase 10**: Shoplifting-specific experiments
- [ ] **Phase 11**: VideoSwin raw video extraction
- [ ] **Phase 12**: End-to-end inference
- [ ] **Phase 13**: Visualization and UI
- [ ] **Phase 14**: Optional extensions (YOLO/tracking)

Track progress in `docs/PROGRESS.md`

## 📝 Implementation Notes

### What's Implemented
- Project structure and environment setup
- Documentation framework
- Dependency management

### What's Based on the Paper
The following components will be implemented according to the AAAI 2022 MSL paper:
- Multi-Sequence Learning mechanism
- Transformer-based architecture
- MSL ranking loss
- Self-training procedure
- Score correction

### Project Extensions (Not in Base Paper)
- Shoplifting-specific classification
- YOLO/DeepSORT integration (optional)
- Visualization interfaces

All assumptions and deviations from the paper are documented in `docs/IMPLEMENTATION_NOTES.md`

## 🎓 Citation

If you use this implementation, please cite the original MSL paper:

```bibtex
@inproceedings{li2022msl,
  title={Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection},
  author={Li, Shuo and Liu, Fang and Jiao, Licheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={2},
  pages={1395--1403},
  year={2022}
}
```

## 📧 Support

For issues or questions:
1. Check `docs/` for detailed documentation
2. Review `docs/PROGRESS.md` for current status
3. Open an issue in the repository

## ⚠️ Important Disclaimers

1. **Research Implementation**: This is a research prototype, not production-ready software
2. **Feature Extraction**: Initial development uses pre-extracted VideoSwin features
3. **Dataset**: Performance metrics are based on UCF-Crime; real-world CCTV performance may vary
4. **Official Code**: The official MSL training code was not available at implementation time

## 📄 License

See LICENSE file for details.

---

**Status**: Phase 1 Complete - Ready for Data Inspection  
**Last Updated**: Initial Setup  
**Next Step**: Provide UCF-Crime and MSL-Features paths, then run `scripts/inspect_features.py`

