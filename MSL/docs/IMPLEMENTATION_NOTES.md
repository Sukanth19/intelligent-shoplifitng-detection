# Implementation Notes

## Research Fidelity

This project implements the MSL (Multi-Sequence Learning) method from:
- **Paper**: "Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection"
- **Authors**: Shuo Li, Fang Liu, Licheng Jiao
- **Conference**: AAAI 2022, Vol. 36, No. 2, pp. 1395–1403
- **DOI**: 10.1609/aaai.v36i2.20028
- **Official Paper**: https://ojs.aaai.org/index.php/AAAI/article/view/20028
- **Official Repository**: https://github.com/LiShuo1001/MSL

## Important Clarifications

### What the Official Repository Provides
- VideoSwin pre-extracted features for UCF-Crime and ShanghaiTech
- Labels for the datasets
- VideoSwin feature extraction configuration
- **NOTE**: The actual MSL training implementation is NOT currently available in the repository

### What We Must Implement
- MSL model architecture (Transformer-based)
- Multi-Sequence Learning mechanism
- MSL ranking loss
- Self-training procedure
- Score correction mechanism
- Training pipeline
- Evaluation pipeline
- Inference pipeline

## Implementation Assumptions

*This section will document any assumptions made where the paper or official materials do not provide explicit implementation details. Each assumption will be clearly labeled.*

### Assumptions Made So Far
None yet - will be added as implementation progresses

## Components NOT in the Base Paper

The following are **project extensions** and are NOT components of the AAAI 2022 MSL method:
- YOLO object detection (if added)
- DeepSORT tracking (if added)
- Shoplifting-specific classification beyond anomaly detection
- Any UI/visualization beyond basic plotting

## Data Sources

### VideoSwin
- **Repository**: https://github.com/SwinTransformer/Video-Swin-Transformer
- **Checkpoint**: swin_base_patch244_window877_kinetics400_1k.pth
- **Pretrained on**: Kinetics-400

### UCF-Crime Configuration
```
Model:
- backbone: patch_size=(2,4,4), drop_path_rate=0.3
- test_cfg: max_testing_views=4

Preprocessing:
- Resize: scale=(-1, 224)
- CenterCrop: crop_size=(288, 224)
- Normalize: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
- FormatShape: NCTHW
```

## Development Strategy

### Phase-by-Phase Approach
1. ✅ Initial documentation
2. Feature-based implementation (using pre-extracted features)
3. MSL model implementation
4. Training and evaluation
5. Raw video pipeline (VideoSwin extraction)
6. Inference and visualization
7. Optional extensions

This ensures we validate the MSL methodology before dealing with video processing complexity.
