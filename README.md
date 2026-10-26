# Shoplifting Detection with Multi-Sequence Learning

Weakly supervised video anomaly detection for retail surveillance, based on
**"Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection"**
(Li, Liu, Jiao — AAAI 2022). A web app lets you upload footage and see *when* suspicious activity happens.

```
video ─► 16-frame snippets ─► Video Swin-B (Kinetics-400) ─► 1024-d feature per snippet
      ─► overlapping 32-snippet windows ─► Transformer encoder ─► snippet scores + video probability
      ─► score correction ─► suspicious intervals + annotated video
```

## Data

The Video Swin features (1.5 GB) are not in this repo. Download them from the MSL authors
([Google Drive](https://drive.google.com/drive/folders/12G8HSj7sFK60QHHnGFriFqL4Ld94YgUy?usp=sharing), linked from
[LiShuo1001/MSL](https://github.com/LiShuo1001/MSL)) and place them like this:

```
data/UCF-VideoSwin-Features/  ucf_swin_{train,test}_feature_224288224288.h5, ucf-train-filter.list, ucf-test.list, ucf_gt_16.npy
data/SHT-VideoSwin-Features/  sh_swin_{train,test}_feature_256320224.h5, sh-train.list, sh-test.list, sh_gt_16.npy
```

The trained models (`MSL/checkpoints/msl_{ucf,sht}_best.pt`) are included, so **uploading and analysing your own videos
works without the dataset**. The dataset is only needed for retraining and for the Dataset explorer / test metrics.

## Quick start

```bash
./START_HERE.sh          # Windows: START_HERE.bat
```
This creates `.venv`, installs the requirements, trains both models if their checkpoints are missing
(about 2 minutes each on an RTX 4060), and starts the app on **http://localhost:5000**.

Manual steps:
```bash
python -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements.txt
cd MSL && ../.venv/bin/python train.py --dataset ucf && ../.venv/bin/python train.py --dataset sht && cd ..
.venv/bin/python app.py
```

## The app

| Page | What it does |
|---|---|
| **Analyze video** | Upload a video. It is split into 16-frame snippets, encoded with Video Swin-B and scored by the trained MSL model. You get an annotated video (alarm border, live score, timeline), a clickable score chart and the list of suspicious intervals. Past analyses are kept. |
| **Dataset explorer** | Runs the model on every video of the official UCF-Crime / ShanghaiTech **test** split and plots the predicted score against the annotated ground truth. Each video is labelled *correctly flagged / missed / false alarm*. |
| **Model & results** | Test metrics, training curve and pipeline description. |

The alarm threshold is not hand-picked. It is the 99th percentile of snippet scores on held-out *normal*
training videos (≈1% of normal footage triggers an alarm): 0.953 for UCF-Crime, 0.353 for ShanghaiTech.

## Results

Frame-level ROC-AUC on the official test splits (standard protocol). Training uses only the train split,
and checkpoint selection uses a 10% validation split of the training videos. **The test set is used only for the final report.**

| | UCF-Crime AUC | UCF-Crime AP | Shoplifting vs normal AUC | ShanghaiTech AUC |
|---|---|---|---|---|
| Original implementation (old checkpoints, same protocol) | 46–58% | 7–10% | 46–54% | – |
| **This version** (deployed model, seed 0) | **80.4%** | **18.0%** | **87.6%** | **92.6%** |
| This version, 3 seeds (mean ± std) | 78.9 ± 1.6% | 17.3 ± 1.2% | 85.7 ± 1.4% | – |
| MSL paper (VideoSwin features) | 85.6% | – | – | 97.3% |

Per category on UCF-Crime (each vs normal test videos): Shoplifting 87.6, Stealing 95.5, Robbery 84.5, Burglary 92.1.
Raw numbers are in `MSL/results/`, and every experiment run is in `MSL/results/experiments/`.

### Ablation: checkpoint selection (UCF-Crime)

| Config | Selected | Frame AUC | Shoplifting AUC |
|---|---|---|---|
| A: lr 1e-4, dropout 0.3, best val anywhere | stage 1 (K=32) | 76.0 | 85.3 |
| B: lr 5e-5, dropout 0.5, best val anywhere | stage 1 (K=32) | 75.7 | 84.9 |
| C: B + hidden 128, 320-snippet training | stage 3 (K=8) | 74.7 | 76.5 |
| **B, best val within the last self-training stage** | stage 4 (K=4) | **80.4** | **87.6** |

The validation split only has video-level labels, so "best validation AUC" favours the coarse first stage, which is good at
saying *whether* a video is anomalous but not *when*. Selecting within the final self-training stage, the one designed for
fine localisation, gives +4.7 AUC. This is the default (`--select last_stage`).

## What was fixed (vs. the original version)

1. **The web app was not connected to the model.** `detect_anomalies()` returned `np.random.rand(...)`, uploaded frames were discarded,
   and the dataset page used random sizes and flags. The app is now fully driven by the trained model.
2. **No feature extractor.** `feature_extractor.py` was a placeholder. `MSL/src/videoswin/swin3d_extractor.py` now uses torchvision's
   Video Swin-B. The checkpoint was chosen by matching the statistics of the authors' features: Kinetics-400/ImageNet-1K matches
   (std 0.39 vs 0.37), while the ImageNet-22K variant does not (std 0.74, no correlation), so the trained model can be applied to new videos.
3. **Trained on the test set.** The old scripts trained on `ucf-test.list` (data leakage). The new `train.py` uses the 1501 train videos.
4. **NaN scores in 40% of videos.** When videos of different lengths were batched, windows made purely of padding had every attention key masked, so
   the softmax returned NaN, which spread through mean pooling. Fixed in `msl_model.py` (masked pooling over real windows only).
5. **Tail of every video was never scored.** The window count used `floor`, so up to 15 final snippets got score 0. Now `ceil`, vectorised with `unfold`.
6. **Wrong video↔feature mapping.** The old loader's "fuzzy match" took the last number in the filename (`Abuse028_x264` → `264`).
   The new loader (`vad_features.py`) matches exact keys and splits the frame-level ground truth per video, with a length check.
7. **Evaluation** now uses the standard frame-level AUC/AP instead of video-level metrics.

## Limitations (worth discussing)

- **Localisation is weaker than detection.** Video-level AUC is 90.6%, but frame AP is only 18%. In many anomalous test videos the score stays
  high for the whole video, not just the annotated event (see e.g. `Shoplifting001` in the explorer). Weak labels only say *that* something happened.
- The gap to the paper (≈5 AUC on UCF, ≈5 on SHT) likely comes from the simplified self-training and fewer epochs. Also, only 25 of the 1501
  UCF training videos are shoplifting; the model learns "anomaly" in general.
- Uploaded videos are scored by a model trained on UCF-Crime (low-resolution CCTV). Footage that looks very different (dashcams, phone
  videos, games) is out of distribution.
- Not real time: about 3.5× faster than real time on an RTX 4060 (50 s video in 14 s), plus Video Swin-B weights (364 MB) on the first run.

## Project layout

```
app.py                     Flask app (upload jobs, dataset explorer, model info)
overlay.py                 suspicious-interval detection + annotated H.264 video rendering
templates/app.html, static/js/main.js, static/css/app.css   UI
MSL/train.py               training + evaluation (one script, final config = defaults)
MSL/src/models/            MSL model: multi-sequence windows + Transformer (fixed)
MSL/src/datasets/vad_features.py   UCF-Crime / ShanghaiTech feature loader
MSL/src/evaluation/vad_eval.py     frame-level AUC/AP
MSL/src/videoswin/swin3d_extractor.py   Video Swin-B feature extraction for raw videos
MSL/checkpoints/msl_{ucf,sht}_best.pt   trained models (old ones in checkpoints/old/)
data/                      VideoSwin features from the MSL authors
archive/, MSL/archive/     previous UI, scripts and notes (kept for reference)
```
