"""
Real VideoSwin feature extraction for raw videos.

Uses torchvision's Video Swin-B (Kinetics-400, ImageNet-1K pretrained), the
same backbone family the MSL authors used for the provided H5 features:
  * one snippet = 16 consecutive frames
  * frames resized to height 224, width scaled and rounded to a multiple of 32
    (UCF-Crime's 320x240 -> 224x288, matching the authors' '224288' features)
  * ImageNet mean/std normalisation
  * output = globally pooled last-stage features, 1024-d per snippet

Checked against the provided H5 features: this checkpoint reproduces their scale
(std 0.39 vs 0.37, L2 12.5 vs 11.8) and per-dimension spread (corr 0.48), while
the ImageNet-22K checkpoint or pre-norm features do not (std 0.74-6.2, corr ~0).
"""

from typing import Callable, Optional

import cv2
import numpy as np
import torch

FRAMES_PER_SNIPPET = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def target_size(width: int, height: int, short: int = 224, max_long: int = 416):
    if height <= width:
        new_w = int(round(width * short / height / 32)) * 32
        return min(max(new_w, short), max_long), short
    new_h = int(round(height * short / width / 32)) * 32
    return short, min(max(new_h, short), max_long)


class Swin3DFeatureExtractor:
    def __init__(self, device: Optional[str] = None, batch_size: int = 4):
        from torchvision.models.video import Swin3D_B_Weights, swin3d_b

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        model = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1)
        model.head = torch.nn.Identity()
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def _embed(self, clips: np.ndarray) -> np.ndarray:
        """clips: (N, 16, H, W, 3) uint8 RGB -> (N, 1024)"""
        x = torch.from_numpy(clips).permute(0, 4, 1, 2, 3).float().div_(255)  # (N, 3, 16, H, W)
        x = ((x - MEAN) / STD).to(self.device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.startswith('cuda')):
            feats = self.model(x)
        return feats.float().cpu().numpy()

    def extract(self, video_path: str, progress: Optional[Callable[[int, int], None]] = None) -> dict:
        """
        Returns {'features': (T, 1024) float32, 'fps': float, 'num_frames': int}.
        A trailing partial snippet is padded by repeating its last frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w, h = target_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        features, clip, batch, n_frames = [], [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1
            clip.append(cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            if len(clip) == FRAMES_PER_SNIPPET:
                batch.append(np.stack(clip))
                clip = []
            if len(batch) == self.batch_size:
                features.append(self._embed(np.stack(batch)))
                batch = []
                if progress:
                    progress(n_frames, total)
        cap.release()

        if clip:
            clip += [clip[-1]] * (FRAMES_PER_SNIPPET - len(clip))
            batch.append(np.stack(clip))
        if batch:
            features.append(self._embed(np.stack(batch)))
        if progress:
            progress(n_frames, max(total, n_frames))
        if not features:
            raise ValueError('Video has no readable frames')

        return {'features': np.concatenate(features).astype(np.float32), 'fps': float(fps), 'num_frames': n_frames}
