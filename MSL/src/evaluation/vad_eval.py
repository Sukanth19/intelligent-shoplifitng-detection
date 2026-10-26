"""
Standard frame-level evaluation for weakly supervised video anomaly detection.

Snippet scores are repeated 16x to frame level and compared with the frame
GT of all test videos concatenated (ROC-AUC and AP), as in the MSL paper.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..datasets.vad_features import FRAMES_PER_SNIPPET, VideoFeatureDataset


@torch.no_grad()
def predict_features(model, features: torch.Tensor, device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    Score one video. features: (T, D).
    Returns snippet scores, the video-level probability and the corrected
    snippet scores (snippet * video probability, the MSL score correction).
    """
    model.eval()
    out = model(features.unsqueeze(0).to(device))
    snippet = out['snippet_scores'][0].float().cpu().numpy()
    video = float(out['video_scores'][0])
    return {
        'snippet': snippet,
        'video': video,
        'corrected': snippet * video,
    }


def frame_level(scores: np.ndarray) -> np.ndarray:
    return np.repeat(scores, FRAMES_PER_SNIPPET)


def _auc_ap(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        'auc': float(roc_auc_score(gt, pred)),
        'ap': float(average_precision_score(gt, pred)),
    }


@torch.no_grad()
def evaluate(model, dataset: VideoFeatureDataset, device: str = 'cpu',
             categories: Optional[List[str]] = None) -> Dict:
    """
    Frame-level AUC/AP over the whole test set, plus per-category AUC
    (that category's videos + all normal videos) for each name in `categories`.
    """
    assert dataset.frame_gt is not None, 'evaluation needs the test split'
    preds = [predict_features(model, dataset[i]['features'], device) for i in range(len(dataset))]

    results = {}
    for key in ('snippet', 'corrected'):
        frame_pred = [frame_level(p[key]) for p in preds]
        results[key] = _auc_ap(np.concatenate(dataset.frame_gt), np.concatenate(frame_pred))
        for cat in categories or []:
            idx = [i for i, c in enumerate(dataset.categories) if c in (cat, 'Normal')]
            if any(dataset.categories[i] == cat for i in idx):
                results[key][f'auc_{cat}'] = float(roc_auc_score(
                    np.concatenate([dataset.frame_gt[i] for i in idx]),
                    np.concatenate([frame_pred[i] for i in idx])))

    video_scores = np.array([p['video'] for p in preds])
    results['video_auc'] = float(roc_auc_score(dataset.labels, video_scores))
    results['nan_videos'] = int(sum(np.isnan(p['snippet']).any() for p in preds))
    return results
