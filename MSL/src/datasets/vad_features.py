"""
VideoSwin feature datasets for UCF-Crime and ShanghaiTech (MSL authors' H5 files).

Each video is a (T, 1024) array of snippet features, one snippet = 16 frames.
The *_gt_16.npy files hold frame-level labels for the test videos, concatenated
in .list order (len = sum(T_i) * 16).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FRAMES_PER_SNIPPET = 16

DATA_ROOT = Path(__file__).resolve().parents[3] / 'data'

DATASETS = {
    'ucf': {
        'dir': 'UCF-VideoSwin-Features',
        'train_h5': 'ucf_swin_train_feature_224288224288.h5',
        'test_h5': 'ucf_swin_test_feature_224288224288.h5',
        'train_list': 'ucf-train-filter.list',
        'test_list': 'ucf-test.list',
        'gt': 'ucf_gt_16.npy',
    },
    'sht': {
        'dir': 'SHT-VideoSwin-Features',
        'train_h5': 'sh_swin_train_feature_256320224.h5',
        'test_h5': 'sh_swin_test_feature_256320224.h5',
        'train_list': 'sh-train.list',
        'test_list': 'sh-test.list',
        'gt': 'sh_gt_16.npy',
    },
}


def video_key(entry: str) -> str:
    """'Abuse/Abuse028_x264.mp4' -> 'Abuse028_x264', '01_0015.npy' -> '01_0015'"""
    return Path(entry).stem


def video_category(dataset: str, entry: str) -> str:
    if dataset == 'ucf':
        if 'Normal' in entry:
            return 'Normal'
        return entry.split('/')[0]
    return 'Anomaly' if re.search(r'_\d{4}$', video_key(entry)) else 'Normal'


class VideoFeatureDataset(Dataset):
    """
    Loads every video of one split into memory (float16 to keep RAM low).

    Video labels:
      UCF  - 'Normal' in the path means normal, anything else is an anomaly class.
      SHT  - train: the standard weakly-supervised split names the 63 anomalous
             videos XX_XXXX (4 digits); test: anomalous if any GT frame is 1.
    """

    def __init__(self, dataset: str = 'ucf', split: str = 'train', data_root: Optional[Path] = None):
        cfg = DATASETS[dataset]
        root = Path(data_root or DATA_ROOT) / cfg['dir']
        self.dataset = dataset
        self.split = split

        entries = [l.strip() for l in open(root / cfg[f'{split}_list']) if l.strip()]
        self.names: List[str] = []
        self.categories: List[str] = []
        self.features: List[np.ndarray] = []
        with h5py.File(root / cfg[f'{split}_h5'], 'r') as f:
            for entry in entries:
                key = video_key(entry)
                if key not in f:
                    raise KeyError(f'{key} from {cfg[f"{split}_list"]} not found in H5 file')
                self.names.append(key)
                self.categories.append(video_category(dataset, entry))
                self.features.append(f[key][:].astype(np.float16))

        self.lengths = np.array([len(x) for x in self.features])

        # Frame-level ground truth, split per video (test only)
        self.frame_gt: Optional[List[np.ndarray]] = None
        if split == 'test':
            gt = np.load(root / cfg['gt'], allow_pickle=True).astype(np.uint8)
            expected = int(self.lengths.sum()) * FRAMES_PER_SNIPPET
            if len(gt) != expected:
                raise ValueError(f'GT length {len(gt)} != sum(T)*16 = {expected}')
            bounds = np.cumsum(self.lengths * FRAMES_PER_SNIPPET)[:-1]
            self.frame_gt = np.split(gt, bounds)

        if dataset == 'sht' and split == 'test':
            self.labels = np.array([int(g.any()) for g in self.frame_gt])
            self.categories = ['Anomaly' if y else 'Normal' for y in self.labels]
        else:
            self.labels = np.array([int(c != 'Normal') for c in self.categories])

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict:
        return {
            'features': torch.from_numpy(self.features[idx].astype(np.float32)),
            'label': int(self.labels[idx]),
            'name': self.names[idx],
            'category': self.categories[idx],
        }

    def index_of(self, name: str) -> int:
        return self.names.index(name)


def resample_snippets(features: torch.Tensor, max_len: int) -> torch.Tensor:
    """Average-pool a long video down to max_len snippets (keeps the whole video)."""
    if features.shape[0] <= max_len:
        return features
    return F.adaptive_avg_pool1d(features.T.unsqueeze(0), max_len)[0].T


def pad_collate(samples: List[Dict], max_len: Optional[int] = None) -> Dict:
    """Pad a list of samples to (B, T, D) with a validity mask (True = real snippet)."""
    feats = [s['features'] if max_len is None else resample_snippets(s['features'], max_len) for s in samples]
    T = max(len(x) for x in feats)
    D = feats[0].shape[1]
    out = torch.zeros(len(feats), T, D)
    mask = torch.zeros(len(feats), T, dtype=torch.bool)
    for i, x in enumerate(feats):
        out[i, :len(x)] = x
        mask[i, :len(x)] = True
    return {
        'features': out,
        'mask': mask,
        'label': torch.tensor([s['label'] for s in samples]),
        'name': [s['name'] for s in samples],
    }
