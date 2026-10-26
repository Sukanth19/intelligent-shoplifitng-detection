"""
UCF-Crime Feature Dataset Loader
Loads pre-extracted VideoSwin features for MSL training
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class UCFCrimeFeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted VideoSwin features from UCF-Crime.
    
    Expected structure:
        feature_dir/
        ├── train/
        │   ├── Normal_000.npy
        │   ├── Shoplifting_000.npy
        │   └── ...
        ├── test/
        └── labels/
            ├── train_labels.json
            └── test_labels.json
    
    Features:
        - Shape: (T, D) where T=temporal snippets, D=feature_dim
        - Format: .npy files
        - Normalized: L2-normalized per snippet
    
    Labels:
        - Video-level only (weakly supervised)
        - 0: Normal, 1: Anomaly
    """
    
    def __init__(
        self,
        feature_dir: Union[str, Path],
        split: str = "train",
        label_file: Optional[Union[str, Path]] = None,
        feature_dim: int = 1024,
        max_len: Optional[int] = None,
        normalize: bool = False,  # Already normalized
        return_video_id: bool = True,
    ):
        """
        Args:
            feature_dir: Path to directory containing train/ and test/
            split: 'train' or 'test'
            label_file: Optional path to label JSON file (auto-detected if None)
            feature_dim: Expected feature dimension (default: 1024 for VideoSwin-Base)
            max_len: Maximum sequence length (None = no truncation)
            normalize: Whether to L2-normalize features (default: False, already normalized)
            return_video_id: Whether to return video ID along with features
        """
        super().__init__()
        
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.feature_dim = feature_dim
        self.max_len = max_len
        self.normalize = normalize
        self.return_video_id = return_video_id
        
        # Determine paths
        self.split_dir = self.feature_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        # Load labels
        if label_file is None:
            label_file = self.feature_dir / "labels" / f"{split}_labels.json"
        
        self.label_file = Path(label_file)
        if not self.label_file.exists():
            raise FileNotFoundError(f"Label file not found: {self.label_file}")
        
        with open(self.label_file, 'r') as f:
            self.labels_data = json.load(f)
        
        # Create mapping
        self.samples = []
        for item in self.labels_data:
            filename = item['filename']
            feature_path = self.split_dir / filename
            
            if not feature_path.exists():
                logger.warning(f"Feature file not found: {feature_path}")
                continue
            
            self.samples.append({
                'feature_path': feature_path,
                'label': item['label'],
                'category': item.get('category', 'Unknown'),
                'video_id': filename.replace('.npy', ''),
                'num_snippets': item.get('num_snippets', None)
            })
        
        logger.info(f"Loaded {len(self.samples)} samples from {split} split")
        logger.info(f"Feature dir: {self.feature_dir}")
        logger.info(f"Split dir: {self.split_dir}")
        
        # Statistics
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute dataset statistics"""
        labels = [s['label'] for s in self.samples]
        categories = [s['category'] for s in self.samples]
        
        self.num_normal = sum(1 for l in labels if l == 0)
        self.num_anomaly = sum(1 for l in labels if l == 1)
        
        from collections import Counter
        self.category_counts = Counter(categories)
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Normal: {self.num_normal}")
        logger.info(f"  Anomaly: {self.num_anomaly}")
        logger.info(f"  Categories: {dict(self.category_counts)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str]]:
        """
        Returns:
            features: Tensor of shape (T, D)
            label: int (0=normal, 1=anomaly)
            video_id: str (if return_video_id=True)
        """
        sample = self.samples[idx]
        
        # Load features
        features = np.load(sample['feature_path'])  # (T, D)
        
        # Validate shape
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {features.shape[1]} for {sample['video_id']}"
            )
        
        # Truncate if needed
        if self.max_len is not None and features.shape[0] > self.max_len:
            features = features[:self.max_len]
        
        # Normalize if requested
        if self.normalize:
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        label = sample['label']
        
        if self.return_video_id:
            return features, label, sample['video_id']
        else:
            return features, label
    
    def get_video_info(self, idx: int) -> Dict:
        """Get metadata for a video"""
        sample = self.samples[idx]
        return {
            'video_id': sample['video_id'],
            'label': sample['label'],
            'category': sample['category'],
            'num_snippets': sample.get('num_snippets'),
            'feature_path': str(sample['feature_path'])
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total': len(self),
            'normal': self.num_normal,
            'anomaly': self.num_anomaly,
            'categories': dict(self.category_counts),
            'split': self.split,
            'feature_dim': self.feature_dim
        }


def collate_fn_pad(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
    """
    Collate function with padding for variable-length sequences.
    
    Args:
        batch: List of (features, label, video_id) tuples
    
    Returns:
        Dictionary with:
            - features: (B, T_max, D) padded features
            - labels: (B,) labels
            - lengths: (B,) original sequence lengths
            - mask: (B, T_max) padding mask (1=valid, 0=padding)
            - video_ids: List of video IDs
    """
    # Separate components
    if len(batch[0]) == 3:
        features_list, labels, video_ids = zip(*batch)
    else:
        features_list, labels = zip(*batch)
        video_ids = [f"video_{i}" for i in range(len(batch))]
    
    # Get lengths
    lengths = torch.tensor([f.shape[0] for f in features_list])
    max_len = lengths.max().item()
    batch_size = len(features_list)
    feature_dim = features_list[0].shape[1]
    
    # Pad features
    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, features in enumerate(features_list):
        length = features.shape[0]
        padded_features[i, :length] = features
        mask[i, :length] = 1
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {
        'features': padded_features,
        'labels': labels,
        'lengths': lengths,
        'mask': mask,
        'video_ids': list(video_ids)
    }


def collate_fn_list(batch: List[Tuple]) -> Dict:
    """
    Collate function that keeps sequences as list (no padding).
    Useful for models that can handle variable lengths directly.
    
    Args:
        batch: List of (features, label, video_id) tuples
    
    Returns:
        Dictionary with features as list and other fields as tensors/lists
    """
    if len(batch[0]) == 3:
        features_list, labels, video_ids = zip(*batch)
    else:
        features_list, labels = zip(*batch)
        video_ids = [f"video_{i}" for i in range(len(batch))]
    
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor([f.shape[0] for f in features_list])
    
    return {
        'features': list(features_list),  # List of tensors
        'labels': labels,
        'lengths': lengths,
        'video_ids': list(video_ids)
    }


if __name__ == "__main__":
    # Test the dataset
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("TESTING UCF-CRIME FEATURE DATASET")
    print("=" * 80)
    
    # Test with synthetic data
    feature_dir = Path(__file__).parent.parent.parent / "data" / "synthetic"
    
    if not feature_dir.exists():
        print(f"\n❌ Synthetic data not found: {feature_dir}")
        print("Run: python scripts/create_synthetic_data.py")
        exit(1)
    
    # Test train split
    print("\n1. Loading train dataset...")
    train_dataset = UCFCrimeFeatureDataset(
        feature_dir=feature_dir,
        split="train",
        return_video_id=True
    )
    
    print(f"\n2. Dataset statistics:")
    stats = train_dataset.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n3. Sample data:")
    for i in range(min(3, len(train_dataset))):
        features, label, video_id = train_dataset[i]
        print(f"   [{i}] {video_id}: shape={features.shape}, label={label}")
    
    # Test dataloader with padding
    print(f"\n4. Testing DataLoader with padding:")
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_pad
    )
    
    batch = next(iter(train_loader))
    print(f"   Batch keys: {batch.keys()}")
    print(f"   features shape: {batch['features'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   lengths: {batch['lengths']}")
    print(f"   mask shape: {batch['mask'].shape}")
    print(f"   video_ids: {batch['video_ids']}")
    
    # Test test split
    print(f"\n5. Loading test dataset...")
    test_dataset = UCFCrimeFeatureDataset(
        feature_dir=feature_dir,
        split="test"
    )
    print(f"   Test samples: {len(test_dataset)}")
    
    print("\n" + "=" * 80)
    print("✓ DATASET TESTS PASSED")
    print("=" * 80)
