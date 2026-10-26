"""
UCF-Crime and SHT H5 Dataset Loader

Loads VideoSwin features from H5 files provided by MSL authors.
Supports both UCF-Crime and Shanghai Tech (SHT) datasets.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class UCFH5Dataset(Dataset):
    """
    Dataset loader for UCF-Crime and SHT VideoSwin features in H5 format.
    
    H5 file structure:
    - Keys: Video IDs (e.g., '04_017', '05_0019')
    - Values: Feature arrays of shape (T, D) where T = temporal length, D = 1024
    
    Args:
        h5_path: Path to H5 feature file
        list_path: Path to .list file with video names
        gt_path: Path to .npy ground truth file (optional)
        max_length: Maximum sequence length (videos will be truncated/padded)
        mode: 'train' or 'test'
        return_video_name: If True, return video name in output dict
    """
    
    def __init__(
        self,
        h5_path: str,
        list_path: str,
        gt_path: Optional[str] = None,
        max_length: int = 320,
        mode: str = 'train',
        return_video_name: bool = True
    ):
        self.h5_path = Path(h5_path)
        self.list_path = Path(list_path)
        self.gt_path = Path(gt_path) if gt_path else None
        self.max_length = max_length
        self.mode = mode
        self.return_video_name = return_video_name
        
        # Load video list
        self.video_names = self._load_video_list()
        
        # Open H5 file (keep open for efficient access)
        self.h5_file = h5py.File(self.h5_path, 'r')
        
        # Load ground truth if provided
        self.gt_labels = self._load_ground_truth() if self.gt_path else None
        
        # Map video names to H5 keys
        self.video_to_key = self._map_video_names_to_keys()
        
        logger.info(f"Loaded {len(self)} videos from {self.h5_path.name}")
        logger.info(f"Mode: {mode}, Max length: {max_length}")
        
    def _load_video_list(self) -> List[str]:
        """Load video names from .list file."""
        with open(self.list_path, 'r') as f:
            video_names = [line.strip() for line in f.readlines()]
        return video_names
    
    def _load_ground_truth(self) -> np.ndarray:
        """
        Load ground truth labels from .npy file.
        
        Note: GT file contains snippet-level labels concatenated for all videos.
        This is a flattened array where labels for video i are followed by labels for video i+1.
        """
        gt = np.load(self.gt_path, allow_pickle=True)
        logger.info(f"Loaded ground truth: shape {gt.shape}, dtype {gt.dtype}")
        return gt
    
    def _map_video_names_to_keys(self) -> Dict[str, str]:
        """
        Map video names from .list file to H5 keys.
        
        Video names in .list: 'Training_Normal_Videos_Anomaly/Normal_Videos505_x264.mp4'
        H5 keys: Vary by dataset, need flexible matching
        """
        video_to_key = {}
        h5_keys = list(self.h5_file.keys())
        
        logger.info(f"H5 file has {len(h5_keys)} keys")
        logger.info(f"List file has {len(self.video_names)} videos")
        
        for video_name in self.video_names:
            # Extract just the filename without path and extension
            # From: Training_Normal_Videos_Anomaly/Normal_Videos505_x264.mp4
            # To: Normal_Videos505
            
            basename = Path(video_name).stem  # Gets filename without extension
            basename_no_x264 = basename.replace('_x264', '')  # Remove _x264 suffix
            
            # Try exact match first
            if basename in h5_keys:
                video_to_key[video_name] = basename
                continue
            
            if basename_no_x264 in h5_keys:
                video_to_key[video_name] = basename_no_x264
                continue
            
            # Try fuzzy matching - find H5 key that contains the base number
            # Extract number from video name (e.g., 505 from Normal_Videos505)
            import re
            numbers = re.findall(r'\d+', basename)
            if numbers:
                number = numbers[-1]  # Get last number
                
                # Look for H5 keys containing this number
                for h5_key in h5_keys:
                    if number in h5_key:
                        video_to_key[video_name] = h5_key
                        break
        
        logger.info(f"Successfully mapped {len(video_to_key)}/{len(self.video_names)} videos to H5 keys")
        
        if len(video_to_key) < len(self.video_names):
            missing_count = len(self.video_names) - len(video_to_key)
            logger.warning(f"{missing_count} videos not found in H5 file - will be skipped")
        
        return video_to_key
    
    def _get_video_label(self, idx: int) -> int:
        """
        Get video-level label (0=normal, 1=abnormal).
        
        Uses video name pattern to determine label:
        - Videos with "Normal" in name → label = 0 (normal)
        - All other videos → label = 1 (abnormal/anomaly)
        
        UCF-Crime anomaly categories:
        Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
        RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
        """
        if self.gt_labels is None:
            return 0  # Default to normal if no GT
        
        # Get video name
        video_name = self.video_names[idx]
        if video_name not in self.video_to_key:
            return 0
        
        # Check if video is normal based on name pattern
        video_name_lower = video_name.lower()
        
        # Normal videos contain "normal" in their name
        if 'normal' in video_name_lower:
            return 0  # Normal video
        
        # All other videos are anomalies
        return 1  # Abnormal/anomaly video
    
    def __len__(self) -> int:
        return len(self.video_names)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video sample.
        
        Returns:
            Dictionary containing:
            - features: (T, D) tensor of features (may be padded)
            - label: Video-level label (0 or 1)
            - length: Actual sequence length before padding
            - mask: (T,) boolean mask (True for valid positions)
            - video_name: Video identifier (if return_video_name=True)
        """
        video_name = self.video_names[idx]
        
        # Check if video exists in H5
        if video_name not in self.video_to_key:
            # Return zero features if not found
            logger.warning(f"Video {video_name} not found, returning zeros")
            features = np.zeros((self.max_length, 1024), dtype=np.float32)
            length = 0
            mask = np.zeros(self.max_length, dtype=bool)
            label = 0
        else:
            # Load features from H5
            key = self.video_to_key[video_name]
            features = self.h5_file[key][:]  # Shape: (T, D)
            
            T, D = features.shape
            length = T
            
            # Get label
            label = self._get_video_label(idx)
            
            # Handle variable length
            if T > self.max_length:
                # Truncate
                features = features[:self.max_length]
                length = self.max_length
                mask = np.ones(self.max_length, dtype=bool)
            elif T < self.max_length:
                # Pad
                padding = np.zeros((self.max_length - T, D), dtype=np.float32)
                features = np.vstack([features, padding])
                mask = np.concatenate([
                    np.ones(T, dtype=bool),
                    np.zeros(self.max_length - T, dtype=bool)
                ])
            else:
                mask = np.ones(self.max_length, dtype=bool)
        
        # Convert to tensors
        result = {
            'features': torch.from_numpy(features).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'mask': torch.from_numpy(mask)
        }
        
        if self.return_video_name:
            result['video_name'] = video_name
        
        return result
    
    def close(self):
        """Close H5 file."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass  # Ignore errors during cleanup
    
    def __del__(self):
        """Cleanup when dataset is destroyed."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with stacked tensors
    """
    # Stack all tensors
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    result = {
        'features': features,
        'label': labels,
        'length': lengths,
        'mask': masks
    }
    
    # Include video names if present
    if 'video_name' in batch[0]:
        result['video_name'] = [item['video_name'] for item in batch]
    
    return result


# Alias for Shanghai Tech dataset (same format)
class SHTH5Dataset(UCFH5Dataset):
    """Shanghai Tech H5 dataset (same format as UCF)."""
    pass


if __name__ == "__main__":
    """Test the dataset loader."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Test on SHT data
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "SHT-VideoSwin-Features"
    
    if data_dir.exists():
        print(f"Testing SHT dataset from {data_dir}")
        
        dataset = UCFH5Dataset(
            h5_path=str(data_dir / "sh_swin_test_feature_256320224.h5"),
            list_path=str(data_dir / "sh-test.list"),
            gt_path=str(data_dir / "sh_gt_16.npy"),
            max_length=200,
            mode='test'
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test __getitem__
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Label: {sample['label']}")
        print(f"  Length: {sample['length']}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Video name: {sample.get('video_name', 'N/A')}")
        
        # Test batching
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        batch = next(iter(loader))
        print(f"\nBatch:")
        print(f"  Features shape: {batch['features'].shape}")
        print(f"  Labels shape: {batch['label'].shape}")
        print(f"  Lengths: {batch['length']}")
        print(f"  Masks shape: {batch['mask'].shape}")
        
        dataset.close()
        print("\n✅ Dataset loader test passed!")
    else:
        print(f"❌ Data directory not found: {data_dir}")
