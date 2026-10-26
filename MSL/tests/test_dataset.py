"""
Unit tests for dataset loaders
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import UCFCrimeFeatureDataset, collate_fn_pad, collate_fn_list


@pytest.fixture
def synthetic_data_dir():
    """Path to synthetic data"""
    data_dir = Path(__file__).parent.parent / "data" / "synthetic"
    if not data_dir.exists():
        pytest.skip("Synthetic data not found. Run: python scripts/create_synthetic_data.py")
    return data_dir


class TestUCFCrimeFeatureDataset:
    """Tests for UCFCrimeFeatureDataset"""
    
    def test_dataset_init(self, synthetic_data_dir):
        """Test dataset initialization"""
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train"
        )
        assert len(dataset) > 0
        assert dataset.feature_dim == 1024
        assert dataset.split == "train"
    
    def test_dataset_splits(self, synthetic_data_dir):
        """Test both train and test splits"""
        train_dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train"
        )
        test_dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="test"
        )
        
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        assert len(train_dataset) != len(test_dataset)
    
    def test_getitem(self, synthetic_data_dir):
        """Test __getitem__ method"""
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train",
            return_video_id=True
        )
        
        features, label, video_id = dataset[0]
        
        # Check types
        assert isinstance(features, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(video_id, str)
        
        # Check shapes
        assert features.ndim == 2
        assert features.shape[1] == 1024
        
        # Check label values
        assert label in [0, 1]
    
    def test_getitem_without_video_id(self, synthetic_data_dir):
        """Test __getitem__ without video_id"""
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train",
            return_video_id=False
        )
        
        result = dataset[0]
        assert len(result) == 2  # features, label only
        features, label = result
        assert isinstance(features, torch.Tensor)
        assert isinstance(label, int)
    
    def test_max_len_truncation(self, synthetic_data_dir):
        """Test max_len truncation"""
        max_len = 50
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train",
            max_len=max_len
        )
        
        for i in range(min(5, len(dataset))):
            features, label = dataset[i]
            assert features.shape[0] <= max_len
    
    def test_normalize(self, synthetic_data_dir):
        """Test normalization"""
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train",
            normalize=True
        )
        
        features, label = dataset[0]
        # Check L2 normalization
        norms = torch.norm(features, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_get_video_info(self, synthetic_data_dir):
        """Test get_video_info method"""
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train"
        )
        
        info = dataset.get_video_info(0)
        
        assert 'video_id' in info
        assert 'label' in info
        assert 'category' in info
        assert 'feature_path' in info
    
    def test_get_statistics(self, synthetic_data_dir):
        """Test get_statistics method"""
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train"
        )
        
        stats = dataset.get_statistics()
        
        assert 'total' in stats
        assert 'normal' in stats
        assert 'anomaly' in stats
        assert 'categories' in stats
        assert stats['total'] == len(dataset)
        assert stats['normal'] + stats['anomaly'] == stats['total']


class TestCollateFunctions:
    """Tests for collate functions"""
    
    def create_sample_batch(self):
        """Create a sample batch for testing"""
        batch = [
            (torch.randn(50, 1024), 0, "video_1"),
            (torch.randn(75, 1024), 1, "video_2"),
            (torch.randn(60, 1024), 0, "video_3"),
        ]
        return batch
    
    def test_collate_fn_pad(self):
        """Test collate_fn_pad"""
        batch = self.create_sample_batch()
        result = collate_fn_pad(batch)
        
        # Check keys
        assert 'features' in result
        assert 'labels' in result
        assert 'lengths' in result
        assert 'mask' in result
        assert 'video_ids' in result
        
        # Check shapes
        batch_size = len(batch)
        max_len = 75  # Longest sequence
        
        assert result['features'].shape == (batch_size, max_len, 1024)
        assert result['labels'].shape == (batch_size,)
        assert result['lengths'].shape == (batch_size,)
        assert result['mask'].shape == (batch_size, max_len)
        assert len(result['video_ids']) == batch_size
        
        # Check mask validity
        for i, length in enumerate(result['lengths']):
            # Valid positions should be True
            assert result['mask'][i, :length].all()
            # Padding positions should be False
            if length < max_len:
                assert not result['mask'][i, length:].any()
    
    def test_collate_fn_pad_without_video_id(self):
        """Test collate_fn_pad with tuples without video_id"""
        batch = [
            (torch.randn(50, 1024), 0),
            (torch.randn(75, 1024), 1),
        ]
        result = collate_fn_pad(batch)
        
        assert 'features' in result
        assert 'labels' in result
        assert len(result['video_ids']) == len(batch)
    
    def test_collate_fn_list(self):
        """Test collate_fn_list"""
        batch = self.create_sample_batch()
        result = collate_fn_list(batch)
        
        # Check keys
        assert 'features' in result
        assert 'labels' in result
        assert 'lengths' in result
        assert 'video_ids' in result
        
        # Features should be a list
        assert isinstance(result['features'], list)
        assert len(result['features']) == len(batch)
        
        # Each item should maintain its original shape
        assert result['features'][0].shape[0] == 50
        assert result['features'][1].shape[0] == 75
        assert result['features'][2].shape[0] == 60


class TestDatasetIntegration:
    """Integration tests"""
    
    def test_dataloader_integration(self, synthetic_data_dir):
        """Test dataset with PyTorch DataLoader"""
        from torch.utils.data import DataLoader
        
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train"
        )
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn_pad
        )
        
        # Get one batch
        batch = next(iter(loader))
        
        # Verify batch structure
        assert isinstance(batch, dict)
        assert batch['features'].ndim == 3
        assert batch['labels'].ndim == 1
        assert batch['features'].shape[0] == batch['labels'].shape[0]
    
    def test_full_epoch(self, synthetic_data_dir):
        """Test iterating through full epoch"""
        from torch.utils.data import DataLoader
        
        dataset = UCFCrimeFeatureDataset(
            feature_dir=synthetic_data_dir,
            split="train"
        )
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn_pad
        )
        
        total_samples = 0
        for batch in loader:
            total_samples += batch['labels'].shape[0]
        
        assert total_samples == len(dataset)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
