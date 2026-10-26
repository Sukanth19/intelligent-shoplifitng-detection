"""
VideoSwin Feature Extractor
Extracts features from videos using pretrained Video Swin Transformer
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VideoSwinFeatureExtractor:
    """
    Feature extraction using Video Swin Transformer.
    
    Note: This is a placeholder implementation. For production:
    1. Install mmaction2 or timm with video models
    2. Load pretrained VideoSwin checkpoint (swin_base_patch244_window877_kinetics400_1k.pth)
    3. Extract features from intermediate layers
    
    For now, this provides the interface and can work with synthetic features.
    """
    
    def __init__(
        self,
        model_name: str = 'swin_base_patch244_window877',
        checkpoint_path: Optional[Path] = None,
        device: str = 'cuda',
        feature_layer: str = 'layer3',  # Which layer to extract features from
        batch_size: int = 8
    ):
        """
        Args:
            model_name: VideoSwin model name
            checkpoint_path: Path to pretrained checkpoint
            device: Device for inference
            feature_layer: Which layer to extract features from
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.feature_layer = feature_layer
        self.batch_size = batch_size
        
        self.model = None
        self.feature_dim = 1024  # Default VideoSwin base feature dimension
        
        logger.info(f"VideoSwinFeatureExtractor initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Feature layer: {feature_layer}")
    
    def load_model(self):
        """
        Load pretrained VideoSwin model.
        
        Note: This is a placeholder. Real implementation would:
        1. Import from mmaction2 or similar
        2. Load pretrained checkpoint
        3. Set up feature extraction hooks
        """
        logger.warning("VideoSwin model loading not implemented - using placeholder")
        logger.warning("For production, install mmaction2 and load pretrained checkpoint")
        
        # Placeholder: Create a simple model that generates random features
        # This allows testing the pipeline without actual VideoSwin
        self.model = PlaceholderVideoSwin(self.feature_dim)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Placeholder model loaded (for testing)")
    
    def extract_features(
        self,
        video_clips: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from video clips.
        
        Args:
            video_clips: Tensor of shape (N, C, T, H, W)
        
        Returns:
            features: Tensor of shape (N, feature_dim)
        """
        if self.model is None:
            self.load_model()
        
        video_clips = video_clips.to(self.device)
        
        with torch.no_grad():
            features = self.model(video_clips)
        
        return features.cpu()
    
    def extract_from_video(
        self,
        video_clips: torch.Tensor,
        aggregate: str = 'mean'
    ) -> np.ndarray:
        """
        Extract features from multiple clips of a video and aggregate.
        
        Args:
            video_clips: Tensor of shape (N, C, T, H, W) - N clips from one video
            aggregate: Aggregation method ('mean', 'max', 'none')
        
        Returns:
            features: Array of shape (feature_dim,) if aggregated, else (N, feature_dim)
        """
        N = video_clips.shape[0]
        
        # Process in batches
        all_features = []
        for i in range(0, N, self.batch_size):
            batch = video_clips[i:i + self.batch_size]
            batch_features = self.extract_features(batch)
            all_features.append(batch_features)
        
        features = torch.cat(all_features, dim=0).numpy()  # (N, feature_dim)
        
        # Aggregate if requested
        if aggregate == 'mean':
            features = features.mean(axis=0)  # (feature_dim,)
        elif aggregate == 'max':
            features = features.max(axis=0)  # (feature_dim,)
        elif aggregate == 'none':
            pass  # Keep all features
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        return features
    
    def extract_temporal_features(
        self,
        video_clips: torch.Tensor
    ) -> np.ndarray:
        """
        Extract features preserving temporal dimension.
        
        Args:
            video_clips: Tensor of shape (N, C, T, H, W)
        
        Returns:
            features: Array of shape (N, feature_dim) - one feature per clip
        """
        return self.extract_from_video(video_clips, aggregate='none')


class PlaceholderVideoSwin(nn.Module):
    """
    Placeholder VideoSwin model for testing.
    
    Generates realistic-looking features without actual VideoSwin.
    Real implementation should use pretrained VideoSwin from mmaction2.
    """
    
    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple conv layers to process input and generate features
        self.conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(64, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, H, W)
        
        Returns:
            features: (N, feature_dim)
        """
        # Simple forward pass
        x = self.conv(x)  # (N, 64, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 64)
        x = self.fc(x)  # (N, feature_dim)
        return x


def extract_features_from_video(
    video_path: Path,
    output_path: Path,
    preprocessor,
    extractor: VideoSwinFeatureExtractor,
    save_format: str = 'npy'
):
    """
    Complete pipeline: video → features → save.
    
    Args:
        video_path: Path to video file
        output_path: Path to save features
        preprocessor: VideoPreprocessor instance
        extractor: VideoSwinFeatureExtractor instance
        save_format: Format to save features ('npy' or 'npz')
    """
    logger.info(f"Extracting features from: {video_path}")
    
    # Preprocess video
    video_clips, metadata = preprocessor.process_video(video_path)
    logger.info(f"  Preprocessed into {metadata['num_clips']} clips")
    
    # Extract features
    features = extractor.extract_temporal_features(video_clips)
    logger.info(f"  Extracted features shape: {features.shape}")
    
    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_format == 'npy':
        np.save(output_path, features)
    elif save_format == 'npz':
        np.savez(
            output_path,
            features=features,
            metadata=metadata
        )
    else:
        raise ValueError(f"Unknown save format: {save_format}")
    
    logger.info(f"  Saved to: {output_path}")
    
    return features, metadata


if __name__ == "__main__":
    # Test feature extractor
    print("=" * 80)
    print("TESTING VIDEO SWIN FEATURE EXTRACTOR")
    print("=" * 80)
    
    # Create extractor
    print("\n1. Creating feature extractor...")
    extractor = VideoSwinFeatureExtractor(
        model_name='swin_base_patch244_window877',
        device='cpu',
        batch_size=4
    )
    print("   ✓ Extractor created")
    
    # Load model
    print("\n2. Loading model...")
    extractor.load_model()
    print("   ✓ Model loaded (placeholder)")
    
    # Test feature extraction
    print("\n3. Testing feature extraction...")
    # Synthetic video clips (N=8, C=3, T=32, H=224, W=224)
    test_clips = torch.randn(8, 3, 32, 224, 224)
    features = extractor.extract_features(test_clips)
    print(f"   Input shape: {test_clips.shape}")
    print(f"   Output shape: {features.shape}")
    print(f"   Expected: (8, {extractor.feature_dim})")
    assert features.shape == (8, extractor.feature_dim)
    print("   ✓ Feature extraction works")
    
    # Test aggregation
    print("\n4. Testing feature aggregation...")
    # Mean aggregation
    agg_features = extractor.extract_from_video(test_clips, aggregate='mean')
    print(f"   Mean aggregated shape: {agg_features.shape}")
    assert agg_features.shape == (extractor.feature_dim,)
    print("   ✓ Mean aggregation works")
    
    # No aggregation
    temporal_features = extractor.extract_temporal_features(test_clips)
    print(f"   Temporal features shape: {temporal_features.shape}")
    assert temporal_features.shape == (8, extractor.feature_dim)
    print("   ✓ Temporal extraction works")
    
    # Test batch processing
    print("\n5. Testing batch processing...")
    large_clips = torch.randn(20, 3, 32, 224, 224)
    large_features = extractor.extract_temporal_features(large_clips)
    print(f"   Large input: {large_clips.shape}")
    print(f"   Output: {large_features.shape}")
    assert large_features.shape == (20, extractor.feature_dim)
    print("   ✓ Batch processing works")
    
    print("\n" + "=" * 80)
    print("✓ ALL FEATURE EXTRACTOR TESTS PASSED")
    print("=" * 80)
    print("\nNote: This uses a placeholder model.")
    print("For production, install mmaction2 and load pretrained VideoSwin checkpoint.")
