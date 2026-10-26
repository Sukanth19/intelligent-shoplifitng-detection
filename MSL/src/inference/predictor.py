"""
Inference Predictor
Unified interface for video anomaly prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import logging

from .score_correction import ScoreCorrection, apply_temporal_smoothing, detect_anomaly_intervals

logger = logging.getLogger(__name__)


class MSLPredictor:
    """
    Predictor for video anomaly detection using trained MSL model.
    
    Handles:
    - Single video prediction
    - Batch prediction
    - Score correction
    - Temporal smoothing
    - Anomaly interval detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        score_correction_method: str = 'linear',
        score_correction_alpha: float = 0.5,
        temporal_smoothing: bool = True,
        smoothing_window: int = 5,
        anomaly_threshold: float = 0.5
    ):
        """
        Args:
            model: Trained MSL model
            device: Device for inference ('cuda' or 'cpu')
            score_correction_method: Correction method ('linear', 'multiplicative', 'threshold', 'none')
            score_correction_alpha: Alpha for linear blending (default: 0.5)
            temporal_smoothing: Whether to apply temporal smoothing
            smoothing_window: Window size for smoothing
            anomaly_threshold: Threshold for anomaly detection
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Score correction
        self.score_corrector = ScoreCorrection(
            method=score_correction_method,
            alpha=score_correction_alpha
        )
        
        # Temporal smoothing
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        
        # Anomaly detection
        self.anomaly_threshold = anomaly_threshold
        
        logger.info(f"Predictor initialized on {device}")
        logger.info(f"  Score correction: {score_correction_method}")
        logger.info(f"  Temporal smoothing: {temporal_smoothing}")
        logger.info(f"  Anomaly threshold: {anomaly_threshold}")
    
    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Predict anomaly for a single video or batch.
        
        Args:
            features: Video features (B, T, D) or (T, D)
            mask: Optional mask (B, T) or (T,)
        
        Returns:
            Dictionary with:
                - video_score: Video-level anomaly probability (B,) or scalar
                - snippet_scores: Original snippet scores (B, T) or (T,)
                - corrected_scores: Corrected snippet scores (B, T) or (T,)
                - smoothed_scores: Smoothed scores if enabled (B, T) or (T,)
                - anomaly_intervals: List of detected intervals
                - prediction: Binary prediction (0=normal, 1=anomaly)
        """
        # Handle single video (T, D) -> (1, T, D)
        single_video = False
        if features.ndim == 2:
            features = features.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            single_video = True
        
        # Move to device
        features = features.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        # Forward pass
        output = self.model(features, mask)
        
        video_scores = output['video_scores']  # (B,)
        snippet_scores = output['snippet_scores']  # (B, T)
        
        # Score correction
        corrected_scores = self.score_corrector(video_scores, snippet_scores)
        
        # Temporal smoothing (optional)
        if self.temporal_smoothing:
            smoothed_scores = apply_temporal_smoothing(
                corrected_scores,
                window_size=self.smoothing_window,
                method='gaussian'
            )
        else:
            smoothed_scores = corrected_scores
        
        # Detect anomaly intervals
        intervals = detect_anomaly_intervals(
            smoothed_scores,
            threshold=self.anomaly_threshold,
            min_length=3,
            merge_gap=2
        )
        
        # Binary prediction
        prediction = (video_scores >= self.anomaly_threshold).long()
        
        # Convert to CPU for output
        result = {
            'video_score': video_scores.cpu(),
            'snippet_scores': snippet_scores.cpu(),
            'corrected_scores': corrected_scores.cpu(),
            'smoothed_scores': smoothed_scores.cpu(),
            'anomaly_intervals': intervals,
            'prediction': prediction.cpu()
        }
        
        # If single video, remove batch dimension
        if single_video:
            result['video_score'] = result['video_score'].item()
            result['snippet_scores'] = result['snippet_scores'].squeeze(0)
            result['corrected_scores'] = result['corrected_scores'].squeeze(0)
            result['smoothed_scores'] = result['smoothed_scores'].squeeze(0)
            result['prediction'] = result['prediction'].item()
            result['anomaly_intervals'] = intervals[0]
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        features_list: List[torch.Tensor]
    ) -> List[Dict[str, any]]:
        """
        Predict anomaly for multiple videos (variable length).
        
        Args:
            features_list: List of feature tensors [(T1, D), (T2, D), ...]
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        
        return results
    
    def format_prediction(
        self,
        result: Dict[str, any],
        video_name: Optional[str] = None,
        fps: float = 30.0
    ) -> str:
        """
        Format prediction result as human-readable string.
        
        Args:
            result: Prediction result dictionary
            video_name: Optional video name
            fps: Frames per second for time conversion
        
        Returns:
            Formatted string
        """
        lines = []
        
        if video_name:
            lines.append(f"Video: {video_name}")
        
        video_score = result['video_score']
        if isinstance(video_score, torch.Tensor):
            video_score = video_score.item()
        
        prediction = result['prediction']
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.item()
        
        lines.append(f"Video Anomaly Score: {video_score:.4f}")
        lines.append(f"Prediction: {'ANOMALY' if prediction == 1 else 'NORMAL'}")
        
        intervals = result['anomaly_intervals']
        if intervals:
            lines.append(f"\nDetected Anomaly Intervals: {len(intervals)}")
            for i, (start, end) in enumerate(intervals, 1):
                start_time = start / fps
                end_time = end / fps
                duration = end_time - start_time
                lines.append(f"  {i}. Snippets {start}-{end} ({start_time:.1f}s - {end_time:.1f}s, duration: {duration:.1f}s)")
        else:
            lines.append("\nNo anomalous intervals detected")
        
        return "\n".join(lines)
    
    def save_predictions(
        self,
        results: List[Dict[str, any]],
        video_names: List[str],
        output_path: Path
    ):
        """
        Save predictions to file.
        
        Args:
            results: List of prediction dictionaries
            video_names: List of video names
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for video_name, result in zip(video_names, results):
                f.write(self.format_prediction(result, video_name))
                f.write("\n" + "=" * 80 + "\n\n")
        
        logger.info(f"Predictions saved to {output_path}")


def load_predictor_from_checkpoint(
    checkpoint_path: Path,
    model_class,
    model_config: Dict,
    device: str = 'cuda',
    **predictor_kwargs
) -> MSLPredictor:
    """
    Load predictor from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class (e.g., MSLModel)
        model_config: Model configuration dictionary
        device: Device for inference
        **predictor_kwargs: Additional arguments for predictor
    
    Returns:
        Initialized predictor
    """
    # Create model
    model = model_class(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Loss: {checkpoint.get('loss', 'unknown')}")
    
    # Create predictor
    predictor = MSLPredictor(model, device=device, **predictor_kwargs)
    
    return predictor


if __name__ == "__main__":
    # Test predictor
    print("=" * 80)
    print("TESTING MSL PREDICTOR")
    print("=" * 80)
    
    from ..models import MSLModel
    
    # Create model
    print("\n1. Creating model...")
    model = MSLModel(
        input_dim=1024,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        sequence_length=32,
        sequence_stride=16
    )
    print(f"   ✓ Model created")
    
    # Create predictor
    print("\n2. Creating predictor...")
    predictor = MSLPredictor(
        model=model,
        device='cpu',
        score_correction_method='linear',
        score_correction_alpha=0.5,
        temporal_smoothing=True,
        smoothing_window=5,
        anomaly_threshold=0.5
    )
    print(f"   ✓ Predictor created")
    
    # Test single video prediction
    print("\n3. Testing single video prediction...")
    features = torch.randn(50, 1024)  # (T, D)
    result = predictor.predict(features)
    
    print(f"   Video score: {result['video_score']:.4f}")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Snippet scores shape: {result['snippet_scores'].shape}")
    print(f"   Corrected scores shape: {result['corrected_scores'].shape}")
    print(f"   Smoothed scores shape: {result['smoothed_scores'].shape}")
    print(f"   Anomaly intervals: {len(result['anomaly_intervals'])}")
    print(f"   ✓ Single video prediction works")
    
    # Test batch prediction
    print("\n4. Testing batch prediction...")
    batch_features = torch.randn(4, 50, 1024)  # (B, T, D)
    batch_result = predictor.predict(batch_features)
    
    print(f"   Batch video scores shape: {batch_result['video_score'].shape}")
    print(f"   Batch predictions shape: {batch_result['prediction'].shape}")
    print(f"   ✓ Batch prediction works")
    
    # Test formatted output
    print("\n5. Testing formatted output...")
    formatted = predictor.format_prediction(result, video_name="test_video.mp4", fps=30.0)
    print("   Formatted output:")
    print("   " + "\n   ".join(formatted.split("\n")))
    print(f"   ✓ Formatted output works")
    
    # Test variable-length batch
    print("\n6. Testing variable-length batch...")
    features_list = [
        torch.randn(30, 1024),
        torch.randn(50, 1024),
        torch.randn(40, 1024)
    ]
    results = predictor.predict_batch(features_list)
    print(f"   Number of results: {len(results)}")
    for i, r in enumerate(results):
        print(f"   Video {i}: score={r['video_score']:.4f}, pred={r['prediction']}")
    print(f"   ✓ Variable-length batch prediction works")
    
    print("\n" + "=" * 80)
    print("✓ ALL PREDICTOR TESTS PASSED")
    print("=" * 80)
