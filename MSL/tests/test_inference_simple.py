"""
Simple inference tests without pytest
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MSLModel
from src.inference import (
    ScoreCorrection,
    apply_temporal_smoothing,
    detect_anomaly_intervals,
    MSLPredictor
)


def test_score_correction():
    """Test score correction."""
    print("Testing score correction...")
    
    # Linear correction
    corrector = ScoreCorrection(method='linear', alpha=0.5)
    video_scores = torch.tensor([0.8, 0.3])
    snippet_scores = torch.tensor([[0.9, 0.8], [0.4, 0.6]])
    corrected = corrector(video_scores, snippet_scores)
    assert corrected.shape == snippet_scores.shape
    print("  ✓ Linear correction works")
    
    # Multiplicative correction
    corrector = ScoreCorrection(method='multiplicative')
    corrected = corrector(video_scores, snippet_scores)
    assert corrected.shape == snippet_scores.shape
    print("  ✓ Multiplicative correction works")
    
    # Threshold correction
    corrector = ScoreCorrection(method='threshold', threshold=0.5)
    corrected = corrector(video_scores, snippet_scores)
    assert corrected[1, 0].item() == 0.0  # Suppressed
    print("  ✓ Threshold correction works")


def test_temporal_smoothing():
    """Test temporal smoothing."""
    print("Testing temporal smoothing...")
    
    scores = torch.tensor([[0.1, 0.9, 0.1, 0.1, 0.1]])
    
    # Gaussian
    smoothed = apply_temporal_smoothing(scores, window_size=3, method='gaussian')
    assert smoothed.shape == scores.shape
    assert smoothed[0, 1] < scores[0, 1]  # Peak reduced
    print("  ✓ Gaussian smoothing works")
    
    # Uniform
    smoothed = apply_temporal_smoothing(scores, window_size=3, method='uniform')
    assert smoothed.shape == scores.shape
    print("  ✓ Uniform smoothing works")
    
    # Median
    smoothed = apply_temporal_smoothing(scores, window_size=3, method='median')
    assert smoothed.shape == scores.shape
    print("  ✓ Median smoothing works")


def test_interval_detection():
    """Test interval detection."""
    print("Testing interval detection...")
    
    # Single interval
    scores = torch.tensor([[0.1, 0.8, 0.9, 0.8, 0.1]])
    intervals = detect_anomaly_intervals(scores, threshold=0.7, min_length=2)
    assert len(intervals[0]) == 1
    print("  ✓ Single interval detection works")
    
    # Multiple intervals (separated enough not to merge)
    scores = torch.tensor([[0.8, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.9, 0.8]])
    intervals = detect_anomaly_intervals(scores, threshold=0.7, min_length=2, merge_gap=1)
    assert len(intervals[0]) == 2
    print("  ✓ Multiple interval detection works")
    
    # No intervals
    scores = torch.tensor([[0.1, 0.2, 0.1, 0.2]])
    intervals = detect_anomaly_intervals(scores, threshold=0.7)
    assert len(intervals[0]) == 0
    print("  ✓ No interval case works")


def test_predictor():
    """Test MSL predictor."""
    print("Testing MSL predictor...")
    
    # Create model
    model = MSLModel(
        input_dim=1024,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        sequence_length=16,
        sequence_stride=8
    )
    
    # Create predictor
    predictor = MSLPredictor(
        model=model,
        device='cpu',
        score_correction_method='linear',
        score_correction_alpha=0.5,
        temporal_smoothing=True,
        smoothing_window=3,
        anomaly_threshold=0.5
    )
    print("  ✓ Predictor created")
    
    # Single video prediction
    features = torch.randn(30, 1024)
    result = predictor.predict(features)
    
    assert 'video_score' in result
    assert 'snippet_scores' in result
    assert 'corrected_scores' in result
    assert 'smoothed_scores' in result
    assert 'anomaly_intervals' in result
    assert 'prediction' in result
    assert isinstance(result['video_score'], float)
    assert result['prediction'] in [0, 1]
    print(f"  ✓ Single video prediction works (score={result['video_score']:.4f})")
    
    # Batch prediction
    batch_features = torch.randn(4, 30, 1024)
    batch_result = predictor.predict(batch_features)
    assert batch_result['video_score'].shape[0] == 4
    assert batch_result['snippet_scores'].shape == (4, 30)
    print("  ✓ Batch prediction works")
    
    # Variable-length batch
    features_list = [
        torch.randn(20, 1024),
        torch.randn(30, 1024),
        torch.randn(25, 1024)
    ]
    results = predictor.predict_batch(features_list)
    assert len(results) == 3
    assert results[0]['snippet_scores'].shape[0] == 20
    assert results[1]['snippet_scores'].shape[0] == 30
    assert results[2]['snippet_scores'].shape[0] == 25
    print("  ✓ Variable-length batch works")
    
    # Formatted output
    formatted = predictor.format_prediction(result, video_name="test.mp4", fps=30.0)
    assert isinstance(formatted, str)
    assert "test.mp4" in formatted
    assert "Video Anomaly Score" in formatted
    print("  ✓ Formatted output works")


def main():
    print("=" * 80)
    print("TESTING INFERENCE COMPONENTS")
    print("=" * 80)
    
    print("\n1. Score Correction")
    test_score_correction()
    
    print("\n2. Temporal Smoothing")
    test_temporal_smoothing()
    
    print("\n3. Interval Detection")
    test_interval_detection()
    
    print("\n4. MSL Predictor")
    test_predictor()
    
    print("\n" + "=" * 80)
    print("✓ ALL INFERENCE TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
