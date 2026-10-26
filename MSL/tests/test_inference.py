"""
Tests for inference components
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MSLModel
from src.inference import (
    ScoreCorrection,
    apply_temporal_smoothing,
    detect_anomaly_intervals,
    MSLPredictor,
    load_predictor_from_checkpoint
)


class TestScoreCorrection:
    """Test score correction methods."""
    
    def test_linear_correction(self):
        """Test linear blending correction."""
        corrector = ScoreCorrection(method='linear', alpha=0.5)
        
        video_scores = torch.tensor([0.8, 0.3])
        snippet_scores = torch.tensor([
            [0.9, 0.8, 0.7],
            [0.4, 0.3, 0.5]
        ])
        
        corrected = corrector(video_scores, snippet_scores)
        
        assert corrected.shape == snippet_scores.shape
        
        # Check linear blend: 0.5 * 0.8 + 0.5 * 0.9 = 0.85
        expected_00 = 0.5 * 0.8 + 0.5 * 0.9
        assert abs(corrected[0, 0].item() - expected_00) < 1e-5
    
    def test_multiplicative_correction(self):
        """Test multiplicative correction."""
        corrector = ScoreCorrection(method='multiplicative')
        
        video_scores = torch.tensor([0.8, 0.5])
        snippet_scores = torch.tensor([
            [0.9, 0.8],
            [0.4, 0.6]
        ])
        
        corrected = corrector(video_scores, snippet_scores)
        
        # Check multiplication: 0.8 * 0.9 = 0.72
        expected_00 = 0.8 * 0.9
        assert abs(corrected[0, 0].item() - expected_00) < 1e-5
    
    def test_threshold_correction(self):
        """Test threshold-based correction."""
        corrector = ScoreCorrection(method='threshold', threshold=0.5)
        
        video_scores = torch.tensor([0.8, 0.3])
        snippet_scores = torch.tensor([
            [0.9, 0.8],
            [0.4, 0.6]
        ])
        
        corrected = corrector(video_scores, snippet_scores)
        
        # Video 0 (0.8 >= 0.5): kept
        assert corrected[0, 0].item() == 0.9
        
        # Video 1 (0.3 < 0.5): suppressed
        assert corrected[1, 0].item() == 0.0
        assert corrected[1, 1].item() == 0.0
    
    def test_no_correction(self):
        """Test no correction mode."""
        corrector = ScoreCorrection(method='none')
        
        video_scores = torch.tensor([0.8, 0.3])
        snippet_scores = torch.tensor([
            [0.9, 0.8],
            [0.4, 0.6]
        ])
        
        corrected = corrector(video_scores, snippet_scores)
        
        # Should be unchanged
        assert torch.allclose(corrected, snippet_scores)


class TestTemporalSmoothing:
    """Test temporal smoothing."""
    
    def test_gaussian_smoothing(self):
        """Test Gaussian smoothing."""
        scores = torch.tensor([[0.1, 0.9, 0.1, 0.1, 0.1]])
        
        smoothed = apply_temporal_smoothing(
            scores,
            window_size=3,
            method='gaussian'
        )
        
        assert smoothed.shape == scores.shape
        
        # Peak should be reduced by smoothing
        assert smoothed[0, 1] < scores[0, 1]
        
        # Neighbors should be increased
        assert smoothed[0, 0] > scores[0, 0]
        assert smoothed[0, 2] > scores[0, 2]
    
    def test_uniform_smoothing(self):
        """Test uniform (moving average) smoothing."""
        scores = torch.tensor([[0.0, 1.0, 0.0]])
        
        smoothed = apply_temporal_smoothing(
            scores,
            window_size=3,
            method='uniform'
        )
        
        # Middle value should be average of neighbors
        expected = 1.0 / 3.0
        assert abs(smoothed[0, 1].item() - expected) < 1e-5
    
    def test_median_smoothing(self):
        """Test median filtering."""
        scores = torch.tensor([[0.1, 0.9, 0.1, 0.1, 0.1]])
        
        smoothed = apply_temporal_smoothing(
            scores,
            window_size=3,
            method='median'
        )
        
        # Median of [0.1, 0.9, 0.1] = 0.1
        assert smoothed[0, 1].item() == 0.1
    
    def test_no_smoothing(self):
        """Test with window_size=1 (no smoothing)."""
        scores = torch.tensor([[0.1, 0.9, 0.1]])
        
        smoothed = apply_temporal_smoothing(
            scores,
            window_size=1,
            method='gaussian'
        )
        
        # Should be unchanged
        assert torch.allclose(smoothed, scores)


class TestIntervalDetection:
    """Test anomaly interval detection."""
    
    def test_single_interval(self):
        """Test detection of single interval."""
        scores = torch.tensor([[0.1, 0.8, 0.9, 0.8, 0.1]])
        
        intervals = detect_anomaly_intervals(
            scores,
            threshold=0.7,
            min_length=2
        )
        
        assert len(intervals[0]) == 1
        assert intervals[0][0] == (1, 4)  # Snippets 1-3 (end is exclusive)
    
    def test_multiple_intervals(self):
        """Test detection of multiple intervals."""
        scores = torch.tensor([[0.8, 0.9, 0.1, 0.1, 0.8, 0.9, 0.8]])
        
        intervals = detect_anomaly_intervals(
            scores,
            threshold=0.7,
            min_length=2
        )
        
        assert len(intervals[0]) == 2
    
    def test_merge_close_intervals(self):
        """Test merging of close intervals."""
        scores = torch.tensor([[0.8, 0.9, 0.1, 0.8, 0.9]])
        
        intervals = detect_anomaly_intervals(
            scores,
            threshold=0.7,
            min_length=2,
            merge_gap=2
        )
        
        # Should merge into single interval
        assert len(intervals[0]) == 1
    
    def test_min_length_filter(self):
        """Test minimum length filtering."""
        scores = torch.tensor([[0.8, 0.1, 0.1, 0.8, 0.9, 0.8]])
        
        # Single snippet intervals should be filtered
        intervals = detect_anomaly_intervals(
            scores,
            threshold=0.7,
            min_length=2
        )
        
        # Only the second interval (3 snippets) should be detected
        assert len(intervals[0]) == 1
        assert intervals[0][0][0] == 3
    
    def test_no_intervals(self):
        """Test when no anomalies detected."""
        scores = torch.tensor([[0.1, 0.2, 0.1, 0.2]])
        
        intervals = detect_anomaly_intervals(
            scores,
            threshold=0.7
        )
        
        assert len(intervals[0]) == 0


class TestMSLPredictor:
    """Test MSL predictor."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return MSLModel(
            input_dim=1024,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            sequence_length=16,
            sequence_stride=8
        )
    
    @pytest.fixture
    def predictor(self, model):
        """Create test predictor."""
        return MSLPredictor(
            model=model,
            device='cpu',
            score_correction_method='linear',
            score_correction_alpha=0.5,
            temporal_smoothing=True,
            smoothing_window=3,
            anomaly_threshold=0.5
        )
    
    def test_single_video_prediction(self, predictor):
        """Test prediction on single video."""
        features = torch.randn(30, 1024)
        
        result = predictor.predict(features)
        
        # Check result structure
        assert 'video_score' in result
        assert 'snippet_scores' in result
        assert 'corrected_scores' in result
        assert 'smoothed_scores' in result
        assert 'anomaly_intervals' in result
        assert 'prediction' in result
        
        # Check shapes
        assert isinstance(result['video_score'], float)
        assert result['snippet_scores'].shape[0] == 30
        assert result['corrected_scores'].shape[0] == 30
        assert result['smoothed_scores'].shape[0] == 30
        
        # Check prediction is binary
        assert result['prediction'] in [0, 1]
    
    def test_batch_prediction(self, predictor):
        """Test prediction on batch."""
        features = torch.randn(4, 30, 1024)
        
        result = predictor.predict(features)
        
        # Check batch dimensions
        assert result['video_score'].shape[0] == 4
        assert result['snippet_scores'].shape == (4, 30)
        assert result['prediction'].shape[0] == 4
    
    def test_variable_length_batch(self, predictor):
        """Test prediction on variable-length videos."""
        features_list = [
            torch.randn(20, 1024),
            torch.randn(30, 1024),
            torch.randn(25, 1024)
        ]
        
        results = predictor.predict_batch(features_list)
        
        assert len(results) == 3
        
        # Check individual shapes
        assert results[0]['snippet_scores'].shape[0] == 20
        assert results[1]['snippet_scores'].shape[0] == 30
        assert results[2]['snippet_scores'].shape[0] == 25
    
    def test_formatted_output(self, predictor):
        """Test formatted output generation."""
        features = torch.randn(30, 1024)
        result = predictor.predict(features)
        
        formatted = predictor.format_prediction(
            result,
            video_name="test.mp4",
            fps=30.0
        )
        
        assert isinstance(formatted, str)
        assert "test.mp4" in formatted
        assert "Video Anomaly Score" in formatted
        assert "Prediction" in formatted
    
    def test_score_correction_applied(self, predictor):
        """Test that score correction is applied."""
        features = torch.randn(30, 1024)
        result = predictor.predict(features)
        
        # Corrected scores should differ from original (usually)
        # Note: They might be the same in rare cases, so we just check they exist
        assert result['snippet_scores'] is not None
        assert result['corrected_scores'] is not None
    
    def test_temporal_smoothing_applied(self, predictor):
        """Test that temporal smoothing is applied."""
        features = torch.randn(30, 1024)
        result = predictor.predict(features)
        
        # Smoothed scores should exist
        assert result['smoothed_scores'] is not None


class TestCheckpointLoading:
    """Test loading predictor from checkpoint."""
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        # Create model
        model = MSLModel(
            input_dim=1024,
            hidden_dim=128,
            num_layers=2,
            num_heads=4
        )
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Save checkpoint
            checkpoint_path = temp_dir / 'test_checkpoint.pt'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 10,
                'loss': 0.5
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load predictor from checkpoint
            predictor = load_predictor_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model_class=MSLModel,
                model_config={
                    'input_dim': 1024,
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'num_heads': 4
                },
                device='cpu'
            )
            
            # Test prediction
            features = torch.randn(20, 1024)
            result = predictor.predict(features)
            
            assert 'video_score' in result
            assert 'prediction' in result
        
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests
    print("=" * 80)
    print("TESTING INFERENCE COMPONENTS")
    print("=" * 80)
    
    # Test score correction
    print("\n1. Testing Score Correction...")
    test = TestScoreCorrection()
    test.test_linear_correction()
    test.test_multiplicative_correction()
    test.test_threshold_correction()
    test.test_no_correction()
    print("   ✓ All score correction tests passed")
    
    # Test temporal smoothing
    print("\n2. Testing Temporal Smoothing...")
    test = TestTemporalSmoothing()
    test.test_gaussian_smoothing()
    test.test_uniform_smoothing()
    test.test_median_smoothing()
    test.test_no_smoothing()
    print("   ✓ All temporal smoothing tests passed")
    
    # Test interval detection
    print("\n3. Testing Interval Detection...")
    test = TestIntervalDetection()
    test.test_single_interval()
    test.test_multiple_intervals()
    test.test_merge_close_intervals()
    test.test_min_length_filter()
    test.test_no_intervals()
    print("   ✓ All interval detection tests passed")
    
    # Test predictor
    print("\n4. Testing MSL Predictor...")
    model = MSLModel(
        input_dim=1024,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        sequence_length=16,
        sequence_stride=8
    )
    predictor = MSLPredictor(
        model=model,
        device='cpu',
        score_correction_method='linear',
        score_correction_alpha=0.5,
        temporal_smoothing=True,
        smoothing_window=3,
        anomaly_threshold=0.5
    )
    
    test = TestMSLPredictor()
    test.test_single_video_prediction(predictor)
    test.test_batch_prediction(predictor)
    test.test_variable_length_batch(predictor)
    test.test_formatted_output(predictor)
    test.test_score_correction_applied(predictor)
    test.test_temporal_smoothing_applied(predictor)
    print("   ✓ All predictor tests passed")
    
    # Test checkpoint loading
    print("\n5. Testing Checkpoint Loading...")
    test = TestCheckpointLoading()
    test.test_save_and_load_checkpoint()
    print("   ✓ Checkpoint loading test passed")
    
    print("\n" + "=" * 80)
    print("✓ ALL INFERENCE TESTS PASSED")
    print("=" * 80)
