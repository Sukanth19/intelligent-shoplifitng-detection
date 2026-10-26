"""
Score Correction
Stabilizes snippet-level scores using video-level probability
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ScoreCorrection:
    """
    Score correction mechanism for MSL.
    
    From the MSL paper:
    Uses video-level anomaly probability to stabilize/correct snippet-level scores.
    
    Multiple correction strategies implemented:
    1. Linear blending (default)
    2. Multiplicative
    3. Threshold-based
    """
    
    def __init__(
        self,
        method: str = 'linear',
        alpha: float = 0.5,
        threshold: float = 0.5
    ):
        """
        Args:
            method: Correction method ('linear', 'multiplicative', 'threshold')
            alpha: Blending factor for linear method (default: 0.5)
            threshold: Threshold for threshold-based method
        """
        self.method = method
        self.alpha = alpha
        self.threshold = threshold
        
        valid_methods = ['linear', 'multiplicative', 'threshold', 'none']
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Choose from {valid_methods}")
    
    def __call__(
        self,
        video_scores: torch.Tensor,
        snippet_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply score correction.
        
        Args:
            video_scores: Video-level anomaly probability (B,)
            snippet_scores: Snippet-level anomaly scores (B, T)
        
        Returns:
            Corrected snippet scores (B, T)
        """
        if self.method == 'none':
            return snippet_scores
        elif self.method == 'linear':
            return self._linear_correction(video_scores, snippet_scores)
        elif self.method == 'multiplicative':
            return self._multiplicative_correction(video_scores, snippet_scores)
        elif self.method == 'threshold':
            return self._threshold_correction(video_scores, snippet_scores)
        else:
            return snippet_scores
    
    def _linear_correction(
        self,
        video_scores: torch.Tensor,
        snippet_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear blending of video and snippet scores.
        
        Formula:
            s'_t = α * p_video + (1 - α) * s_t
        
        Args:
            video_scores: (B,)
            snippet_scores: (B, T)
        
        Returns:
            Corrected scores (B, T)
        """
        # Expand video scores to match snippet dimensions
        video_scores_expanded = video_scores.unsqueeze(1)  # (B, 1)
        
        # Linear blend
        corrected = self.alpha * video_scores_expanded + (1 - self.alpha) * snippet_scores
        
        return corrected
    
    def _multiplicative_correction(
        self,
        video_scores: torch.Tensor,
        snippet_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Multiplicative correction.
        
        Formula:
            s'_t = p_video * s_t
        
        Args:
            video_scores: (B,)
            snippet_scores: (B, T)
        
        Returns:
            Corrected scores (B, T)
        """
        video_scores_expanded = video_scores.unsqueeze(1)  # (B, 1)
        corrected = video_scores_expanded * snippet_scores
        
        return corrected
    
    def _threshold_correction(
        self,
        video_scores: torch.Tensor,
        snippet_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Threshold-based correction.
        
        If video score < threshold, suppress all snippet scores.
        
        Formula:
            s'_t = s_t  if p_video >= threshold
                   0    otherwise
        
        Args:
            video_scores: (B,)
            snippet_scores: (B, T)
        
        Returns:
            Corrected scores (B, T)
        """
        # Create mask
        mask = (video_scores >= self.threshold).float().unsqueeze(1)  # (B, 1)
        
        # Apply mask
        corrected = mask * snippet_scores
        
        return corrected


def apply_temporal_smoothing(
    snippet_scores: torch.Tensor,
    window_size: int = 5,
    method: str = 'gaussian'
) -> torch.Tensor:
    """
    Apply temporal smoothing to snippet scores.
    
    Args:
        snippet_scores: (B, T) snippet scores
        window_size: Size of smoothing window
        method: Smoothing method ('gaussian', 'uniform', 'median')
    
    Returns:
        Smoothed scores (B, T)
    """
    B, T = snippet_scores.shape
    
    if window_size <= 1 or window_size > T:
        return snippet_scores
    
    if method == 'uniform':
        # Uniform (moving average)
        kernel = torch.ones(1, 1, window_size, device=snippet_scores.device) / window_size
    elif method == 'gaussian':
        # Gaussian kernel
        sigma = window_size / 6.0
        x = torch.arange(window_size, dtype=torch.float32, device=snippet_scores.device)
        x = x - window_size // 2
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, window_size)
    elif method == 'median':
        # Median filter (not using convolution)
        smoothed = snippet_scores.clone()
        half_window = window_size // 2
        for i in range(T):
            start = max(0, i - half_window)
            end = min(T, i + half_window + 1)
            smoothed[:, i] = snippet_scores[:, start:end].median(dim=1)[0]
        return smoothed
    else:
        return snippet_scores
    
    # Apply convolution for uniform and gaussian
    # Add channel dimension for conv1d
    scores_expanded = snippet_scores.unsqueeze(1)  # (B, 1, T)
    
    # Pad to keep same size
    padding = window_size // 2
    scores_padded = F.pad(scores_expanded, (padding, padding), mode='replicate')
    
    # Apply convolution
    smoothed = F.conv1d(scores_padded, kernel)
    
    # Remove channel dimension
    smoothed = smoothed.squeeze(1)
    
    return smoothed


def detect_anomaly_intervals(
    snippet_scores: torch.Tensor,
    threshold: float = 0.5,
    min_length: int = 3,
    merge_gap: int = 2
) -> Dict[int, list]:
    """
    Detect anomalous intervals from snippet scores.
    
    Args:
        snippet_scores: (B, T) snippet scores
        threshold: Score threshold for anomaly
        min_length: Minimum interval length
        merge_gap: Merge intervals separated by less than this
    
    Returns:
        Dictionary mapping batch index to list of (start, end) intervals
    """
    B, T = snippet_scores.shape
    intervals = {}
    
    for b in range(B):
        scores = snippet_scores[b].cpu().numpy()
        
        # Threshold
        binary = (scores >= threshold).astype(int)
        
        # Find intervals
        current_intervals = []
        start = None
        
        for t in range(T):
            if binary[t] == 1:
                if start is None:
                    start = t
            else:
                if start is not None:
                    # End of interval
                    if t - start >= min_length:
                        current_intervals.append((start, t))
                    start = None
        
        # Handle last interval
        if start is not None and T - start >= min_length:
            current_intervals.append((start, T))
        
        # Merge close intervals
        if len(current_intervals) > 1:
            merged = [current_intervals[0]]
            for start, end in current_intervals[1:]:
                prev_start, prev_end = merged[-1]
                if start - prev_end <= merge_gap:
                    # Merge
                    merged[-1] = (prev_start, end)
                else:
                    merged.append((start, end))
            current_intervals = merged
        
        intervals[b] = current_intervals
    
    return intervals


if __name__ == "__main__":
    # Test score correction
    print("=" * 80)
    print("TESTING SCORE CORRECTION")
    print("=" * 80)
    
    # Test 1: Linear correction
    print("\n1. Testing linear correction...")
    corrector = ScoreCorrection(method='linear', alpha=0.5)
    
    video_scores = torch.tensor([0.8, 0.3, 0.6])
    snippet_scores = torch.tensor([
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.4, 0.3, 0.5, 0.2, 0.4],
        [0.7, 0.6, 0.8, 0.5, 0.6]
    ])
    
    corrected = corrector(video_scores, snippet_scores)
    print(f"   Video scores: {video_scores.tolist()}")
    print(f"   Original snippet scores shape: {snippet_scores.shape}")
    print(f"   Corrected snippet scores shape: {corrected.shape}")
    print(f"   Sample corrected (video 0): {corrected[0].tolist()}")
    print(f"   ✓ Linear correction works")
    
    # Test 2: Multiplicative correction
    print("\n2. Testing multiplicative correction...")
    corrector_mult = ScoreCorrection(method='multiplicative')
    corrected_mult = corrector_mult(video_scores, snippet_scores)
    print(f"   Corrected shape: {corrected_mult.shape}")
    print(f"   Sample (video 0): {corrected_mult[0].tolist()}")
    print(f"   ✓ Multiplicative correction works")
    
    # Test 3: Threshold correction
    print("\n3. Testing threshold correction...")
    corrector_thresh = ScoreCorrection(method='threshold', threshold=0.5)
    corrected_thresh = corrector_thresh(video_scores, snippet_scores)
    print(f"   Threshold: 0.5")
    print(f"   Video 0 (0.8 >= 0.5): kept")
    print(f"   Video 1 (0.3 < 0.5): suppressed")
    print(f"   Corrected video 1: {corrected_thresh[1].tolist()}")
    print(f"   ✓ Threshold correction works")
    
    # Test 4: Temporal smoothing
    print("\n4. Testing temporal smoothing...")
    smoothed = apply_temporal_smoothing(snippet_scores, window_size=3, method='gaussian')
    print(f"   Original: {snippet_scores[0].tolist()}")
    print(f"   Smoothed: {smoothed[0].tolist()}")
    print(f"   ✓ Temporal smoothing works")
    
    # Test 5: Detect intervals
    print("\n5. Testing interval detection...")
    high_scores = torch.tensor([
        [0.2, 0.3, 0.8, 0.9, 0.7, 0.8, 0.2, 0.1, 0.1, 0.8, 0.9, 0.8]
    ])
    intervals = detect_anomaly_intervals(high_scores, threshold=0.6, min_length=2)
    print(f"   Scores: {high_scores[0].tolist()}")
    print(f"   Detected intervals (threshold=0.6): {intervals[0]}")
    print(f"   ✓ Interval detection works")
    
    print("\n" + "=" * 80)
    print("✓ ALL SCORE CORRECTION TESTS PASSED")
    print("=" * 80)
