"""
Multi-Sequence Learning Module
Constructs and processes multiple temporal sequences from video features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class MultiSequenceLearning(nn.Module):
    """
    Multi-Sequence Learning module for temporal anomaly detection.
    
    From the MSL paper:
    - Constructs multiple overlapping sequences from video features
    - Each sequence captures local temporal patterns
    - Progressive sequence length during self-training
    
    Implementation details:
    - Uses sliding window to create sequences
    - Configurable sequence length and stride
    - Supports variable-length videos through padding/masking
    """
    
    def __init__(
        self,
        sequence_length: int = 32,
        stride: int = 16,
        pad_mode: str = "replicate"
    ):
        """
        Args:
            sequence_length: Number of snippets per sequence
            stride: Stride between consecutive sequences
            pad_mode: Padding mode ('replicate', 'zeros', 'circular')
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.stride = stride
        self.pad_mode = pad_mode
        
        assert sequence_length > 0, "Sequence length must be positive"
        assert stride > 0, "Stride must be positive"
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract multiple sequences from video features.
        
        Args:
            features: Video features (B, T, D)
            mask: Padding mask (B, T) - True for valid positions
        
        Returns:
            sequences: Extracted sequences (B, N, L, D)
                where N = number of sequences, L = sequence_length
            sequence_mask: Mask for sequences (B, N, L)
            num_sequences: Number of sequences per video (B,)
        """
        B, T, D = features.shape
        L = self.sequence_length
        S = self.stride
        
        # If no mask provided, assume all positions are valid
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=features.device)
        
        # Number of windows per video. Uses ceil so the last window covers the
        # tail of the video (a floor here silently left the last snippets unscored).
        lengths = mask.sum(dim=1)
        num_sequences = torch.where(
            lengths > L,
            (lengths - L + S - 1) // S + 1,
            torch.ones_like(lengths)
        )
        max_num_seq = int(num_sequences.max().item())

        # Pad on the time axis so every window fits, then slice windows with unfold
        padded_len = (max_num_seq - 1) * S + L
        pad = padded_len - T
        if pad > 0:
            features = F.pad(features, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        else:
            features = features[:, :padded_len]
            mask = mask[:, :padded_len]

        sequences = features.unfold(1, L, S).permute(0, 1, 3, 2).contiguous()  # (B, N, L, D)
        sequence_mask = mask.unfold(1, L, S).contiguous()  # (B, N, L)

        return sequences, sequence_mask, num_sequences
    
    def reconstruct_from_sequences(
        self,
        sequence_scores: torch.Tensor,
        num_sequences: torch.Tensor,
        original_length: int
    ) -> torch.Tensor:
        """
        Reconstruct snippet-level scores from sequence-level scores.
        Uses averaging for overlapping regions.
        
        Args:
            sequence_scores: Scores for each sequence (B, N, L)
            num_sequences: Number of valid sequences per video (B,)
            original_length: Original video length (T)
        
        Returns:
            snippet_scores: Reconstructed scores (B, T)
        """
        B, N, L = sequence_scores.shape
        S = self.stride
        padded_len = max((N - 1) * S + L, original_length)

        snippet_scores = sequence_scores.new_zeros(B, padded_len)
        snippet_counts = sequence_scores.new_zeros(B, padded_len)

        # Only accumulate from real windows of each video
        valid = (torch.arange(N, device=sequence_scores.device)[None, :] < num_sequences[:, None]).float()
        for i in range(N):
            start = i * S
            snippet_scores[:, start:start + L] += sequence_scores[:, i] * valid[:, i:i + 1]
            snippet_counts[:, start:start + L] += valid[:, i:i + 1]

        # Average overlapping regions
        snippet_scores = snippet_scores / snippet_counts.clamp(min=1)

        return snippet_scores[:, :original_length]
    
    def update_sequence_params(self, sequence_length: int, stride: Optional[int] = None):
        """
        Update sequence parameters (useful for self-training stages).
        
        Args:
            sequence_length: New sequence length
            stride: New stride (if None, keep current or set to length//2)
        """
        self.sequence_length = sequence_length
        if stride is not None:
            self.stride = stride
        else:
            self.stride = max(1, sequence_length // 2)


if __name__ == "__main__":
    # Test Multi-Sequence Learning
    print("=" * 80)
    print("TESTING MULTI-SEQUENCE LEARNING")
    print("=" * 80)
    
    # Configuration
    batch_size = 4
    feature_dim = 1024
    sequence_length = 32
    stride = 16
    
    # Create module
    print(f"\n1. Creating MultiSequenceLearning module...")
    print(f"   sequence_length={sequence_length}, stride={stride}")
    
    msl = MultiSequenceLearning(
        sequence_length=sequence_length,
        stride=stride
    )
    
    # Test case 1: Fixed length videos
    print(f"\n2. Test with fixed-length videos...")
    T = 100
    features = torch.randn(batch_size, T, feature_dim)
    sequences, seq_mask, num_seq = msl(features)
    
    print(f"   Input shape: {features.shape}")
    print(f"   Output sequences shape: {sequences.shape}")
    print(f"   Sequence mask shape: {seq_mask.shape}")
    print(f"   Number of sequences: {num_seq.tolist()}")
    
    expected_num = ((T - sequence_length) // stride) + 1
    print(f"   Expected num sequences: {expected_num}")
    assert sequences.shape[1] >= expected_num, "Not enough sequences extracted"
    print(f"   ✓ Correct number of sequences")
    
    # Test case 2: Variable length videos
    print(f"\n3. Test with variable-length videos...")
    lengths = torch.tensor([100, 80, 60, 50])
    mask = torch.zeros(batch_size, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    sequences, seq_mask, num_seq = msl(features, mask)
    
    print(f"   Video lengths: {lengths.tolist()}")
    print(f"   Number of sequences: {num_seq.tolist()}")
    print(f"   Sequences shape: {sequences.shape}")
    print(f"   ✓ Variable lengths handled")
    
    # Test case 3: Short videos (< sequence_length)
    print(f"\n4. Test with short videos...")
    T_short = 20  # Shorter than sequence_length
    features_short = torch.randn(2, T_short, feature_dim)
    mask_short = torch.ones(2, T_short, dtype=torch.bool)
    mask_short[1, 15:] = False  # Second video has only 15 valid frames
    
    sequences, seq_mask, num_seq = msl(features_short, mask_short)
    
    print(f"   Short video length: {T_short}")
    print(f"   Number of sequences: {num_seq.tolist()}")
    print(f"   Sequences shape: {sequences.shape}")
    assert num_seq[0] >= 1, "Should have at least one sequence"
    print(f"   ✓ Short videos handled with padding")
    
    # Test case 4: Reconstruction
    print(f"\n5. Test sequence score reconstruction...")
    B = 2
    T_test = 100
    features_test = torch.randn(B, T_test, feature_dim)
    
    # Extract sequences
    sequences, seq_mask, num_seq = msl(features_test)
    N = sequences.shape[1]
    L = sequences.shape[2]
    
    # Create dummy sequence scores
    sequence_scores = torch.randn(B, N, L)
    
    # Reconstruct snippet scores
    snippet_scores = msl.reconstruct_from_sequences(sequence_scores, num_seq, T_test)
    
    print(f"   Sequence scores shape: {sequence_scores.shape}")
    print(f"   Reconstructed scores shape: {snippet_scores.shape}")
    assert snippet_scores.shape == (B, T_test), "Reconstruction shape mismatch"
    print(f"   ✓ Reconstruction works")
    
    # Test case 5: Update parameters (for self-training)
    print(f"\n6. Test parameter update...")
    old_length = msl.sequence_length
    new_length = 48
    msl.update_sequence_params(new_length)
    
    print(f"   Old sequence_length: {old_length}")
    print(f"   New sequence_length: {msl.sequence_length}")
    print(f"   New stride: {msl.stride}")
    
    # Test with new parameters
    features_new = torch.randn(2, 100, feature_dim)
    sequences, seq_mask, num_seq = msl(features_new)
    assert sequences.shape[2] == new_length, "Parameter update failed"
    print(f"   ✓ Parameter update works")
    
    print("\n" + "=" * 80)
    print("✓ ALL MULTI-SEQUENCE LEARNING TESTS PASSED")
    print("=" * 80)
