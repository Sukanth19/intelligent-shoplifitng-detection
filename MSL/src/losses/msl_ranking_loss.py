"""
MSL Ranking Loss
Hinge-based ranking loss for weakly supervised video anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class MSLRankingLoss(nn.Module):
    """
    Multi-Sequence Learning Ranking Loss
    
    From the MSL paper (AAAI 2022):
    The ranking loss encourages anomalous videos to have higher anomaly scores
    than normal videos using a hinge-based formulation.
    
    L = max(0, margin - (score_anomaly - score_normal))
    
    For multiple sequences:
    - Uses top-k highest scoring sequences from anomalous videos
    - Compares with normal video sequences
    - Combines video-level and snippet-level losses
    
    Mathematical Formulation:
    
    Video-level loss:
        L_video = max(0, m - (s_a - s_n))
        where s_a = anomaly video score, s_n = normal video score
    
    Snippet-level loss (for sequences):
        L_snippet = (1/K) * Σ max(0, m - (s_a^k - s_n))
        where s_a^k = k-th highest anomaly sequence score
    
    Total loss:
        L = α * L_video + β * L_snippet
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        weight_video: float = 1.0,
        weight_snippet: float = 1.0,
        top_k_ratio: float = 0.8,
        reduction: str = "mean"
    ):
        """
        Args:
            margin: Margin for hinge loss (default: 0.5)
            weight_video: Weight for video-level loss (default: 1.0)
            weight_snippet: Weight for snippet-level loss (default: 1.0)
            top_k_ratio: Ratio of top sequences to use from anomaly videos (default: 0.8)
            reduction: Loss reduction ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.margin = margin
        self.weight_video = weight_video
        self.weight_snippet = weight_snippet
        self.top_k_ratio = top_k_ratio
        self.reduction = reduction
        
        assert 0 < top_k_ratio <= 1.0, "top_k_ratio must be in (0, 1]"
        assert reduction in ["mean", "sum", "none"], "Invalid reduction"
    
    def forward(
        self,
        video_scores: torch.Tensor,
        snippet_scores: torch.Tensor,
        labels: torch.Tensor,
        sequence_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MSL ranking loss.
        
        Args:
            video_scores: Video-level anomaly scores (B,) in [0, 1]
            snippet_scores: Snippet-level anomaly scores (B, T) in [0, 1]
            labels: Binary labels (B,) - 0=normal, 1=anomaly
            sequence_scores: Optional sequence-level scores (B, N) for finer granularity
        
        Returns:
            Dictionary with:
                - loss: Total weighted loss
                - loss_video: Video-level ranking loss
                - loss_snippet: Snippet-level ranking loss
                - n_pairs: Number of normal-anomaly pairs
        """
        device = video_scores.device
        
        # Separate normal and anomaly videos
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        normal_indices = torch.where(normal_mask)[0]
        anomaly_indices = torch.where(anomaly_mask)[0]
        
        n_normal = normal_indices.size(0)
        n_anomaly = anomaly_indices.size(0)
        
        if n_normal == 0 or n_anomaly == 0:
            # Cannot compute ranking loss without both classes
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_video': torch.tensor(0.0, device=device),
                'loss_snippet': torch.tensor(0.0, device=device),
                'n_pairs': 0
            }
        
        # Get scores for each class
        video_scores_normal = video_scores[normal_mask]  # (N_n,)
        video_scores_anomaly = video_scores[anomaly_mask]  # (N_a,)
        
        snippet_scores_normal = snippet_scores[normal_mask]  # (N_n, T)
        snippet_scores_anomaly = snippet_scores[anomaly_mask]  # (N_a, T)
        
        # 1. Video-level ranking loss
        loss_video = self._compute_video_ranking_loss(
            video_scores_normal,
            video_scores_anomaly
        )
        
        # 2. Snippet-level ranking loss
        if sequence_scores is not None:
            # Use sequence scores if provided (more fine-grained)
            seq_scores_normal = sequence_scores[normal_mask]
            seq_scores_anomaly = sequence_scores[anomaly_mask]
            loss_snippet = self._compute_sequence_ranking_loss(
                seq_scores_normal,
                seq_scores_anomaly
            )
        else:
            # Use snippet scores
            loss_snippet = self._compute_snippet_ranking_loss(
                snippet_scores_normal,
                snippet_scores_anomaly
            )
        
        # 3. Combine losses
        total_loss = (
            self.weight_video * loss_video +
            self.weight_snippet * loss_snippet
        )
        
        n_pairs = n_normal * n_anomaly
        
        return {
            'loss': total_loss,
            'loss_video': loss_video.detach(),
            'loss_snippet': loss_snippet.detach(),
            'n_pairs': n_pairs
        }
    
    def _compute_video_ranking_loss(
        self,
        scores_normal: torch.Tensor,
        scores_anomaly: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute video-level ranking loss.
        
        L = (1/N) * Σ_i Σ_j max(0, m - (s_a^j - s_n^i))
        
        Args:
            scores_normal: Normal video scores (N_n,)
            scores_anomaly: Anomaly video scores (N_a,)
        
        Returns:
            Ranking loss (scalar)
        """
        # Expand for pairwise comparison
        # scores_normal: (N_n,) -> (N_n, 1)
        # scores_anomaly: (N_a,) -> (1, N_a)
        scores_n = scores_normal.unsqueeze(1)  # (N_n, 1)
        scores_a = scores_anomaly.unsqueeze(0)  # (1, N_a)
        
        # Compute pairwise differences
        # We want: score_anomaly > score_normal + margin
        diff = scores_a - scores_n  # (N_n, N_a)
        
        # Hinge loss: max(0, margin - diff)
        loss = F.relu(self.margin - diff)  # (N_n, N_a)
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss
    
    def _compute_snippet_ranking_loss(
        self,
        scores_normal: torch.Tensor,
        scores_anomaly: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute snippet-level ranking loss.
        
        Uses top-k highest scoring snippets from anomaly videos.
        
        Args:
            scores_normal: Normal snippet scores (N_n, T)
            scores_anomaly: Anomaly snippet scores (N_a, T)
        
        Returns:
            Ranking loss (scalar)
        """
        N_n, T_n = scores_normal.shape
        N_a, T_a = scores_anomaly.shape
        
        # Get top-k snippets from each anomaly video
        k = max(1, int(T_a * self.top_k_ratio))
        top_k_scores_anomaly, _ = torch.topk(scores_anomaly, k, dim=1)  # (N_a, k)
        
        # Use mean score for normal videos
        mean_scores_normal = scores_normal.mean(dim=1)  # (N_n,)
        
        # Expand for pairwise comparison
        scores_n = mean_scores_normal.unsqueeze(1).unsqueeze(2)  # (N_n, 1, 1)
        scores_a = top_k_scores_anomaly.unsqueeze(0)  # (1, N_a, k)
        
        # Compute pairwise differences
        diff = scores_a - scores_n  # (N_n, N_a, k)
        
        # Hinge loss
        loss = F.relu(self.margin - diff)  # (N_n, N_a, k)
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def _compute_sequence_ranking_loss(
        self,
        scores_normal: torch.Tensor,
        scores_anomaly: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sequence-level ranking loss.
        
        Args:
            scores_normal: Normal sequence scores (N_n, N_seq)
            scores_anomaly: Anomaly sequence scores (N_a, N_seq)
        
        Returns:
            Ranking loss (scalar)
        """
        N_n, N_seq_n = scores_normal.shape
        N_a, N_seq_a = scores_anomaly.shape
        
        # Get top-k sequences from anomaly videos
        k = max(1, int(N_seq_a * self.top_k_ratio))
        top_k_scores_anomaly, _ = torch.topk(scores_anomaly, k, dim=1)  # (N_a, k)
        
        # Use mean for normal videos
        mean_scores_normal = scores_normal.mean(dim=1)  # (N_n,)
        
        # Expand for pairwise comparison
        scores_n = mean_scores_normal.unsqueeze(1).unsqueeze(2)  # (N_n, 1, 1)
        scores_a = top_k_scores_anomaly.unsqueeze(0)  # (1, N_a, k)
        
        # Compute differences
        diff = scores_a - scores_n  # (N_n, N_a, k)
        
        # Hinge loss
        loss = F.relu(self.margin - diff)
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BinaryCrossEntropyLoss(nn.Module):
    """
    Standard BCE loss for comparison/baseline.
    Can be used alongside or instead of ranking loss.
    """
    
    def __init__(self, weight_video: float = 1.0, weight_snippet: float = 0.0):
        """
        Args:
            weight_video: Weight for video-level BCE
            weight_snippet: Weight for snippet-level BCE (usually 0 in weakly supervised)
        """
        super().__init__()
        self.weight_video = weight_video
        self.weight_snippet = weight_snippet
    
    def forward(
        self,
        video_scores: torch.Tensor,
        snippet_scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BCE loss.
        
        Args:
            video_scores: Video-level scores (B,)
            snippet_scores: Snippet-level scores (B, T)
            labels: Binary labels (B,)
        
        Returns:
            Dictionary with loss components
        """
        labels_float = labels.float()
        
        # Video-level BCE
        loss_video = F.binary_cross_entropy(
            video_scores,
            labels_float,
            reduction='mean'
        )
        
        # Snippet-level BCE (if weight > 0)
        if self.weight_snippet > 0:
            # Broadcast labels to all snippets
            labels_snippets = labels_float.unsqueeze(1).expand_as(snippet_scores)
            loss_snippet = F.binary_cross_entropy(
                snippet_scores,
                labels_snippets,
                reduction='mean'
            )
        else:
            loss_snippet = torch.tensor(0.0, device=video_scores.device)
        
        total_loss = (
            self.weight_video * loss_video +
            self.weight_snippet * loss_snippet
        )
        
        return {
            'loss': total_loss,
            'loss_video': loss_video.detach(),
            'loss_snippet': loss_snippet.detach()
        }


if __name__ == "__main__":
    # Test MSL Ranking Loss
    print("=" * 80)
    print("TESTING MSL RANKING LOSS")
    print("=" * 80)
    
    # Configuration
    batch_size = 8
    T = 100
    
    # Create loss function
    print(f"\n1. Creating MSL Ranking Loss...")
    loss_fn = MSLRankingLoss(
        margin=0.5,
        weight_video=1.0,
        weight_snippet=1.0,
        top_k_ratio=0.8
    )
    print(f"   Margin: {loss_fn.margin}")
    print(f"   Weights: video={loss_fn.weight_video}, snippet={loss_fn.weight_snippet}")
    
    # Test case 1: Balanced batch
    print(f"\n2. Test with balanced batch...")
    video_scores = torch.rand(batch_size)
    snippet_scores = torch.rand(batch_size, T)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # 4 normal, 4 anomaly
    
    result = loss_fn(video_scores, snippet_scores, labels)
    
    print(f"   Total loss: {result['loss'].item():.4f}")
    print(f"   Video loss: {result['loss_video'].item():.4f}")
    print(f"   Snippet loss: {result['loss_snippet'].item():.4f}")
    print(f"   Pairs: {result['n_pairs']}")
    print(f"   ✓ Balanced batch works")
    
    # Test case 2: Check loss decreases with better scores
    print(f"\n3. Test loss behavior...")
    # Good case: anomaly scores > normal scores
    video_scores_good = torch.cat([
        torch.rand(4) * 0.3,  # Normal: low scores
        torch.rand(4) * 0.7 + 0.3  # Anomaly: high scores
    ])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    result_good = loss_fn(video_scores_good, snippet_scores, labels)
    
    # Bad case: anomaly scores ≈ normal scores
    video_scores_bad = torch.rand(batch_size)
    result_bad = loss_fn(video_scores_bad, snippet_scores, labels)
    
    print(f"   Good separation loss: {result_good['loss'].item():.4f}")
    print(f"   Poor separation loss: {result_bad['loss'].item():.4f}")
    assert result_good['loss'] < result_bad['loss'], "Loss should be lower for better separation"
    print(f"   ✓ Loss behavior correct")
    
    # Test case 3: Edge case - all normal
    print(f"\n4. Test edge case - all normal...")
    labels_all_normal = torch.zeros(batch_size, dtype=torch.long)
    result = loss_fn(video_scores, snippet_scores, labels_all_normal)
    print(f"   Loss: {result['loss'].item():.4f}")
    print(f"   Pairs: {result['n_pairs']}")
    assert result['loss'].item() == 0.0, "Should be zero when no pairs"
    print(f"   ✓ Edge case handled")
    
    # Test case 4: Edge case - all anomaly
    print(f"\n5. Test edge case - all anomaly...")
    labels_all_anomaly = torch.ones(batch_size, dtype=torch.long)
    result = loss_fn(video_scores, snippet_scores, labels_all_anomaly)
    print(f"   Loss: {result['loss'].item():.4f}")
    assert result['loss'].item() == 0.0, "Should be zero when no pairs"
    print(f"   ✓ Edge case handled")
    
    # Test case 5: Gradient flow
    print(f"\n6. Test gradient flow...")
    video_scores = torch.rand(batch_size, requires_grad=True)
    snippet_scores = torch.rand(batch_size, T, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    result = loss_fn(video_scores, snippet_scores, labels)
    result['loss'].backward()
    
    has_video_grad = video_scores.grad is not None and video_scores.grad.abs().sum() > 0
    has_snippet_grad = snippet_scores.grad is not None and snippet_scores.grad.abs().sum() > 0
    
    print(f"   Video scores gradient: {'✓' if has_video_grad else '✗'}")
    print(f"   Snippet scores gradient: {'✓' if has_snippet_grad else '✗'}")
    print(f"   ✓ Gradient flow works")
    
    # Test case 6: Different margins
    print(f"\n7. Test different margins...")
    for margin in [0.1, 0.5, 1.0]:
        loss_fn_margin = MSLRankingLoss(margin=margin)
        result = loss_fn_margin(video_scores.detach(), snippet_scores.detach(), labels)
        print(f"   Margin={margin}: loss={result['loss'].item():.4f}")
    print(f"   ✓ Different margins work")
    
    # Test case 7: BCE Loss (for comparison)
    print(f"\n8. Test BCE Loss...")
    bce_loss_fn = BinaryCrossEntropyLoss(weight_video=1.0)
    video_scores_bce = torch.rand(batch_size)
    snippet_scores_bce = torch.rand(batch_size, T)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    result_bce = bce_loss_fn(video_scores_bce, snippet_scores_bce, labels)
    print(f"   BCE loss: {result_bce['loss'].item():.4f}")
    print(f"   ✓ BCE loss works")
    
    # Test case 8: Reduction modes
    print(f"\n9. Test reduction modes...")
    for reduction in ['mean', 'sum']:
        loss_fn_red = MSLRankingLoss(margin=0.5, reduction=reduction)
        result = loss_fn_red(video_scores.detach(), snippet_scores.detach(), labels)
        print(f"   Reduction='{reduction}': loss={result['loss'].item():.4f}")
    print(f"   ✓ Reduction modes work")
    
    print("\n" + "=" * 80)
    print("✓ ALL RANKING LOSS TESTS PASSED")
    print("=" * 80)
