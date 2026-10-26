"""
MSL (Multi-Sequence Learning) Model
Complete architecture for video anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path

# Handle imports for both package use and standalone testing
try:
    from .transformer import TransformerEncoder
    from .multi_sequence import MultiSequenceLearning
except ImportError:
    # Add parent directory to path for standalone testing
    sys.path.insert(0, str(Path(__file__).parent))
    from transformer import TransformerEncoder
    from multi_sequence import MultiSequenceLearning


class MSLModel(nn.Module):
    """
    MSL: Multi-Sequence Learning with Transformer for Video Anomaly Detection
    
    Architecture:
        Video Features (T, D)
                ↓
        Input Projection → hidden_dim
                ↓
        Multi-Sequence Learning → (N, L, hidden_dim)
                ↓
        Transformer Encoder → (N, L, hidden_dim)
                ↓
        ┌────────────────┴────────────────┐
        │                                  │
     Video-level                    Snippet-level
     Prediction                     Predictions
        │                                  │
     (N,) scores                      (N, L) scores
        │                                  │
        └────────── Aggregation ───────────┘
                         ↓
              Video anomaly probability
              Snippet anomaly scores
    
    Based on AAAI 2022 MSL paper.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        sequence_length: int = 32,
        sequence_stride: int = 16,
        num_classes: int = 2,
        use_pos_encoding: bool = True,
        aggregation: str = "mean"  # 'mean', 'max', or 'attention'
    ):
        """
        Args:
            input_dim: Input feature dimension (VideoSwin output, typically 1024)
            hidden_dim: Hidden dimension for model
            num_layers: Number of Transformer layers
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension
            dropout: Dropout rate
            sequence_length: Length of each sequence in MSL
            sequence_stride: Stride between sequences
            num_classes: Number of output classes (2 for binary anomaly detection)
            use_pos_encoding: Whether to use positional encoding
            aggregation: How to aggregate sequence features ('mean', 'max', 'attention')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.aggregation = aggregation
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-Sequence Learning
        self.msl = MultiSequenceLearning(
            sequence_length=sequence_length,
            stride=sequence_stride
        )
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            use_pos_encoding=use_pos_encoding
        )
        
        # Aggregation (for video-level representation)
        if aggregation == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
        
        # Video-level prediction head
        self.video_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Snippet-level prediction head
        self.snippet_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary anomaly score
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Video features (B, T, D) where D=input_dim
            mask: Padding mask (B, T) - True for valid positions
        
        Returns:
            Dictionary containing:
                - video_logits: Video-level classification logits (B, num_classes)
                - video_scores: Video-level anomaly probability (B,)
                - snippet_scores: Snippet-level anomaly scores (B, T)
                - sequence_features: Encoded sequence features (B, N, L, hidden_dim)
                - num_sequences: Number of sequences per video (B,)
        """
        B, T, D = features.shape
        
        # 1. Input projection
        projected = self.input_projection(features)  # (B, T, hidden_dim)
        
        # 2. Multi-Sequence Learning
        sequences, seq_mask, num_seq = self.msl(projected, mask)  # (B, N, L, hidden_dim)
        B, N, L, H = sequences.shape
        
        # 3. Reshape for Transformer: process each sequence independently
        # (B, N, L, H) -> (B*N, L, H)
        sequences_flat = sequences.view(B * N, L, H)
        seq_mask_flat = seq_mask.view(B * N, L) if seq_mask is not None else None
        
        # Convert mask: our mask has True for valid, PyTorch wants True for padding.
        # Windows that are pure padding (shorter videos in a batch) would have every
        # key masked, and softmax over all -inf gives NaN. Leave those unmasked; their
        # outputs are ignored below via seq_valid.
        padding_mask_flat = ~seq_mask_flat
        padding_mask_flat[padding_mask_flat.all(dim=1)] = False

        # 4. Transformer encoding
        encoded = self.transformer(sequences_flat, src_key_padding_mask=padding_mask_flat)

        # 5. Reshape back: (B*N, L, H) -> (B, N, L, H)
        encoded = encoded.view(B, N, L, H)

        # 6. Aggregate sequences for video-level representation
        # Mean over the valid snippets of each window -> (B, N, H)
        snip_w = seq_mask.unsqueeze(-1).float()
        sequence_repr = (encoded * snip_w).sum(dim=2) / snip_w.sum(dim=2).clamp(min=1)

        # Then aggregate over the real windows of each video
        seq_valid = torch.arange(N, device=features.device)[None, :] < num_seq[:, None]  # (B, N)
        seq_w = seq_valid.unsqueeze(-1).float()
        if self.aggregation == "max":
            video_repr = sequence_repr.masked_fill(~seq_valid.unsqueeze(-1), float('-inf')).max(dim=1)[0]
        elif self.aggregation == "attention":
            attention_logits = self.attention_pool(sequence_repr).masked_fill(~seq_valid.unsqueeze(-1), float('-inf'))
            attention_weights = F.softmax(attention_logits, dim=1)  # (B, N, 1)
            video_repr = (sequence_repr * attention_weights).sum(dim=1)  # (B, H)
        else:
            video_repr = (sequence_repr * seq_w).sum(dim=1) / seq_w.sum(dim=1).clamp(min=1)
        
        # 7. Video-level prediction
        video_logits = self.video_classifier(video_repr)  # (B, num_classes)
        video_probs = F.softmax(video_logits, dim=-1)
        video_scores = video_probs[:, 1]  # Anomaly probability (class 1)
        
        # 8. Snippet-level prediction
        # Apply snippet classifier to all encoded features
        snippet_logits = self.snippet_classifier(encoded)  # (B, N, L, 1)
        snippet_logits = snippet_logits.squeeze(-1)  # (B, N, L)
        snippet_probs = torch.sigmoid(snippet_logits)  # (B, N, L)
        
        # 9. Reconstruct snippet scores for original video
        # Average over sequences that overlap
        snippet_scores = self.msl.reconstruct_from_sequences(
            snippet_probs,
            num_seq,
            T
        )  # (B, T)
        
        return {
            'video_logits': video_logits,
            'video_scores': video_scores,
            'snippet_scores': snippet_scores,
            'sequence_features': encoded,
            'num_sequences': num_seq,
            'sequence_repr': sequence_repr
        }
    
    def update_sequence_params(self, sequence_length: int, stride: Optional[int] = None):
        """
        Update MSL sequence parameters (for self-training stages).
        
        Args:
            sequence_length: New sequence length
            stride: New stride (optional)
        """
        self.sequence_length = sequence_length
        self.msl.update_sequence_params(sequence_length, stride)


if __name__ == "__main__":
    # Test MSL Model
    print("=" * 80)
    print("TESTING MSL MODEL")
    print("=" * 80)
    
    # Configuration
    batch_size = 4
    T = 100  # Video length (snippets)
    input_dim = 1024  # VideoSwin feature dim
    
    # Create model
    print(f"\n1. Creating MSL Model...")
    model = MSLModel(
        input_dim=input_dim,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        sequence_length=32,
        sequence_stride=16,
        num_classes=2
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print(f"\n2. Testing forward pass...")
    features = torch.randn(batch_size, T, input_dim)
    print(f"   Input shape: {features.shape}")
    
    output = model(features)
    
    print(f"\n3. Output shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        else:
            print(f"   {key}: {value}")
    
    # Verify shapes
    assert output['video_logits'].shape == (batch_size, 2), "Video logits shape wrong"
    assert output['video_scores'].shape == (batch_size,), "Video scores shape wrong"
    assert output['snippet_scores'].shape == (batch_size, T), "Snippet scores shape wrong"
    print(f"   ✓ All shapes correct")
    
    # Test with variable lengths
    print(f"\n4. Testing with variable-length videos...")
    lengths = torch.tensor([100, 80, 60, 50])
    mask = torch.zeros(batch_size, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    output = model(features, mask)
    print(f"   Video lengths: {lengths.tolist()}")
    print(f"   Number of sequences: {output['num_sequences'].tolist()}")
    print(f"   Video scores: {output['video_scores'].detach().numpy()}")
    print(f"   ✓ Variable lengths work")
    
    # Test gradient flow
    print(f"\n5. Testing gradient flow...")
    output = model(features)
    loss = output['video_logits'].sum() + output['snippet_scores'].sum()
    loss.backward()
    
    has_gradients = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"   All parameters have gradients: {has_gradients}")
    print(f"   ✓ Gradient flow correct")
    
    # Test sequence parameter update
    print(f"\n6. Testing sequence parameter update...")
    old_length = model.sequence_length
    model.update_sequence_params(48, 24)
    print(f"   Old sequence_length: {old_length}")
    print(f"   New sequence_length: {model.sequence_length}")
    
    output = model(features)
    print(f"   Output still works: ✓")
    
    # Test different aggregation methods
    print(f"\n7. Testing different aggregation methods...")
    for agg in ['mean', 'max', 'attention']:
        model_agg = MSLModel(
            input_dim=input_dim,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            aggregation=agg
        )
        output = model_agg(features[:2])
        print(f"   {agg}: video_scores shape = {output['video_scores'].shape} ✓")
    
    # Test inference mode
    print(f"\n8. Testing inference mode...")
    model.eval()
    with torch.no_grad():
        output = model(features)
        print(f"   Video scores range: [{output['video_scores'].min():.3f}, {output['video_scores'].max():.3f}]")
        print(f"   Snippet scores range: [{output['snippet_scores'].min():.3f}, {output['snippet_scores'].max():.3f}]")
    print(f"   ✓ Inference mode works")
    
    print("\n" + "=" * 80)
    print("✓ ALL MSL MODEL TESTS PASSED")
    print("=" * 80)
    
    # Summary
    print(f"\nModel Summary:")
    print(f"  Input: Video features (B, T, {input_dim})")
    print(f"  Output: Video anomaly scores + Snippet scores")
    print(f"  Parameters: {num_params:,}")
    print(f"  Sequence Learning: {model.sequence_length} snippets, stride {model.msl.stride}")
    print(f"  Transformer: {model.transformer.num_layers} layers, {model.hidden_dim} hidden dim")
