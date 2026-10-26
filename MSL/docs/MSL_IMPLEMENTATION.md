# MSL Implementation Details

**Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection**

This document describes the complete implementation of the MSL methodology from the AAAI 2022 paper.

## Overview

The MSL method addresses video anomaly detection in a weakly supervised setting where only video-level labels (normal/anomaly) are available, without temporal annotations indicating when anomalies occur.

### Key Innovation

**Multi-Sequence Learning**: Instead of processing the entire video as one sequence, MSL extracts multiple overlapping temporal sequences, enabling:
- Better temporal localization of anomalies
- Progressive refinement through self-training
- More robust learning from weak supervision

---

## Architecture

### Complete Pipeline

```
Input Video Features (T × D)
         ↓
   Input Projection
         ↓
Multi-Sequence Learning
  - Extract N sequences of length L
  - Sliding window with stride S
         ↓
Transformer Encoder
  - Multi-head self-attention
  - Positional encoding
  - Process each sequence
         ↓
    Aggregation
         ↓
  ┌──────┴──────┐
  │             │
Video-level   Snippet-level
Prediction    Predictions
  │             │
  ├─ Logits    ├─ Scores
  └─ Scores    └─ Per snippet
```

### Components

#### 1. Input Projection
**Input**: VideoSwin features (T, 1024)  
**Output**: Projected features (T, 512)

```python
self.input_projection = nn.Sequential(
    nn.Linear(1024, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Dropout(0.1)
)
```

#### 2. Multi-Sequence Learning
**Purpose**: Extract overlapping temporal sequences

**Parameters**:
- `sequence_length` (L): Snippets per sequence (default: 32)
- `stride` (S): Overlap between sequences (default: 16)

**Formula**:
```
Number of sequences: N = floor((T - L) / S) + 1
```

**Output**: (B, N, L, D) tensor of sequences

#### 3. Transformer Encoder
**Architecture**:
- Layers: 3
- Hidden dimension: 512
- Attention heads: 8
- Feedforward dimension: 2048

**Components**:
- Multi-head self-attention
- Positional encoding (sinusoidal)
- Layer normalization
- Residual connections

**Formula** (single layer):
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

x' = LayerNorm(x + MultiHead(x))
x'' = LayerNorm(x' + FFN(x'))
```

#### 4. Aggregation
Three methods implemented:

**Mean Aggregation**:
```
v = (1/N) Σ_{i=1}^N (1/L) Σ_{j=1}^L h_{ij}
```

**Max Aggregation**:
```
v = max_{i,j} h_{ij}
```

**Attention Aggregation**:
```
α_i = softmax(W * h_i)
v = Σ_{i=1}^N α_i * h_i
```

#### 5. Prediction Heads

**Video-level**:
```python
video_classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 2)  # Binary classification
)
```

**Snippet-level**:
```python
snippet_classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)  # Anomaly score
)
```

---

## Loss Function

### MSL Ranking Loss

The core training objective that encourages anomalous videos to receive higher scores than normal videos.

#### Mathematical Formulation

**Video-level Ranking Loss**:
```
L_video = (1/|N||A|) Σ_{i∈N} Σ_{j∈A} max(0, m - (s_j^a - s_i^n))
```

where:
- N: Set of normal videos
- A: Set of anomaly videos
- s_i^n: Score for normal video i
- s_j^a: Score for anomaly video j
- m: Margin (default: 0.5)

**Snippet-level Ranking Loss**:
```
L_snippet = (1/|N||A|K) Σ_{i∈N} Σ_{j∈A} Σ_{k=1}^K max(0, m - (s_jk^a - s̄_i^n))
```

where:
- K: Number of top-k sequences from anomaly videos (K = ⌊top_k_ratio × N_seq⌋)
- s_jk^a: k-th highest score from anomaly video j
- s̄_i^n: Mean score of normal video i

**Total Loss**:
```
L_total = α * L_video + β * L_snippet
```

Default weights: α = β = 1.0

#### Implementation

```python
class MSLRankingLoss(nn.Module):
    def forward(self, video_scores, snippet_scores, labels):
        # Separate normal and anomaly
        normal_mask = (labels == 0)
        anomaly_mask = (labels == 1)
        
        # Video-level ranking
        diff_video = scores_anomaly - scores_normal
        loss_video = max(0, margin - diff_video)
        
        # Snippet-level ranking (top-k from anomaly)
        top_k_anomaly = topk(scores_anomaly, k)
        diff_snippet = top_k_anomaly - mean(scores_normal)
        loss_snippet = max(0, margin - diff_snippet)
        
        return α * loss_video + β * loss_snippet
```

#### Properties

1. **Differentiable**: Hinge loss is piecewise differentiable
2. **Pairwise**: Compares all normal-anomaly pairs
3. **Margin-based**: Enforces minimum separation
4. **Top-k**: Focuses on most anomalous sequences

---

## Self-Training

Progressive refinement of temporal localization through multiple training stages.

### Concept

Start with longer sequences (easier to detect anomalies) and progressively shorten them (better localization).

### Stages

**Stage 1** (Initial):
```yaml
sequence_length: 64
stride: 32
epochs: 15
```
- Easier anomaly detection
- Coarse temporal localization

**Stage 2** (Refinement):
```yaml
sequence_length: 48
stride: 24
epochs: 15
```
- Balanced detection and localization

**Stage 3** (Fine-tuning):
```yaml
sequence_length: 32
stride: 16
epochs: 20
```
- Fine-grained temporal localization
- Most challenging

### Algorithm

```
1. Initialize model M
2. For each stage s:
    a. Update sequence parameters (L_s, S_s)
    b. For epoch e in 1..E_s:
        i. Forward pass with current sequence length
        ii. Compute MSL ranking loss
        iii. Backpropagation
        iv. Update parameters
    c. Save checkpoint
3. Return final model
```

### Implementation

```python
# Stage 1
model = MSLModel(sequence_length=64, stride=32)
train(model, epochs=15)

# Stage 2
model.update_sequence_params(sequence_length=48, stride=24)
train(model, epochs=15)

# Stage 3
model.update_sequence_params(sequence_length=32, stride=16)
train(model, epochs=20)
```

---

## Score Correction

Post-processing to stabilize snippet-level scores using video-level probability.

### Motivation

- Video-level predictions are generally more reliable (trained on labels)
- Snippet-level scores may be noisy (no snippet-level supervision)
- Use video-level confidence to correct snippet scores

### Formula

```
s'_t = α * p_video + (1 - α) * s_t

where:
- s'_t: Corrected snippet score at time t
- p_video: Video-level anomaly probability
- s_t: Original snippet score
- α: Blending factor (default: 0.5)
```

### Alternative Methods

**Multiplicative**:
```
s'_t = p_video * s_t
```

**Threshold-based**:
```
s'_t = s_t  if p_video > threshold
       0    otherwise
```

### Implementation

```python
def score_correction(video_prob, snippet_scores, alpha=0.5):
    """
    Args:
        video_prob: (B,) video-level probability
        snippet_scores: (B, T) snippet scores
        alpha: blending factor
    
    Returns:
        corrected_scores: (B, T)
    """
    # Broadcast video probability to all snippets
    video_prob_expanded = video_prob.unsqueeze(1)  # (B, 1)
    
    # Blend
    corrected = alpha * video_prob_expanded + (1 - alpha) * snippet_scores
    
    return corrected
```

---

## Training Procedure

### Complete Training Loop

```python
def train_msl(model, train_loader, val_loader, config):
    """
    Complete MSL training with self-training stages
    """
    optimizer = Adam(model.parameters(), lr=config.lr)
    loss_fn = MSLRankingLoss(margin=config.margin)
    
    for stage in config.self_training_stages:
        # Update sequence parameters
        model.update_sequence_params(
            stage.sequence_length,
            stage.stride
        )
        
        print(f"Stage: {stage.name}")
        print(f"  Sequence length: {stage.sequence_length}")
        print(f"  Epochs: {stage.epochs}")
        
        for epoch in range(stage.epochs):
            # Training
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                features = batch['features']  # (B, T, D)
                labels = batch['labels']      # (B,)
                mask = batch['mask']          # (B, T)
                
                # Forward pass
                output = model(features, mask)
                
                # Compute loss
                loss_dict = loss_fn(
                    output['video_scores'],
                    output['snippet_scores'],
                    labels
                )
                loss = loss_dict['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss, val_auc = validate(model, val_loader, loss_fn)
            
            print(f"Epoch {epoch}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"val_auc={val_auc:.4f}")
            
            # Save checkpoint
            if val_auc > best_auc:
                save_checkpoint(model, f"best_{stage.name}.pt")
    
    return model
```

### Hyperparameters

**Model**:
- Input dim: 1024 (VideoSwin-Base)
- Hidden dim: 512
- Transformer layers: 3
- Attention heads: 8
- Dropout: 0.1

**Training**:
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 5e-5
- Batch size: 16
- Gradient clipping: 1.0

**Loss**:
- Margin: 0.5
- Video weight: 1.0
- Snippet weight: 1.0
- Top-k ratio: 0.8

**Self-training**:
- Stage 1: L=64, S=32, 15 epochs
- Stage 2: L=48, S=24, 15 epochs
- Stage 3: L=32, S=16, 20 epochs

---

## Evaluation

### Metrics

**Video-level**:
- AUC-ROC: Area under ROC curve
- Average Precision (AP)
- Accuracy at threshold

**Snippet-level**:
- Frame-level AUC (if annotations available)
- Temporal IoU with ground truth

### Evaluation Procedure

```python
def evaluate(model, test_loader):
    model.eval()
    
    all_video_scores = []
    all_snippet_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch['features'], batch['mask'])
            
            all_video_scores.append(output['video_scores'])
            all_snippet_scores.append(output['snippet_scores'])
            all_labels.append(batch['labels'])
    
    # Concatenate
    video_scores = torch.cat(all_video_scores)
    labels = torch.cat(all_labels)
    
    # Compute metrics
    auc = roc_auc_score(labels.cpu(), video_scores.cpu())
    ap = average_precision_score(labels.cpu(), video_scores.cpu())
    
    return {
        'auc': auc,
        'ap': ap,
        'video_scores': video_scores,
        'snippet_scores': all_snippet_scores,
        'labels': labels
    }
```

---

## Implementation Assumptions

All assumptions are documented here for transparency.

### Assumption 1: Transformer Configuration
**Assumption**: 3 layers, 8 heads, 512 hidden dim  
**Source**: Not specified in paper  
**Rationale**: Standard Transformer configuration for this scale  
**Impact**: Model capacity and training time  
**Risk**: Low - can be tuned via config

### Assumption 2: Aggregation Method
**Assumption**: Mean aggregation as default  
**Source**: Not specified in paper  
**Rationale**: Simple and effective  
**Impact**: Video-level representation  
**Risk**: Low - implemented multiple methods  
**Documented in**: `configs/default.yaml` as `aggregation: mean`

### Assumption 3: Top-k Ratio
**Assumption**: 80% of sequences (top-k ratio = 0.8)  
**Source**: Not specified in paper  
**Rationale**: Focus on most anomalous sequences while avoiding outliers  
**Impact**: Snippet-level loss computation  
**Risk**: Medium - affects learning  
**Documented in**: `configs/default.yaml` as `top_k_ratio: 0.8`

### Assumption 4: Score Correction Formula
**Assumption**: Linear blending with α=0.5  
**Source**: Concept in paper, formula not specified  
**Rationale**: Simple weighted average  
**Impact**: Final snippet scores  
**Risk**: Medium - affects localization  
**Documented in**: `configs/default.yaml` as `score_correction.alpha: 0.5`

### Assumption 5: Self-training Stages
**Assumption**: 3 stages with L={64, 48, 32}  
**Source**: Progressive refinement concept from paper, exact values not specified  
**Rationale**: Gradual reduction for smooth transition  
**Impact**: Training schedule  
**Risk**: Medium - affects convergence  
**Documented in**: `configs/default.yaml` under `self_training.stages`

---

## Differences from Paper

### What Matches the Paper
- ✅ Multi-Sequence Learning concept
- ✅ Ranking loss formulation
- ✅ Self-training progressive refinement
- ✅ Video and snippet-level predictions
- ✅ Weakly supervised setting
- ✅ Transformer-based encoding

### Implementation Details
- Specific network architectures (heads, layers, dimensions)
- Exact hyperparameter values
- Aggregation method choice
- Top-k selection strategy
- Score correction formula

These are standard choices where the paper doesn't provide specific values. All are configurable and documented.

---

## References

1. **MSL Paper**: Li, S., Liu, F., & Jiao, L. (2022). Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection. AAAI 2022.
   - DOI: 10.1609/aaai.v36i2.20028
   - URL: https://ojs.aaai.org/index.php/AAAI/article/view/20028

2. **Transformer**: Vaswani, A., et al. (2017). Attention is all you need. NeurIPS 2017.

3. **VideoSwin**: Liu, Z., et al. (2022). Video Swin Transformer. CVPR 2022.
   - Repository: https://github.com/SwinTransformer/Video-Swin-Transformer

4. **UCF-Crime**: Sultani, W., Chen, C., & Shah, M. (2018). Real-world anomaly detection in surveillance videos. CVPR 2018.

---

**Document Version**: 1.0  
**Last Updated**: Phase 5 Implementation  
**Status**: Complete MSL methodology documented
