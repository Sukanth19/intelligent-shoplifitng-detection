"""
Evaluate MSL Model - Compute AUC and other metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.datasets import UCFH5Dataset, collate_fn_h5
from src.models import MSLModel
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import argparse
import matplotlib.pyplot as plt

print("="*80)
print("MSL MODEL EVALUATION")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate MSL model')
parser.add_argument('--checkpoint', type=str, default='checkpoints/fullscale_final.pt',
                    help='Path to model checkpoint')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for evaluation')
args = parser.parse_args()

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {DEVICE}")
print(f"Checkpoint: {args.checkpoint}")

# Load checkpoint
checkpoint_path = Path(args.checkpoint)
if not checkpoint_path.exists():
    print(f"\n❌ Checkpoint not found: {checkpoint_path}")
    print("\nAvailable checkpoints:")
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for ckpt in checkpoints_dir.glob("*.pt"):
            print(f"  - {ckpt}")
    sys.exit(1)

print(f"\nLoading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Load test dataset
print(f"\nLoading test dataset...")
data_dir = Path("../data/UCF-VideoSwin-Features")
h5_path = data_dir / "ucf_swin_test_feature_224288224288.h5"
list_path = data_dir / "ucf-test.list"
gt_path = data_dir / "ucf_gt_16.npy"

test_dataset = UCFH5Dataset(
    h5_path=str(h5_path),
    list_path=str(list_path),
    gt_path=str(gt_path),
    max_length=100,
    mode='test'
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Mapped videos: {len(test_dataset.video_to_key)}")

# Create test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn_h5
)

print(f"Test batches: {len(test_loader)}")

# Create model
print(f"\nCreating model...")
model = MSLModel(
    input_dim=1024,
    hidden_dim=256,
    num_layers=2,
    num_heads=4,
    sequence_length=32,
    sequence_stride=16,
    dropout=0.1,
    aggregation='mean'
).to(DEVICE)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")
print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

# Run inference
print("\n" + "="*80)
print("RUNNING INFERENCE ON TEST SET")
print("="*80)

all_scores = []
all_labels = []
all_video_names = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        try:
            features = batch['features'].to(DEVICE)
            labels = batch['label']
            masks = batch['mask'].to(DEVICE)
            video_names = batch.get('video_name', [f"video_{i}" for i in range(len(labels))])
            
            # Forward pass
            outputs = model(features, mask=masks)
            scores = outputs['video_scores'].cpu().numpy()
            
            # Check for NaN in this batch
            if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                print(f"  ⚠️  Batch {batch_idx} contains NaN/Inf scores")
            
            # Store results
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_video_names.extend(video_names)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")
                
        except Exception as e:
            print(f"  Error in batch {batch_idx}: {e}")
            # Add neutral scores for failed batch
            batch_size = len(batch['label'])
            all_scores.extend([0.5] * batch_size)
            all_labels.extend(batch['label'].numpy().tolist())
            all_video_names.extend([f"error_video_{i}" for i in range(batch_size)])
            continue

print(f"\n✅ Inference complete!")
print(f"Total videos processed: {len(all_scores)}")

# Convert to numpy arrays
all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# Check for NaN or Inf values
nan_mask = np.isnan(all_scores) | np.isinf(all_scores)
if np.any(nan_mask):
    print(f"\n⚠️  Warning: Found {np.sum(nan_mask)} videos with NaN/Inf scores")
    print(f"   Replacing with 0.5 (neutral score)")
    all_scores[nan_mask] = 0.5  # Replace with neutral score

# Clip scores to valid range [0, 1]
all_scores = np.clip(all_scores, 0.0, 1.0)
print(f"\nScore range: [{np.min(all_scores):.4f}, {np.max(all_scores):.4f}]")

# Count labels
n_normal = np.sum(all_labels == 0)
n_anomaly = np.sum(all_labels == 1)

print(f"\nLabel distribution:")
print(f"  Normal (0): {n_normal}")
print(f"  Anomaly (1): {n_anomaly}")

# Compute metrics
print("\n" + "="*80)
print("COMPUTING METRICS")
print("="*80)

if n_normal == 0 or n_anomaly == 0:
    print("\n⚠️  Warning: Only one class present in test set!")
    print("Cannot compute AUC without both classes.")
    auc = 0.0
    ap = 0.0
else:
    # AUC
    auc = roc_auc_score(all_labels, all_scores)
    print(f"\n✅ AUC (Area Under ROC Curve): {auc:.4f} ({auc*100:.2f}%)")
    
    # Average Precision
    ap = average_precision_score(all_labels, all_scores)
    print(f"✅ AP (Average Precision): {ap:.4f} ({ap*100:.2f}%)")
    
    # ROC curve for analysis
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    # Find optimal threshold (closest to top-left corner)
    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    print(f"\n📊 ROC Analysis:")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  True Positive Rate: {optimal_tpr:.4f} ({optimal_tpr*100:.2f}%)")
    print(f"  False Positive Rate: {optimal_fpr:.4f} ({optimal_fpr*100:.2f}%)")

# Score statistics
print(f"\n📊 Score Statistics:")
print(f"  Mean score: {np.mean(all_scores):.4f}")
print(f"  Std score: {np.std(all_scores):.4f}")
print(f"  Min score: {np.min(all_scores):.4f}")
print(f"  Max score: {np.max(all_scores):.4f}")

normal_scores = all_scores[all_labels == 0]
anomaly_scores = all_scores[all_labels == 1]

if len(normal_scores) > 0:
    print(f"\n  Normal videos:")
    print(f"    Mean: {np.mean(normal_scores):.4f}")
    print(f"    Std: {np.std(normal_scores):.4f}")

if len(anomaly_scores) > 0:
    print(f"\n  Anomaly videos:")
    print(f"    Mean: {np.mean(anomaly_scores):.4f}")
    print(f"    Std: {np.std(anomaly_scores):.4f}")

# Comparison with base paper
print("\n" + "="*80)
print("COMPARISON WITH BASE PAPER")
print("="*80)

paper_auc = 0.83  # Average of 82-84%
paper_ap = 0.765   # Average of 75-78%

print(f"\nBASE PAPER (AAAI 2022):")
print(f"  AUC: 82-84% (avg: {paper_auc*100:.1f}%)")
print(f"  AP: 75-78% (avg: {paper_ap*100:.1f}%)")

print(f"\nOUR IMPLEMENTATION:")
print(f"  AUC: {auc*100:.2f}%")
print(f"  AP: {ap*100:.2f}%")

print(f"\nPERFORMANCE GAP:")
auc_gap = (paper_auc - auc) * 100
ap_gap = (paper_ap - ap) * 100
print(f"  AUC gap: {auc_gap:+.2f}%")
print(f"  AP gap: {ap_gap:+.2f}%")

if auc >= 0.80:
    print(f"\n🎉 EXCELLENT: AUC matches paper performance!")
elif auc >= 0.75:
    print(f"\n✅ GOOD: AUC is within 5-8% of paper")
elif auc >= 0.65:
    print(f"\n⚠️  ACCEPTABLE: AUC is reasonable for limited training")
else:
    print(f"\n⚠️  BELOW EXPECTED: Consider more training or debugging")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Save numerical results
results_file = results_dir / "evaluation_results.txt"
with open(results_file, 'w') as f:
    f.write("MSL MODEL EVALUATION RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Checkpoint: {args.checkpoint}\n")
    f.write(f"Test videos: {len(all_scores)}\n")
    f.write(f"Normal videos: {n_normal}\n")
    f.write(f"Anomaly videos: {n_anomaly}\n\n")
    f.write(f"METRICS:\n")
    f.write(f"  AUC: {auc:.4f} ({auc*100:.2f}%)\n")
    f.write(f"  AP: {ap:.4f} ({ap*100:.2f}%)\n\n")
    f.write(f"COMPARISON WITH PAPER:\n")
    f.write(f"  Paper AUC: 82-84%\n")
    f.write(f"  Our AUC: {auc*100:.2f}%\n")
    f.write(f"  Gap: {auc_gap:+.2f}%\n")

print(f"\n✅ Results saved to: {results_file}")

# Save scores for further analysis
scores_file = results_dir / "scores.npz"
np.savez(scores_file,
         scores=all_scores,
         labels=all_labels,
         video_names=all_video_names)
print(f"✅ Scores saved to: {scores_file}")

# Plot ROC curve
if n_normal > 0 and n_anomaly > 0:
    print(f"\n📊 Generating ROC curve plot...")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'MSL Model (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    plt.scatter([optimal_fpr], [optimal_tpr], c='red', s=100, zorder=5, 
                label=f'Optimal (threshold={optimal_threshold:.3f})')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - MSL Model on UCF-Crime', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_file = results_dir / "roc_curve.png"
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    print(f"✅ ROC curve saved to: {roc_file}")
    plt.close()

# Final summary
print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print(f"\n📊 FINAL RESULTS:")
print(f"  AUC: {auc:.4f} ({auc*100:.2f}%)")
print(f"  AP: {ap:.4f} ({ap*100:.2f}%)")
print(f"  Paper AUC: 82-84%")
print(f"  Performance: {auc/paper_auc*100:.1f}% of paper")
print(f"\n✅ All results saved to: {results_dir}/")
print("="*80)
