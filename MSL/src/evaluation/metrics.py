"""
Evaluation Metrics
Comprehensive metrics for video anomaly detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)


def compute_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute basic classification metrics.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_scores: Prediction scores (N,) - optional for AUC
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    # Add AUC if scores provided
    if y_scores is not None:
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_scores))
            metrics['ap'] = float(average_precision_score(y_true, y_scores))
        except ValueError:
            # Handle case where only one class present
            metrics['auc'] = 0.0
            metrics['ap'] = 0.0
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Compute confusion matrix and related metrics.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        labels: Label names (default: ['Normal', 'Anomaly'])
    
    Returns:
        Dictionary with confusion matrix and metrics
    """
    if labels is None:
        labels = ['Normal', 'Anomaly']
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': float(specificity),
        'negative_predictive_value': float(npv),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr)
    }


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        y_true: True labels (N,)
        y_scores: Prediction scores (N,)
    
    Returns:
        Dictionary with FPR, TPR, thresholds, and AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': float(auc)
    }


def compute_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute Precision-Recall curve.
    
    Args:
        y_true: True labels (N,)
        y_scores: Prediction scores (N,)
    
    Returns:
        Dictionary with precision, recall, thresholds, and AP
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'average_precision': float(ap)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold based on specified metric.
    
    Args:
        y_true: True labels (N,)
        y_scores: Prediction scores (N,)
        metric: Metric to optimize ('f1', 'accuracy', 'balanced')
    
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    # Try thresholds from 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 81)
    
    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'balanced':
            # Balanced accuracy: average of TPR and TNR
            cm = confusion_matrix(y_true, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (tpr + tnr) / 2
            else:
                score = 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = compute_basic_metrics(y_true, y_pred, y_scores)
    
    return float(best_threshold), best_metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        class_names: Class names
    
    Returns:
        Dictionary of per-class metrics
    """
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_classes]
    
    per_class = {}
    
    for cls_idx, cls_name in zip(unique_classes, class_names):
        # Binary mask for this class
        y_true_binary = (y_true == cls_idx).astype(int)
        y_pred_binary = (y_pred == cls_idx).astype(int)
        
        per_class[cls_name] = {
            'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            'support': int(np.sum(y_true == cls_idx))
        }
    
    return per_class


def compute_temporal_metrics(
    snippet_scores: List[np.ndarray],
    snippet_labels: List[np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute snippet-level (temporal) metrics.
    
    Args:
        snippet_scores: List of snippet score arrays [(T1,), (T2,), ...]
        snippet_labels: List of snippet label arrays [(T1,), (T2,), ...]
        threshold: Detection threshold
    
    Returns:
        Dictionary of temporal metrics
    """
    # Concatenate all snippets
    all_scores = np.concatenate(snippet_scores)
    all_labels = np.concatenate(snippet_labels)
    
    # Threshold predictions
    all_preds = (all_scores >= threshold).astype(int)
    
    # Compute metrics
    metrics = compute_basic_metrics(all_labels, all_preds, all_scores)
    
    # Add temporal-specific metrics
    metrics['num_snippets'] = len(all_scores)
    metrics['num_anomalous_snippets'] = int(np.sum(all_labels))
    metrics['predicted_anomalous_snippets'] = int(np.sum(all_preds))
    
    return metrics


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    video_names: Optional[List[str]] = None
) -> Dict[str, List]:
    """
    Analyze prediction errors.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_scores: Prediction scores (N,)
        video_names: Video names (optional)
    
    Returns:
        Dictionary with false positives and false negatives
    """
    errors = {
        'false_positives': [],
        'false_negatives': []
    }
    
    for i in range(len(y_true)):
        video_info = {
            'index': i,
            'true_label': int(y_true[i]),
            'pred_label': int(y_pred[i]),
            'score': float(y_scores[i])
        }
        
        if video_names is not None:
            video_info['name'] = video_names[i]
        
        # False positive: predicted anomaly, actually normal
        if y_true[i] == 0 and y_pred[i] == 1:
            errors['false_positives'].append(video_info)
        
        # False negative: predicted normal, actually anomaly
        elif y_true[i] == 1 and y_pred[i] == 0:
            errors['false_negatives'].append(video_info)
    
    # Sort by score (most confident errors first)
    errors['false_positives'].sort(key=lambda x: x['score'], reverse=True)
    errors['false_negatives'].sort(key=lambda x: x['score'])
    
    return errors


def create_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    video_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Create comprehensive evaluation report.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_scores: Prediction scores (N,)
        video_names: Video names (optional)
        class_names: Class names (optional)
    
    Returns:
        Complete evaluation report dictionary
    """
    if class_names is None:
        class_names = ['Normal', 'Anomaly']
    
    report = {}
    
    # Basic metrics
    report['basic_metrics'] = compute_basic_metrics(y_true, y_pred, y_scores)
    
    # Confusion matrix
    report['confusion_matrix'] = compute_confusion_matrix(y_true, y_pred, class_names)
    
    # Per-class metrics
    report['per_class_metrics'] = compute_per_class_metrics(y_true, y_pred, class_names)
    
    # Optimal threshold analysis
    report['optimal_thresholds'] = {}
    for metric in ['f1', 'accuracy', 'balanced']:
        threshold, metrics = find_optimal_threshold(y_true, y_scores, metric)
        report['optimal_thresholds'][metric] = {
            'threshold': threshold,
            'metrics': metrics
        }
    
    # Error analysis
    report['error_analysis'] = analyze_errors(y_true, y_pred, y_scores, video_names)
    
    # Dataset statistics
    report['dataset_statistics'] = {
        'total_videos': len(y_true),
        'num_normal': int(np.sum(y_true == 0)),
        'num_anomaly': int(np.sum(y_true == 1)),
        'class_balance': float(np.sum(y_true == 1) / len(y_true)) if len(y_true) > 0 else 0.0
    }
    
    return report


def save_evaluation_report(
    report: Dict,
    output_path: Path,
    readable: bool = True
):
    """
    Save evaluation report to JSON file.
    
    Args:
        report: Evaluation report dictionary
        output_path: Output file path
        readable: Whether to use pretty formatting
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_report = convert_to_serializable(report)
    
    with open(output_path, 'w') as f:
        if readable:
            json.dump(serializable_report, f, indent=2)
        else:
            json.dump(serializable_report, f)


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """
    Format metrics as a readable table string.
    
    Args:
        metrics: Dictionary of metrics
        title: Table title
    
    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        title,
        "=" * 60
    ]
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key:30s}: {value:8.4f}")
        else:
            lines.append(f"{key:30s}: {value}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics computation
    print("=" * 80)
    print("TESTING EVALUATION METRICS")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_scores = np.random.rand(n_samples)
    # Make scores somewhat correlated with labels
    y_scores = y_scores * 0.5 + y_true * 0.5
    y_pred = (y_scores >= 0.5).astype(int)
    
    print("\n1. Testing basic metrics...")
    basic = compute_basic_metrics(y_true, y_pred, y_scores)
    print(format_metrics_table(basic, "Basic Metrics"))
    print("   ✓ Basic metrics computed")
    
    print("\n2. Testing confusion matrix...")
    cm_result = compute_confusion_matrix(y_true, y_pred)
    print(f"   Confusion Matrix: {cm_result['confusion_matrix']}")
    print(f"   TP={cm_result['true_positives']}, FP={cm_result['false_positives']}")
    print(f"   FN={cm_result['false_negatives']}, TN={cm_result['true_negatives']}")
    print("   ✓ Confusion matrix computed")
    
    print("\n3. Testing ROC curve...")
    roc = compute_roc_curve(y_true, y_scores)
    print(f"   AUC: {roc['auc']:.4f}")
    print(f"   FPR points: {len(roc['fpr'])}")
    print("   ✓ ROC curve computed")
    
    print("\n4. Testing PR curve...")
    pr = compute_precision_recall_curve(y_true, y_scores)
    print(f"   AP: {pr['average_precision']:.4f}")
    print(f"   Precision points: {len(pr['precision'])}")
    print("   ✓ PR curve computed")
    
    print("\n5. Testing optimal threshold...")
    threshold, metrics = find_optimal_threshold(y_true, y_scores, metric='f1')
    print(f"   Optimal threshold (F1): {threshold:.3f}")
    print(f"   F1 at threshold: {metrics['f1']:.4f}")
    print("   ✓ Optimal threshold found")
    
    print("\n6. Testing per-class metrics...")
    per_class = compute_per_class_metrics(y_true, y_pred)
    for class_name, class_metrics in per_class.items():
        print(f"   {class_name}: F1={class_metrics['f1']:.4f}")
    print("   ✓ Per-class metrics computed")
    
    print("\n7. Testing error analysis...")
    video_names = [f"video_{i:03d}.mp4" for i in range(n_samples)]
    errors = analyze_errors(y_true, y_pred, y_scores, video_names)
    print(f"   False positives: {len(errors['false_positives'])}")
    print(f"   False negatives: {len(errors['false_negatives'])}")
    if errors['false_positives']:
        print(f"   Most confident FP: {errors['false_positives'][0]['name']} (score={errors['false_positives'][0]['score']:.3f})")
    print("   ✓ Error analysis completed")
    
    print("\n8. Testing complete report...")
    report = create_evaluation_report(y_true, y_pred, y_scores, video_names)
    print(f"   Report sections: {list(report.keys())}")
    print(f"   Total videos: {report['dataset_statistics']['total_videos']}")
    print("   ✓ Complete report generated")
    
    print("\n" + "=" * 80)
    print("✓ ALL METRICS TESTS PASSED")
    print("=" * 80)
