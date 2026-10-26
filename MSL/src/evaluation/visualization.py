"""
Visualization
Plotting functions for evaluation results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_precision_recall_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    ap: float,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        recall: Recall values
        precision: Precision values
        ap: Average precision
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot PR curve
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap:.4f})')
    
    # Plot baseline (random classifier)
    baseline = np.sum(recall > 0) / len(recall) if len(recall) > 0 else 0.5
    ax.axhline(y=baseline, color='r', linestyle='--', linewidth=1, label=f'Baseline (AP = {baseline:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
        normalize: Whether to normalize by row
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        vmin=0,
        vmax=vmax,
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_score_distribution(
    scores_normal: np.ndarray,
    scores_anomaly: np.ndarray,
    threshold: float = 0.5,
    title: str = "Anomaly Score Distribution",
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot distribution of anomaly scores for each class.
    
    Args:
        scores_normal: Scores for normal videos
        scores_anomaly: Scores for anomalous videos
        threshold: Detection threshold
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(scores_normal, bins=30, alpha=0.6, color='blue', label='Normal', density=True)
    ax.hist(scores_anomaly, bins=30, alpha=0.6, color='red', label='Anomaly', density=True)
    
    # Plot threshold line
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
    
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_temporal_anomaly_scores(
    snippet_scores: np.ndarray,
    snippet_labels: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    intervals: Optional[List[Tuple[int, int]]] = None,
    video_name: str = "Video",
    fps: float = 30.0,
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot temporal anomaly scores for a single video.
    
    Args:
        snippet_scores: Snippet-level scores (T,)
        snippet_labels: Ground truth labels (T,) - optional
        threshold: Detection threshold
        intervals: Detected anomaly intervals [(start, end), ...]
        video_name: Video name for title
        fps: Frames per second for time axis
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    T = len(snippet_scores)
    time_axis = np.arange(T) / fps  # Convert to seconds
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot scores
    ax.plot(time_axis, snippet_scores, 'b-', linewidth=2, label='Anomaly Score')
    
    # Plot threshold
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {threshold:.2f}')
    
    # Highlight ground truth anomalies if available
    if snippet_labels is not None:
        anomaly_mask = snippet_labels > 0
        if np.any(anomaly_mask):
            ax.fill_between(time_axis, 0, 1, where=anomaly_mask, alpha=0.2, color='red', label='Ground Truth Anomaly')
    
    # Highlight detected intervals
    if intervals:
        for start, end in intervals:
            ax.axvspan(start / fps, end / fps, alpha=0.3, color='yellow', label='Detected Interval' if start == intervals[0][0] else '')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.set_title(f'Temporal Anomaly Scores: {video_name}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Metrics Comparison",
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot bar chart comparing metrics across different methods/thresholds.
    
    Args:
        metrics_dict: Dictionary mapping method name to metrics
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Extract metric names and values
    methods = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    # Prepare data
    x = np.arange(len(metric_names))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        values = [metrics_dict[method][m] for m in metric_names]
        offset = width * (i - len(methods) / 2 + 0.5)
        ax.bar(x + offset, values, width, label=method)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_threshold_analysis(
    thresholds: np.ndarray,
    accuracies: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1_scores: np.ndarray,
    optimal_threshold: float,
    title: str = "Threshold Analysis",
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot metrics vs threshold.
    
    Args:
        thresholds: Threshold values
        accuracies: Accuracy at each threshold
        precisions: Precision at each threshold
        recalls: Recall at each threshold
        f1_scores: F1 score at each threshold
        optimal_threshold: Optimal threshold value
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot metrics
    ax.plot(thresholds, accuracies, 'b-', linewidth=2, label='Accuracy')
    ax.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
    ax.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
    ax.plot(thresholds, f1_scores, 'm-', linewidth=2, label='F1 Score')
    
    # Mark optimal threshold
    ax.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Optimal = {optimal_threshold:.3f}')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([thresholds.min(), thresholds.max()])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def create_evaluation_dashboard(
    evaluation_data: Dict,
    output_dir: Path,
    dataset_name: str = "Dataset"
):
    """
    Create complete evaluation dashboard with all plots.
    
    Args:
        evaluation_data: Dictionary containing all evaluation data
        output_dir: Output directory for plots
        dataset_name: Name of dataset for titles
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating evaluation dashboard for {dataset_name}...")
    
    # 1. ROC Curve
    if 'roc' in evaluation_data:
        roc = evaluation_data['roc']
        plot_roc_curve(
            roc['fpr'],
            roc['tpr'],
            roc['auc'],
            title=f"ROC Curve - {dataset_name}",
            save_path=output_dir / 'roc_curve.png'
        )
        print("  ✓ ROC curve saved")
    
    # 2. Precision-Recall Curve
    if 'pr_curve' in evaluation_data:
        pr = evaluation_data['pr_curve']
        plot_precision_recall_curve(
            pr['recall'],
            pr['precision'],
            pr['average_precision'],
            title=f"Precision-Recall Curve - {dataset_name}",
            save_path=output_dir / 'pr_curve.png'
        )
        print("  ✓ PR curve saved")
    
    # 3. Confusion Matrix
    if 'confusion_matrix' in evaluation_data:
        cm_data = evaluation_data['confusion_matrix']
        cm = np.array(cm_data['confusion_matrix'])
        labels = cm_data['labels']
        
        # Raw counts
        plot_confusion_matrix(
            cm,
            labels,
            title=f"Confusion Matrix - {dataset_name}",
            normalize=False,
            save_path=output_dir / 'confusion_matrix.png'
        )
        
        # Normalized
        plot_confusion_matrix(
            cm,
            labels,
            title=f"Normalized Confusion Matrix - {dataset_name}",
            normalize=True,
            save_path=output_dir / 'confusion_matrix_normalized.png'
        )
        print("  ✓ Confusion matrices saved")
    
    # 4. Score Distribution
    if 'scores_by_class' in evaluation_data:
        scores_data = evaluation_data['scores_by_class']
        plot_score_distribution(
            scores_data['normal'],
            scores_data['anomaly'],
            threshold=evaluation_data.get('threshold', 0.5),
            title=f"Anomaly Score Distribution - {dataset_name}",
            save_path=output_dir / 'score_distribution.png'
        )
        print("  ✓ Score distribution saved")
    
    print(f"Dashboard created at: {output_dir}")


if __name__ == "__main__":
    # Test visualization functions
    print("=" * 80)
    print("TESTING VISUALIZATION")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    
    print("\n1. Testing ROC curve...")
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr) + np.random.rand(100) * 0.1
    tpr = np.clip(tpr, 0, 1)
    auc = 0.85
    fig = plot_roc_curve(fpr, tpr, auc)
    plt.close(fig)
    print("   ✓ ROC curve generated")
    
    print("\n2. Testing PR curve...")
    recall = np.linspace(0, 1, 100)
    precision = 1 - recall * 0.5 + np.random.rand(100) * 0.1
    precision = np.clip(precision, 0, 1)
    ap = 0.78
    fig = plot_precision_recall_curve(recall, precision, ap)
    plt.close(fig)
    print("   ✓ PR curve generated")
    
    print("\n3. Testing confusion matrix...")
    cm = np.array([[45, 5], [8, 42]])
    labels = ['Normal', 'Anomaly']
    fig = plot_confusion_matrix(cm, labels)
    plt.close(fig)
    print("   ✓ Confusion matrix generated")
    
    print("\n4. Testing score distribution...")
    scores_normal = np.random.beta(2, 5, 500)
    scores_anomaly = np.random.beta(5, 2, 500)
    fig = plot_score_distribution(scores_normal, scores_anomaly, threshold=0.5)
    plt.close(fig)
    print("   ✓ Score distribution generated")
    
    print("\n5. Testing temporal scores...")
    T = 100
    snippet_scores = np.random.rand(T) * 0.3
    snippet_scores[30:50] = np.random.rand(20) * 0.4 + 0.6  # Anomaly region
    snippet_labels = np.zeros(T)
    snippet_labels[30:50] = 1
    intervals = [(30, 50)]
    fig = plot_temporal_anomaly_scores(
        snippet_scores,
        snippet_labels,
        threshold=0.5,
        intervals=intervals,
        video_name="test_video.mp4"
    )
    plt.close(fig)
    print("   ✓ Temporal scores plotted")
    
    print("\n6. Testing metrics comparison...")
    metrics_dict = {
        'Method 1': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85},
        'Method 2': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.86, 'f1': 0.86}
    }
    fig = plot_metrics_comparison(metrics_dict)
    plt.close(fig)
    print("   ✓ Metrics comparison generated")
    
    print("\n7. Testing threshold analysis...")
    thresholds = np.linspace(0.1, 0.9, 50)
    accuracies = 1 - np.abs(thresholds - 0.5)
    precisions = np.clip(thresholds + np.random.rand(50) * 0.1, 0, 1)
    recalls = np.clip(1 - thresholds + np.random.rand(50) * 0.1, 0, 1)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-7)
    fig = plot_threshold_analysis(thresholds, accuracies, precisions, recalls, f1_scores, optimal_threshold=0.5)
    plt.close(fig)
    print("   ✓ Threshold analysis generated")
    
    print("\n" + "=" * 80)
    print("✓ ALL VISUALIZATION TESTS PASSED")
    print("=" * 80)
