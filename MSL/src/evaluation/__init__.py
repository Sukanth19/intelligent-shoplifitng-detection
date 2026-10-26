"""
Evaluation module
"""

from .metrics import (
    compute_basic_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    compute_precision_recall_curve,
    find_optimal_threshold,
    compute_per_class_metrics,
    compute_temporal_metrics,
    analyze_errors,
    create_evaluation_report,
    save_evaluation_report,
    format_metrics_table
)

from .visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_temporal_anomaly_scores,
    plot_metrics_comparison,
    plot_threshold_analysis,
    create_evaluation_dashboard
)

__all__ = [
    # Metrics
    'compute_basic_metrics',
    'compute_confusion_matrix',
    'compute_roc_curve',
    'compute_precision_recall_curve',
    'find_optimal_threshold',
    'compute_per_class_metrics',
    'compute_temporal_metrics',
    'analyze_errors',
    'create_evaluation_report',
    'save_evaluation_report',
    'format_metrics_table',
    # Visualization
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
    'plot_score_distribution',
    'plot_temporal_anomaly_scores',
    'plot_metrics_comparison',
    'plot_threshold_analysis',
    'create_evaluation_dashboard'
]
