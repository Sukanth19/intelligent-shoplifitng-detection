"""
Simple evaluation tests without pytest
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    compute_basic_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    compute_precision_recall_curve,
    find_optimal_threshold,
    compute_per_class_metrics,
    analyze_errors,
    create_evaluation_report
)


def test_basic_metrics():
    """Test basic metrics computation."""
    print("Testing basic metrics...")
    
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.85])
    
    metrics = compute_basic_metrics(y_true, y_pred, y_scores)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['auc'] <= 1
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print("  ✓ Basic metrics work")


def test_confusion_matrix():
    """Test confusion matrix computation."""
    print("Testing confusion matrix...")
    
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    
    cm_result = compute_confusion_matrix(y_true, y_pred)
    
    assert 'confusion_matrix' in cm_result
    assert 'true_positives' in cm_result
    assert 'false_positives' in cm_result
    assert 'false_negatives' in cm_result
    assert 'true_negatives' in cm_result
    
    tp = cm_result['true_positives']
    fp = cm_result['false_positives']
    fn = cm_result['false_negatives']
    tn = cm_result['true_negatives']
    
    assert tp + fp + fn + tn == len(y_true)
    
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("  ✓ Confusion matrix works")


def test_roc_curve():
    """Test ROC curve computation."""
    print("Testing ROC curve...")
    
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    
    roc = compute_roc_curve(y_true, y_scores)
    
    assert 'fpr' in roc
    assert 'tpr' in roc
    assert 'thresholds' in roc
    assert 'auc' in roc
    assert len(roc['fpr']) == len(roc['tpr'])
    assert 0 <= roc['auc'] <= 1
    
    print(f"  AUC: {roc['auc']:.4f}")
    print(f"  Points: {len(roc['fpr'])}")
    print("  ✓ ROC curve works")


def test_pr_curve():
    """Test Precision-Recall curve."""
    print("Testing PR curve...")
    
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    
    pr = compute_precision_recall_curve(y_true, y_scores)
    
    assert 'precision' in pr
    assert 'recall' in pr
    assert 'thresholds' in pr
    assert 'average_precision' in pr
    assert 0 <= pr['average_precision'] <= 1
    
    print(f"  AP: {pr['average_precision']:.4f}")
    print("  ✓ PR curve works")


def test_optimal_threshold():
    """Test optimal threshold finding."""
    print("Testing optimal threshold...")
    
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    
    # Test F1 optimization
    threshold, metrics = find_optimal_threshold(y_true, y_scores, metric='f1')
    assert 0 <= threshold <= 1
    assert 'f1' in metrics
    print(f"  F1 optimal: threshold={threshold:.3f}, F1={metrics['f1']:.4f}")
    
    # Test accuracy optimization
    threshold, metrics = find_optimal_threshold(y_true, y_scores, metric='accuracy')
    assert 0 <= threshold <= 1
    assert 'accuracy' in metrics
    print(f"  Acc optimal: threshold={threshold:.3f}, Acc={metrics['accuracy']:.4f}")
    
    print("  ✓ Optimal threshold works")


def test_per_class_metrics():
    """Test per-class metrics."""
    print("Testing per-class metrics...")
    
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    
    per_class = compute_per_class_metrics(y_true, y_pred, class_names=['Normal', 'Anomaly'])
    
    assert 'Normal' in per_class
    assert 'Anomaly' in per_class
    assert 'precision' in per_class['Normal']
    assert 'recall' in per_class['Normal']
    assert 'f1' in per_class['Normal']
    
    print(f"  Normal F1: {per_class['Normal']['f1']:.4f}")
    print(f"  Anomaly F1: {per_class['Anomaly']['f1']:.4f}")
    print("  ✓ Per-class metrics work")


def test_error_analysis():
    """Test error analysis."""
    print("Testing error analysis...")
    
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.85])
    video_names = [f"video_{i}.mp4" for i in range(len(y_true))]
    
    errors = analyze_errors(y_true, y_pred, y_scores, video_names)
    
    assert 'false_positives' in errors
    assert 'false_negatives' in errors
    assert isinstance(errors['false_positives'], list)
    assert isinstance(errors['false_negatives'], list)
    
    # Check that FPs are sorted by score (descending)
    if len(errors['false_positives']) > 1:
        scores = [fp['score'] for fp in errors['false_positives']]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    print(f"  False positives: {len(errors['false_positives'])}")
    print(f"  False negatives: {len(errors['false_negatives'])}")
    print("  ✓ Error analysis works")


def test_complete_report():
    """Test complete evaluation report."""
    print("Testing complete report...")
    
    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_scores = np.random.rand(n)
    y_pred = (y_scores >= 0.5).astype(int)
    video_names = [f"video_{i:03d}.mp4" for i in range(n)]
    
    report = create_evaluation_report(
        y_true, y_pred, y_scores,
        video_names=video_names,
        class_names=['Normal', 'Anomaly']
    )
    
    # Check all sections present
    assert 'basic_metrics' in report
    assert 'confusion_matrix' in report
    assert 'per_class_metrics' in report
    assert 'optimal_thresholds' in report
    assert 'error_analysis' in report
    assert 'dataset_statistics' in report
    
    # Check optimal thresholds
    assert 'f1' in report['optimal_thresholds']
    assert 'accuracy' in report['optimal_thresholds']
    assert 'balanced' in report['optimal_thresholds']
    
    # Check dataset stats
    stats = report['dataset_statistics']
    assert stats['total_videos'] == n
    assert stats['num_normal'] + stats['num_anomaly'] == n
    
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Normal: {stats['num_normal']}, Anomaly: {stats['num_anomaly']}")
    print(f"  Sections: {len(report)}")
    print("  ✓ Complete report works")


def main():
    print("=" * 80)
    print("TESTING EVALUATION COMPONENTS")
    print("=" * 80)
    
    print("\n1. Basic Metrics")
    test_basic_metrics()
    
    print("\n2. Confusion Matrix")
    test_confusion_matrix()
    
    print("\n3. ROC Curve")
    test_roc_curve()
    
    print("\n4. PR Curve")
    test_pr_curve()
    
    print("\n5. Optimal Threshold")
    test_optimal_threshold()
    
    print("\n6. Per-Class Metrics")
    test_per_class_metrics()
    
    print("\n7. Error Analysis")
    test_error_analysis()
    
    print("\n8. Complete Report")
    test_complete_report()
    
    print("\n" + "=" * 80)
    print("✓ ALL EVALUATION TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
