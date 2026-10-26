"""
Evaluation Script
Comprehensive evaluation with metrics and visualization
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MSLModel
from src.inference import load_predictor_from_checkpoint
from src.datasets import UCFCrimeFeatureDataset
from src.evaluation import (
    compute_basic_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    compute_precision_recall_curve,
    find_optimal_threshold,
    compute_per_class_metrics,
    analyze_errors,
    create_evaluation_report,
    save_evaluation_report,
    format_metrics_table,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_temporal_anomaly_scores,
    plot_threshold_analysis,
    create_evaluation_dashboard
)


def setup_logging(log_file: Path = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def evaluate_dataset(
    feature_dir: Path,
    checkpoint_path: Path,
    config: dict,
    output_dir: Path,
    device: str = 'cuda',
    split: str = 'test',
    create_visualizations: bool = True,
    save_predictions: bool = True
):
    """
    Complete evaluation on dataset.
    
    Args:
        feature_dir: Directory containing features
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        output_dir: Output directory
        device: Device for inference
        split: Dataset split
        create_visualizations: Whether to create plots
        save_predictions: Whether to save predictions
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"EVALUATING ON {split.upper()} SET")
    logger.info("=" * 80)
    
    # Load dataset
    logger.info(f"Loading dataset from {feature_dir}...")
    dataset = UCFCrimeFeatureDataset(feature_dir=feature_dir, split=split)
    logger.info(f"  Found {len(dataset)} videos")
    
    # Load predictor
    logger.info("Loading model...")
    predictor = load_predictor_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_class=MSLModel,
        model_config=config['model'],
        device=device,
        score_correction_method=config.get('score_correction', {}).get('method', 'linear'),
        score_correction_alpha=config.get('score_correction', {}).get('alpha', 0.5),
        temporal_smoothing=config.get('temporal_smoothing', {}).get('enabled', True),
        smoothing_window=config.get('temporal_smoothing', {}).get('window_size', 5),
        anomaly_threshold=config.get('anomaly_threshold', 0.5)
    )
    
    # Run predictions
    logger.info("Running predictions...")
    predictions = []
    true_labels = []
    video_scores = []
    video_names = []
    snippet_scores_list = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        features = sample['features']
        label = sample['label']
        video_name = sample['video_name']
        
        result = predictor.predict(features)
        
        predictions.append(result['prediction'])
        true_labels.append(label)
        video_scores.append(result['video_score'])
        video_names.append(video_name)
        snippet_scores_list.append(result['smoothed_scores'].numpy())
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(dataset)} videos")
    
    # Convert to numpy
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    y_scores = np.array(video_scores)
    
    logger.info(f"✓ Completed predictions for {len(y_true)} videos")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === COMPUTE METRICS ===
    logger.info("\nComputing metrics...")
    
    # Basic metrics
    basic_metrics = compute_basic_metrics(y_true, y_pred, y_scores)
    print("\n" + format_metrics_table(basic_metrics, "Basic Metrics"))
    
    # Confusion matrix
    cm_result = compute_confusion_matrix(y_true, y_pred, labels=['Normal', 'Anomaly'])
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm_result['true_negatives']:4d}")
    print(f"  False Positives: {cm_result['false_positives']:4d}")
    print(f"  False Negatives: {cm_result['false_negatives']:4d}")
    print(f"  True Positives:  {cm_result['true_positives']:4d}")
    print(f"  Specificity:     {cm_result['specificity']:.4f}")
    print(f"  FPR:             {cm_result['false_positive_rate']:.4f}")
    
    # ROC curve
    roc_result = compute_roc_curve(y_true, y_scores)
    logger.info(f"  ROC AUC: {roc_result['auc']:.4f}")
    
    # PR curve
    pr_result = compute_precision_recall_curve(y_true, y_scores)
    logger.info(f"  Average Precision: {pr_result['average_precision']:.4f}")
    
    # Optimal thresholds
    logger.info("\nOptimal thresholds:")
    optimal_thresholds = {}
    for metric_name in ['f1', 'accuracy', 'balanced']:
        threshold, metrics = find_optimal_threshold(y_true, y_scores, metric=metric_name)
        optimal_thresholds[metric_name] = {'threshold': threshold, 'metrics': metrics}
        logger.info(f"  {metric_name.upper():10s}: threshold={threshold:.3f}, {metric_name}={metrics.get(metric_name, metrics.get('accuracy', 0)):.4f}")
    
    # Per-class metrics
    per_class = compute_per_class_metrics(y_true, y_pred, class_names=['Normal', 'Anomaly'])
    print("\nPer-Class Metrics:")
    for class_name, class_metrics in per_class.items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1:        {class_metrics['f1']:.4f}")
        print(f"    Support:   {class_metrics['support']}")
    
    # Error analysis
    errors = analyze_errors(y_true, y_pred, y_scores, video_names)
    print(f"\nError Analysis:")
    print(f"  False Positives: {len(errors['false_positives'])}")
    print(f"  False Negatives: {len(errors['false_negatives'])}")
    
    if errors['false_positives']:
        print(f"\n  Top False Positives (most confident):")
        for i, fp in enumerate(errors['false_positives'][:5], 1):
            print(f"    {i}. {fp['name']:30s} score={fp['score']:.4f}")
    
    if errors['false_negatives']:
        print(f"\n  Top False Negatives (least confident):")
        for i, fn in enumerate(errors['false_negatives'][:5], 1):
            print(f"    {i}. {fn['name']:30s} score={fn['score']:.4f}")
    
    # === SAVE RESULTS ===
    logger.info("\nSaving results...")
    
    # Save complete report
    report = create_evaluation_report(y_true, y_pred, y_scores, video_names, ['Normal', 'Anomaly'])
    save_evaluation_report(report, output_dir / f'{split}_evaluation_report.json')
    logger.info(f"  ✓ Evaluation report saved")
    
    # Save predictions
    if save_predictions:
        np.savez(
            output_dir / f'{split}_predictions.npz',
            video_names=video_names,
            video_scores=y_scores,
            predictions=y_pred,
            true_labels=y_true
        )
        logger.info(f"  ✓ Predictions saved")
    
    # === CREATE VISUALIZATIONS ===
    if create_visualizations:
        logger.info("\nCreating visualizations...")
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # ROC curve
        plot_roc_curve(
            roc_result['fpr'],
            roc_result['tpr'],
            roc_result['auc'],
            title=f"ROC Curve - {split.upper()}",
            save_path=viz_dir / 'roc_curve.png'
        )
        logger.info("  ✓ ROC curve saved")
        
        # PR curve
        plot_precision_recall_curve(
            pr_result['recall'],
            pr_result['precision'],
            pr_result['average_precision'],
            title=f"Precision-Recall Curve - {split.upper()}",
            save_path=viz_dir / 'pr_curve.png'
        )
        logger.info("  ✓ PR curve saved")
        
        # Confusion matrices
        cm = np.array(cm_result['confusion_matrix'])
        plot_confusion_matrix(
            cm,
            ['Normal', 'Anomaly'],
            title=f"Confusion Matrix - {split.upper()}",
            normalize=False,
            save_path=viz_dir / 'confusion_matrix.png'
        )
        plot_confusion_matrix(
            cm,
            ['Normal', 'Anomaly'],
            title=f"Normalized Confusion Matrix - {split.upper()}",
            normalize=True,
            save_path=viz_dir / 'confusion_matrix_normalized.png'
        )
        logger.info("  ✓ Confusion matrices saved")
        
        # Score distribution
        scores_normal = y_scores[y_true == 0]
        scores_anomaly = y_scores[y_true == 1]
        plot_score_distribution(
            scores_normal,
            scores_anomaly,
            threshold=config.get('anomaly_threshold', 0.5),
            title=f"Anomaly Score Distribution - {split.upper()}",
            save_path=viz_dir / 'score_distribution.png'
        )
        logger.info("  ✓ Score distribution saved")
        
        # Threshold analysis
        thresholds_test = np.linspace(0.1, 0.9, 81)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds_test:
            y_pred_temp = (y_scores >= threshold).astype(int)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracies.append(accuracy_score(y_true, y_pred_temp))
            precisions.append(precision_score(y_true, y_pred_temp, zero_division=0))
            recalls.append(recall_score(y_true, y_pred_temp, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred_temp, zero_division=0))
        
        plot_threshold_analysis(
            thresholds_test,
            np.array(accuracies),
            np.array(precisions),
            np.array(recalls),
            np.array(f1_scores),
            optimal_threshold=optimal_thresholds['f1']['threshold'],
            title=f"Threshold Analysis - {split.upper()}",
            save_path=viz_dir / 'threshold_analysis.png'
        )
        logger.info("  ✓ Threshold analysis saved")
        
        # Sample temporal plots (first 5 videos)
        temporal_dir = viz_dir / 'temporal'
        temporal_dir.mkdir(exist_ok=True)
        
        for i in range(min(5, len(dataset))):
            video_name = video_names[i]
            snippet_scores = snippet_scores_list[i]
            
            plot_temporal_anomaly_scores(
                snippet_scores,
                threshold=config.get('anomaly_threshold', 0.5),
                video_name=video_name,
                fps=config.get('fps', 30.0),
                save_path=temporal_dir / f'{Path(video_name).stem}_temporal.png'
            )
        
        logger.info(f"  ✓ Temporal plots saved (first {min(5, len(dataset))} videos)")
    
    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Dataset: {split.upper()}")
    print(f"Total Videos: {len(y_true)}")
    print(f"Normal: {np.sum(y_true == 0)}, Anomaly: {np.sum(y_true == 1)}")
    print(f"\nPerformance:")
    print(f"  Accuracy:  {basic_metrics['accuracy']:.4f}")
    print(f"  Precision: {basic_metrics['precision']:.4f}")
    print(f"  Recall:    {basic_metrics['recall']:.4f}")
    print(f"  F1 Score:  {basic_metrics['f1']:.4f}")
    print(f"  AUC:       {basic_metrics['auc']:.4f}")
    print(f"  AP:        {basic_metrics['ap']:.4f}")
    print(f"\nOptimal Thresholds:")
    for metric_name, opt_data in optimal_thresholds.items():
        print(f"  {metric_name.upper():10s}: {opt_data['threshold']:.3f}")
    print(f"\nResults saved to: {output_dir}")
    if create_visualizations:
        print(f"Visualizations: {viz_dir}")
    print("=" * 80)
    
    logger.info(f"\n✓ Evaluation complete!")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate MSL model on dataset')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Directory containing feature files')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Disable visualization creation')
    parser.add_argument('--no-save-predictions', action='store_true',
                        help='Disable saving predictions')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert paths
    checkpoint_path = Path(args.checkpoint)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    if not feature_dir.exists():
        logger.error(f"Feature directory not found: {feature_dir}")
        return
    
    # Run evaluation
    evaluate_dataset(
        feature_dir=feature_dir,
        checkpoint_path=checkpoint_path,
        config=config,
        output_dir=output_dir,
        device=args.device,
        split=args.split,
        create_visualizations=not args.no_visualizations,
        save_predictions=not args.no_save_predictions
    )


if __name__ == "__main__":
    main()
