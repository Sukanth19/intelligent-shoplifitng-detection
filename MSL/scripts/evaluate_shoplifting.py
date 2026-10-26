"""
Shoplifting-Specific Evaluation Script
Focused evaluation on Shoplifting class detection
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
    analyze_errors,
    create_evaluation_report,
    save_evaluation_report,
    format_metrics_table,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_temporal_anomaly_scores,
    plot_metrics_comparison
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


def filter_shoplifting_videos(
    dataset,
    predictor,
    shoplifting_keywords: list = None
):
    """
    Filter dataset to identify shoplifting and normal videos.
    
    Args:
        dataset: UCFCrimeFeatureDataset
        predictor: MSLPredictor
        shoplifting_keywords: Keywords to identify shoplifting videos
    
    Returns:
        Dictionary with filtered data
    """
    if shoplifting_keywords is None:
        shoplifting_keywords = ['shoplifting', 'stealing', 'theft']
    
    logger = logging.getLogger(__name__)
    logger.info("Filtering shoplifting videos...")
    
    normal_data = []
    shoplifting_data = []
    other_anomaly_data = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        video_name = sample['video_name'].lower()
        label = sample['label']
        
        # Check if video name contains shoplifting keywords
        is_shoplifting = any(keyword in video_name for keyword in shoplifting_keywords)
        
        if label == 0:
            # Normal video
            normal_data.append((i, sample, 'Normal'))
        elif is_shoplifting:
            # Shoplifting video
            shoplifting_data.append((i, sample, 'Shoplifting'))
        else:
            # Other anomaly type
            other_anomaly_data.append((i, sample, 'Other Anomaly'))
    
    logger.info(f"  Normal videos: {len(normal_data)}")
    logger.info(f"  Shoplifting videos: {len(shoplifting_data)}")
    logger.info(f"  Other anomaly videos: {len(other_anomaly_data)}")
    
    return {
        'normal': normal_data,
        'shoplifting': shoplifting_data,
        'other_anomaly': other_anomaly_data
    }


def evaluate_normal_vs_shoplifting(
    filtered_data: dict,
    predictor,
    output_dir: Path,
    config: dict
):
    """
    Evaluate Normal vs Shoplifting binary classification.
    
    Args:
        filtered_data: Filtered dataset
        predictor: MSLPredictor
        output_dir: Output directory
        config: Configuration dict
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("NORMAL vs SHOPLIFTING EVALUATION")
    logger.info("=" * 80)
    
    normal_data = filtered_data['normal']
    shoplifting_data = filtered_data['shoplifting']
    
    if len(shoplifting_data) == 0:
        logger.warning("No shoplifting videos found in dataset!")
        return None
    
    # Combine normal and shoplifting
    combined_data = normal_data + shoplifting_data
    
    # Run predictions
    logger.info("Running predictions...")
    predictions = []
    true_labels = []
    video_scores = []
    video_names = []
    snippet_scores_list = []
    
    for idx, sample, category in combined_data:
        features = sample['features']
        result = predictor.predict(features)
        
        # Label: 0 = Normal, 1 = Shoplifting
        true_label = 1 if category == 'Shoplifting' else 0
        
        predictions.append(result['prediction'])
        true_labels.append(true_label)
        video_scores.append(result['video_score'])
        video_names.append(sample['video_name'])
        snippet_scores_list.append(result['smoothed_scores'].numpy())
    
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    y_scores = np.array(video_scores)
    
    logger.info(f"✓ Completed predictions for {len(y_true)} videos")
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    basic_metrics = compute_basic_metrics(y_true, y_pred, y_scores)
    print("\n" + format_metrics_table(basic_metrics, "Normal vs Shoplifting Metrics"))
    
    # Confusion matrix
    cm_result = compute_confusion_matrix(y_true, y_pred, labels=['Normal', 'Shoplifting'])
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (Normal):     {cm_result['true_negatives']:4d}")
    print(f"  False Positives (Normal→SL): {cm_result['false_positives']:4d}")
    print(f"  False Negatives (SL→Normal): {cm_result['false_negatives']:4d}")
    print(f"  True Positives (Shoplifting): {cm_result['true_positives']:4d}")
    
    # ROC curve
    roc_result = compute_roc_curve(y_true, y_scores)
    logger.info(f"  ROC AUC: {roc_result['auc']:.4f}")
    
    # PR curve
    pr_result = compute_precision_recall_curve(y_true, y_scores)
    logger.info(f"  Average Precision: {pr_result['average_precision']:.4f}")
    
    # Optimal threshold
    threshold, metrics = find_optimal_threshold(y_true, y_scores, metric='f1')
    logger.info(f"\nOptimal threshold (F1): {threshold:.3f}")
    logger.info(f"  Metrics at optimal threshold:")
    logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"    Precision: {metrics['precision']:.4f}")
    logger.info(f"    Recall:    {metrics['recall']:.4f}")
    logger.info(f"    F1:        {metrics['f1']:.4f}")
    
    # Error analysis
    errors = analyze_errors(y_true, y_pred, y_scores, video_names)
    print(f"\nError Analysis:")
    print(f"  False Positives (Normal predicted as Shoplifting): {len(errors['false_positives'])}")
    print(f"  False Negatives (Shoplifting predicted as Normal): {len(errors['false_negatives'])}")
    
    if errors['false_positives']:
        print(f"\n  Top False Positives:")
        for i, fp in enumerate(errors['false_positives'][:5], 1):
            print(f"    {i}. {fp['name']:40s} score={fp['score']:.4f}")
    
    if errors['false_negatives']:
        print(f"\n  Top False Negatives (Missed Shoplifting):")
        for i, fn in enumerate(errors['false_negatives'][:5], 1):
            print(f"    {i}. {fn['name']:40s} score={fn['score']:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'evaluation_type': 'Normal vs Shoplifting',
        'basic_metrics': basic_metrics,
        'confusion_matrix': cm_result,
        'roc': {'auc': roc_result['auc']},
        'pr': {'ap': pr_result['average_precision']},
        'optimal_threshold': {'threshold': threshold, 'metrics': metrics},
        'error_analysis': errors,
        'dataset_statistics': {
            'total_videos': len(y_true),
            'normal_videos': len(normal_data),
            'shoplifting_videos': len(shoplifting_data)
        }
    }
    
    save_evaluation_report(report, output_dir / 'normal_vs_shoplifting_report.json')
    logger.info(f"\n✓ Report saved")
    
    # Save predictions
    np.savez(
        output_dir / 'normal_vs_shoplifting_predictions.npz',
        video_names=video_names,
        video_scores=y_scores,
        predictions=y_pred,
        true_labels=y_true
    )
    
    # Visualizations
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    logger.info("\nCreating visualizations...")
    
    # ROC curve
    plot_roc_curve(
        roc_result['fpr'],
        roc_result['tpr'],
        roc_result['auc'],
        title="ROC Curve - Normal vs Shoplifting",
        save_path=viz_dir / 'roc_curve_shoplifting.png'
    )
    
    # PR curve
    plot_precision_recall_curve(
        pr_result['recall'],
        pr_result['precision'],
        pr_result['average_precision'],
        title="Precision-Recall Curve - Normal vs Shoplifting",
        save_path=viz_dir / 'pr_curve_shoplifting.png'
    )
    
    # Confusion matrix
    cm = np.array(cm_result['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        ['Normal', 'Shoplifting'],
        title="Confusion Matrix - Normal vs Shoplifting",
        normalize=False,
        save_path=viz_dir / 'confusion_matrix_shoplifting.png'
    )
    
    # Score distribution
    scores_normal = y_scores[y_true == 0]
    scores_shoplifting = y_scores[y_true == 1]
    plot_score_distribution(
        scores_normal,
        scores_shoplifting,
        threshold=config.get('anomaly_threshold', 0.5),
        title="Score Distribution - Normal vs Shoplifting",
        save_path=viz_dir / 'score_distribution_shoplifting.png'
    )
    
    # Temporal plots for shoplifting videos (first 5)
    temporal_dir = viz_dir / 'temporal_shoplifting'
    temporal_dir.mkdir(exist_ok=True)
    
    shoplifting_indices = [i for i, (_, _, cat) in enumerate(combined_data) if cat == 'Shoplifting']
    for i in shoplifting_indices[:5]:
        video_name = video_names[i]
        snippet_scores = snippet_scores_list[i]
        
        plot_temporal_anomaly_scores(
            snippet_scores,
            threshold=config.get('anomaly_threshold', 0.5),
            video_name=video_name,
            fps=config.get('fps', 30.0),
            save_path=temporal_dir / f'{Path(video_name).stem}_temporal.png'
        )
    
    logger.info(f"✓ Visualizations saved to {viz_dir}")
    
    return report


def evaluate_multiclass(
    filtered_data: dict,
    predictor,
    output_dir: Path,
    config: dict
):
    """
    Evaluate 3-class classification: Normal, Shoplifting, Other Anomaly.
    
    Args:
        filtered_data: Filtered dataset
        predictor: MSLPredictor
        output_dir: Output directory
        config: Configuration dict
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("MULTICLASS EVALUATION (Normal / Shoplifting / Other)")
    logger.info("=" * 80)
    
    # Combine all categories
    all_data = (
        filtered_data['normal'] +
        filtered_data['shoplifting'] +
        filtered_data['other_anomaly']
    )
    
    if len(all_data) == 0:
        logger.warning("No data available!")
        return None
    
    # Run predictions
    logger.info("Running predictions...")
    predictions = []
    true_labels = []
    video_scores = []
    video_names = []
    categories = []
    
    for idx, sample, category in all_data:
        features = sample['features']
        result = predictor.predict(features)
        
        # Map category to label
        # 0 = Normal, 1 = Shoplifting, 2 = Other Anomaly
        if category == 'Normal':
            true_label = 0
        elif category == 'Shoplifting':
            true_label = 1
        else:  # Other Anomaly
            true_label = 2
        
        # For binary predictor, we need to interpret results
        # High score + category determines class
        pred_binary = result['prediction']
        score = result['video_score']
        
        # Simple heuristic: if predicted as anomaly, check category
        if pred_binary == 1:
            # Anomaly - check which type based on video name
            video_name = sample['video_name'].lower()
            if any(kw in video_name for kw in ['shoplifting', 'stealing', 'theft']):
                pred_label = 1  # Shoplifting
            else:
                pred_label = 2  # Other anomaly
        else:
            pred_label = 0  # Normal
        
        predictions.append(pred_label)
        true_labels.append(true_label)
        video_scores.append(score)
        video_names.append(sample['video_name'])
        categories.append(category)
    
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    y_scores = np.array(video_scores)
    
    logger.info(f"✓ Completed predictions for {len(y_true)} videos")
    
    # Compute per-class metrics
    from sklearn.metrics import classification_report
    
    print("\n" + "=" * 80)
    print("MULTICLASS CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Shoplifting', 'Other Anomaly'],
        digits=4
    ))
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               N    SL   Other")
    print(f"Actual N     {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
    print(f"       SL    {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
    print(f"       Other {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'evaluation_type': 'Multiclass (Normal / Shoplifting / Other)',
        'confusion_matrix': cm.tolist(),
        'class_names': ['Normal', 'Shoplifting', 'Other Anomaly'],
        'dataset_statistics': {
            'total_videos': len(y_true),
            'normal_videos': len(filtered_data['normal']),
            'shoplifting_videos': len(filtered_data['shoplifting']),
            'other_anomaly_videos': len(filtered_data['other_anomaly'])
        }
    }
    
    save_evaluation_report(report, output_dir / 'multiclass_report.json')
    
    # Visualize multiclass confusion matrix
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    plot_confusion_matrix(
        cm,
        ['Normal', 'Shoplifting', 'Other'],
        title="Multiclass Confusion Matrix",
        normalize=False,
        save_path=viz_dir / 'confusion_matrix_multiclass.png'
    )
    
    plot_confusion_matrix(
        cm,
        ['Normal', 'Shoplifting', 'Other'],
        title="Multiclass Confusion Matrix (Normalized)",
        normalize=True,
        save_path=viz_dir / 'confusion_matrix_multiclass_normalized.png'
    )
    
    logger.info(f"✓ Multiclass results saved")
    
    return report


def compare_with_general_anomaly(
    filtered_data: dict,
    predictor,
    output_dir: Path,
    config: dict
):
    """
    Compare shoplifting detection with general anomaly detection.
    
    Args:
        filtered_data: Filtered dataset
        predictor: MSLPredictor
        output_dir: Output directory
        config: Configuration dict
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Shoplifting vs General Anomaly Detection")
    logger.info("=" * 80)
    
    # Strategy 1: Binary (Normal vs All Anomalies)
    all_data_binary = filtered_data['normal'] + filtered_data['shoplifting'] + filtered_data['other_anomaly']
    
    y_true_binary = []
    y_pred_binary = []
    y_scores_binary = []
    
    logger.info("Evaluating general anomaly detection (Normal vs Any Anomaly)...")
    for idx, sample, category in all_data_binary:
        features = sample['features']
        result = predictor.predict(features)
        
        true_label = 0 if category == 'Normal' else 1
        y_true_binary.append(true_label)
        y_pred_binary.append(result['prediction'])
        y_scores_binary.append(result['video_score'])
    
    y_true_binary = np.array(y_true_binary)
    y_pred_binary = np.array(y_pred_binary)
    y_scores_binary = np.array(y_scores_binary)
    
    metrics_general = compute_basic_metrics(y_true_binary, y_pred_binary, y_scores_binary)
    
    # Strategy 2: Binary (Normal vs Shoplifting only)
    shoplifting_only = filtered_data['normal'] + filtered_data['shoplifting']
    
    y_true_sl = []
    y_pred_sl = []
    y_scores_sl = []
    
    logger.info("Evaluating shoplifting-specific detection (Normal vs Shoplifting)...")
    for idx, sample, category in shoplifting_only:
        features = sample['features']
        result = predictor.predict(features)
        
        true_label = 1 if category == 'Shoplifting' else 0
        y_true_sl.append(true_label)
        y_pred_sl.append(result['prediction'])
        y_scores_sl.append(result['video_score'])
    
    y_true_sl = np.array(y_true_sl)
    y_pred_sl = np.array(y_pred_sl)
    y_scores_sl = np.array(y_scores_sl)
    
    metrics_shoplifting = compute_basic_metrics(y_true_sl, y_pred_sl, y_scores_sl)
    
    # Compare
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'General Anomaly':<20} {'Shoplifting-Specific':<20} {'Difference':<15}")
    print("-" * 80)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        general_val = metrics_general[metric]
        shoplifting_val = metrics_shoplifting[metric]
        diff = shoplifting_val - general_val
        diff_str = f"{diff:+.4f}"
        
        print(f"{metric:<20} {general_val:<20.4f} {shoplifting_val:<20.4f} {diff_str:<15}")
    
    print("=" * 80)
    
    # Visualize comparison
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    metrics_dict = {
        'General Anomaly': {k: metrics_general[k] for k in ['accuracy', 'precision', 'recall', 'f1']},
        'Shoplifting-Specific': {k: metrics_shoplifting[k] for k in ['accuracy', 'precision', 'recall', 'f1']}
    }
    
    plot_metrics_comparison(
        metrics_dict,
        title="Performance Comparison: General vs Shoplifting-Specific",
        save_path=viz_dir / 'comparison_general_vs_shoplifting.png'
    )
    
    # Save comparison report
    comparison_report = {
        'general_anomaly_detection': metrics_general,
        'shoplifting_specific_detection': metrics_shoplifting,
        'improvements': {
            metric: float(metrics_shoplifting[metric] - metrics_general[metric])
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    }
    
    save_evaluation_report(comparison_report, output_dir / 'comparison_report.json')
    
    logger.info(f"✓ Comparison results saved")
    
    return comparison_report


def main():
    parser = argparse.ArgumentParser(description='Evaluate MSL model on shoplifting detection')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Directory containing feature files')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default='results/shoplifting_evaluation',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--shoplifting-keywords', type=str, nargs='+',
                        default=['shoplifting', 'stealing', 'theft'],
                        help='Keywords to identify shoplifting videos')
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
    
    # Load dataset
    logger.info(f"Loading dataset from {feature_dir}...")
    dataset = UCFCrimeFeatureDataset(feature_dir=feature_dir, split=args.split)
    logger.info(f"  Found {len(dataset)} videos")
    
    # Load predictor
    logger.info("Loading model...")
    predictor = load_predictor_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_class=MSLModel,
        model_config=config['model'],
        device=args.device,
        score_correction_method=config.get('score_correction', {}).get('method', 'linear'),
        score_correction_alpha=config.get('score_correction', {}).get('alpha', 0.5),
        temporal_smoothing=config.get('temporal_smoothing', {}).get('enabled', True),
        smoothing_window=config.get('temporal_smoothing', {}).get('window_size', 5),
        anomaly_threshold=config.get('anomaly_threshold', 0.5)
    )
    
    # Filter dataset
    filtered_data = filter_shoplifting_videos(dataset, predictor, args.shoplifting_keywords)
    
    # Run evaluations
    logger.info("\n" + "=" * 80)
    logger.info("SHOPLIFTING-SPECIFIC EVALUATION PIPELINE")
    logger.info("=" * 80)
    
    # 1. Normal vs Shoplifting
    if len(filtered_data['shoplifting']) > 0:
        evaluate_normal_vs_shoplifting(filtered_data, predictor, output_dir, config)
    else:
        logger.warning("No shoplifting videos found! Skipping Normal vs Shoplifting evaluation.")
    
    # 2. Multiclass (if we have other anomalies)
    if len(filtered_data['other_anomaly']) > 0:
        evaluate_multiclass(filtered_data, predictor, output_dir, config)
    else:
        logger.info("No other anomaly videos found. Skipping multiclass evaluation.")
    
    # 3. Comparison with general anomaly detection
    if len(filtered_data['shoplifting']) > 0:
        compare_with_general_anomaly(filtered_data, predictor, output_dir, config)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SHOPLIFTING EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations: {output_dir / 'visualizations'}")
    print("=" * 80)
    
    logger.info("\n✓ Shoplifting evaluation complete!")


if __name__ == "__main__":
    main()
