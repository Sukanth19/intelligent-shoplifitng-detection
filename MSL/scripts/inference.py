"""
Inference Script
Run anomaly detection on videos using trained MSL model
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
from src.inference import MSLPredictor, load_predictor_from_checkpoint
from src.datasets import UCFCrimeFeatureDataset


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


def load_feature_file(feature_path: Path) -> torch.Tensor:
    """
    Load feature file.
    
    Supports:
    - .npy files
    - .pt/.pth files
    - .npz files
    
    Args:
        feature_path: Path to feature file
    
    Returns:
        Feature tensor (T, D)
    """
    suffix = feature_path.suffix.lower()
    
    if suffix == '.npy':
        features = np.load(feature_path)
        features = torch.from_numpy(features).float()
    elif suffix in ['.pt', '.pth']:
        features = torch.load(feature_path, weights_only=True)
    elif suffix == '.npz':
        data = np.load(feature_path)
        # Try common keys
        if 'features' in data:
            features = torch.from_numpy(data['features']).float()
        elif 'feature' in data:
            features = torch.from_numpy(data['feature']).float()
        else:
            # Use first array
            key = list(data.keys())[0]
            features = torch.from_numpy(data[key]).float()
    else:
        raise ValueError(f"Unsupported feature file format: {suffix}")
    
    # Ensure 2D: (T, D)
    if features.ndim == 1:
        features = features.unsqueeze(0)
    
    return features


def run_single_video_inference(
    feature_path: Path,
    checkpoint_path: Path,
    config: dict,
    output_dir: Path,
    device: str = 'cuda'
):
    """
    Run inference on a single video.
    
    Args:
        feature_path: Path to video features
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        output_dir: Output directory
        device: Device for inference
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running inference on: {feature_path.name}")
    
    # Load features
    logger.info("Loading features...")
    features = load_feature_file(feature_path)
    logger.info(f"  Feature shape: {features.shape}")
    
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
    
    # Predict
    logger.info("Running prediction...")
    result = predictor.predict(features)
    
    # Format and print
    formatted = predictor.format_prediction(
        result,
        video_name=feature_path.name,
        fps=config.get('fps', 30.0)
    )
    print("\n" + "=" * 80)
    print(formatted)
    print("=" * 80)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save formatted text
    output_text = output_dir / f"{feature_path.stem}_prediction.txt"
    with open(output_text, 'w') as f:
        f.write(formatted)
    logger.info(f"Saved text prediction to: {output_text}")
    
    # Save numpy arrays
    output_npz = output_dir / f"{feature_path.stem}_scores.npz"
    np.savez(
        output_npz,
        video_score=result['video_score'],
        snippet_scores=result['snippet_scores'].numpy(),
        corrected_scores=result['corrected_scores'].numpy(),
        smoothed_scores=result['smoothed_scores'].numpy(),
        prediction=result['prediction']
    )
    logger.info(f"Saved score arrays to: {output_npz}")
    
    return result


def run_batch_inference(
    feature_dir: Path,
    checkpoint_path: Path,
    config: dict,
    output_dir: Path,
    device: str = 'cuda',
    split: str = 'test'
):
    """
    Run inference on multiple videos from dataset.
    
    Args:
        feature_dir: Directory containing features
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        output_dir: Output directory
        device: Device for inference
        split: Dataset split ('train' or 'test')
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running batch inference on {split} set")
    logger.info(f"Feature directory: {feature_dir}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = UCFCrimeFeatureDataset(
        feature_dir=feature_dir,
        split=split
    )
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
    results = []
    video_names = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        features = sample['features']
        label = sample['label']
        video_name = sample['video_name']
        
        result = predictor.predict(features)
        result['true_label'] = label
        results.append(result)
        video_names.append(video_name)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(dataset)} videos")
    
    logger.info(f"Completed predictions for {len(results)} videos")
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    predictions = np.array([r['prediction'] for r in results])
    true_labels = np.array([r['true_label'] for r in results])
    video_scores = np.array([r['video_score'] for r in results])
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    auc = roc_auc_score(true_labels, video_scores)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'num_videos': len(results),
        'num_anomalies': int(true_labels.sum()),
        'num_predicted_anomalies': int(predictions.sum())
    }
    
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"\nTotal videos: {len(results)}")
    print(f"True anomalies: {int(true_labels.sum())}")
    print(f"Predicted anomalies: {int(predictions.sum())}")
    print("=" * 80)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    import json
    metrics_file = output_dir / f"{split}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_file}")
    
    # Save predictions
    predictions_file = output_dir / f"{split}_predictions.npz"
    np.savez(
        predictions_file,
        video_names=video_names,
        video_scores=video_scores,
        predictions=predictions,
        true_labels=true_labels
    )
    logger.info(f"Saved predictions to: {predictions_file}")
    
    # Save detailed text report
    report_file = output_dir / f"{split}_report.txt"
    predictor.save_predictions(results, video_names, report_file)
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description='Run MSL inference on videos')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    
    # Input mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--feature', type=str,
                       help='Path to single feature file')
    group.add_argument('--feature-dir', type=str,
                       help='Directory containing feature files for batch inference')
    
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test'],
                        help='Dataset split for batch inference')
    parser.add_argument('--output', type=str, default='results/inference',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
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
    output_dir = Path(args.output)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Run inference
    if args.feature:
        # Single video inference
        feature_path = Path(args.feature)
        if not feature_path.exists():
            logger.error(f"Feature file not found: {feature_path}")
            return
        
        run_single_video_inference(
            feature_path=feature_path,
            checkpoint_path=checkpoint_path,
            config=config,
            output_dir=output_dir,
            device=args.device
        )
    
    elif args.feature_dir:
        # Batch inference
        feature_dir = Path(args.feature_dir)
        if not feature_dir.exists():
            logger.error(f"Feature directory not found: {feature_dir}")
            return
        
        run_batch_inference(
            feature_dir=feature_dir,
            checkpoint_path=checkpoint_path,
            config=config,
            output_dir=output_dir,
            device=args.device,
            split=args.split
        )
    
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
