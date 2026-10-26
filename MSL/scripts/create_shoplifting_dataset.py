"""
Create Shoplifting-Specific Synthetic Dataset
For testing shoplifting evaluation pipeline
"""

import numpy as np
from pathlib import Path
import json
import argparse


def create_shoplifting_synthetic_dataset(
    output_dir: Path,
    num_normal: int = 10,
    num_shoplifting: int = 10,
    num_other_anomaly: int = 5,
    feature_dim: int = 1024,
    min_length: int = 30,
    max_length: int = 80
):
    """
    Create synthetic dataset with shoplifting videos.
    
    Args:
        output_dir: Output directory
        num_normal: Number of normal videos
        num_shoplifting: Number of shoplifting videos
        num_other_anomaly: Number of other anomaly videos
        feature_dim: Feature dimension
        min_length: Minimum video length (snippets)
        max_length: Maximum video length (snippets)
    """
    print("=" * 80)
    print("CREATING SHOPLIFTING-SPECIFIC SYNTHETIC DATASET")
    print("=" * 80)
    
    # Create directories
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    labels_dir = output_dir / 'labels'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    train_labels = []
    test_labels = []
    
    np.random.seed(42)
    
    # === TRAINING SET ===
    print("\nCreating training set...")
    
    # Normal videos
    print(f"  Creating {num_normal} normal videos...")
    for i in range(num_normal):
        length = np.random.randint(min_length, max_length)
        # Normal: low values, little variation
        features = np.random.randn(length, feature_dim) * 0.3
        
        filename = f"Normal_{i:03d}.npy"
        np.save(train_dir / filename, features.astype(np.float32))
        train_labels.append({
            'filename': filename,
            'label': 0,
            'category': 'Normal',
            'length': length
        })
    
    # Shoplifting videos
    print(f"  Creating {num_shoplifting} shoplifting videos...")
    for i in range(num_shoplifting):
        length = np.random.randint(min_length, max_length)
        
        # Shoplifting: high values in middle section (anomalous behavior)
        features = np.random.randn(length, feature_dim) * 0.3
        
        # Add anomalous region in the middle third
        anomaly_start = length // 3
        anomaly_end = 2 * length // 3
        features[anomaly_start:anomaly_end] += np.random.randn(anomaly_end - anomaly_start, feature_dim) * 1.5
        
        filename = f"Shoplifting_{i:03d}.npy"
        np.save(train_dir / filename, features.astype(np.float32))
        train_labels.append({
            'filename': filename,
            'label': 1,
            'category': 'Shoplifting',
            'length': length,
            'anomaly_region': [anomaly_start, anomaly_end]
        })
    
    # Other anomaly videos (for multiclass testing)
    print(f"  Creating {num_other_anomaly} other anomaly videos...")
    anomaly_types = ['Robbery', 'Burglary', 'Assault', 'Vandalism', 'Stealing']
    for i in range(num_other_anomaly):
        length = np.random.randint(min_length, max_length)
        
        # Other anomalies: different pattern than shoplifting
        features = np.random.randn(length, feature_dim) * 0.3
        
        # Add anomalous spikes (different from shoplifting pattern)
        num_spikes = np.random.randint(2, 5)
        for _ in range(num_spikes):
            spike_pos = np.random.randint(0, length)
            spike_width = np.random.randint(3, 8)
            spike_end = min(spike_pos + spike_width, length)
            features[spike_pos:spike_end] += np.random.randn(spike_end - spike_pos, feature_dim) * 2.0
        
        anomaly_type = anomaly_types[i % len(anomaly_types)]
        filename = f"{anomaly_type}_{i:03d}.npy"
        np.save(train_dir / filename, features.astype(np.float32))
        train_labels.append({
            'filename': filename,
            'label': 1,
            'category': anomaly_type,
            'length': length
        })
    
    # === TEST SET ===
    print("\nCreating test set...")
    
    # Normal videos (fewer)
    num_test_normal = max(3, num_normal // 2)
    print(f"  Creating {num_test_normal} normal videos...")
    for i in range(num_test_normal):
        length = np.random.randint(min_length, max_length)
        features = np.random.randn(length, feature_dim) * 0.3
        
        filename = f"Normal_{i:03d}.npy"
        np.save(test_dir / filename, features.astype(np.float32))
        test_labels.append({
            'filename': filename,
            'label': 0,
            'category': 'Normal',
            'length': length
        })
    
    # Shoplifting videos (fewer)
    num_test_shoplifting = max(3, num_shoplifting // 2)
    print(f"  Creating {num_test_shoplifting} shoplifting videos...")
    for i in range(num_test_shoplifting):
        length = np.random.randint(min_length, max_length)
        features = np.random.randn(length, feature_dim) * 0.3
        
        anomaly_start = length // 3
        anomaly_end = 2 * length // 3
        features[anomaly_start:anomaly_end] += np.random.randn(anomaly_end - anomaly_start, feature_dim) * 1.5
        
        filename = f"Shoplifting_{i:03d}.npy"
        np.save(test_dir / filename, features.astype(np.float32))
        test_labels.append({
            'filename': filename,
            'label': 1,
            'category': 'Shoplifting',
            'length': length,
            'anomaly_region': [anomaly_start, anomaly_end]
        })
    
    # Other anomalies (fewer)
    num_test_other = max(2, num_other_anomaly // 2)
    print(f"  Creating {num_test_other} other anomaly videos...")
    for i in range(num_test_other):
        length = np.random.randint(min_length, max_length)
        features = np.random.randn(length, feature_dim) * 0.3
        
        num_spikes = np.random.randint(2, 5)
        for _ in range(num_spikes):
            spike_pos = np.random.randint(0, length)
            spike_width = np.random.randint(3, 8)
            spike_end = min(spike_pos + spike_width, length)
            features[spike_pos:spike_end] += np.random.randn(spike_end - spike_pos, feature_dim) * 2.0
        
        anomaly_type = anomaly_types[i % len(anomaly_types)]
        filename = f"{anomaly_type}_{i:03d}.npy"
        np.save(test_dir / filename, features.astype(np.float32))
        test_labels.append({
            'filename': filename,
            'label': 1,
            'category': anomaly_type,
            'length': length
        })
    
    # Save labels
    print("\nSaving labels...")
    with open(labels_dir / 'train_labels.json', 'w') as f:
        json.dump(train_labels, f, indent=2)
    
    with open(labels_dir / 'test_labels.json', 'w') as f:
        json.dump(test_labels, f, indent=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("DATASET CREATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTraining set:")
    print(f"  Normal videos:        {num_normal}")
    print(f"  Shoplifting videos:   {num_shoplifting}")
    print(f"  Other anomaly videos: {num_other_anomaly}")
    print(f"  Total:                {num_normal + num_shoplifting + num_other_anomaly}")
    print(f"\nTest set:")
    print(f"  Normal videos:        {num_test_normal}")
    print(f"  Shoplifting videos:   {num_test_shoplifting}")
    print(f"  Other anomaly videos: {num_test_other}")
    print(f"  Total:                {num_test_normal + num_test_shoplifting + num_test_other}")
    print("\n" + "=" * 80)
    
    return {
        'train': {'normal': num_normal, 'shoplifting': num_shoplifting, 'other': num_other_anomaly},
        'test': {'normal': num_test_normal, 'shoplifting': num_test_shoplifting, 'other': num_test_other}
    }


def main():
    parser = argparse.ArgumentParser(description='Create shoplifting-specific synthetic dataset')
    
    parser.add_argument('--output', type=str, default='data/synthetic_shoplifting',
                        help='Output directory')
    parser.add_argument('--num-normal', type=int, default=10,
                        help='Number of normal videos (training)')
    parser.add_argument('--num-shoplifting', type=int, default=10,
                        help='Number of shoplifting videos (training)')
    parser.add_argument('--num-other', type=int, default=5,
                        help='Number of other anomaly videos (training)')
    parser.add_argument('--feature-dim', type=int, default=1024,
                        help='Feature dimension')
    parser.add_argument('--min-length', type=int, default=30,
                        help='Minimum video length (snippets)')
    parser.add_argument('--max-length', type=int, default=80,
                        help='Maximum video length (snippets)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    create_shoplifting_synthetic_dataset(
        output_dir=output_dir,
        num_normal=args.num_normal,
        num_shoplifting=args.num_shoplifting,
        num_other_anomaly=args.num_other,
        feature_dim=args.feature_dim,
        min_length=args.min_length,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
