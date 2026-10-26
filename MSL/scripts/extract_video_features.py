"""
Video Feature Extraction Script
Extract VideoSwin features from raw videos
"""

import argparse
from pathlib import Path
import sys
import logging
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.videoswin import VideoPreprocessor, VideoSwinFeatureExtractor


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


def extract_single_video(
    video_path: Path,
    output_path: Path,
    preprocessor: VideoPreprocessor,
    extractor: VideoSwinFeatureExtractor,
    save_format: str = 'npy'
):
    """
    Extract features from a single video.
    
    Args:
        video_path: Path to video file
        output_path: Path to save features
        preprocessor: VideoPreprocessor instance
        extractor: VideoSwinFeatureExtractor instance
        save_format: Format to save ('npy' or 'npz')
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing: {video_path.name}")
    
    try:
        # Preprocess video
        video_clips, metadata = preprocessor.process_video(video_path)
        logger.info(f"  Clips extracted: {metadata['num_clips']}")
        
        # Extract features
        features = extractor.extract_temporal_features(video_clips)
        logger.info(f"  Features shape: {features.shape}")
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import numpy as np
        if save_format == 'npy':
            np.save(output_path, features)
        elif save_format == 'npz':
            np.savez(output_path, features=features, **metadata)
        
        logger.info(f"  Saved to: {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"  Error processing {video_path.name}: {e}")
        return False


def extract_batch(
    video_dir: Path,
    output_dir: Path,
    video_extensions: list = ['.mp4', '.avi', '.mov', '.mkv'],
    preprocessor_config: dict = None,
    extractor_config: dict = None,
    save_format: str = 'npy'
):
    """
    Extract features from all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save features
        video_extensions: List of video file extensions
        preprocessor_config: Preprocessor configuration
        extractor_config: Extractor configuration
        save_format: Save format
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("BATCH VIDEO FEATURE EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Input directory: {video_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find all videos
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    video_files = sorted(set(video_files))
    
    logger.info(f"Found {len(video_files)} videos")
    
    if len(video_files) == 0:
        logger.warning("No videos found!")
        return
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(**(preprocessor_config or {}))
    
    # Create extractor
    extractor = VideoSwinFeatureExtractor(**(extractor_config or {}))
    extractor.load_model()
    
    # Process each video
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"\nProcessing {i}/{len(video_files)}: {video_path.name}")
        
        # Output path
        output_filename = video_path.stem + '.npy'
        output_path = output_dir / output_filename
        
        # Skip if already exists
        if output_path.exists():
            logger.info(f"  Skipping (already exists)")
            success_count += 1
            continue
        
        # Extract
        success = extract_single_video(
            video_path,
            output_path,
            preprocessor,
            extractor,
            save_format
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total videos: {len(video_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Extract VideoSwin features from videos')
    
    # Input/output
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str,
                       help='Path to single video file')
    group.add_argument('--video-dir', type=str,
                       help='Directory containing videos')
    
    parser.add_argument('--output', type=str, required=True,
                        help='Output path (file or directory)')
    
    # Preprocessor config
    parser.add_argument('--num-frames', type=int, default=32,
                        help='Number of frames per clip')
    parser.add_argument('--frame-interval', type=int, default=2,
                        help='Frame sampling interval')
    parser.add_argument('--target-height', type=int, default=224,
                        help='Target height after resize')
    parser.add_argument('--target-width', type=int, default=288,
                        help='Target width after resize')
    
    # Extractor config
    parser.add_argument('--model', type=str, default='swin_base_patch244_window877',
                        help='VideoSwin model name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to VideoSwin checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for extraction')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing')
    
    # Other options
    parser.add_argument('--save-format', type=str, default='npy',
                        choices=['npy', 'npz'],
                        help='Format to save features')
    parser.add_argument('--video-extensions', type=str, nargs='+',
                        default=['.mp4', '.avi', '.mov', '.mkv'],
                        help='Video file extensions to process')
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
    
    # Preprocessor config
    preprocessor_config = {
        'num_frames': args.num_frames,
        'frame_interval': args.frame_interval,
        'target_height': args.target_height,
        'target_width': args.target_width
    }
    
    # Extractor config
    extractor_config = {
        'model_name': args.model,
        'checkpoint_path': Path(args.checkpoint) if args.checkpoint else None,
        'device': args.device,
        'batch_size': args.batch_size
    }
    
    # Single video or batch
    if args.video:
        # Single video
        video_path = Path(args.video)
        output_path = Path(args.output)
        
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return
        
        preprocessor = VideoPreprocessor(**preprocessor_config)
        extractor = VideoSwinFeatureExtractor(**extractor_config)
        extractor.load_model()
        
        extract_single_video(
            video_path,
            output_path,
            preprocessor,
            extractor,
            args.save_format
        )
    
    elif args.video_dir:
        # Batch processing
        video_dir = Path(args.video_dir)
        output_dir = Path(args.output)
        
        if not video_dir.exists():
            logger.error(f"Video directory not found: {video_dir}")
            return
        
        extract_batch(
            video_dir,
            output_dir,
            args.video_extensions,
            preprocessor_config,
            extractor_config,
            args.save_format
        )


if __name__ == "__main__":
    main()
