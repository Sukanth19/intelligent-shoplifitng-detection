"""
Video Preprocessor
Handles video loading, frame sampling, and preprocessing for VideoSwin
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """
    Video preprocessing for VideoSwin Transformer.
    
    Based on MSL authors' configuration:
    - Resize: scale=(-1, 224)
    - CenterCrop: crop_size=(288, 224)
    - Normalize: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    - Format: NCTHW
    """
    
    def __init__(
        self,
        target_height: int = 224,
        target_width: int = 288,
        crop_height: int = 224,
        crop_width: int = 224,
        mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
        num_frames: int = 32,
        frame_interval: int = 2
    ):
        """
        Args:
            target_height: Resize height
            target_width: Resize width (for shorter side)
            crop_height: Center crop height
            crop_width: Center crop width
            mean: RGB mean for normalization
            std: RGB std for normalization
            num_frames: Number of frames to sample per clip
            frame_interval: Frame sampling interval
        """
        self.target_height = target_height
        self.target_width = target_width
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
    
    def load_video(self, video_path: Path) -> Tuple[np.ndarray, dict]:
        """
        Load video and extract metadata.
        
        Args:
            video_path: Path to video file
        
        Returns:
            frames: numpy array of shape (T, H, W, C)
            metadata: dict with fps, duration, num_frames
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        metadata = {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration
        }
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        frames = np.array(frames)  # (T, H, W, C)
        
        logger.info(f"Loaded video: {video_path.name}")
        logger.info(f"  Shape: {frames.shape}")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Duration: {duration:.2f}s")
        
        return frames, metadata
    
    def sample_frames(
        self,
        frames: np.ndarray,
        num_clips: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Sample frames into clips.
        
        Args:
            frames: Array of shape (T, H, W, C)
            num_clips: Number of clips to sample (None = all possible)
        
        Returns:
            List of clip arrays, each of shape (num_frames, H, W, C)
        """
        T = frames.shape[0]
        clip_length = self.num_frames * self.frame_interval
        
        if T < clip_length:
            # Video too short - pad or repeat
            logger.warning(f"Video has only {T} frames, need {clip_length}. Padding...")
            # Repeat last frame
            padding = clip_length - T
            last_frame = frames[-1:].repeat(padding, axis=0)
            frames = np.concatenate([frames, last_frame], axis=0)
            T = frames.shape[0]
        
        # Calculate number of possible clips
        max_clips = (T - clip_length) // clip_length + 1
        
        if num_clips is None:
            num_clips = max_clips
        else:
            num_clips = min(num_clips, max_clips)
        
        clips = []
        for i in range(num_clips):
            start_idx = i * clip_length
            # Sample frames with interval
            indices = start_idx + np.arange(self.num_frames) * self.frame_interval
            clip = frames[indices]  # (num_frames, H, W, C)
            clips.append(clip)
        
        logger.debug(f"Sampled {len(clips)} clips from {T} frames")
        
        return clips
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame maintaining aspect ratio.
        
        Args:
            frame: Frame of shape (H, W, C)
        
        Returns:
            Resized frame
        """
        H, W = frame.shape[:2]
        
        # Resize shorter side to target_height
        if H < W:
            new_H = self.target_height
            new_W = int(W * (self.target_height / H))
        else:
            new_W = self.target_width
            new_H = int(H * (self.target_width / W))
        
        resized = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def center_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Center crop frame.
        
        Args:
            frame: Frame of shape (H, W, C)
        
        Returns:
            Cropped frame of shape (crop_height, crop_width, C)
        """
        H, W = frame.shape[:2]
        
        # Calculate crop coordinates
        start_h = (H - self.crop_height) // 2
        start_w = (W - self.crop_width) // 2
        
        # Handle case where frame is smaller than crop size
        if H < self.crop_height or W < self.crop_width:
            # Pad with zeros
            pad_h = max(0, self.crop_height - H)
            pad_w = max(0, self.crop_width - W)
            frame = np.pad(
                frame,
                ((pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2),
                 (0, 0)),
                mode='constant'
            )
            start_h = 0
            start_w = 0
        
        cropped = frame[start_h:start_h + self.crop_height,
                       start_w:start_w + self.crop_width]
        
        return cropped
    
    def normalize(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalize frames.
        
        Args:
            frames: Array of shape (T, H, W, C) or (H, W, C)
        
        Returns:
            Normalized frames
        """
        # Normalize: (x - mean) / std
        normalized = (frames - self.mean) / self.std
        return normalized.astype(np.float32)
    
    def preprocess_clip(self, clip: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single clip.
        
        Args:
            clip: Array of shape (T, H, W, C)
        
        Returns:
            Tensor of shape (C, T, H, W) - ready for VideoSwin
        """
        # Resize each frame
        resized_frames = []
        for frame in clip:
            resized = self.resize_frame(frame)
            cropped = self.center_crop(resized)
            resized_frames.append(cropped)
        
        resized_clip = np.stack(resized_frames)  # (T, H, W, C)
        
        # Normalize
        normalized = self.normalize(resized_clip)  # (T, H, W, C)
        
        # Convert to tensor and rearrange to (C, T, H, W)
        tensor = torch.from_numpy(normalized).permute(3, 0, 1, 2)  # (C, T, H, W)
        
        return tensor
    
    def process_video(
        self,
        video_path: Path,
        num_clips: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Complete preprocessing pipeline for a video.
        
        Args:
            video_path: Path to video file
            num_clips: Number of clips to extract (None = all)
        
        Returns:
            clips: Tensor of shape (N, C, T, H, W) where N is number of clips
            metadata: Video metadata dict
        """
        # Load video
        frames, metadata = self.load_video(video_path)
        
        # Sample into clips
        clips = self.sample_frames(frames, num_clips)
        
        # Preprocess each clip
        processed_clips = []
        for clip in clips:
            processed = self.preprocess_clip(clip)
            processed_clips.append(processed)
        
        # Stack into batch
        clips_tensor = torch.stack(processed_clips)  # (N, C, T, H, W)
        
        metadata['num_clips'] = len(processed_clips)
        
        return clips_tensor, metadata


if __name__ == "__main__":
    # Test video preprocessor
    print("=" * 80)
    print("TESTING VIDEO PREPROCESSOR")
    print("=" * 80)
    
    # Create synthetic video (for testing without real video)
    print("\n1. Creating synthetic video...")
    synthetic_frames = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
    print(f"   Synthetic video shape: {synthetic_frames.shape}")
    print("   ✓ Synthetic video created")
    
    # Create preprocessor
    print("\n2. Creating preprocessor...")
    preprocessor = VideoPreprocessor(
        target_height=224,
        target_width=288,
        crop_height=224,
        crop_width=224,
        num_frames=32,
        frame_interval=2
    )
    print("   ✓ Preprocessor created")
    
    # Test frame sampling
    print("\n3. Testing frame sampling...")
    clips = preprocessor.sample_frames(synthetic_frames, num_clips=2)
    print(f"   Number of clips: {len(clips)}")
    print(f"   Clip shape: {clips[0].shape}")
    print("   ✓ Frame sampling works")
    
    # Test resize
    print("\n4. Testing resize...")
    test_frame = synthetic_frames[0]
    resized = preprocessor.resize_frame(test_frame)
    print(f"   Original: {test_frame.shape}")
    print(f"   Resized: {resized.shape}")
    print("   ✓ Resize works")
    
    # Test center crop
    print("\n5. Testing center crop...")
    cropped = preprocessor.center_crop(resized)
    print(f"   Cropped: {cropped.shape}")
    print("   ✓ Center crop works")
    
    # Test normalization
    print("\n6. Testing normalization...")
    normalized = preprocessor.normalize(cropped)
    print(f"   Normalized shape: {normalized.shape}")
    print(f"   Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    print("   ✓ Normalization works")
    
    # Test clip preprocessing
    print("\n7. Testing clip preprocessing...")
    clip_tensor = preprocessor.preprocess_clip(clips[0])
    print(f"   Clip tensor shape: {clip_tensor.shape}")
    print(f"   Expected: (C=3, T=32, H=224, W=224)")
    assert clip_tensor.shape == (3, 32, 224, 224)
    print("   ✓ Clip preprocessing works")
    
    print("\n" + "=" * 80)
    print("✓ ALL PREPROCESSOR TESTS PASSED")
    print("=" * 80)
