"""
Video processing utilities for soccer player recognition.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import tempfile
import os


class VideoProcessor:
    """Video processing utilities for soccer player recognition."""
    
    def __init__(self, target_fps: int = 30, target_resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize video processor.
        
        Args:
            target_fps: Target frames per second
            target_resolution: Target resolution (width, height)
        """
        self.target_fps = target_fps
        self.target_resolution = target_resolution
    
    def extract_frames(self, video_path: str, 
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      sample_rate: int = 1) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            sample_rate: Extract every nth frame
            
        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame ranges
        start_frame = 0
        end_frame = total_frames
        
        if start_time is not None:
            start_frame = int(start_time * fps)
        
        if end_time is not None:
            end_frame = int(end_time * fps)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on sample_rate
            if frame_count < start_frame:
                frame_count += 1
                continue
            
            if frame_count >= end_frame:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def resize_frame(self, frame: np.ndarray, 
                    target_resolution: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize frame to target resolution.
        
        Args:
            frame: Input frame
            target_resolution: Target resolution (width, height)
            
        Returns:
            Resized frame
        """
        if target_resolution is None:
            target_resolution = self.target_resolution
            
        return cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
    
    def normalize_frame(self, frame: np.ndarray, 
                       mean: Optional[Tuple[float, float, float]] = None,
                       std: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Normalize frame pixel values.
        
        Args:
            frame: Input frame
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            
        Returns:
            Normalized frame
        """
        if mean is None:
            mean = (0.485, 0.456, 0.406)  # ImageNet mean
        if std is None:
            std = (0.229, 0.224, 0.225)  # ImageNet std
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Apply normalization
        frame_normalized = (frame_normalized - mean) / std
        
        return frame_normalized
    
    def create_video(self, frames: List[np.ndarray], 
                    output_path: str,
                    fps: Optional[int] = None) -> None:
        """
        Create video from frames.
        
        Args:
            frames: List of frames
            output_path: Output video path
            fps: Frames per second
        """
        if fps is None:
            fps = self.target_fps
        
        if not frames:
            raise ValueError("No frames provided")
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
    
    def sample_frames_uniformly(self, frames: List[np.ndarray], 
                               target_count: int) -> List[np.ndarray]:
        """
        Sample frames uniformly from a list.
        
        Args:
            frames: List of frames
            target_count: Number of frames to sample
            
        Returns:
            Sampled frames
        """
        if len(frames) <= target_count:
            return frames
        
        indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
        return [frames[i] for i in indices]
    
    def detect_scene_changes(self, frames: List[np.ndarray], 
                            threshold: float = 0.3) -> List[int]:
        """
        Detect scene changes in video frames.
        
        Args:
            frames: List of frames
            threshold: Similarity threshold for scene change detection
            
        Returns:
            List of frame indices where scene changes occur
        """
        if len(frames) < 2:
            return []
        
        scene_changes = []
        
        # Convert frames to grayscale for comparison
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram similarity
            hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
            hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
            
            similarity = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
            
            # If similarity is below threshold, it's a scene change
            if similarity < (1 - threshold):
                scene_changes.append(i)
            
            prev_gray = curr_gray
        
        return scene_changes
    
    def enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality using basic image processing techniques.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # Apply histogram equalization for better contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced


def load_video_frames(video_path: str, 
                     frame_processor: Optional[VideoProcessor] = None,
                     **kwargs) -> torch.Tensor:
    """
    Load video frames as tensor for model input.
    
    Args:
        video_path: Path to video file
        frame_processor: VideoProcessor instance
        **kwargs: Additional arguments for frame processing
        
    Returns:
        Tensor of processed frames
    """
    if frame_processor is None:
        frame_processor = VideoProcessor()
    
    # Extract frames
    frames = frame_processor.extract_frames(video_path, **kwargs)
    
    # Process frames
    processed_frames = []
    for frame in frames:
        # Resize
        frame = frame_processor.resize_frame(frame)
        
        # Enhance quality
        frame = frame_processor.enhance_frame_quality(frame)
        
        # Normalize
        frame = frame_processor.normalize_frame(frame)
        
        processed_frames.append(frame)
    
    # Convert to tensor (C, T, H, W) format
    if processed_frames:
        processed_frames = np.array(processed_frames)
        processed_frames = torch.from_numpy(processed_frames)
        # Convert from (T, H, W, C) to (C, T, H, W)
        processed_frames = processed_frames.permute(3, 0, 1, 2)
    
    return processed_frames