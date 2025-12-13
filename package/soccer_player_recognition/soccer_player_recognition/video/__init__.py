"""Video processing module for handling video streams and files.

This module provides utilities for video input/output, streaming,
and frame processing.
"""

from .video_processor import VideoProcessor
from .stream_processor import StreamProcessor
from .frame_extractor import FrameExtractor
from .video_writer import VideoWriter

__all__ = [
    "VideoProcessor",
    "StreamProcessor",
    "FrameExtractor", 
    "VideoWriter"
]