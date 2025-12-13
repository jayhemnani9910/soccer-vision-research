"""Tracking module for multi-object tracking of soccer players.

This module implements tracking algorithms to maintain consistent
player IDs across video frames.
"""

from .deepsort_tracker import DeepSortTracker
from .tracking_engine import TrackingEngine
from .kalman_filter import KalmanFilter
from .feature_extractor import FeatureExtractor

__all__ = [
    "DeepSortTracker",
    "TrackingEngine",
    "KalmanFilter", 
    "FeatureExtractor"
]