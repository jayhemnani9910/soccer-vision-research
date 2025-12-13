"""Data types and structures for Soccer Player Recognition System.

This module defines the core data structures used throughout the system
for representing detection results, player information, and analysis outputs.
"""

from .detection_result import DetectionResult, BoundingBox
from .identification_result import IdentificationResult, PlayerMatch
from .tracking_result import TrackingResult, Track
from .classification_result import ClassificationResult, PlayerClass
from .player_profile import PlayerProfile, PlayerMetadata

__all__ = [
    # Detection
    "DetectionResult",
    "BoundingBox",
    
    # Identification  
    "IdentificationResult",
    "PlayerMatch",
    
    # Tracking
    "TrackingResult", 
    "Track",
    
    # Classification
    "ClassificationResult",
    "PlayerClass",
    
    # Player profiles
    "PlayerProfile",
    "PlayerMetadata"
]