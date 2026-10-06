"""Soccer Player Recognition System

An advanced computer vision system for soccer player recognition, tracking, 
and analysis using deep learning techniques.

Main components:
- Detection: Player detection using YOLO and other models
- Identification: Player recognition using facial features and jersey analysis  
- Classification: Team and position classification
- Tracking: Multi-object tracking across video frames
- Analytics: Performance metrics and statistics

Example usage:
    from soccer_player_recognition import PlayerRecognizer
    
    recognizer = PlayerRecognizer()
    results = recognizer.process_video('match.mp4')

Author: AI Development Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "AI Development Team"
__email__ = "ai@example.com"
__license__ = "MIT"

# Import main classes (only modules that exist in this package)
from .core import PlayerRecognizer
from .core.results import (
    DetectionResult,
    IdentificationResult,
    SegmentationResult,
    TrackingResult
)
from .models.identification.siglip_model import SigLIPModel as IdentificationEngine

__all__ = [
    # Core classes
    "PlayerRecognizer",
    "IdentificationEngine",  # SigLIP Model

    # Data types
    "DetectionResult",
    "IdentificationResult",
    "SegmentationResult",
    "TrackingResult",

    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__"
]

# Package metadata
PACKAGE_INFO = {
    "name": "soccer-player-recognition",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "description": "Advanced soccer player recognition and tracking system",
    "keywords": [
        "computer vision",
        "machine learning", 
        "deep learning",
        "soccer",
        "football",
        "player recognition",
        "object detection",
        "tracking",
        "sports analytics"
    ]
}