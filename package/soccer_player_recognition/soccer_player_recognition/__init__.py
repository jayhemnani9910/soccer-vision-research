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

# Import main classes
from .core import PlayerRecognizer
from .models.detection.rf_detr_model import RF_DETRModel as DetectionEngine
from .models.identification.siglip_model import SigLIPModel as IdentificationEngine
from .models.classification.resnet_model import ResNetModel as ClassificationEngine
from .models.segmentation.sam2_model import SAM2Model as SegmentationEngine
from .video import VideoProcessor, StreamProcessor

# Import utilities
from .utils import (
    load_config,
    save_results,
    visualize_results,
    create_output_dirs,
    validate_input,
    get_device_info
)

# Import datatypes
from .datatypes import (
    DetectionResult,
    IdentificationResult,
    TrackingResult,
    ClassificationResult,
    PlayerProfile
)

__all__ = [
    # Core classes
    "PlayerRecognizer",
    "DetectionEngine",  # RF-DETR Model
    "IdentificationEngine",  # SigLIP Model
    "ClassificationEngine",  # ResNet Model
    "SegmentationEngine",  # SAM2 Model
    "VideoProcessor",
    "StreamProcessor",
    
    # Utility functions
    "load_config",
    "save_results", 
    "visualize_results",
    "create_output_dirs",
    "validate_input",
    "get_device_info",
    
    # Data types
    "DetectionResult",
    "IdentificationResult", 
    "TrackingResult",
    "ClassificationResult",
    "PlayerProfile",
    
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