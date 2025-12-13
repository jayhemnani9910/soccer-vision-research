"""
RF-DETR Model Configuration for Soccer Player Detection

This module contains configuration settings for the RF-DETR model
specifically optimized for soccer player detection tasks.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RFDETRConfig:
    """Configuration class for RF-DETR model parameters."""
    
    # Model architecture parameters
    backbone_model_name: str = "resnet50"
    d_model: int = 256
    nhead: int = 8
    dropout: float = 0.1
    num_feature_levels: int = 3
    dec_n_points: int = 4
    enc_n_points: int = 4
    
    # Input preprocessing parameters
    input_size: Tuple[int, int] = (640, 640)  # width, height
    mean: List[float] = None
    std: List[float] = None
    
    # Training parameters
    weight_decay: float = 0.05
    learning_rate: float = 2e-4
    clip_max_norm: float = 0.1
    num_epochs: int = 100
    
    # Soccer-specific detection parameters
    num_classes: int = 4  # players, goalkeepers, referees, ball
    score_threshold: float = 0.7
    nms_threshold: float = 0.5
    max_detections: int = 100
    
    # Class mapping for soccer detection
    class_names: List[str] = None
    
    # Model loading
    pretrained_model_path: str = "weights/rf_detr_soccer.pth"
    model_device: str = "cuda"
    
    # Output formatting
    output_format: str = "xyxy"  # 'xyxy' or 'xywh'
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]  # ImageNet statistics
            
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]  # ImageNet statistics
            
        if self.class_names is None:
            self.class_names = [
                "background",  # class 0 - background
                "player",      # class 1 - regular player
                "goalkeeper",  # class 2 - goalkeeper
                "referee",     # class 3 - referee
                "ball"         # class 4 - soccer ball
            ]
    
    def get_class_id_mapping(self) -> Dict[str, int]:
        """Get mapping from class names to IDs."""
        return {name: idx for idx, name in enumerate(self.class_names)}
    
    def get_id_to_class_mapping(self) -> Dict[int, str]:
        """Get mapping from class IDs to names."""
        return {idx: name for idx, name in enumerate(self.class_names)}


# Predefined configurations for different use cases
class RFDETRSoccerConfigs:
    """Predefined configuration sets for different soccer detection scenarios."""
    
    # High accuracy configuration for offline analysis
    HIGH_ACCURACY = RFDETRConfig(
        input_size=(800, 800),
        score_threshold=0.8,
        nms_threshold=0.4,
        d_model=512,
        nhead=8,
        max_detections=200,
        num_epochs=150
    )
    
    # Real-time configuration for live processing
    REAL_TIME = RFDETRConfig(
        input_size=(416, 416),
        score_threshold=0.6,
        nms_threshold=0.6,
        d_model=256,
        nhead=8,
        max_detections=50,
        num_epochs=50
    )
    
    # Balanced configuration for general use
    BALANCED = RFDETRConfig(
        input_size=(640, 640),
        score_threshold=0.7,
        nms_threshold=0.5,
        d_model=256,
        nhead=8,
        max_detections=100,
        num_epochs=100
    )
    
    # Training configuration with data augmentation
    TRAINING = RFDETRConfig(
        input_size=(640, 640),
        score_threshold=0.3,
        nms_threshold=0.8,
        d_model=256,
        nhead=8,
        max_detections=500,
        learning_rate=1e-4,
        num_epochs=200,
        weight_decay=0.05
    )


# Global configuration instance
DEFAULT_CONFIG = RFDETRSoccerConfigs.BALANCED