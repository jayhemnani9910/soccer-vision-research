"""
Utility modules for soccer player recognition system.

This package contains various utility classes and functions for:
- Base model classes
- Video processing
- Image processing  
- Logging
- Drawing and visualization
"""

from .base_model import BaseModel
from .video_utils import VideoProcessor, load_video_frames
from .image_utils import ImageProcessor, create_torch_tensor, load_and_process_image, batch_process_images
from .logger import SoccerPlayerRecognitionLogger, get_logger, setup_logging, log_system_info, log_training_progress, log_model_info, log_error, log_performance_metrics
from .draw_utils import Visualizer, create_confusion_matrix_visualization, plot_training_metrics, create_comparison_visualization

# RF-DETR utilities
from .rf_detr_utils import (
    RFDETRPreprocessor,
    RFDETRPostprocessor,
    preprocess_for_soccer_detection,
    postprocess_soccer_detection,
    load_and_preprocess_image,
    batch_process_images
)

# ResNet utilities
from .resnet_utils import (
    ResNetPreprocessor,
    ResNetFeatureExtractor,
    ResNetDatasetProcessor,
    ResNetModelEvaluator,
    load_and_preprocess_image as load_and_preprocess_resnet_image,
    save_image_tensor,
    compute_model_flops,
    test_resnet_utils
)

__all__ = [
    # Base model
    'BaseModel',
    
    # Video utilities
    'VideoProcessor',
    'load_video_frames',
    
    # Image utilities
    'ImageProcessor',
    'create_torch_tensor',
    'load_and_process_image',
    'batch_process_images',
    
    # Logging utilities
    'SoccerPlayerRecognitionLogger',
    'get_logger',
    'setup_logging',
    'log_system_info',
    'log_training_progress',
    'log_model_info',
    'log_error',
    'log_performance_metrics',
    
    # Drawing utilities
    'Visualizer',
    'create_confusion_matrix_visualization',
    'plot_training_metrics',
    'create_comparison_visualization',
    
    # RF-DETR utilities
    'RFDETRPreprocessor',
    'RFDETRPostprocessor',
    'preprocess_for_soccer_detection',
    'postprocess_soccer_detection',
    'load_and_preprocess_image',
    'batch_process_images',
    
    # ResNet utilities
    'ResNetPreprocessor',
    'ResNetFeatureExtractor',
    'ResNetDatasetProcessor',
    'ResNetModelEvaluator',
    'load_and_preprocess_resnet_image',
    'save_image_tensor',
    'compute_model_flops',
    'test_resnet_utils'
]

__version__ = "1.0.0"