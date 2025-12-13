"""Utility functions and helpers for Soccer Player Recognition System.

This module contains common utility functions, helper classes, and 
general-purpose tools used throughout the system.
"""

from .config_loader import load_config, save_config
from .file_utils import (
    create_output_dirs,
    validate_input,
    get_file_info,
    ensure_dir
)
from .image_utils import (
    resize_image,
    normalize_image,
    draw_bounding_box,
    draw_text,
    apply_color_filter
)
from .video_utils import (
    get_video_info,
    extract_frames,
    calculate_fps,
    resize_video
)
from .model_utils import (
    load_model,
    save_model,
    get_device_info,
    optimize_model,
    convert_to_onnx
)
from .metrics_utils import (
    calculate_map,
    calculate_iou,
    calculate_accuracy,
    generate_confusion_matrix
)
from .visualization_utils import (
    plot_detection_results,
    plot_tracking_results,
    plot_classification_results,
    create_dashboard
)
from .siglip_utils import (
    PlayerNameNormalizer,
    TextPromptProcessor,
    SimilaritySearch,
    EmbeddingUtils,
    MatchScoreAggregator,
    create_name_normalizer,
    create_prompt_processor,
    create_similarity_search,
    create_match_aggregator
)

__all__ = [
    # Config utilities
    "load_config",
    "save_config",
    
    # File utilities
    "create_output_dirs",
    "validate_input", 
    "get_file_info",
    "ensure_dir",
    
    # Image utilities
    "resize_image",
    "normalize_image",
    "draw_bounding_box",
    "draw_text",
    "apply_color_filter",
    
    # Video utilities
    "get_video_info",
    "extract_frames", 
    "calculate_fps",
    "resize_video",
    
    # Model utilities
    "load_model",
    "save_model",
    "get_device_info",
    "optimize_model",
    "convert_to_onnx",
    
    # Metrics utilities
    "calculate_map",
    "calculate_iou",
    "calculate_accuracy",
    "generate_confusion_matrix",
    
    # Visualization utilities
    "plot_detection_results",
    "plot_tracking_results",
    "plot_classification_results", 
    "create_dashboard",
    
    # SigLIP utilities
    "PlayerNameNormalizer",
    "TextPromptProcessor",
    "SimilaritySearch",
    "EmbeddingUtils",
    "MatchScoreAggregator",
    "create_name_normalizer",
    "create_prompt_processor",
    "create_similarity_search",
    "create_match_aggregator"
]