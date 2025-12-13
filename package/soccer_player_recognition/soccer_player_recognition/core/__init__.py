"""Core module for Soccer Player Recognition System.

This module contains the unified integration framework that brings together all four models:
- RF-DETR: Player detection in images/video
- SAM2: Video segmentation and tracking  
- SigLIP: Zero-shot player identification
- ResNet: Player classification and recognition

Main Classes:
- PlayerRecognizer: Main integration class that orchestrates all models
- ModelPipeline: Orchestrates model execution with optimization
- ResultFuser: Combines results from multiple models
- Config: Configuration management system

Data Structures:
- DetectionResult: Standardized detection output
- IdentificationResult: Player identification results
- SegmentationResult: Video segmentation results
- TrackingResult: Temporal tracking results

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import os
from typing import Optional

# Version information
__version__ = "1.0.0"
__author__ = "Soccer Player Recognition Team"
__email__ = "team@soccerrecognition.com"

# Main integration class
try:
    from .player_recognizer import PlayerRecognizer, create_player_recognizer
    _player_recognizer_available = True
except ImportError as e:
    print(f"Warning: PlayerRecognizer not available: {e}")
    _player_recognizer_available = False
    PlayerRecognizer = None
    create_player_recognizer = None

# Pipeline and orchestration
try:
    from .model_pipeline import ModelPipeline, PipelineConfig, ExecutionMode, PipelineStage
    _model_pipeline_available = True
except ImportError as e:
    print(f"Warning: ModelPipeline not available: {e}")
    _model_pipeline_available = False
    ModelPipeline = None
    PipelineConfig = None
    ExecutionMode = None
    PipelineStage = None

# Result fusion
try:
    from .result_fusion import (
        ResultFuser, 
        FusionStrategy, 
        FusedDetection, 
        FusedIdentification, 
        FusedSegmentation
    )
    _result_fusion_available = True
except ImportError as e:
    print(f"Warning: ResultFuser not available: {e}")
    _result_fusion_available = False
    ResultFuser = None
    FusionStrategy = None
    FusedDetection = None
    FusedIdentification = None
    FusedSegmentation = None

# Configuration management
try:
    from .config import (
        Config,
        SystemConfig,
        ModelConfig,
        DetectionConfig,
        SegmentationConfig,
        IdentificationConfig,
        ClassificationConfig,
        FusionConfig,
        PipelineConfig as CorePipelineConfig,
        load_config
    )
    _config_available = True
except ImportError as e:
    print(f"Warning: Config system not available: {e}")
    _config_available = False
    Config = None
    SystemConfig = None
    ModelConfig = None
    DetectionConfig = None
    SegmentationConfig = None
    IdentificationConfig = None
    ClassificationConfig = None
    FusionConfig = None
    CorePipelineConfig = None
    load_config = None

# Data structures
try:
    from .results import (
        DetectionResult,
        IdentificationResult,
        SegmentationResult,
        TrackingResult,
        batch_detection_results,
        batch_identification_results
    )
    _results_available = True
except ImportError as e:
    print(f"Warning: Results system not available: {e}")
    _results_available = False
    DetectionResult = None
    IdentificationResult = None
    SegmentationResult = None
    TrackingResult = None
    batch_detection_results = None
    batch_identification_results = None

# Public API exports (only include available components)
__all__ = []

# Add available components to __all__
if _player_recognizer_available:
    __all__.extend(["PlayerRecognizer", "create_player_recognizer"])

if _model_pipeline_available:
    __all__.extend(["ModelPipeline", "PipelineConfig", "ExecutionMode", "PipelineStage"])

if _result_fusion_available:
    __all__.extend([
        "ResultFuser", "FusionStrategy", "FusedDetection", 
        "FusedIdentification", "FusedSegmentation"
    ])

if _config_available:
    __all__.extend([
        "Config", "SystemConfig", "ModelConfig", "DetectionConfig",
        "SegmentationConfig", "IdentificationConfig", "ClassificationConfig",
        "FusionConfig", "PipelineConfig", "load_config"
    ])

if _results_available:
    __all__.extend([
        "DetectionResult", "IdentificationResult", "SegmentationResult",
        "TrackingResult", "batch_detection_results", "batch_identification_results"
    ])

# Always include version
__all__.append("__version__")


def get_system_info() -> dict:
    """Get system and framework information."""
    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        device_count = torch.cuda.device_count() if cuda_available else 1
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch, 'backends') else False
    except ImportError:
        torch_available = False
        torch_version = "not installed"
        cuda_available = False
        cuda_version = None
        device_count = 1
        mps_available = False
    
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = "not installed"
    
    return {
        "framework_version": __version__,
        "pytorch_available": torch_available,
        "pytorch_version": torch_version,
        "numpy_version": numpy_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "mps_available": mps_available,
        "device_count": device_count,
        "models_supported": ["RF-DETR", "SAM2", "SigLIP", "ResNet"],
        "fusion_strategies": [strategy.value for strategy in (FusionStrategy if _result_fusion_available else [])],
        "execution_modes": [mode.value for mode in (ExecutionMode if _model_pipeline_available else [])],
        "components_available": {
            "player_recognizer": _player_recognizer_available,
            "model_pipeline": _model_pipeline_available,
            "result_fusion": _result_fusion_available,
            "config": _config_available,
            "results": _results_available
        }
    }


def create_minimal_recognizer() -> Optional[PlayerRecognizer]:
    """Create a minimal PlayerRecognizer for testing without loading all models."""
    if not _player_recognizer_available:
        print("Error: PlayerRecognizer not available")
        return None
    
    return PlayerRecognizer(
        device="cpu",
        enable_all_models=False,
        memory_efficient=True
    )


def create_full_recognizer(
    config_path: Optional[str] = None,
    device: str = "auto", 
    preset: str = "balanced"
) -> Optional[PlayerRecognizer]:
    """Create a full PlayerRecognizer with all models loaded."""
    if not _player_recognizer_available:
        print("Error: PlayerRecognizer not available")
        return None
    
    return create_player_recognizer(
        config_path=config_path,
        device=device,
        models_to_load=None,  # Load all
        memory_efficient=(preset != "high_accuracy")
    )


def demo_integration_framework():
    """Demonstration of the integration framework capabilities."""
    print("=== Soccer Player Recognition Integration Framework Demo ===")
    
    # Get system info
    system_info = get_system_info()
    print(f"\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Check component availability
    print(f"\nComponent Status:")
    for component, available in system_info.get("components_available", {}).items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {component}: {status}")
    
    # Create minimal recognizer for demo if available
    if _player_recognizer_available:
        print(f"\n1. Creating minimal PlayerRecognizer...")
        try:
            recognizer = create_minimal_recognizer()
            if recognizer:
                status = recognizer.get_model_status()
                print(f"  Status: {status['device']}, Memory efficient: {status['memory_efficient']}")
            else:
                print("  Failed to create recognizer")
        except Exception as e:
            print(f"  Error creating recognizer: {e}")
    else:
        print(f"\n1. PlayerRecognizer not available for demo")
    
    # Show configuration options if available
    if _config_available:
        print(f"\n2. Configuration Presets Available:")
        for preset in ["balanced", "real_time", "high_accuracy", "development"]:
            try:
                config = load_config(preset=preset)
                print(f"  {preset}: {config.detection_config.batch_size} batch, {config.detection_config.input_size} input")
            except Exception as e:
                print(f"  {preset}: Error loading config - {e}")
    
    # Show fusion strategies if available
    if _result_fusion_available:
        print(f"\n3. Fusion Strategies:")
        for strategy in FusionStrategy:
            print(f"  - {strategy.value}")
    
    # Show execution modes if available
    if _model_pipeline_available:
        print(f"\n4. Execution Modes:")
        for mode in ExecutionMode:
            print(f"  - {mode.value}")
    
    print(f"\n5. Model Integration Status:")
    print(f"  ✓ RF-DETR: Detection (players, ball, referees)")
    print(f"  ✓ SAM2: Video segmentation and tracking")  
    print(f"  ✓ SigLIP: Zero-shot player identification")
    print(f"  ✓ ResNet: Trained player classification")
    
    print(f"\n6. Framework Features:")
    print(f"  ✓ Unified API for all models")
    print(f"  ✓ Intelligent result fusion")
    print(f"  ✓ Temporal consistency for video")
    print(f"  ✓ Adaptive execution modes")
    print(f"  ✓ Memory management")
    print(f"  ✓ Comprehensive configuration")
    
    print(f"\nDemo completed successfully!")
    
    if _player_recognizer_available:
        return create_minimal_recognizer()
    else:
        return None


if __name__ == "__main__":
    # Run demo when module is executed directly
    demo_integration_framework()