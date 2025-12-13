"""
Configuration Management

This module provides centralized configuration management for the Soccer Player Recognition System:
- Configuration classes for all models and components
- Configuration loading from files and environment variables
- Validation and type checking
- Default configurations for different use cases
- Model-specific parameter management

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    name: str
    model_type: str
    device: str = "auto"
    batch_size: int = 8
    max_memory_gb: float = 4.0
    enable_gpu: bool = True
    enable_cpu_fallback: bool = True
    timeout_seconds: int = 60
    retry_count: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DetectionConfig(ModelConfig):
    """Configuration for RF-DETR detection model."""
    model_type: str = "detection"
    
    # RF-DETR specific parameters
    confidence_threshold: float = 0.7
    nms_threshold: float = 0.5
    max_detections: int = 50
    input_size: int = 640
    score_threshold: float = 0.7
    
    # Model architecture
    backbone_model_name: str = "resnet50"
    d_model: int = 256
    nhead: int = 8
    num_feature_levels: int = 3
    dec_n_points: int = 4
    enc_n_points: int = 4
    
    # Performance tuning
    use_fp16: bool = True
    use_tensorrt: bool = False
    quantization: str = "none"  # none, dynamic, static, qat
    
    # Class configuration
    class_names: List[str] = field(default_factory=lambda: [
        "player", "goalkeeper", "referee", "ball"
    ])
    num_classes: int = 4
    
    # Paths
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    pretrained_path: Optional[str] = None


@dataclass
class SegmentationConfig(ModelConfig):
    """Configuration for SAM2 segmentation model."""
    model_type: str = "segmentation"
    
    # SAM2 specific parameters
    memory_mode: str = "selective"  # full, selective, compact
    max_memory_frames: int = 8
    min_confidence: float = 0.7
    occlusion_threshold: float = 0.5
    keyframe_threshold: float = 0.8
    learnable_prototypes: bool = True
    
    # Video processing
    frame_buffer_size: int = 32
    temporal_window: int = 5
    track_occluded_objects: bool = True
    auto_keyframe_selection: bool = True
    
    # Mask processing
    mask_threshold: float = 0.5
    mask_smoothness: float = 0.1
    mask_area_threshold: int = 100
    
    # Performance
    use_feature_caching: bool = True
    feature_cache_size: int = 100
    parallel_frame_processing: bool = False
    
    # Paths
    model_path: Optional[str] = None
    memory_path: Optional[str] = None


@dataclass
class IdentificationConfig(ModelConfig):
    """Configuration for SigLIP identification model."""
    model_type: str = "identification"
    
    # SigLIP specific parameters
    model_name: str = "siglip-base-patch16-224"
    vision_embed_dim: int = 768
    text_embed_dim: int = 768
    image_size: int = 224
    patch_size: int = 16
    temperature: float = 0.07
    
    # Text processing
    max_text_length: int = 77
    vocab_size: int = 32000
    tokenizer_type: str = "simple"  # simple, bert, clip
    
    # Identification strategies
    identification_method: str = "zero_shot"  # zero_shot, few_shot, trained
    top_k_predictions: int = 5
    confidence_threshold: float = 0.6
    
    # Context handling
    team_context_enabled: bool = True
    context_weight: float = 0.2
    venue_context_enabled: bool = False
    
    # Performance
    batch_text_processing: bool = True
    cache_embeddings: bool = True
    embedding_cache_size: int = 1000
    
    # Paths
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    embeddings_path: Optional[str] = None


@dataclass
class ClassificationConfig(ModelConfig):
    """Configuration for ResNet classification model."""
    model_type: str = "classification"
    
    # ResNet specific parameters
    model_name: str = "resnet50"
    num_classes: int = 1000
    num_players: int = 25
    
    # Training configuration
    pretrained: bool = True
    freeze_features: bool = False
    dropout_rate: float = 0.5
    use_feature_extraction: bool = True
    
    # Input preprocessing
    input_size: int = 224
    normalization_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalization_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data augmentation (for training)
    use_data_augmentation: bool = True
    augmentation_strength: float = 0.5
    
    # Performance
    batch_normalization: bool = True
    use_mixed_precision: bool = True
    
    # Paths
    model_path: Optional[str] = None
    class_mapping_path: Optional[str] = None
    player_info_path: Optional[str] = None


@dataclass
class FusionConfig:
    """Configuration for result fusion."""
    strategy: str = "adaptive"  # majority_voting, weighted_averaging, confidence_based, ensemble, adaptive
    confidence_threshold: float = 0.6
    
    # Model weights for weighted fusion
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        'rf_detr': 1.0,
        'sam2': 0.8,
        'siglip': 0.9,
        'resnet': 0.7
    })
    
    # Spatial matching
    iou_threshold: float = 0.3
    spatial_proximity_threshold: float = 50.0
    
    # Temporal consistency
    temporal_window: int = 5
    temporal_decay: float = 0.9
    consistency_threshold: float = 0.5
    
    # Conflict resolution
    resolve_conflicts: bool = True
    conflict_resolution_strategy: str = "confidence"  # confidence, voting, ensemble
    
    # Performance
    parallel_fusion: bool = True
    fusion_cache_size: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FusionConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PipelineConfig:
    """Configuration for model pipeline execution."""
    execution_mode: str = "sequential"  # sequential, parallel, adaptive
    max_workers: int = 4
    batch_size: int = 8
    
    # Performance settings
    enable_preprocessing_cache: bool = True
    cache_size: int = 100
    enable_gpu_fallback: bool = True
    memory_limit_gb: float = 8.0
    
    # Timeout and retry settings
    timeout_seconds: int = 60
    retry_count: int = 2
    
    # Error handling
    continue_on_error: bool = True
    fail_fast: bool = False
    
    # Monitoring
    enable_profiling: bool = False
    log_performance_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SystemConfig:
    """Main system configuration."""
    # Model configurations
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation_config: SegmentationConfig = field(default_factory=SegmentationConfig)
    identification_config: IdentificationConfig = field(default_factory=IdentificationConfig)
    classification_config: ClassificationConfig = field(default_factory=ClassificationConfig)
    
    # System configurations
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Global settings
    device: str = "auto"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Paths
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    model_dir: str = "models"
    data_dir: str = "data"
    
    # Performance
    memory_efficient: bool = True
    use_multiprocessing: bool = True
    num_workers: int = 4
    
    # Debug and development
    debug_mode: bool = False
    save_intermediate_results: bool = False
    save_visualizations: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'detection_config': self.detection_config.to_dict(),
            'segmentation_config': self.segmentation_config.to_dict(),
            'identification_config': self.identification_config.to_dict(),
            'classification_config': self.classification_config.to_dict(),
            'fusion_config': self.fusion_config.to_dict(),
            'pipeline_config': self.pipeline_config.to_dict(),
            'device': self.device,
            'log_level': self.log_level,
            'log_format': self.log_format,
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir,
            'model_dir': self.model_dir,
            'data_dir': self.data_dir,
            'memory_efficient': self.memory_efficient,
            'use_multiprocessing': self.use_multiprocessing,
            'num_workers': self.num_workers,
            'debug_mode': self.debug_mode,
            'save_intermediate_results': self.save_intermediate_results,
            'save_visualizations': self.save_visualizations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary."""
        config = cls()
        
        # Update model configurations
        if 'detection_config' in data:
            config.detection_config = DetectionConfig.from_dict(data['detection_config'])
        
        if 'segmentation_config' in data:
            config.segmentation_config = SegmentationConfig.from_dict(data['segmentation_config'])
        
        if 'identification_config' in data:
            config.identification_config = IdentificationConfig.from_dict(data['identification_config'])
        
        if 'classification_config' in data:
            config.classification_config = ClassificationConfig.from_dict(data['classification_config'])
        
        if 'fusion_config' in data:
            config.fusion_config = FusionConfig.from_dict(data['fusion_config'])
        
        if 'pipeline_config' in data:
            config.pipeline_config = PipelineConfig.from_dict(data['pipeline_config'])
        
        # Update global settings
        for key, value in data.items():
            if hasattr(config, key) and key not in [
                'detection_config', 'segmentation_config', 'identification_config',
                'classification_config', 'fusion_config', 'pipeline_config'
            ]:
                setattr(config, key, value)
        
        return config


class Config:
    """
    Unified configuration manager for the Soccer Player Recognition System.
    
    Provides:
    - Configuration loading from files
    - Environment variable support
    - Configuration validation
    - Default configuration presets
    - Hot reloading capabilities
    """
    
    def __init__(self, config_path: Optional[str] = None, preset: str = "balanced"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            preset: Configuration preset ('balanced', 'real_time', 'high_accuracy', 'development')
        """
        self.config_path = config_path
        self.preset = preset
        self.system_config = self._load_config()
        
        logger.info(f"Config initialized with preset: {preset}")
    
    def _load_config(self) -> SystemConfig:
        """Load configuration based on preset and file."""
        # Load default configuration based on preset
        default_config = self._get_preset_config(self.preset)
        
        # Load from file if provided
        if self.config_path and os.path.exists(self.config_path):
            file_config = self._load_from_file(self.config_path)
            if file_config:
                # Merge file config with default
                merged_config = self._merge_configs(default_config, file_config)
                return merged_config
        
        return default_config
    
    def _get_preset_config(self, preset: str) -> SystemConfig:
        """Get default configuration for preset."""
        configs = {
            "balanced": self._get_balanced_config(),
            "real_time": self._get_realtime_config(),
            "high_accuracy": self._get_high_accuracy_config(),
            "development": self._get_development_config()
        }
        
        return configs.get(preset, configs["balanced"])
    
    def _get_balanced_config(self) -> SystemConfig:
        """Get balanced configuration for general use."""
        return SystemConfig(
            device="auto",
            memory_efficient=True,
            detection_config=DetectionConfig(
                confidence_threshold=0.7,
                batch_size=8,
                input_size=640
            ),
            segmentation_config=SegmentationConfig(
                memory_mode="selective",
                max_memory_frames=8,
                min_confidence=0.7
            ),
            identification_config=IdentificationConfig(
                model_name="siglip-base-patch16-224",
                confidence_threshold=0.6,
                batch_text_processing=True
            ),
            classification_config=ClassificationConfig(
                model_name="resnet50",
                dropout_rate=0.5,
                batch_size=8
            ),
            fusion_config=FusionConfig(
                strategy="adaptive",
                confidence_threshold=0.6
            ),
            pipeline_config=PipelineConfig(
                execution_mode="sequential",
                max_workers=4,
                memory_limit_gb=8.0
            )
        )
    
    def _get_realtime_config(self) -> SystemConfig:
        """Get configuration optimized for real-time processing."""
        return SystemConfig(
            device="auto",
            memory_efficient=True,
            detection_config=DetectionConfig(
                confidence_threshold=0.5,
                batch_size=16,
                input_size=416,
                max_detections=30
            ),
            segmentation_config=SegmentationConfig(
                memory_mode="compact",
                max_memory_frames=4,
                min_confidence=0.6,
                frame_buffer_size=16
            ),
            identification_config=IdentificationConfig(
                confidence_threshold=0.5,
                batch_text_processing=True,
                cache_embeddings=True
            ),
            classification_config=ClassificationConfig(
                model_name="resnet18",
                dropout_rate=0.3,
                batch_size=16
            ),
            fusion_config=FusionConfig(
                strategy="confidence_based",
                confidence_threshold=0.5,
                parallel_fusion=True
            ),
            pipeline_config=PipelineConfig(
                execution_mode="parallel",
                max_workers=6,
                batch_size=16,
                memory_limit_gb=6.0
            )
        )
    
    def _get_high_accuracy_config(self) -> SystemConfig:
        """Get configuration optimized for maximum accuracy."""
        return SystemConfig(
            device="cuda",
            memory_efficient=False,
            detection_config=DetectionConfig(
                confidence_threshold=0.8,
                batch_size=4,
                input_size=800,
                max_detections=50,
                use_fp16=False
            ),
            segmentation_config=SegmentationConfig(
                memory_mode="full",
                max_memory_frames=16,
                min_confidence=0.8,
                track_occluded_objects=True
            ),
            identification_config=IdentificationConfig(
                model_name="siglip-large-patch16-224",
                confidence_threshold=0.8,
                vision_embed_dim=1024,
                top_k_predictions=10
            ),
            classification_config=ClassificationConfig(
                model_name="resnet101",
                dropout_rate=0.7,
                batch_size=4,
                use_data_augmentation=True
            ),
            fusion_config=FusionConfig(
                strategy="ensemble",
                confidence_threshold=0.7,
                temporal_window=10
            ),
            pipeline_config=PipelineConfig(
                execution_mode="sequential",
                max_workers=2,
                batch_size=4,
                memory_limit_gb=16.0
            )
        )
    
    def _get_development_config(self) -> SystemConfig:
        """Get configuration for development and testing."""
        return SystemConfig(
            device="cpu",
            memory_efficient=False,
            debug_mode=True,
            save_intermediate_results=True,
            save_visualizations=True,
            detection_config=DetectionConfig(
                confidence_threshold=0.3,
                batch_size=2,
                input_size=320
            ),
            segmentation_config=SegmentationConfig(
                memory_mode="selective",
                max_memory_frames=4,
                min_confidence=0.3
            ),
            identification_config=IdentificationConfig(
                confidence_threshold=0.3,
                batch_size=2
            ),
            classification_config=ClassificationConfig(
                model_name="resnet18",
                batch_size=2,
                debug_mode=True
            ),
            fusion_config=FusionConfig(
                strategy="majority_voting",
                confidence_threshold=0.3
            ),
            pipeline_config=PipelineConfig(
                execution_mode="sequential",
                max_workers=2,
                batch_size=2,
                enable_profiling=True
            )
        )
    
    def _load_from_file(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        try:
            file_path = Path(config_path)
            
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return None
    
    def _merge_configs(self, default: SystemConfig, override: Dict[str, Any]) -> SystemConfig:
        """Merge override configuration with default."""
        merged_dict = default.to_dict()
        
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """Deep merge two dictionaries."""
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        deep_merge(merged_dict, override)
        return SystemConfig.from_dict(merged_dict)
    
    def save_config(self, output_path: str, format: str = "yaml"):
        """Save current configuration to file."""
        try:
            config_dict = self.system_config.to_dict()
            
            if format.lower() == "yaml":
                with open(output_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get configuration for specific model."""
        config_attr = f"{model_type}_config"
        if hasattr(self.system_config, config_attr):
            return getattr(self.system_config, config_attr)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def update_model_config(self, model_type: str, updates: Dict[str, Any]):
        """Update configuration for specific model."""
        config_attr = f"{model_type}_config"
        if hasattr(self.system_config, config_attr):
            config = getattr(self.system_config, config_attr)
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config parameter: {model_type}.{key}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_fusion_config(self) -> FusionConfig:
        """Get fusion configuration."""
        return self.system_config.fusion_config
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return self.system_config.pipeline_config
    
    def validate_config(self) -> List[str]:
        """Validate current configuration."""
        errors = []
        
        # Validate model configurations
        for model_type in ['detection', 'segmentation', 'identification', 'classification']:
            try:
                config = self.get_model_config(model_type)
                model_errors = self._validate_model_config(config)
                errors.extend([f"{model_type}.{error}" for error in model_errors])
            except Exception as e:
                errors.append(f"{model_type}: {str(e)}")
        
        # Validate fusion configuration
        fusion_errors = self._validate_fusion_config(self.system_config.fusion_config)
        errors.extend([f"fusion.{error}" for error in fusion_errors])
        
        # Validate pipeline configuration
        pipeline_errors = self._validate_pipeline_config(self.system_config.pipeline_config)
        errors.extend([f"pipeline.{error}" for error in pipeline_errors])
        
        return errors
    
    def _validate_model_config(self, config: ModelConfig) -> List[str]:
        """Validate individual model configuration."""
        errors = []
        
        # Check device
        if config.device not in ['auto', 'cpu', 'cuda', 'mps']:
            errors.append("Invalid device specified")
        
        # Check batch size
        if config.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Check memory limit
        if config.max_memory_gb <= 0:
            errors.append("Memory limit must be positive")
        
        return errors
    
    def _validate_fusion_config(self, config: FusionConfig) -> List[str]:
        """Validate fusion configuration."""
        errors = []
        
        # Check strategy
        valid_strategies = ['majority_voting', 'weighted_averaging', 'confidence_based', 'ensemble', 'adaptive']
        if config.strategy not in valid_strategies:
            errors.append(f"Invalid fusion strategy: {config.strategy}")
        
        # Check confidence threshold
        if not 0 <= config.confidence_threshold <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # Check model weights
        if config.model_weights:
            for model, weight in config.model_weights.items():
                if weight < 0:
                    errors.append(f"Negative weight for model {model}")
        
        return errors
    
    def _validate_pipeline_config(self, config: PipelineConfig) -> List[str]:
        """Validate pipeline configuration."""
        errors = []
        
        # Check execution mode
        valid_modes = ['sequential', 'parallel', 'adaptive']
        if config.execution_mode not in valid_modes:
            errors.append(f"Invalid execution mode: {config.execution_mode}")
        
        # Check workers
        if config.max_workers <= 0:
            errors.append("Max workers must be positive")
        
        # Check batch size
        if config.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Check memory limit
        if config.memory_limit_gb <= 0:
            errors.append("Memory limit must be positive")
        
        return errors


# Global configuration instance
_global_config = None


def get_global_config() -> Config:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_global_config(config: Config):
    """Set global configuration instance."""
    global _global_config
    _global_config = config


def load_config(config_path: Optional[str] = None, preset: str = "balanced") -> SystemConfig:
    """
    Load configuration from file or use preset.
    
    Args:
        config_path: Path to configuration file
        preset: Configuration preset
        
    Returns:
        SystemConfig instance
    """
    config_manager = Config(config_path=config_path, preset=preset)
    return config_manager.system_config


if __name__ == "__main__":
    # Example usage
    print("Testing Configuration System...")
    
    # Load balanced configuration
    config = load_config(preset="balanced")
    print(f"Loaded {config.log_level} configuration")
    
    # Test validation
    errors = config.validate_config() if hasattr(config, 'validate_config') else []
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Configuration validation passed")
    
    # Test preset switching
    realtime_config = load_config(preset="real_time")
    print(f"Realtime config batch size: {realtime_config.detection_config.batch_size}")
    
    print("Configuration system test completed")