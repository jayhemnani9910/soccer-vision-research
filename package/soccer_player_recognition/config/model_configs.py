"""
Model-Specific Configuration Manager
====================================

This module provides specialized configuration classes and schemas for different models
used in the soccer player recognition system. Each model has its own configuration
schema with validation, defaults, and compatibility checking.

Models supported:
- RF-DETR (Real-time Football Detection)
- SAM2 (Segment Anything Model 2)
- SigLIP (Sigmoid Linear projections for player recognition)
- ResNet (Player classification)
- Custom ensemble configurations

Author: Advanced AI Assistant
Date: 2025-11-04
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from enum import Enum

# Import the config manager
from .config_manager import ConfigManager, ConfigScope, ConfigFormat


class ModelType(Enum):
    """Supported model types."""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    IDENTIFICATION = "identification"
    ENSEMBLE = "ensemble"


class DeviceType(Enum):
    """Supported device types."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


@dataclass
class ModelMetadata:
    """Metadata for model configurations."""
    name: str
    version: str
    author: str
    created: str
    description: str
    input_format: str = "RGB"
    output_format: str = "feature_map"
    supported_formats: List[str] = field(default_factory=lambda: ["RGB", "BGR"])
    framework: str = "pytorch"
    license: str = "MIT"


@dataclass
class DetectionConfig:
    """RF-DETR Detection Model Configuration."""
    model_name: str = "rf-detr"
    model_path: str = ""
    config_path: str = ""
    
    # Input/output settings
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    
    # Model settings
    device: DeviceType = DeviceType.AUTO
    batch_size: int = 1
    num_workers: int = 4
    
    # Classes
    class_names: List[str] = field(default_factory=lambda: [
        "player", "ball", "referee", "goalkeeper"
    ])
    
    # Training parameters
    pretrained: bool = True
    freeze_backbone: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Post-processing
    use_nms: bool = True
    sort_tracking: bool = False
    tracking_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "input_size": list(self.input_size),
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "max_detections": self.max_detections,
            "device": self.device.value,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "class_names": self.class_names,
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "use_nms": self.use_nms,
            "sort_tracking": self.sort_tracking,
            "tracking_config": self.tracking_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionConfig':
        """Create from dictionary."""
        return cls(
            model_name=data.get("model_name", "rf-detr"),
            model_path=data.get("model_path", ""),
            config_path=data.get("config_path", ""),
            input_size=tuple(data.get("input_size", [640, 640])),
            confidence_threshold=data.get("confidence_threshold", 0.5),
            nms_threshold=data.get("nms_threshold", 0.4),
            max_detections=data.get("max_detections", 100),
            device=DeviceType(data.get("device", "auto")),
            batch_size=data.get("batch_size", 1),
            num_workers=data.get("num_workers", 4),
            class_names=data.get("class_names", ["player", "ball", "referee", "goalkeeper"]),
            pretrained=data.get("pretrained", True),
            freeze_backbone=data.get("freeze_backbone", False),
            learning_rate=data.get("learning_rate", 1e-4),
            weight_decay=data.get("weight_decay", 1e-4),
            momentum=data.get("momentum", 0.9),
            use_nms=data.get("use_nms", True),
            sort_tracking=data.get("sort_tracking", False),
            tracking_config=data.get("tracking_config", {})
        )


@dataclass
class SegmentationConfig:
    """SAM2 Segmentation Model Configuration."""
    model_name: str = "sam2"
    model_path: str = ""
    config_path: str = ""
    
    # Input/output settings
    input_size: Tuple[int, int] = (1024, 1024)
    confidence_threshold: float = 0.7
    iou_threshold: float = 0.8
    
    # Model settings
    device: DeviceType = DeviceType.AUTO
    batch_size: int = 1
    num_workers: int = 2
    
    # Model components
    predictor_type: str = "SAM2Predictor"
    image_encoder: Dict[str, Any] = field(default_factory=lambda: {
        "type": "vit_h",
        "checkpoint": "sam2_hiera_large.pt"
    })
    prompt_encoder: Dict[str, Any] = field(default_factory=lambda: {
        "type": "SAM2PromptEncoder"
    })
    mask_decoder: Dict[str, Any] = field(default_factory=lambda: {
        "type": "SAM2MaskDecoder"
    })
    
    # Memory bank
    memory_bank: Dict[str, int] = field(default_factory=lambda: {
        "size": 8,
        "feature_dim": 256
    })
    
    # Prompt settings
    point_params: Dict[str, Any] = field(default_factory=lambda: {
        "num_points": 4,
        "threshold": 0.8
    })
    box_params: Dict[str, Any] = field(default_factory=lambda: {
        "iou_threshold": 0.8,
        "stability_score_threshold": 0.7
    })
    
    # Advanced settings
    use_multimask: bool = True
    stability_score_offset: float = 1.0
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "input_size": list(self.input_size),
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device.value,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "predictor_type": self.predictor_type,
            "image_encoder": self.image_encoder,
            "prompt_encoder": self.prompt_encoder,
            "mask_decoder": self.mask_decoder,
            "memory_bank": self.memory_bank,
            "point_params": self.point_params,
            "box_params": self.box_params,
            "use_multimask": self.use_multimask,
            "stability_score_offset": self.stability_score_offset,
            "crop_n_layers": self.crop_n_layers,
            "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SegmentationConfig':
        """Create from dictionary."""
        return cls(
            model_name=data.get("model_name", "sam2"),
            model_path=data.get("model_path", ""),
            config_path=data.get("config_path", ""),
            input_size=tuple(data.get("input_size", [1024, 1024])),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            iou_threshold=data.get("iou_threshold", 0.8),
            device=DeviceType(data.get("device", "auto")),
            batch_size=data.get("batch_size", 1),
            num_workers=data.get("num_workers", 2),
            predictor_type=data.get("predictor_type", "SAM2Predictor"),
            image_encoder=data.get("image_encoder", {"type": "vit_h", "checkpoint": "sam2_hiera_large.pt"}),
            prompt_encoder=data.get("prompt_encoder", {"type": "SAM2PromptEncoder"}),
            mask_decoder=data.get("mask_decoder", {"type": "SAM2MaskDecoder"}),
            memory_bank=data.get("memory_bank", {"size": 8, "feature_dim": 256}),
            point_params=data.get("point_params", {"num_points": 4, "threshold": 0.8}),
            box_params=data.get("box_params", {"iou_threshold": 0.8, "stability_score_threshold": 0.7}),
            use_multimask=data.get("use_multimask", True),
            stability_score_offset=data.get("stability_score_offset", 1.0),
            crop_n_layers=data.get("crop_n_layers", 0),
            crop_n_points_downscale_factor=data.get("crop_n_points_downscale_factor", 1)
        )


@dataclass
class ClassificationConfig:
    """ResNet/SigLIP Classification Model Configuration."""
    model_name: str = "resnet"
    model_path: str = ""
    config_path: str = ""
    
    # Input/output settings
    input_size: Tuple[int, int] = (224, 224)
    
    # Model settings
    device: DeviceType = DeviceType.AUTO
    batch_size: int = 32
    num_workers: int = 4
    
    # Architecture settings
    architecture: str = "resnet50"
    pretrained: bool = True
    freeze_features: bool = False
    num_classes: int = 25
    dropout: float = 0.5
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Scheduler
    scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "type": "StepLR",
        "step_size": 30,
        "gamma": 0.1
    })
    
    # Data augmentation
    augmentations: Dict[str, float] = field(default_factory=lambda: {
        "random_rotation": 15.0,
        "random_color_jitter": 0.4,
        "random_gaussian_blur": 0.1,
        "random_horizontal_flip": 0.5
    })
    
    # SigLIP specific (if using SigLIP instead of ResNet)
    vision_model: Optional[Dict[str, Any]] = None
    text_model: Optional[Dict[str, Any]] = None
    temperature: float = 100.0
    use_softmax: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "input_size": list(self.input_size),
            "device": self.device.value,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "architecture": self.architecture,
            "pretrained": self.pretrained,
            "freeze_features": self.freeze_features,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "scheduler": self.scheduler,
            "augmentations": self.augmentations
        }
        
        if self.vision_model:
            result["vision_model"] = self.vision_model
        if self.text_model:
            result["text_model"] = self.text_model
        
        result["temperature"] = self.temperature
        result["use_softmax"] = self.use_softmax
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationConfig':
        """Create from dictionary."""
        return cls(
            model_name=data.get("model_name", "resnet"),
            model_path=data.get("model_path", ""),
            config_path=data.get("config_path", ""),
            input_size=tuple(data.get("input_size", [224, 224])),
            device=DeviceType(data.get("device", "auto")),
            batch_size=data.get("batch_size", 32),
            num_workers=data.get("num_workers", 4),
            architecture=data.get("architecture", "resnet50"),
            pretrained=data.get("pretrained", True),
            freeze_features=data.get("freeze_features", False),
            num_classes=data.get("num_classes", 25),
            dropout=data.get("dropout", 0.5),
            learning_rate=data.get("learning_rate", 1e-3),
            weight_decay=data.get("weight_decay", 1e-4),
            momentum=data.get("momentum", 0.9),
            scheduler=data.get("scheduler", {"type": "StepLR", "step_size": 30, "gamma": 0.1}),
            augmentations=data.get("augmentations", {
                "random_rotation": 15.0,
                "random_color_jitter": 0.4,
                "random_gaussian_blur": 0.1,
                "random_horizontal_flip": 0.5
            }),
            vision_model=data.get("vision_model"),
            text_model=data.get("text_model"),
            temperature=data.get("temperature", 100.0),
            use_softmax=data.get("use_softmax", True)
        )


@dataclass
class EnsembleConfig:
    """Ensemble Model Configuration."""
    model_name: str = "ensemble"
    enabled: bool = False
    
    # Voting strategy
    voting_strategy: str = "soft"  # "soft", "hard", "majority"
    
    # Model weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "rf_detr": 0.3,
        "sam2": 0.2,
        "siglip": 0.3,
        "resnet": 0.2
    })
    
    # Threshold calibration
    threshold_calibration: Dict[str, Any] = field(default_factory=lambda: {
        "method": "tpr",  # "fpr", "tpr", "balanced"
        "target_tpr": 0.95
    })
    
    # Ensemble specific settings
    consensus_threshold: float = 0.6
    diversity_factor: float = 0.1
    bootstrap_samples: int = 1000
    
    # Meta-learning settings
    use_stacking: bool = False
    meta_learner_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "enabled": self.enabled,
            "voting_strategy": self.voting_strategy,
            "weights": self.weights,
            "threshold_calibration": self.threshold_calibration,
            "consensus_threshold": self.consensus_threshold,
            "diversity_factor": self.diversity_factor,
            "bootstrap_samples": self.bootstrap_samples,
            "use_stacking": self.use_stacking,
            "meta_learner_config": self.meta_learner_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnsembleConfig':
        """Create from dictionary."""
        return cls(
            model_name=data.get("model_name", "ensemble"),
            enabled=data.get("enabled", False),
            voting_strategy=data.get("voting_strategy", "soft"),
            weights=data.get("weights", {"rf_detr": 0.3, "sam2": 0.2, "siglip": 0.3, "resnet": 0.2}),
            threshold_calibration=data.get("threshold_calibration", {"method": "tpr", "target_tpr": 0.95}),
            consensus_threshold=data.get("consensus_threshold", 0.6),
            diversity_factor=data.get("diversity_factor", 0.1),
            bootstrap_samples=data.get("bootstrap_samples", 1000),
            use_stacking=data.get("use_stacking", False),
            meta_learner_config=data.get("meta_learner_config", {})
        )


class ModelConfigManager:
    """
    Specialized configuration manager for model-specific settings.
    Provides validation, compatibility checking, and schema enforcement.
    """
    
    def __init__(self, config_dir: Union[str, Path]):
        """Initialize the model config manager."""
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.detection_config: Optional[DetectionConfig] = None
        self.segmentation_config: Optional[SegmentationConfig] = None
        self.classification_config: Optional[ClassificationConfig] = None
        self.ensemble_config: Optional[EnsembleConfig] = None
        
        # Validation schemas
        self._init_validation_schemas()
    
    def _init_validation_schemas(self):
        """Initialize validation schemas for each model type."""
        self.validation_schemas = {
            ModelType.DETECTION: {
                "required_fields": ["model_name", "model_path", "input_size"],
                "field_types": {
                    "input_size": list,
                    "confidence_threshold": (int, float),
                    "nms_threshold": (int, float),
                    "batch_size": int,
                    "num_workers": int
                },
                "constraints": {
                    "confidence_threshold": lambda x: 0 <= x <= 1,
                    "nms_threshold": lambda x: 0 <= x <= 1,
                    "batch_size": lambda x: x > 0,
                    "num_workers": lambda x: x >= 0
                }
            },
            ModelType.SEGMENTATION: {
                "required_fields": ["model_name", "model_path", "input_size"],
                "field_types": {
                    "input_size": list,
                    "confidence_threshold": (int, float),
                    "iou_threshold": (int, float)
                },
                "constraints": {
                    "confidence_threshold": lambda x: 0 <= x <= 1,
                    "iou_threshold": lambda x: 0 <= x <= 1
                }
            },
            ModelType.CLASSIFICATION: {
                "required_fields": ["model_name", "num_classes"],
                "field_types": {
                    "input_size": list,
                    "num_classes": int,
                    "dropout": (int, float),
                    "learning_rate": (int, float)
                },
                "constraints": {
                    "num_classes": lambda x: x > 0,
                    "dropout": lambda x: 0 <= x <= 1,
                    "learning_rate": lambda x: x > 0
                }
            }
        }
    
    def load_model_config(self, model_type: ModelType) -> Union[
        DetectionConfig, SegmentationConfig, ClassificationConfig, EnsembleConfig
    ]:
        """
        Load configuration for a specific model type.
        
        Args:
            model_type: Type of model to load configuration for
            
        Returns:
            Model configuration object
        """
        config_file_map = {
            ModelType.DETECTION: "detection_config.yaml",
            ModelType.SEGMENTATION: "segmentation_config.yaml", 
            ModelType.CLASSIFICATION: "classification_config.yaml",
            ModelType.IDENTIFICATION: "identification_config.yaml",
            ModelType.ENSEMBLE: "ensemble_config.yaml"
        }
        
        config_file = self.config_dir / config_file_map.get(model_type, f"{model_type.value}_config.yaml")
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            if model_type == ModelType.DETECTION:
                config = DetectionConfig.from_dict(config_data)
                self.detection_config = config
            elif model_type == ModelType.SEGMENTATION:
                config = SegmentationConfig.from_dict(config_data)
                self.segmentation_config = config
            elif model_type == ModelType.CLASSIFICATION:
                config = ClassificationConfig.from_dict(config_data)
                self.classification_config = config
            elif model_type == ModelType.ENSEMBLE:
                config = EnsembleConfig.from_dict(config_data)
                self.ensemble_config = config
            
            self.logger.info(f"Loaded {model_type.value} configuration from {config_file}")
            return config
            
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {config_file}")
            return self._create_default_config(model_type)
        except Exception as e:
            self.logger.error(f"Error loading {model_type.value} configuration: {e}")
            return self._create_default_config(model_type)
    
    def save_model_config(self, config: Union[
        DetectionConfig, SegmentationConfig, ClassificationConfig, EnsembleConfig
    ]):
        """Save model configuration to file."""
        config_file_map = {
            ModelType.DETECTION: "detection_config.yaml",
            ModelType.SEGMENTATION: "segmentation_config.yaml",
            ModelType.CLASSIFICATION: "classification_config.yaml",
            ModelType.ENSEMBLE: "ensemble_config.yaml"
        }
        
        config_type = type(config).__name__.replace("Config", "").upper()
        model_type = ModelType(config_type.lower())
        
        config_file = self.config_dir / config_file_map[model_type]
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved {config_type} configuration to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving {config_type} configuration: {e}")
            raise
    
    def _create_default_config(self, model_type: ModelType) -> Union[
        DetectionConfig, SegmentationConfig, ClassificationConfig, EnsembleConfig
    ]:
        """Create default configuration for model type."""
        if model_type == ModelType.DETECTION:
            return DetectionConfig()
        elif model_type == ModelType.SEGMENTATION:
            return SegmentationConfig()
        elif model_type == ModelType.CLASSIFICATION:
            return ClassificationConfig()
        elif model_type == ModelType.ENSEMBLE:
            return EnsembleConfig()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def validate_model_config(self, config: Union[
        DetectionConfig, SegmentationConfig, ClassificationConfig, EnsembleConfig
    ]) -> bool:
        """
        Validate model configuration against schema.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            True if valid, False otherwise
        """
        config_type = type(config).__name__.replace("Config", "").upper()
        
        try:
            model_type = ModelType(config_type.lower())
            schema = self.validation_schemas.get(model_type)
            
            if not schema:
                self.logger.warning(f"No validation schema found for {model_type}")
                return True
            
            # Check required fields
            config_dict = config.to_dict()
            missing_fields = [field for field in schema["required_fields"] 
                            if field not in config_dict]
            
            if missing_fields:
                self.logger.error(f"Missing required fields: {missing_fields}")
                return False
            
            # Check field types
            for field_name, expected_type in schema["field_types"].items():
                if field_name in config_dict:
                    value = config_dict[field_name]
                    if not isinstance(value, expected_type):
                        self.logger.error(f"Field {field_name} has wrong type: "
                                        f"expected {expected_type}, got {type(value)}")
                        return False
            
            # Check constraints
            for field_name, constraint in schema["constraints"].items():
                if field_name in config_dict:
                    value = config_dict[field_name]
                    if not constraint(value):
                        self.logger.error(f"Field {field_name} failed constraint: {value}")
                        return False
            
            self.logger.info(f"Configuration validation passed for {model_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            return False
    
    def get_model_metadata(self, model_name: str) -> ModelMetadata:
        """Get metadata for a specific model."""
        metadata_map = {
            "rf-detr": ModelMetadata(
                name="rf-detr",
                version="1.0.0",
                author="Research Team",
                created="2025-11-04",
                description="Real-time Football Detection with Transformer"
            ),
            "sam2": ModelMetadata(
                name="sam2",
                version="2.0.0", 
                author="Meta AI",
                created="2025-11-04",
                description="Segment Anything Model 2 for video segmentation"
            ),
            "siglip": ModelMetadata(
                name="siglip",
                version="1.0.0",
                author="Google Research",
                created="2025-11-04",
                description="Sigmoid Linear Projections for Player Recognition"
            ),
            "resnet": ModelMetadata(
                name="resnet",
                version="1.0.0",
                author="Microsoft Research",
                created="2025-11-04", 
                description="Residual Networks for Player Classification"
            )
        }
        
        return metadata_map.get(model_name, ModelMetadata(
            name=model_name,
            version="unknown",
            author="unknown",
            created="unknown",
            description="Unknown model"
        ))
    
    def create_model_config_template(self, model_type: ModelType, file_path: Union[str, Path]):
        """Create a template configuration file for a model type."""
        config = self._create_default_config(model_type)
        template_path = Path(file_path)
        
        with open(template_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        self.logger.info(f"Created template configuration for {model_type} at {template_path}")
    
    def migrate_config(self, old_config: Dict[str, Any], target_version: str) -> Dict[str, Any]:
        """Migrate configuration to a newer version."""
        current_version = old_config.get("version", "1.0.0")
        
        migrations = {
            ("1.0.0", "1.1.0"): self._migrate_v1_0_to_v1_1,
            ("1.1.0", "1.2.0"): self._migrate_v1_1_to_v1_2
        }
        
        migration_key = (current_version, target_version)
        migration_func = migrations.get(migration_key)
        
        if migration_func:
            new_config = migration_func(old_config)
            new_config["version"] = target_version
            self.logger.info(f"Migrated configuration from {current_version} to {target_version}")
            return new_config
        else:
            self.logger.warning(f"No migration path from {current_version} to {target_version}")
            return old_config
    
    def _migrate_v1_0_to_v1_1(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migration from version 1.0.0 to 1.1.0."""
        # Example migration: add new field with default value
        config["new_feature"] = True
        return config
    
    def _migrate_v1_1_to_v1_2(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migration from version 1.1.0 to 1.2.0."""
        # Example migration: rename field
        if "old_field_name" in config:
            config["new_field_name"] = config.pop("old_field_name")
        return config


# Convenience functions
def get_detection_config(config_dir: Union[str, Path]) -> DetectionConfig:
    """Get detection model configuration."""
    manager = ModelConfigManager(config_dir)
    return manager.load_model_config(ModelType.DETECTION)


def get_segmentation_config(config_dir: Union[str, Path]) -> SegmentationConfig:
    """Get segmentation model configuration."""
    manager = ModelConfigManager(config_dir)
    return manager.load_model_config(ModelType.SEGMENTATION)


def get_classification_config(config_dir: Union[str, Path]) -> ClassificationConfig:
    """Get classification model configuration."""
    manager = ModelConfigManager(config_dir)
    return manager.load_model_config(ModelType.CLASSIFICATION)


def get_ensemble_config(config_dir: Union[str, Path]) -> EnsembleConfig:
    """Get ensemble model configuration."""
    manager = ModelConfigManager(config_dir)
    return manager.load_model_config(ModelType.ENSEMBLE)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model config manager
    config_dir = Path("config")
    manager = ModelConfigManager(config_dir)
    
    # Load all model configurations
    detection_config = manager.load_model_config(ModelType.DETECTION)
    segmentation_config = manager.load_model_config(ModelType.SEGMENTATION)
    classification_config = manager.load_model_config(ModelType.CLASSIFICATION)
    
    # Validate configurations
    manager.validate_model_config(detection_config)
    manager.validate_model_config(segmentation_config)
    manager.validate_model_config(classification_config)
    
    print(f"Detection config: {detection_config.to_dict()}")
    print(f"Segmentation config: {segmentation_config.to_dict()}")
    print(f"Classification config: {classification_config.to_dict()}")
    
    # Create template
    manager.create_model_config_template(ModelType.DETECTION, "config/detection_template.yaml")
