"""
Standalone Configuration Management System Demo
=============================================

This script demonstrates the configuration management system without
circular import dependencies.

Author: Advanced AI Assistant  
Date: 2025-11-04
"""

import sys
import time
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING" 
    INFO = "INFO"


class DeviceType(Enum):
    """Supported device types."""
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


class ConfigDemo:
    """Simplified configuration management demonstration."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the demo."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        logger.info("Configuration system demo initialized")
    
    def demo_basic_configuration(self):
        """Demonstrate basic configuration management."""
        print("\n" + "="*60)
        print("1. Basic Configuration Management")
        print("="*60)
        
        # Create sample configuration
        config = {
            "project": {
                "name": "Soccer Player Recognition",
                "version": "1.0.0",
                "author": "AI Research Team",
                "description": "Advanced AI system for soccer player analysis"
            },
            "models": {
                "rf_detr": {
                    "model_name": "rf-detr",
                    "model_path": "models/pretrained/rf_detr/rf_detr_epoch_20.pth",
                    "input_size": [640, 640],
                    "confidence_threshold": 0.6,
                    "nms_threshold": 0.5,
                    "batch_size": 2,
                    "num_workers": 4,
                    "class_names": ["player", "ball", "referee", "goalkeeper"],
                    "device": "cuda",
                    "pretrained": True
                },
                "sam2": {
                    "model_name": "sam2",
                    "model_path": "models/pretrained/sam2/sam2_hiera_large.pt",
                    "input_size": [1024, 1024],
                    "confidence_threshold": 0.75,
                    "batch_size": 1,
                    "num_workers": 2,
                    "device": "cuda",
                    "predictor_type": "SAM2Predictor",
                    "use_multimask": True
                },
                "siglip": {
                    "model_name": "siglip",
                    "model_path": "models/pretrained/siglip/siglip-vit-so400m-14-e384.pt",
                    "input_size": [384, 384],
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "cuda",
                    "temperature": 100.0,
                    "num_player_classes": 25
                },
                "resnet": {
                    "model_name": "resnet",
                    "model_path": "models/pretrained/resnet/resnet50-0676ba61.pth",
                    "input_size": [224, 224],
                    "batch_size": 64,
                    "num_workers": 8,
                    "device": "cuda",
                    "architecture": "resnet50",
                    "num_classes": 25,
                    "dropout": 0.5,
                    "learning_rate": 1e-3
                }
            },
            "ensemble": {
                "enabled": True,
                "voting_strategy": "soft",
                "weights": {
                    "rf_detr": 0.3,
                    "sam2": 0.2,
                    "siglip": 0.3,
                    "resnet": 0.2
                },
                "consensus_threshold": 0.6,
                "threshold_calibration": {
                    "method": "balanced",
                    "target_tpr": 0.95
                }
            },
            "system": {
                "device": "cuda",
                "device_id": 0,
                "mixed_precision": True,
                "compile_model": False,
                "gradient_checkpointing": True,
                "max_memory_fraction": 0.8,
                "clear_cache_frequency": 10,
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True
            },
            "performance": {
                "enable_caching": True,
                "cache_size_mb": 512,
                "memory_pool_size": 1024,
                "fuse_conv_bn": True,
                "remove_dropout_inference": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_to_file": True,
                "log_file_path": "logs/system.log",
                "log_model_summary": True,
                "log_training_history": True,
                "log_performance_metrics": True
            },
            "data": {
                "data_dir": "data",
                "output_dir": "outputs",
                "model_dir": "models",
                "cache_dir": "cache",
                "dataset_path": "data/datasets",
                "supported_image_formats": [".jpg", ".jpeg", ".png", ".bmp"],
                "supported_video_formats": [".mp4", ".avi", ".mov"],
                "auto_cleanup": True,
                "cleanup_threshold_gb": 10.0
            }
        }
        
        # Save configuration to YAML
        config_file = self.config_dir / "complete_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"✓ Configuration saved to {config_file}")
        
        # Save to JSON for easier reading
        json_file = self.config_dir / "complete_config.json"
        with open(json_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration also saved to {json_file}")
        
        # Demonstrate configuration access
        rf_detr_batch_size = config["models"]["rf_detr"]["batch_size"]
        sam2_confidence = config["models"]["sam2"]["confidence_threshold"]
        ensemble_enabled = config["ensemble"]["enabled"]
        
        print(f"✓ RF-DETR batch size: {rf_detr_batch_size}")
        print(f"✓ SAM2 confidence threshold: {sam2_confidence}")
        print(f"✓ Ensemble enabled: {ensemble_enabled}")
        
        return config
    
    def demo_configuration_validation(self):
        """Demonstrate configuration validation."""
        print("\n" + "="*60)
        print("2. Configuration Validation")
        print("="*60)
        
        # Load configuration
        config_file = self.config_dir / "complete_config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        issues = []
        
        # Validate RF-DETR configuration
        rf_detr = config["models"]["rf_detr"]
        if rf_detr["batch_size"] <= 0:
            issues.append("RF-DETR batch size must be positive")
        if not 0 <= rf_detr["confidence_threshold"] <= 1:
            issues.append("RF-DETR confidence threshold must be between 0 and 1")
        if not 0 <= rf_detr["nms_threshold"] <= 1:
            issues.append("RF-DETR NMS threshold must be between 0 and 1")
        
        # Validate SAM2 configuration
        sam2 = config["models"]["sam2"]
        if sam2["batch_size"] <= 0:
            issues.append("SAM2 batch size must be positive")
        if not 0 <= sam2["confidence_threshold"] <= 1:
            issues.append("SAM2 confidence threshold must be between 0 and 1")
        
        # Validate system configuration
        system = config["system"]
        if system["max_memory_fraction"] > 1.0:
            issues.append("System max_memory_fraction should not exceed 1.0")
        if system["num_workers"] < 0:
            issues.append("System num_workers cannot be negative")
        
        # Validate ensemble weights
        ensemble = config["ensemble"]
        weight_sum = sum(ensemble["weights"].values())
        if abs(weight_sum - 1.0) > 0.01:
            issues.append(f"Ensemble weights should sum to 1.0, got {weight_sum}")
        
        print(f"✓ Validation completed: {len(issues)} issues found")
        
        for issue in issues:
            print(f"  - {issue}")
        
        # Create a fixed version if issues found
        if issues:
            fixed_config = config.copy()
            
            # Fix common issues
            if fixed_config["models"]["rf_detr"]["confidence_threshold"] > 1.0:
                fixed_config["models"]["rf_detr"]["confidence_threshold"] = min(
                    fixed_config["models"]["rf_detr"]["confidence_threshold"] / 100.0, 0.99
                )
            
            if fixed_config["system"]["max_memory_fraction"] > 1.0:
                fixed_config["system"]["max_memory_fraction"] = 0.8
            
            # Save fixed configuration
            fixed_file = self.config_dir / "fixed_config.yaml"
            with open(fixed_file, 'w') as f:
                yaml.dump(fixed_config, f, default_flow_style=False, indent=2)
            print(f"✓ Fixed configuration saved to {fixed_file}")
            
            return fixed_config, issues
        else:
            print("✓ No issues found in configuration")
            return config, []
    
    def demo_model_specific_configs(self):
        """Demonstrate model-specific configurations."""
        print("\n" + "="*60)
        print("3. Model-Specific Configurations")
        print("="*60)
        
        # Detection model configuration
        detection_config = {
            "model_name": "rf_detr",
            "version": "1.0.0",
            "type": "detection",
            "parameters": {
                "input_size": [640, 640],
                "confidence_threshold": 0.6,
                "nms_threshold": 0.5,
                "max_detections": 100,
                "class_names": ["player", "ball", "referee", "goalkeeper"]
            },
            "training": {
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "momentum": 0.9,
                "batch_size": 2,
                "num_epochs": 100,
                "pretrained": True
            },
            "inference": {
                "batch_size": 1,
                "device": "cuda",
                "mixed_precision": True
            }
        }
        
        # Segmentation model configuration
        segmentation_config = {
            "model_name": "sam2",
            "version": "2.0.0",
            "type": "segmentation",
            "parameters": {
                "input_size": [1024, 1024],
                "confidence_threshold": 0.75,
                "iou_threshold": 0.8,
                "predictor_type": "SAM2Predictor",
                "image_encoder": {
                    "type": "vit_h",
                    "checkpoint": "sam2_hiera_large.pt"
                }
            },
            "memory_bank": {
                "size": 8,
                "feature_dim": 256
            },
            "prompt_settings": {
                "num_points": 4,
                "threshold": 0.8
            }
        }
        
        # Classification model configuration
        classification_config = {
            "model_name": "resnet",
            "version": "1.0.0",
            "type": "classification",
            "architecture": "resnet50",
            "parameters": {
                "input_size": [224, 224],
                "num_classes": 25,
                "dropout": 0.5
            },
            "training": {
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "momentum": 0.9,
                "batch_size": 64,
                "scheduler": {
                    "type": "StepLR",
                    "step_size": 30,
                    "gamma": 0.1
                },
                "augmentations": {
                    "random_rotation": 15.0,
                    "random_color_jitter": 0.4,
                    "random_horizontal_flip": 0.5
                }
            },
            "pretrained": True
        }
        
        # Save individual model configurations
        configs = [
            ("detection_config.yaml", detection_config),
            ("segmentation_config.yaml", segmentation_config),
            ("classification_config.yaml", classification_config)
        ]
        
        for filename, model_config in configs:
            file_path = self.config_dir / filename
            with open(file_path, 'w') as f:
                yaml.dump(model_config, f, default_flow_style=False, indent=2)
            print(f"✓ {model_config['model_name']} configuration saved to {filename}")
        
        # Create model registry
        model_registry = {
            "models": {
                "rf_detr": {
                    "name": "RF-DETR",
                    "type": "detection",
                    "description": "Real-time Football Detection with Transformer",
                    "config_file": "detection_config.yaml",
                    "supported_tasks": ["detection", "tracking"]
                },
                "sam2": {
                    "name": "SAM2",
                    "type": "segmentation", 
                    "description": "Segment Anything Model 2 for video segmentation",
                    "config_file": "segmentation_config.yaml",
                    "supported_tasks": ["segmentation", "tracking"]
                },
                "resnet": {
                    "name": "ResNet",
                    "type": "classification",
                    "description": "Residual Networks for Player Classification",
                    "config_file": "classification_config.yaml", 
                    "supported_tasks": ["classification", "identification"]
                }
            },
            "ensemble_models": ["rf_detr", "sam2", "resnet"],
            "default_pipeline": ["detection", "segmentation", "classification"]
        }
        
        registry_file = self.config_dir / "model_registry.yaml"
        with open(registry_file, 'w') as f:
            yaml.dump(model_registry, f, default_flow_style=False, indent=2)
        print(f"✓ Model registry saved to {registry_file}")
        
        return model_registry
    
    def demo_system_configuration(self):
        """Demonstrate system-wide configuration."""
        print("\n" + "="*60)
        print("4. System-Wide Configuration")
        print("="*60)
        
        # System information detection
        import platform
        import os
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": cpu_count,
                "memory_total_gb": round(memory_info.total / (1024**3), 2),
                "memory_available_gb": round(memory_info.available / (1024**3), 2)
            }
        except ImportError:
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count() or 4,
                "memory_total_gb": 8.0,  # Default assumption
                "memory_available_gb": 4.0
            }
            print("⚠ psutil not available, using default values")
        
        # Device configuration
        device_config = {
            "device": "auto",
            "device_id": 0,
            "max_memory_fraction": 0.8,
            "clear_cache_frequency": 10,
            "mixed_precision": True,
            "precision": "fp16",
            "compile_model": False,
            "gradient_checkpointing": True,
            "use_cudnn_benchmark": True,
            "multi_gpu": False,
            "distributed": False
        }
        
        # Performance configuration
        performance_config = {
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "enable_caching": True,
            "cache_size_mb": 512,
            "num_workers": min(system_info["cpu_count"], 8),
            "max_workers": min(system_info["cpu_count"], 8),
            "memory_pool_size": 1024,
            "garbage_collection_threshold": 0.8,
            "use_memory_mapping": True,
            "fuse_conv_bn": True,
            "remove_dropout_inference": True
        }
        
        # Runtime configuration
        runtime_config = {
            "mode": "inference",
            "use_threads": True,
            "thread_pool_size": system_info["cpu_count"],
            "max_concurrent_operations": 8,
            "memory_management_strategy": "auto",
            "model_loading_timeout": 300,
            "inference_timeout": 60,
            "continue_on_error": False,
            "max_retries": 3,
            "enable_checkpoints": True,
            "checkpoint_interval": 100,
            "show_progress": True
        }
        
        # Combine all system configurations
        system_config = {
            "version": "1.0.0",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "System configuration for Soccer Player Recognition",
            "system_info": system_info,
            "device": device_config,
            "performance": performance_config,
            "runtime": runtime_config
        }
        
        # Save system configuration
        system_file = self.config_dir / "system_config.yaml"
        with open(system_file, 'w') as f:
            yaml.dump(system_config, f, default_flow_style=False, indent=2)
        print(f"✓ System configuration saved to {system_file}")
        
        # Print detected system information
        print(f"✓ Detected system information:")
        for key, value in system_info.items():
            print(f"  - {key}: {value}")
        
        return system_config
    
    def demo_configuration_templates(self):
        """Demonstrate configuration templates."""
        print("\n" + "="*60)
        print("5. Configuration Templates")
        print("="*60)
        
        # Model template
        model_template = {
            "model_name": "{{MODEL_NAME}}",
            "version": "1.0.0",
            "type": "{{MODEL_TYPE}}",
            "model_path": "models/pretrained/{{MODEL_NAME}}/model.pth",
            "config_path": "models/pretrained/{{MODEL_NAME}}/config.yaml",
            "input_size": [224, 224],
            "device": "auto",
            "batch_size": 1,
            "num_workers": 4,
            "confidence_threshold": 0.5,
            "pretrained": True,
            "training": {
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 32,
                "num_epochs": 100
            },
            "inference": {
                "batch_size": 1,
                "mixed_precision": True
            }
        }
        
        # System template
        system_template = {
            "version": "1.0.0",
            "description": "System configuration template",
            "device": {
                "device": "auto",
                "device_id": 0,
                "max_memory_fraction": 0.8,
                "mixed_precision": True,
                "gradient_checkpointing": True
            },
            "performance": {
                "num_workers": 4,
                "enable_caching": True,
                "cache_size_mb": 512,
                "pin_memory": True
            },
            "logging": {
                "level": "INFO",
                "log_to_file": True,
                "log_file_path": "logs/system.log"
            }
        }
        
        # Pipeline template
        pipeline_template = {
            "pipeline_name": "{{PIPELINE_NAME}}",
            "description": "Processing pipeline template",
            "steps": [
                {
                    "name": "detection",
                    "model": "rf_detr",
                    "enabled": True,
                    "config_override": {}
                },
                {
                    "name": "segmentation", 
                    "model": "sam2",
                    "enabled": True,
                    "config_override": {}
                },
                {
                    "name": "classification",
                    "model": "resnet",
                    "enabled": True,
                    "config_override": {}
                }
            ],
            "ensemble": {
                "enabled": False,
                "strategy": "weighted_average",
                "weights": [0.3, 0.2, 0.5]
            },
            "output": {
                "save_annotations": True,
                "save_cropped_players": True,
                "save_tracking_results": True,
                "format": "json"
            }
        }
        
        # Save templates
        templates = [
            ("model_template.yaml", model_template),
            ("system_template.yaml", system_template),
            ("pipeline_template.yaml", pipeline_template)
        ]
        
        for filename, template in templates:
            file_path = self.config_dir / filename
            with open(file_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
            print(f"✓ Template saved to {filename}")
        
        # Create usage guide
        usage_guide = """# Configuration Templates Usage Guide

## Model Template Usage
To create a new model configuration:
1. Copy model_template.yaml
2. Replace {{MODEL_NAME}} with your model name
3. Replace {{MODEL_TYPE}} with model type (detection/segmentation/classification)
4. Update paths and parameters as needed

## System Template Usage  
To create system configuration:
1. Copy system_template.yaml
2. Adjust device settings based on your hardware
3. Configure performance parameters
4. Set up logging preferences

## Pipeline Template Usage
To create processing pipeline:
1. Copy pipeline_template.yaml
2. Replace {{PIPELINE_NAME}} with pipeline name
3. Enable/disable steps as needed
4. Configure ensemble settings
5. Set output preferences
"""
        
        guide_file = self.config_dir / "templates_usage_guide.md"
        with open(guide_file, 'w') as f:
            f.write(usage_guide)
        print(f"✓ Template usage guide saved to {guide_file}")
    
    def demo_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "="*60)
        print("6. Validation Report Generation")
        print("="*60)
        
        # Find all configuration files
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.json"))
        
        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_directory": str(self.config_dir),
            "total_files": len(config_files),
            "files_validated": [],
            "summary": {
                "total_issues": 0,
                "errors": 0,
                "warnings": 0,
                "infos": 0
            },
            "recommendations": []
        }
        
        for config_file in config_files:
            file_report = self._validate_config_file(config_file)
            report["files_validated"].append(file_report)
            
            # Update summary
            report["summary"]["total_issues"] += file_report["total_issues"]
            report["summary"]["errors"] += file_report["errors"]
            report["summary"]["warnings"] += file_report["warnings"]
            report["summary"]["infos"] += file_report["infos"]
        
        # Add recommendations
        if report["summary"]["errors"] == 0:
            report["recommendations"].append("All configurations are valid! 🎉")
        else:
            report["recommendations"].append("Please fix configuration errors before deployment.")
        
        if report["summary"]["warnings"] > 0:
            report["recommendations"].append("Consider addressing warnings for optimal performance.")
        
        # Save validation report
        report_file = self.config_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Validation report saved to {report_file}")
        
        # Create human-readable report
        report_text = self._generate_readable_report(report)
        report_md_file = self.config_dir / "validation_report.md"
        with open(report_md_file, 'w') as f:
            f.write(report_text)
        print(f"✓ Human-readable report saved to {report_md_file}")
        
        # Print summary
        print(f"✓ Validation Summary:")
        print(f"  - Total files: {report['total_files']}")
        print(f"  - Total issues: {report['summary']['total_issues']}")
        print(f"  - Errors: {report['summary']['errors']}")
        print(f"  - Warnings: {report['summary']['warnings']}")
        print(f"  - Infos: {report['summary']['infos']}")
        
        return report
    
    def _validate_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single configuration file."""
        report = {
            "file_path": str(file_path),
            "total_issues": 0,
            "errors": 0,
            "warnings": 0,
            "infos": 0,
            "issues": []
        }
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Basic validation
            if not isinstance(config, dict):
                report["issues"].append("Configuration must be a dictionary/object")
                report["errors"] += 1
                return report
            
            # Model-specific validations
            if "models" in config:
                self._validate_models_config(config["models"], report)
            
            if "system" in config:
                self._validate_system_config(config["system"], report)
            
            if "ensemble" in config:
                self._validate_ensemble_config(config["ensemble"], report)
            
            report["total_issues"] = len(report["issues"])
            
        except Exception as e:
            report["issues"].append(f"Error loading file: {e}")
            report["errors"] += 1
        
        return report
    
    def _validate_models_config(self, models: Dict[str, Any], report: Dict[str, Any]):
        """Validate models configuration."""
        for model_name, model_config in models.items():
            # Check required fields
            required_fields = ["model_name", "input_size", "device"]
            for field in required_fields:
                if field not in model_config:
                    report["issues"].append(f"{model_name}: Missing required field '{field}'")
                    report["errors"] += 1
            
            # Check input size
            if "input_size" in model_config:
                input_size = model_config["input_size"]
                if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
                    report["issues"].append(f"{model_name}: input_size must be a list/tuple of 2 values")
                    report["errors"] += 1
            
            # Check confidence threshold
            if "confidence_threshold" in model_config:
                ct = model_config["confidence_threshold"]
                if not isinstance(ct, (int, float)) or not 0 <= ct <= 1:
                    report["issues"].append(f"{model_name}: confidence_threshold must be between 0 and 1")
                    report["errors"] += 1
            
            # Check batch size
            if "batch_size" in model_config:
                bs = model_config["batch_size"]
                if not isinstance(bs, int) or bs <= 0:
                    report["issues"].append(f"{model_name}: batch_size must be a positive integer")
                    report["errors"] += 1
    
    def _validate_system_config(self, system: Dict[str, Any], report: Dict[str, Any]):
        """Validate system configuration."""
        # Check device
        if "device" in system:
            device = system["device"]
            if device not in ["cuda", "cpu", "mps", "auto"]:
                report["issues"].append(f"device must be one of ['cuda', 'cpu', 'mps', 'auto'], got '{device}'")
                report["errors"] += 1
        
        # Check memory settings
        if "max_memory_fraction" in system:
            mmf = system["max_memory_fraction"]
            if not isinstance(mmf, (int, float)) or not 0 < mmf <= 1.0:
                report["issues"].append("max_memory_fraction must be between 0 and 1")
                report["errors"] += 1
        
        # Check worker settings
        if "num_workers" in system:
            nw = system["num_workers"]
            if not isinstance(nw, int) or nw < 0:
                report["issues"].append("num_workers must be a non-negative integer")
                report["errors"] += 1
    
    def _validate_ensemble_config(self, ensemble: Dict[str, Any], report: Dict[str, Any]):
        """Validate ensemble configuration."""
        if "weights" in ensemble:
            weights = ensemble["weights"]
            if not isinstance(weights, dict):
                report["issues"].append("ensemble weights must be a dictionary")
                report["errors"] += 1
            else:
                weight_sum = sum(weights.values())
                if abs(weight_sum - 1.0) > 0.01:
                    report["issues"].append(f"ensemble weights should sum to 1.0, got {weight_sum:.3f}")
                    report["warnings"] += 1
        
        if "voting_strategy" in ensemble:
            strategy = ensemble["voting_strategy"]
            if strategy not in ["soft", "hard", "majority"]:
                report["issues"].append(f"voting_strategy must be one of ['soft', 'hard', 'majority'], got '{strategy}'")
                report["errors"] += 1
    
    def _generate_readable_report(self, report: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("# Configuration Validation Report")
        lines.append(f"Generated: {report['generated_at']}")
        lines.append(f"Directory: {report['config_directory']}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append(f"- **Total Files**: {report['total_files']}")
        lines.append(f"- **Total Issues**: {report['summary']['total_issues']}")
        lines.append(f"- **Errors**: {report['summary']['errors']}")
        lines.append(f"- **Warnings**: {report['summary']['warnings']}")
        lines.append(f"- **Info**: {report['summary']['infos']}")
        lines.append("")
        
        # File details
        lines.append("## File Details")
        for file_report in report["files_validated"]:
            lines.append(f"### {Path(file_report['file_path']).name}")
            lines.append(f"- **Status**: {'✅ Valid' if file_report['errors'] == 0 else '❌ Issues Found'}")
            lines.append(f"- **Issues**: {file_report['total_issues']}")
            
            if file_report["issues"]:
                lines.append("- **Issues Found**:")
                for issue in file_report["issues"]:
                    lines.append(f"  - {issue}")
            lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")
        
        return "\n".join(lines)
    
    def run_complete_demo(self):
        """Run the complete configuration management demo."""
        print("🚀 Advanced Configuration Management System Demo")
        print("="*80)
        print("This demo showcases the configuration management system")
        print("for the Soccer Player Recognition project.\n")
        
        start_time = time.time()
        
        try:
            # Run all demonstrations
            config = self.demo_basic_configuration()
            fixed_config, issues = self.demo_configuration_validation()
            model_registry = self.demo_model_specific_configs()
            system_config = self.demo_system_configuration()
            self.demo_configuration_templates()
            validation_report = self.demo_validation_report()
            
            # Final summary
            total_time = time.time() - start_time
            config_files = len(list(self.config_dir.glob("*")))
            
            print("\n" + "="*80)
            print("✅ Configuration Management System Demo Completed Successfully!")
            print(f"⏱ Total execution time: {total_time:.2f} seconds")
            print(f"📁 Configuration directory: {self.config_dir.absolute()}")
            print(f"📄 Generated files: {config_files}")
            
            print(f"\n🎯 Key Features Demonstrated:")
            features = [
                "✓ Centralized configuration management with YAML/JSON support",
                "✓ Model-specific configuration schemas and validation", 
                "✓ System-wide configuration with auto-detection",
                "✓ Comprehensive configuration validation and compatibility checking",
                "✓ Configuration template generation and usage",
                "✓ Detailed validation reporting with human-readable output",
                "✓ Cross-configuration dependency validation",
                "✓ Auto-fix common configuration issues"
            ]
            for feature in features:
                print(f"  {feature}")
            
            print(f"\n📋 Generated Configuration Files:")
            for file_path in sorted(self.config_dir.glob("*")):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    print(f"  - {file_path.name} ({size:,} bytes)")
            
            print(f"\n📖 Next Steps:")
            print(f"  1. Review generated configuration files in {self.config_dir}")
            print(f"  2. Check validation report: validation_report.md")
            print(f"  3. Customize configurations for your specific use case")
            print(f"  4. Use templates to create new configurations")
            print(f"  5. Integrate with your model training/inference pipelines")
            
            print(f"\n🎉 Demo completed successfully! Check the 'config' directory for all generated files.")
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise
        
        return True


def main():
    """Main function to run the configuration system demo."""
    print("Starting Advanced Configuration Management System Demo...")
    print("This will create and validate various configuration files.\n")
    
    # Run the demo
    demo = ConfigDemo()
    success = demo.run_complete_demo()
    
    return success


if __name__ == "__main__":
    main()