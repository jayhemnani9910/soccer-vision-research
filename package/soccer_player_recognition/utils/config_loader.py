"""
Configuration Loader for Soccer Player Recognition Project

This module provides utilities for loading and managing configurations
from both Python settings files and YAML configuration files.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy

# Import project settings
try:
    from config.settings import *
except ImportError:
    # Fallback if import fails
    PROJECT_ROOT = Path(__file__).parent.parent
    LOG_LEVEL = "INFO"

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader for managing project settings."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Path to the configuration directory. If None, uses default.
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent
        else:
            self.config_dir = Path(config_dir)
        
        self.settings = {}
        self.model_configs = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all configuration files."""
        try:
            # Load Python settings
            self._load_python_settings()
            
            # Load YAML model configuration
            self._load_yaml_config()
            
            logger.info("Configurations loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise
    
    def _load_python_settings(self):
        """Load Python settings module."""
        try:
            import sys
            sys.path.insert(0, str(self.config_dir.parent))
            
            # Import settings module
            import config.settings as settings_module
            
            # Get all uppercase attributes from settings
            self.settings = {
                attr: getattr(settings_module, attr)
                for attr in dir(settings_module)
                if attr.isupper() and not attr.startswith('_')
            }
            
            logger.info("Python settings loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load Python settings: {e}")
            self.settings = self._get_default_settings()
    
    def _load_yaml_config(self):
        """Load YAML configuration file."""
        yaml_path = self.config_dir / "model_config.yaml"
        
        if not yaml_path.exists():
            logger.error(f"YAML config file not found: {yaml_path}")
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                self.model_configs = yaml.safe_load(f)
            
            logger.info("YAML configuration loaded successfully")
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading YAML: {e}")
            raise
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings if loading fails."""
        return {
            'PROJECT_ROOT': Path(__file__).parent.parent,
            'DATA_DIR': Path(__file__).parent.parent / "data",
            'MODELS_DIR': Path(__file__).parent.parent / "models",
            'OUTPUTS_DIR': Path(__file__).parent.parent / "outputs",
            'DEFAULT_DEVICE': "cuda",
            'DEFAULT_BATCH_SIZE': 32,
            'LOG_LEVEL': "INFO"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Try Python settings first
        if key in self.settings:
            return self.settings[key]
        
        # Try YAML config
        keys = key.split('.')
        value = self.model_configs
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'rf_detr', 'sam2', 'siglip', 'resnet')
            
        Returns:
            Model configuration dictionary
        """
        return self.model_configs.get(model_name, {})
    
    def get_all_configs(self) -> Dict[str, Any]:
        """
        Get all configurations.
        
        Returns:
            Dictionary containing all configurations
        """
        return {
            'python_settings': deepcopy(self.settings),
            'yaml_config': deepcopy(self.model_configs)
        }
    
    def update_config(self, key: str, value: Any):
        """
        Update a configuration value.
        
        Args:
            key: Configuration key
            value: New value
        """
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            config_dict = self.model_configs if self._is_yaml_key(key) else self.settings
            
            # Navigate to the parent dictionary
            for k in keys[:-1]:
                if k not in config_dict:
                    config_dict[k] = {}
                config_dict = config_dict[k]
            
            # Set the final value
            config_dict[keys[-1]] = value
        else:
            # Simple key
            if self._is_yaml_key(key):
                self.model_configs[key] = value
            else:
                self.settings[key] = value
    
    def _is_yaml_key(self, key: str) -> bool:
        """Check if a key exists in YAML config."""
        keys = key.split('.')
        value = self.model_configs
        
        try:
            for k in keys:
                value = value[k]
            return True
        except (KeyError, TypeError):
            return False
    
    def save_config(self, filepath: Optional[Union[str, Path]] = None):
        """
        Save current configuration to YAML file.
        
        Args:
            filepath: Path to save the configuration. If None, uses default YAML file.
        """
        if filepath is None:
            filepath = self.config_dir / "model_config.yaml"
        else:
            filepath = Path(filepath)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.model_configs, f, default_flow_style=False, 
                         indent=2, allow_unicode=True)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration settings.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required directories
        required_dirs = ['DATA_DIR', 'MODELS_DIR', 'OUTPUTS_DIR']
        for dir_key in required_dirs:
            if dir_key in self.settings:
                dir_path = self.settings[dir_key]
                if not dir_path.exists():
                    validation_results['warnings'].append(
                        f"Directory {dir_key} does not exist: {dir_path}"
                    )
        
        # Check model configurations
        required_models = ['rf_detr', 'sam2', 'siglip', 'resnet']
        for model_name in required_models:
            if model_name not in self.model_configs:
                validation_results['errors'].append(
                    f"Missing configuration for model: {model_name}"
                )
            else:
                model_config = self.model_configs[model_name]
                if 'model_path' not in model_config:
                    validation_results['warnings'].append(
                        f"Model path not specified for: {model_name}"
                    )
        
        if validation_results['errors']:
            validation_results['valid'] = False
        
        return validation_results
    
    def get_device(self, model_name: Optional[str] = None) -> str:
        """
        Get device configuration for model or general use.
        
        Args:
            model_name: Specific model to get device for
            
        Returns:
            Device string ('cuda', 'cpu', etc.)
        """
        if model_name:
            model_config = self.get_model_config(model_name)
            return model_config.get('device', self.get('DEFAULT_DEVICE', 'cpu'))
        
        return self.get('DEFAULT_DEVICE', 'cpu')
    
    def get_batch_size(self, model_name: str) -> int:
        """
        Get batch size for specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Batch size
        """
        model_config = self.get_model_config(model_name)
        return model_config.get('batch_size', self.get('DEFAULT_BATCH_SIZE', 32))


# Global configuration instance
_config_instance = None


def get_config() -> ConfigLoader:
    """
    Get the global configuration instance.
    
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance


def load_config(config_dir: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """
    Load configuration from specified directory.
    
    Args:
        config_dir: Path to configuration directory
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_dir)


def get_model_path(model_name: str) -> str:
    """
    Get model file path for specified model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model file path
    """
    config = get_config()
    model_config = config.get_model_config(model_name)
    model_path = model_config.get('model_path', '')
    
    # If relative path, make it relative to project root
    if model_path and not os.path.isabs(model_path):
        project_root = config.get('PROJECT_ROOT', Path.cwd())
        model_path = str(project_root / model_path)
    
    return model_path


def get_input_paths() -> Dict[str, str]:
    """
    Get all input data paths.
    
    Returns:
        Dictionary of input path configurations
    """
    config = get_config()
    
    return {
        'input_data': str(config.get('INPUT_DATA_DIR', '')),
        'raw_data': str(config.get('RAW_DATA_DIR', '')),
        'processed_data': str(config.get('PROCESSED_DATA_DIR', ''))
    }


def get_output_paths() -> Dict[str, str]:
    """
    Get all output directory paths.
    
    Returns:
        Dictionary of output path configurations
    """
    config = get_config()
    
    return {
        'outputs_dir': str(config.get('OUTPUTS_DIR', '')),
        'detection': str(config.get('DETECTION_OUTPUTS_DIR', '')),
        'segmentation': str(config.get('SEGMENTATION_OUTPUTS_DIR', '')),
        'classification': str(config.get('CLASSIFICATION_OUTPUTS_DIR', '')),
        'identification': str(config.get('IDENTIFICATION_OUTPUTS_DIR', ''))
    }


if __name__ == "__main__":
    # Example usage
    config = load_config()
    
    print("=== Configuration Summary ===")
    print(f"Project Root: {config.get('PROJECT_ROOT')}")
    print(f"Data Directory: {config.get('DATA_DIR')}")
    print(f"Models Directory: {config.get('MODELS_DIR')}")
    print(f"Outputs Directory: {config.get('OUTPUTS_DIR')}")
    print()
    
    print("=== Model Configurations ===")
    for model_name in ['rf_detr', 'sam2', 'siglip', 'resnet']:
        model_config = config.get_model_config(model_name)
        model_path = model_config.get('model_path', 'Not specified')
        print(f"{model_name}: {model_path}")
    
    print()
    print("=== Validation Results ===")
    validation = config.validate_config()
    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print("Errors:", validation['errors'])
    if validation['warnings']:
        print("Warnings:", validation['warnings'])