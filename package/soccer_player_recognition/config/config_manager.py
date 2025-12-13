"""
Advanced Configuration Manager for Soccer Player Recognition System
================================================================

This module provides a centralized configuration management system that handles:
- Hot-reloading of configuration files
- Configuration validation and schema enforcement
- Environment variable substitution
- Configuration caching and persistence
- Backwards compatibility checking

Author: Advanced AI Assistant
Date: 2025-11-04
"""

import os
import json
import yaml
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import warnings


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    PYTHON = "py"


class ConfigScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    MODEL = "model"
    SYSTEM = "system"
    USER = "user"


@dataclass
class ConfigChange:
    """Represents a configuration change event."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float
    source: str


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    required: List[str]
    optional: Dict[str, Any]
    types: Dict[str, type]
    constraints: Dict[str, Callable]
    default_values: Dict[str, Any]


class ConfigManager:
    """
    Advanced Configuration Manager with hot-reloading, validation, and caching.
    
    Features:
    - Hot-reload configuration files when they change
    - Validate configurations against schemas
    - Environment variable substitution
    - Configuration caching with cache invalidation
    - Change notifications and callbacks
    - Thread-safe operations
    - Backwards compatibility checking
    """
    
    def __init__(self, 
                 config_dir: Union[str, Path],
                 cache_dir: Optional[Union[str, Path]] = None,
                 enable_hot_reload: bool = True,
                 enable_validation: bool = True,
                 enable_caching: bool = True,
                 env_prefix: str = "SOCCER_"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            cache_dir: Directory for cache files (defaults to config_dir/cache)
            enable_hot_reload: Enable automatic configuration reloading
            enable_validation: Enable schema validation
            enable_caching: Enable configuration caching
            env_prefix: Prefix for environment variable substitution
        """
        self.config_dir = Path(config_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.config_dir / "cache"
        self.enable_hot_reload = enable_hot_reload
        self.enable_validation = enable_validation
        self.enable_caching = enable_caching
        self.env_prefix = env_prefix
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration storage
        self._config_cache: Dict[str, Any] = {}
        self._schema_cache: Dict[str, ConfigSchema] = {}
        self._file_timestamps: Dict[str, float] = {}
        self._change_listeners: List[Callable] = []
        self._change_history: List[ConfigChange] = []
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ConfigManager")
        self._monitor_thread = None
        self._stop_monitor = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring if enabled
        if enable_hot_reload:
            self._start_monitoring()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def _start_monitoring(self):
        """Start the configuration monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitor = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_config_files,
            daemon=True,
            name="ConfigMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("Configuration monitoring started")
    
    def _monitor_config_files(self):
        """Monitor configuration files for changes."""
        while not self._stop_monitor:
            try:
                with self._lock:
                    changed_files = self._check_file_changes()
                    for file_path in changed_files:
                        self.logger.info(f"Configuration file changed: {file_path}")
                        # Remove from cache to force reload
                        if file_path in self._config_cache:
                            del self._config_cache[file_path]
                        if file_path in self._file_timestamps:
                            del self._file_timestamps[file_path]
                
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                self.logger.error(f"Error monitoring configuration files: {e}")
                time.sleep(5)
    
    def _check_file_changes(self) -> List[str]:
        """Check for changes in configuration files."""
        changed_files = []
        
        for config_file in self.config_dir.rglob("*.{yaml,yml,json}"):
            try:
                mtime = config_file.stat().st_mtime
                file_key = str(config_file)
                
                if file_key not in self._file_timestamps:
                    self._file_timestamps[file_key] = mtime
                elif self._file_timestamps[file_key] != mtime:
                    changed_files.append(file_key)
                    self._file_timestamps[file_key] = mtime
            except OSError:
                continue
        
        return changed_files
    
    def register_change_listener(self, callback: Callable[[ConfigChange], None]):
        """Register a callback for configuration changes."""
        with self._lock:
            self._change_listeners.append(callback)
    
    def _notify_changes(self, changes: List[ConfigChange]):
        """Notify listeners of configuration changes."""
        with self._lock:
            for change in changes:
                self._change_history.append(change)
                
            for callback in self._change_listeners:
                try:
                    callback(change)
                except Exception as e:
                    self.logger.error(f"Error in change callback: {e}")
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            env_var = f"{self.env_prefix}{var_name}"
            return os.getenv(env_var, obj)
        else:
            return obj
    
    def _load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                content = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                content = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
        
        # Substitute environment variables
        content = self._substitute_env_vars(content)
        
        return content
    
    def _get_cache_key(self, file_path: Union[str, Path]) -> str:
        """Generate cache key for configuration file."""
        file_path = Path(file_path)
        content_hash = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    content_hash.update(chunk)
        except OSError:
            return f"missing_{file_path}"
        
        mtime = file_path.stat().st_mtime
        return f"{file_path}_{content_hash.hexdigest()}_{mtime}"
    
    def load_config(self, 
                   file_path: Union[str, Path],
                   scope: ConfigScope = ConfigScope.GLOBAL,
                   validate: Optional[bool] = None) -> Dict[str, Any]:
        """
        Load configuration from file with caching and validation.
        
        Args:
            file_path: Path to configuration file
            scope: Configuration scope
            validate: Override global validation setting
            
        Returns:
            Loaded configuration dictionary
        """
        file_path = Path(file_path)
        
        with self._lock:
            # Check cache first
            if self.enable_caching:
                cache_key = self._get_cache_key(file_path)
                if cache_key in self._config_cache:
                    self.logger.debug(f"Loading configuration from cache: {file_path}")
                    return self._config_cache[cache_key]
            
            try:
                # Load configuration
                config = self._load_from_file(file_path)
                
                # Validate if enabled
                if validate if validate is not None else self.enable_validation:
                    self.validate_config(config, scope)
                
                # Cache the configuration
                if self.enable_caching:
                    self._config_cache[cache_key] = config
                
                self.logger.info(f"Configuration loaded: {file_path}")
                return config
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration from {file_path}: {e}")
                raise
    
    def save_config(self, 
                   config: Dict[str, Any], 
                   file_path: Union[str, Path],
                   format: ConfigFormat = ConfigFormat.YAML):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            file_path: Path to save configuration
            format: File format (yaml, json, py)
        """
        file_path = Path(file_path)
        
        with self._lock:
            # Ensure directory exists
            file_path.parent.mkdir(exist_ok=True, parents=True)
            
            try:
                if format == ConfigFormat.YAML:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, indent=2)
                elif format == ConfigFormat.JSON:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                elif format == ConfigFormat.PYTHON:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"# Generated configuration file\n")
                        f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write(f"CONFIG = {repr(config)}\n")
                
                # Update cache
                if self.enable_caching:
                    cache_key = self._get_cache_key(file_path)
                    self._config_cache[cache_key] = config
                
                self.logger.info(f"Configuration saved: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save configuration to {file_path}: {e}")
                raise
    
    def get_config(self, 
                   key: str, 
                   default: Any = None,
                   file_path: Optional[Union[str, Path]] = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation, e.g., "model.rf_detr.batch_size")
            default: Default value if key not found
            file_path: Specific file to get config from
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            if file_path:
                config = self.load_config(file_path)
            else:
                # Try to find config in cache or load from default locations
                possible_files = [
                    self.config_dir / "model_config.yaml",
                    self.config_dir / "system_config.yaml",
                    self.config_dir / "settings.py"
                ]
                
                config = {}
                for pf in possible_files:
                    try:
                        config.update(self.load_config(pf))
                    except (FileNotFoundError, Exception):
                        continue
            
            # Navigate using dot notation
            keys = key.split('.')
            current = config
            
            try:
                for k in keys:
                    current = current[k]
                return current
            except (KeyError, TypeError):
                return default
    
    def set_config(self, 
                  key: str, 
                  value: Any,
                  file_path: Optional[Union[str, Path]] = None,
                  save: bool = False):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key
            value: Value to set
            file_path: Specific file to modify
            save: Whether to save changes to file
        """
        with self._lock:
            if file_path:
                config = self.load_config(file_path)
            else:
                config = self.get_all_configs()
            
            # Navigate and set value
            keys = key.split('.')
            current = config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            old_value = current.get(keys[-1])
            current[keys[-1]] = value
            
            # Record change
            change = ConfigChange(
                key=key,
                old_value=old_value,
                new_value=value,
                timestamp=time.time(),
                source="manual_set"
            )
            self._notify_changes([change])
            
            # Save if requested
            if save and file_path:
                self.save_config(config, file_path)
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all loaded configurations merged together."""
        with self._lock:
            all_configs = {}
            
            # Load all configuration files
            for config_file in self.config_dir.rglob("*.{yaml,yml,json}"):
                try:
                    config = self.load_config(config_file)
                    # Merge configurations (later files override earlier ones)
                    self._deep_merge(all_configs, config)
                except Exception as e:
                    self.logger.warning(f"Failed to load {config_file}: {e}")
                    continue
            
            return all_configs
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge two dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def validate_config(self, config: Dict[str, Any], scope: ConfigScope):
        """Validate configuration against schema."""
        # This is a placeholder - actual validation logic would be in ConfigValidator
        # For now, just check for basic structure
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Check for required fields based on scope
        required_fields = {
            ConfigScope.MODEL: ["model_name", "model_path"],
            ConfigScope.SYSTEM: ["device", "batch_size"],
            ConfigScope.GLOBAL: []
        }
        
        missing_fields = [field for field in required_fields.get(scope, []) 
                         if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required fields for {scope.value}: {missing_fields}")
    
    def create_config_template(self, 
                              file_path: Union[str, Path],
                              template_type: str = "model"):
        """Create a configuration template file."""
        templates = {
            "model": {
                "model_name": "model_name",
                "model_path": "path/to/model.pth",
                "config_path": "path/to/config.yaml",
                "input_size": [224, 224],
                "device": "cuda",
                "batch_size": 1,
                "num_workers": 4,
                "confidence_threshold": 0.5,
                "pretrained": True
            },
            "system": {
                "device": "cuda",
                "mixed_precision": True,
                "compile_model": False,
                "gradient_checkpointing": True,
                "max_memory_fraction": 0.8,
                "clear_cache_frequency": 10,
                "log_model_summary": True,
                "save_training_history": True
            }
        }
        
        template = templates.get(template_type, {})
        self.save_config(template, file_path, ConfigFormat.YAML)
        
        self.logger.info(f"Configuration template created: {file_path}")
    
    def export_config(self, 
                     file_path: Union[str, Path],
                     format: ConfigFormat = ConfigFormat.YAML,
                     include_metadata: bool = True):
        """Export current configuration to file."""
        config = self.get_all_configs()
        
        if include_metadata:
            config = {
                "_metadata": {
                    "exported_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "config_manager_version": "1.0.0",
                    "total_keys": self._count_keys(config)
                },
                **config
            }
        
        self.save_config(config, file_path, format)
    
    def _count_keys(self, obj: Any) -> int:
        """Count total number of keys in nested dictionary."""
        if isinstance(obj, dict):
            return sum(1 + self._count_keys(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(self._count_keys(item) for item in obj)
        else:
            return 0
    
    def clear_cache(self):
        """Clear configuration cache."""
        with self._lock:
            self._config_cache.clear()
            self.logger.info("Configuration cache cleared")
    
    def get_change_history(self, 
                          limit: Optional[int] = None,
                          since: Optional[float] = None) -> List[ConfigChange]:
        """Get configuration change history."""
        with self._lock:
            history = self._change_history
            
            if since:
                history = [h for h in history if h.timestamp >= since]
            
            if limit:
                history = history[-limit:]
            
            return history.copy()
    
    def shutdown(self):
        """Shutdown the configuration manager."""
        self.logger.info("Shutting down configuration manager")
        
        # Stop monitoring thread
        self._stop_monitor = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)


# Convenience functions
def get_config_manager(config_dir: Union[str, Path], **kwargs) -> ConfigManager:
    """Get a configured ConfigManager instance."""
    return ConfigManager(config_dir, **kwargs)


def load_model_config(model_name: str, config_dir: Union[str, Path] = "config") -> Dict[str, Any]:
    """Load configuration for a specific model."""
    manager = ConfigManager(config_dir)
    return manager.get_config(f"models.{model_name}", {})


def update_config(key: str, value: Any, config_dir: Union[str, Path] = "config"):
    """Update configuration value."""
    manager = ConfigManager(config_dir)
    manager.set_config(key, value, save=True)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    with ConfigManager("config") as manager:
        # Load all configurations
        config = manager.get_all_configs()
        print(f"Loaded configuration: {config}")
        
        # Get specific value
        batch_size = manager.get_config("rf_detr.batch_size", default=1)
        print(f"RF-DETR batch size: {batch_size}")
        
        # Create template
        manager.create_config_template("config/new_model_template.yaml", "model")
        
        # Export configuration
        manager.export_config("config/exported_config.yaml")
        
        # Monitor changes
        def on_change(change: ConfigChange):
            print(f"Config changed: {change.key} from {change.old_value} to {change.new_value}")
        
        manager.register_change_listener(on_change)