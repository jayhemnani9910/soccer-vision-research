"""
System-Wide Configuration Manager
=================================

This module provides system-wide configuration management for the soccer player
recognition system. Handles device settings, performance optimization, logging,
data paths, and other system-level parameters.

System settings categories:
- Device and compute settings
- Performance optimization
- Logging and monitoring
- Data paths and directories
- Runtime behavior
- Security settings

Author: Advanced AI Assistant
Date: 2025-11-04
"""

import os
import json
import logging
import psutil
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import warnings
import threading
import time


class DeviceType(Enum):
    """Supported device types."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class LoggingLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Precision(Enum):
    """Floating point precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class DeviceConfig:
    """Device and compute configuration."""
    # Primary device selection
    device: DeviceType = DeviceType.AUTO
    device_id: int = 0
    
    # Memory settings
    max_memory_fraction: float = 0.8
    clear_cache_frequency: int = 10
    
    # Mixed precision
    mixed_precision: bool = True
    precision: Precision = Precision.FP16
    
    # Compilation and optimization
    compile_model: bool = False
    gradient_checkpointing: bool = True
    use_cudnn_benchmark: bool = True
    
    # Multi-GPU settings
    multi_gpu: bool = False
    distributed: bool = False
    device_ids: List[int] = field(default_factory=lambda: [0])
    
    # CPU settings
    cpu_threads: Optional[int] = None
    cpu_architecture: str = "auto"  # "x86_64", "arm64", "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device": self.device.value,
            "device_id": self.device_id,
            "max_memory_fraction": self.max_memory_fraction,
            "clear_cache_frequency": self.clear_cache_frequency,
            "mixed_precision": self.mixed_precision,
            "precision": self.precision.value,
            "compile_model": self.compile_model,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_cudnn_benchmark": self.use_cudnn_benchmark,
            "multi_gpu": self.multi_gpu,
            "distributed": self.distributed,
            "device_ids": self.device_ids,
            "cpu_threads": self.cpu_threads,
            "cpu_architecture": self.cpu_architecture
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceConfig':
        """Create from dictionary."""
        return cls(
            device=DeviceType(data.get("device", "auto")),
            device_id=data.get("device_id", 0),
            max_memory_fraction=data.get("max_memory_fraction", 0.8),
            clear_cache_frequency=data.get("clear_cache_frequency", 10),
            mixed_precision=data.get("mixed_precision", True),
            precision=Precision(data.get("precision", "fp16")),
            compile_model=data.get("compile_model", False),
            gradient_checkpointing=data.get("gradient_checkpointing", True),
            use_cudnn_benchmark=data.get("use_cudnn_benchmark", True),
            multi_gpu=data.get("multi_gpu", False),
            distributed=data.get("distributed", False),
            device_ids=data.get("device_ids", [0]),
            cpu_threads=data.get("cpu_threads"),
            cpu_architecture=data.get("cpu_architecture", "auto")
        )


@dataclass
class PerformanceConfig:
    """Performance optimization settings."""
    # Data loading
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Caching
    enable_caching: bool = True
    cache_size_mb: int = 512
    cache_dir: str = "cache"
    
    # Parallel processing
    num_workers: int = 4
    max_workers: int = 8
    
    # Memory management
    memory_pool_size: int = 1024  # MB
    garbage_collection_threshold: float = 0.8
    
    # I/O optimization
    use_memory_mapping: bool = True
    read_ahead_size: int = 1024  # KB
    
    # Model optimization
    fuse_conv_bn: bool = True
    remove_dropout_inference: bool = True
    
    # Quantization and pruning
    quantization: bool = False
    pruning: bool = False
    pruning_ratio: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "enable_caching": self.enable_caching,
            "cache_size_mb": self.cache_size_mb,
            "cache_dir": self.cache_dir,
            "num_workers": self.num_workers,
            "max_workers": self.max_workers,
            "memory_pool_size": self.memory_pool_size,
            "garbage_collection_threshold": self.garbage_collection_threshold,
            "use_memory_mapping": self.use_memory_mapping,
            "read_ahead_size": self.read_ahead_size,
            "fuse_conv_bn": self.fuse_conv_bn,
            "remove_dropout_inference": self.remove_dropout_inference,
            "quantization": self.quantization,
            "pruning": self.pruning,
            "pruning_ratio": self.pruning_ratio
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceConfig':
        """Create from dictionary."""
        return cls(
            pin_memory=data.get("pin_memory", True),
            persistent_workers=data.get("persistent_workers", True),
            prefetch_factor=data.get("prefetch_factor", 2),
            enable_caching=data.get("enable_caching", True),
            cache_size_mb=data.get("cache_size_mb", 512),
            cache_dir=data.get("cache_dir", "cache"),
            num_workers=data.get("num_workers", 4),
            max_workers=data.get("max_workers", 8),
            memory_pool_size=data.get("memory_pool_size", 1024),
            garbage_collection_threshold=data.get("garbage_collection_threshold", 0.8),
            use_memory_mapping=data.get("use_memory_mapping", True),
            read_ahead_size=data.get("read_ahead_size", 1024),
            fuse_conv_bn=data.get("fuse_conv_bn", True),
            remove_dropout_inference=data.get("remove_dropout_inference", True),
            quantization=data.get("quantization", False),
            pruning=data.get("pruning", False),
            pruning_ratio=data.get("pruning_ratio", 0.3)
        )


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    # Log level
    level: LoggingLevel = LoggingLevel.INFO
    
    # Log format
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Log files
    log_to_file: bool = True
    log_file_path: str = "logs/system.log"
    max_log_size: int = 10  # MB
    backup_count: int = 5
    
    # Console logging
    log_to_console: bool = True
    console_level: LoggingLevel = LoggingLevel.INFO
    
    # Model-specific logging
    log_model_summary: bool = True
    log_training_history: bool = True
    log_performance_metrics: bool = True
    
    # Debug settings
    log_memory_usage: bool = False
    log_gpu_utilization: bool = False
    log_inference_times: bool = True
    
    # Remote logging
    use_remote_logging: bool = False
    remote_log_endpoint: Optional[str] = None
    remote_log_level: LoggingLevel = LoggingLevel.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "format": self.format,
            "date_format": self.date_format,
            "log_to_file": self.log_to_file,
            "log_file_path": self.log_file_path,
            "max_log_size": self.max_log_size,
            "backup_count": self.backup_count,
            "log_to_console": self.log_to_console,
            "console_level": self.console_level.value,
            "log_model_summary": self.log_model_summary,
            "log_training_history": self.log_training_history,
            "log_performance_metrics": self.log_performance_metrics,
            "log_memory_usage": self.log_memory_usage,
            "log_gpu_utilization": self.log_gpu_utilization,
            "log_inference_times": self.log_inference_times,
            "use_remote_logging": self.use_remote_logging,
            "remote_log_endpoint": self.remote_log_endpoint,
            "remote_log_level": self.remote_log_level.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoggingConfig':
        """Create from dictionary."""
        return cls(
            level=LoggingLevel(data.get("level", "INFO")),
            format=data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            date_format=data.get("date_format", "%Y-%m-%d %H:%M:%S"),
            log_to_file=data.get("log_to_file", True),
            log_file_path=data.get("log_file_path", "logs/system.log"),
            max_log_size=data.get("max_log_size", 10),
            backup_count=data.get("backup_count", 5),
            log_to_console=data.get("log_to_console", True),
            console_level=LoggingLevel(data.get("console_level", "INFO")),
            log_model_summary=data.get("log_model_summary", True),
            log_training_history=data.get("log_training_history", True),
            log_performance_metrics=data.get("log_performance_metrics", True),
            log_memory_usage=data.get("log_memory_usage", False),
            log_gpu_utilization=data.get("log_gpu_utilization", False),
            log_inference_times=data.get("log_inference_times", True),
            use_remote_logging=data.get("use_remote_logging", False),
            remote_log_endpoint=data.get("remote_log_endpoint"),
            remote_log_level=LoggingLevel(data.get("remote_log_level", "WARNING"))
        )


@dataclass
class DataConfig:
    """Data paths and management configuration."""
    # Base directories
    root_dir: str = "./"
    data_dir: str = "data"
    output_dir: str = "outputs"
    model_dir: str = "models"
    cache_dir: str = "cache"
    logs_dir: str = "logs"
    
    # Dataset settings
    dataset_path: str = "data/datasets"
    annotation_path: str = "data/annotations"
    
    # File formats
    supported_image_formats: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
    ])
    supported_video_formats: List[str] = field(default_factory=lambda: [
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"
    ])
    
    # Data management
    auto_cleanup: bool = True
    cleanup_threshold_gb: float = 10.0
    keep_recent_files: int = 10
    
    # Backup settings
    enable_backup: bool = False
    backup_dir: str = "backup"
    backup_frequency: str = "daily"  # "hourly", "daily", "weekly"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root_dir": self.root_dir,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "model_dir": self.model_dir,
            "cache_dir": self.cache_dir,
            "logs_dir": self.logs_dir,
            "dataset_path": self.dataset_path,
            "annotation_path": self.annotation_path,
            "supported_image_formats": self.supported_image_formats,
            "supported_video_formats": self.supported_video_formats,
            "auto_cleanup": self.auto_cleanup,
            "cleanup_threshold_gb": self.cleanup_threshold_gb,
            "keep_recent_files": self.keep_recent_files,
            "enable_backup": self.enable_backup,
            "backup_dir": self.backup_dir,
            "backup_frequency": self.backup_frequency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        """Create from dictionary."""
        return cls(
            root_dir=data.get("root_dir", "./"),
            data_dir=data.get("data_dir", "data"),
            output_dir=data.get("output_dir", "outputs"),
            model_dir=data.get("model_dir", "models"),
            cache_dir=data.get("cache_dir", "cache"),
            logs_dir=data.get("logs_dir", "logs"),
            dataset_path=data.get("dataset_path", "data/datasets"),
            annotation_path=data.get("annotation_path", "data/annotations"),
            supported_image_formats=data.get("supported_image_formats", [
                ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
            ]),
            supported_video_formats=data.get("supported_video_formats", [
                ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"
            ]),
            auto_cleanup=data.get("auto_cleanup", True),
            cleanup_threshold_gb=data.get("cleanup_threshold_gb", 10.0),
            keep_recent_files=data.get("keep_recent_files", 10),
            enable_backup=data.get("enable_backup", False),
            backup_dir=data.get("backup_dir", "backup"),
            backup_frequency=data.get("backup_frequency", "daily")
        )


@dataclass
class RuntimeConfig:
    """Runtime behavior configuration."""
    # Execution mode
    mode: str = "inference"  # "training", "inference", "evaluation", "debug"
    
    # Threading
    use_threads: bool = True
    thread_pool_size: int = 4
    max_concurrent_operations: int = 8
    
    # Memory management
    memory_management_strategy: str = "auto"  # "aggressive", "conservative", "auto"
    memory_limit_gb: Optional[float] = None
    
    # Timeout settings
    model_loading_timeout: int = 300  # seconds
    inference_timeout: int = 60
    training_timeout: Optional[int] = None
    
    # Error handling
    continue_on_error: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_interval: int = 100  # steps
    checkpoint_dir: str = "checkpoints"
    
    # Progress tracking
    show_progress: bool = True
    progress_update_frequency: int = 1  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode,
            "use_threads": self.use_threads,
            "thread_pool_size": self.thread_pool_size,
            "max_concurrent_operations": self.max_concurrent_operations,
            "memory_management_strategy": self.memory_management_strategy,
            "memory_limit_gb": self.memory_limit_gb,
            "model_loading_timeout": self.model_loading_timeout,
            "inference_timeout": self.inference_timeout,
            "training_timeout": self.training_timeout,
            "continue_on_error": self.continue_on_error,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "show_progress": self.show_progress,
            "progress_update_frequency": self.progress_update_frequency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuntimeConfig':
        """Create from dictionary."""
        return cls(
            mode=data.get("mode", "inference"),
            use_threads=data.get("use_threads", True),
            thread_pool_size=data.get("thread_pool_size", 4),
            max_concurrent_operations=data.get("max_concurrent_operations", 8),
            memory_management_strategy=data.get("memory_management_strategy", "auto"),
            memory_limit_gb=data.get("memory_limit_gb"),
            model_loading_timeout=data.get("model_loading_timeout", 300),
            inference_timeout=data.get("inference_timeout", 60),
            training_timeout=data.get("training_timeout"),
            continue_on_error=data.get("continue_on_error", False),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            enable_checkpoints=data.get("enable_checkpoints", True),
            checkpoint_interval=data.get("checkpoint_interval", 100),
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints"),
            show_progress=data.get("show_progress", True),
            progress_update_frequency=data.get("progress_update_frequency", 1)
        )


@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    # Input validation
    validate_inputs: bool = True
    max_input_size_mb: int = 100
    allowed_domains: List[str] = field(default_factory=list)
    
    # Output sanitization
    sanitize_outputs: bool = True
    strip_metadata: bool = True
    
    # Model security
    model_encryption: bool = False
    secure_inference: bool = False
    
    # Data privacy
    anonymize_data: bool = False
    data_retention_days: int = 30
    
    # Access control
    enable_auth: bool = False
    require_https: bool = False
    
    # Audit logging
    audit_logging: bool = False
    audit_log_path: str = "audit.log"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validate_inputs": self.validate_inputs,
            "max_input_size_mb": self.max_input_size_mb,
            "allowed_domains": self.allowed_domains,
            "sanitize_outputs": self.sanitize_outputs,
            "strip_metadata": self.strip_metadata,
            "model_encryption": self.model_encryption,
            "secure_inference": self.secure_inference,
            "anonymize_data": self.anonymize_data,
            "data_retention_days": self.data_retention_days,
            "enable_auth": self.enable_auth,
            "require_https": self.require_https,
            "audit_logging": self.audit_logging,
            "audit_log_path": self.audit_log_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityConfig':
        """Create from dictionary."""
        return cls(
            validate_inputs=data.get("validate_inputs", True),
            max_input_size_mb=data.get("max_input_size_mb", 100),
            allowed_domains=data.get("allowed_domains", []),
            sanitize_outputs=data.get("sanitize_outputs", True),
            strip_metadata=data.get("strip_metadata", True),
            model_encryption=data.get("model_encryption", False),
            secure_inference=data.get("secure_inference", False),
            anonymize_data=data.get("anonymize_data", False),
            data_retention_days=data.get("data_retention_days", 30),
            enable_auth=data.get("enable_auth", False),
            require_https=data.get("require_https", False),
            audit_logging=data.get("audit_logging", False),
            audit_log_path=data.get("audit_log_path", "audit.log")
        )


@dataclass
class SystemConfig:
    """Complete system configuration."""
    # Configuration metadata
    version: str = "1.0.0"
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    description: str = "Soccer Player Recognition System Configuration"
    
    # Subsystem configurations
    device: DeviceConfig = field(default_factory=DeviceConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # System information
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created": self.created,
            "description": self.description,
            "device": self.device.to_dict(),
            "performance": self.performance.to_dict(),
            "logging": self.logging.to_dict(),
            "data": self.data.to_dict(),
            "runtime": self.runtime.to_dict(),
            "security": self.security.to_dict(),
            "system_info": self.system_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary."""
        config = cls()
        config.version = data.get("version", "1.0.0")
        config.created = data.get("created", time.strftime("%Y-%m-%d %H:%M:%S"))
        config.description = data.get("description", "Soccer Player Recognition System Configuration")
        
        if "device" in data:
            config.device = DeviceConfig.from_dict(data["device"])
        if "performance" in data:
            config.performance = PerformanceConfig.from_dict(data["performance"])
        if "logging" in data:
            config.logging = LoggingConfig.from_dict(data["logging"])
        if "data" in data:
            config.data = DataConfig.from_dict(data["data"])
        if "runtime" in data:
            config.runtime = RuntimeConfig.from_dict(data["runtime"])
        if "security" in data:
            config.security = SecurityConfig.from_dict(data["security"])
        
        config.system_info = data.get("system_info", {})
        return config


class SystemConfigManager:
    """
    Manager for system-wide configuration.
    Handles loading, saving, validation, and system detection.
    """
    
    def __init__(self, config_dir: Union[str, Path]):
        """Initialize the system config manager."""
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # System configuration
        self.system_config: Optional[SystemConfig] = None
        self.config_file = self.config_dir / "system_config.json"
        
        # Auto-detection
        self._detect_system_info()
    
    def _detect_system_info(self):
        """Auto-detect system information."""
        try:
            # Hardware information
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            
            # GPU information (if available)
            gpu_info = {}
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_info = {
                        "cuda_available": True,
                        "cuda_version": torch.version.cuda,
                        "gpu_count": torch.cuda.device_count(),
                        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                        "gpu_memory": [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
                    }
                else:
                    gpu_info = {"cuda_available": False}
            except ImportError:
                gpu_info = {"cuda_available": False}
            
            self.auto_detected_info = {
                "cpu_count": cpu_count,
                "memory_total_gb": round(memory_info.total / (1024**3), 2),
                "memory_available_gb": round(memory_info.available / (1024**3), 2),
                "platform": os.name,
                "architecture": os.uname().machine if hasattr(os, 'uname') else "unknown",
                "python_version": os.sys.version,
                **gpu_info
            }
            
            self.logger.info(f"System detected: {self.auto_detected_info}")
            
        except Exception as e:
            self.logger.warning(f"Failed to detect system information: {e}")
            self.auto_detected_info = {}
    
    def load_system_config(self) -> SystemConfig:
        """Load system configuration from file or create default."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                config = SystemConfig.from_dict(config_data)
                self.logger.info("Loaded system configuration from file")
            else:
                # Create default configuration with auto-detection
                config = self._create_default_config()
                self.logger.info("Created default system configuration")
            
            # Merge with auto-detected system info
            config.system_info.update(self.auto_detected_info)
            
            self.system_config = config
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading system configuration: {e}")
            # Fall back to default configuration
            config = self._create_default_config()
            config.system_info.update(self.auto_detected_info)
            self.system_config = config
            return config
    
    def save_system_config(self, config: Optional[SystemConfig] = None):
        """Save system configuration to file."""
        if config is None:
            config = self.system_config
        
        if config is None:
            raise ValueError("No system configuration to save")
        
        try:
            # Ensure directory exists
            self.config_dir.mkdir(exist_ok=True, parents=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            self.logger.info(f"System configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving system configuration: {e}")
            raise
    
    def _create_default_config(self) -> SystemConfig:
        """Create default system configuration with auto-detection."""
        config = SystemConfig()
        
        # Auto-detect optimal settings
        if self.auto_detected_info:
            # Device settings
            if self.auto_detected_info.get("cuda_available", False):
                config.device.device = DeviceType.CUDA
                config.device.mixed_precision = True
            else:
                config.device.device = DeviceType.CPU
                config.device.mixed_precision = False
            
            # CPU thread settings
            cpu_count = self.auto_detected_info.get("cpu_count", 4)
            config.device.cpu_threads = cpu_count
            config.performance.num_workers = min(cpu_count, 8)
            config.runtime.thread_pool_size = cpu_count
            
            # Memory settings
            memory_gb = self.auto_detected_info.get("memory_total_gb", 8)
            config.performance.memory_pool_size = max(512, int(memory_gb * 64))  # ~6.4% of total
            config.performance.cache_size_mb = min(2048, int(memory_gb * 128))  # ~12.5% of total
        
        return config
    
    def validate_system_config(self, config: SystemConfig) -> bool:
        """Validate system configuration for compatibility."""
        try:
            issues = []
            
            # Check device settings
            if config.device.device == DeviceType.CUDA:
                if not torch.cuda.is_available():
                    issues.append("CUDA device selected but not available")
            
            # Check memory settings
            if config.performance.memory_pool_size > self.auto_detected_info.get("memory_total_gb", 8) * 1024:
                issues.append("Memory pool size exceeds available system memory")
            
            # Check thread settings
            max_workers = self.auto_detected_info.get("cpu_count", 4)
            if config.performance.num_workers > max_workers:
                issues.append(f"Number of workers ({config.performance.num_workers}) exceeds CPU cores ({max_workers})")
            
            if issues:
                for issue in issues:
                    self.logger.warning(f"Configuration issue: {issue}")
                return False
            
            self.logger.info("System configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during system configuration validation: {e}")
            return False
    
    def optimize_config_for_system(self, config: SystemConfig) -> SystemConfig:
        """Optimize configuration based on detected system capabilities."""
        if not self.auto_detected_info:
            return config
        
        optimized = config
        
        # Memory optimization
        memory_gb = self.auto_detected_info.get("memory_total_gb", 8)
        if memory_gb < 8:  # Low memory system
            optimized.device.max_memory_fraction = 0.6
            optimized.performance.memory_pool_size = 256
            optimized.performance.num_workers = min(2, optimized.performance.num_workers)
        
        elif memory_gb > 32:  # High memory system
            optimized.device.max_memory_fraction = 0.9
            optimized.performance.memory_pool_size = max(2048, int(memory_gb * 128))
        
        # GPU optimization
        if self.auto_detected_info.get("cuda_available", False):
            gpu_count = self.auto_detected_info.get("gpu_count", 1)
            if gpu_count > 1:
                optimized.device.multi_gpu = True
                optimized.device.device_ids = list(range(gpu_count))
        
        # CPU optimization
        cpu_count = self.auto_detected_info.get("cpu_count", 4)
        optimized.performance.num_workers = min(cpu_count, 8)
        optimized.runtime.thread_pool_size = min(cpu_count, 8)
        
        self.logger.info("Configuration optimized for system capabilities")
        return optimized
    
    def get_recommended_batch_size(self, model_size_mb: float) -> int:
        """Calculate recommended batch size based on system memory and model size."""
        if not self.auto_detected_info:
            return 1
        
        memory_gb = self.auto_detected_info.get("memory_total_gb", 8)
        available_memory_gb = self.auto_detected_info.get("memory_available_gb", 4)
        
        # Estimate memory usage (model + activations + overhead)
        estimated_memory_per_batch = model_size_mb * 3  # Conservative estimate
        
        # Leave 20% memory for system and other processes
        max_batch_memory = available_memory_gb * 1024 * 0.8
        
        if estimated_memory_per_batch > 0:
            recommended_batch_size = max(1, int(max_batch_memory / estimated_memory_per_batch))
            return min(recommended_batch_size, 64)  # Cap at 64
        else:
            return 1
    
    def create_config_template(self, file_path: Union[str, Path]):
        """Create a system configuration template."""
        config = SystemConfig()
        template_path = Path(file_path)
        
        with open(template_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        self.logger.info(f"Created system configuration template at {template_path}")
    
    def benchmark_system_performance(self) -> Dict[str, Any]:
        """Benchmark system performance for configuration tuning."""
        results = {}
        
        try:
            # CPU benchmark
            start_time = time.time()
            for i in range(1000000):
                _ = i ** 2
            cpu_benchmark_time = time.time() - start_time
            results["cpu_benchmark_time"] = cpu_benchmark_time
            
            # Memory benchmark
            start_time = time.time()
            test_array = [i for i in range(1000000)]
            memory_benchmark_time = time.time() - start_time
            results["memory_benchmark_time"] = memory_benchmark_time
            
            # GPU benchmark (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    start_time = time.time()
                    x = torch.randn(1000, 1000).cuda()
                    y = torch.randn(1000, 1000).cuda()
                    z = torch.mm(x, y)
                    torch.cuda.synchronize()
                    gpu_benchmark_time = time.time() - start_time
                    results["gpu_benchmark_time"] = gpu_benchmark_time
            except ImportError:
                results["gpu_benchmark_time"] = None
            
            self.logger.info(f"System benchmark completed: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during system benchmarking: {e}")
            return {}


# Convenience functions
def get_system_config(config_dir: Union[str, Path] = "config") -> SystemConfig:
    """Get system configuration."""
    manager = SystemConfigManager(config_dir)
    return manager.load_system_config()


def optimize_system_config(config_dir: Union[str, Path] = "config") -> SystemConfig:
    """Get optimized system configuration."""
    manager = SystemConfigManager(config_dir)
    config = manager.load_system_config()
    optimized = manager.optimize_config_for_system(config)
    manager.save_system_config(optimized)
    return optimized


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create system config manager
    config_dir = Path("config")
    manager = SystemConfigManager(config_dir)
    
    # Load and optimize configuration
    config = manager.load_system_config()
    config = manager.optimize_config_for_system(config)
    
    # Save optimized configuration
    manager.save_system_config(config)
    
    # Benchmark system
    benchmark_results = manager.benchmark_system_performance()
    print(f"System benchmark results: {benchmark_results}")
    
    # Print recommended settings
    print(f"Recommended batch size for 500MB model: {manager.get_recommended_batch_size(500)}")
    
    print(f"System configuration: {config.to_dict()}")