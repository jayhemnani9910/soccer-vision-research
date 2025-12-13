"""
Logging utilities for soccer player recognition system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import json
import os


class SoccerPlayerRecognitionLogger:
    """
    Custom logger for soccer player recognition system.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if SoccerPlayerRecognitionLogger._initialized:
            return
        
        self.logger = None
        self.log_file = None
        self.log_level = logging.INFO
        self.format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        SoccerPlayerRecognitionLogger._initialized = True
    
    def setup_logger(self, 
                    name: str = 'soccer_player_recognition',
                    log_level: Union[int, str] = logging.INFO,
                    log_file: Optional[str] = None,
                    console_output: bool = True,
                    format_string: Optional[str] = None) -> logging.Logger:
        """
        Setup logger with specified configuration.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file
            console_output: Whether to output to console
            format_string: Custom format string for log messages
            
        Returns:
            Configured logger instance
        """
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
        
        self.log_level = log_level
        
        if format_string:
            self.format_string = format_string
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.format_string)
        
        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler
        if log_file:
            self.log_file = log_file
            # Create directory if it doesn't exist
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        return self.logger
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if self.logger is None:
            self.setup_logger(name)
        
        if name:
            return logging.getLogger(f"soccer_player_recognition.{name}")
        return self.logger
    
    def log_system_info(self) -> None:
        """Log system information."""
        logger = self.get_logger()
        
        logger.info("=" * 60)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 60)
        
        # Python version
        logger.info(f"Python Version: {sys.version}")
        
        # Platform
        logger.info(f"Platform: {sys.platform}")
        
        # Current working directory
        logger.info(f"Working Directory: {os.getcwd()}")
        
        # Environment variables
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            logger.info(f"CUDA Devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # PyTorch information
        try:
            import torch
            logger.info(f"PyTorch Version: {torch.__version__}")
            logger.info(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA Version: {torch.version.cuda}")
                logger.info(f"Device Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        except ImportError:
            logger.warning("PyTorch not available")
        
        # OpenCV version
        try:
            import cv2
            logger.info(f"OpenCV Version: {cv2.__version__}")
        except ImportError:
            logger.warning("OpenCV not available")
        
        logger.info("=" * 60)
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Log model information.
        
        Args:
            model_info: Dictionary containing model information
        """
        logger = self.get_logger('model')
        
        logger.info("MODEL INFORMATION")
        logger.info("-" * 30)
        for key, value in model_info.items():
            logger.info(f"{key}: {value}")
    
    def log_training_progress(self, epoch: int, total_epochs: int, 
                            metrics: Dict[str, float], 
                            phase: str = 'train') -> None:
        """
        Log training progress.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            metrics: Dictionary of metrics
            phase: Training phase ('train', 'val', 'test')
        """
        logger = self.get_logger('training')
        
        progress = f"{epoch}/{total_epochs}"
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        logger.info(f"[{phase.upper()}] Epoch {progress} | {metrics_str}")
    
    def log_inference_results(self, input_info: Dict[str, Any], 
                            results: Dict[str, Any], 
                            inference_time: float) -> None:
        """
        Log inference results.
        
        Args:
            input_info: Information about input data
            results: Inference results
            inference_time: Time taken for inference
        """
        logger = self.get_logger('inference')
        
        logger.info(f"Inference completed in {inference_time:.3f} seconds")
        
        if 'input_size' in input_info:
            logger.info(f"Input size: {input_info['input_size']}")
        
        if 'batch_size' in input_info:
            logger.info(f"Batch size: {input_info['batch_size']}")
        
        # Log results summary
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value}")
            elif isinstance(value, list) and len(value) <= 10:
                logger.info(f"{key}: {value}")
            else:
                logger.info(f"{key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'unknown'} items")
    
    def log_error(self, error: Exception, context: str = "", 
                 include_traceback: bool = True) -> None:
        """
        Log error with optional traceback.
        
        Args:
            error: Exception object
            context: Additional context information
            include_traceback: Whether to include traceback
        """
        logger = self.get_logger()
        
        error_msg = f"ERROR in {context}: {str(error)}" if context else f"ERROR: {str(error)}"
        
        if include_traceback:
            logger.error(error_msg, exc_info=True)
        else:
            logger.error(error_msg)
    
    def log_performance_metrics(self, metrics: Dict[str, float], 
                              phase: str = 'evaluation') -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            phase: Evaluation phase
        """
        logger = self.get_logger('metrics')
        
        logger.info(f"{phase.upper()} METRICS")
        logger.info("-" * 30)
        
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}" if isinstance(value, float) 
                       else f"{metric_name}: {value}")
    
    def create_checkpoint_log(self, checkpoint_path: str, 
                            checkpoint_info: Dict[str, Any]) -> None:
        """
        Log checkpoint information.
        
        Args:
            checkpoint_path: Path to checkpoint file
            checkpoint_info: Checkpoint information
        """
        logger = self.get_logger('checkpoint')
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        for key, value in checkpoint_info.items():
            logger.info(f"  {key}: {value}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration settings.
        
        Args:
            config: Configuration dictionary
        """
        logger = self.get_logger('config')
        
        logger.info("CONFIGURATION")
        logger.info("-" * 30)
        
        def log_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    logger.info("  " * indent + f"{key}:")
                    log_dict(value, indent + 1)
                else:
                    logger.info("  " * indent + f"{key}: {value}")
        
        log_dict(config)
    
    def save_log_config(self, config: Dict[str, Any], 
                       output_path: Optional[str] = None) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            output_path: Output path for JSON file
        """
        if output_path is None and self.log_file:
            # Use same directory as log file
            log_dir = Path(self.log_file).parent
            output_path = log_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        elif output_path is None:
            output_path = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.get_logger().info(f"Configuration saved to: {output_path}")


# Global logger instance
global_logger = SoccerPlayerRecognitionLogger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return global_logger.get_logger(name)


def setup_logging(**kwargs) -> logging.Logger:
    """
    Setup global logging configuration.
    
    Args:
        **kwargs: Arguments for setup_logger
        
    Returns:
        Logger instance
    """
    return global_logger.setup_logger(**kwargs)


def log_system_info():
    """Log system information."""
    global_logger.log_system_info()


def log_training_progress(epoch: int, total_epochs: int, 
                        metrics: Dict[str, float], 
                        phase: str = 'train'):
    """Log training progress."""
    global_logger.log_training_progress(epoch, total_epochs, metrics, phase)


def log_model_info(model_info: Dict[str, Any]):
    """Log model information."""
    global_logger.log_model_info(model_info)


def log_error(error: Exception, context: str = "", include_traceback: bool = True):
    """Log error."""
    global_logger.log_error(error, context, include_traceback)


def log_performance_metrics(metrics: Dict[str, float], phase: str = 'evaluation'):
    """Log performance metrics."""
    global_logger.log_performance_metrics(metrics, phase)