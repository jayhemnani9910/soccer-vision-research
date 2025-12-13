"""
Model Manager for Soccer Player Recognition System

Provides functionality for loading, managing, and using different types of models
including detection, classification, identification, and segmentation models.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from .model_registry import ModelRegistry, ModelType

logger = logging.getLogger(__name__)


class ModelInstance:
    """
    Wrapper class for managing individual model instances.
    
    Features:
    - Model loading and unloading
    - Memory management
    - Performance tracking
    - Thread safety
    """
    
    def __init__(
        self,
        model_id: str,
        model_path: str,
        config: Dict[str, Any],
        model_type: ModelType,
        device: str = "auto"
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.config = config
        self.model_type = model_type
        self.device = self._determine_device(device)
        
        self.model = None
        self.is_loaded = False
        self.load_time = 0
        self.inference_count = 0
        self.total_inference_time = 0
        self._lock = threading.Lock()
        
        logger.info(f"Created ModelInstance for {model_id}")
    
    def _determine_device(self, device: str) -> torch.device:
        """Determine the appropriate device for the model."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        return torch.device(device)
    
    def load_model(self) -> bool:
        """
        Load the model based on its type.
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        with self._lock:
            if self.is_loaded:
                logger.warning(f"Model {self.model_id} is already loaded")
                return True
            
            start_time = time.time()
            
            try:
                self.model = self._load_by_type()
                self.is_loaded = True
                self.load_time = time.time() - start_time
                
                logger.info(f"Model {self.model_id} loaded successfully in {self.load_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_id}: {e}")
                self.model = None
                self.is_loaded = False
                return False
    
    def _load_by_type(self) -> Any:
        """Load model based on its type."""
        model_factory = {
            ModelType.DETECTION: self._load_detection_model,
            ModelType.CLASSIFICATION: self._load_classification_model,
            ModelType.IDENTIFICATION: self._load_identification_model,
            ModelType.SEGMENTATION: self._load_segmentation_model,
            ModelType.POSE: self._load_pose_model,
        }
        
        if self.model_type not in model_factory:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model_factory[self.model_type]()
    
    def _load_detection_model(self) -> Any:
        """Load detection model (e.g., YOLO, Faster R-CNN)."""
        # Example implementation for YOLO
        try:
            import ultralytics
            model = ultralytics.YOLO(self.model_path)
            model.to(self.device)
            return model
        except ImportError:
            # Fallback for other detection frameworks
            logger.warning(f"Ultralytics not available, using generic loader for {self.model_path}")
            return self._load_generic_model()
    
    def _load_classification_model(self) -> Any:
        """Load classification model (e.g., ResNet, EfficientNet)."""
        try:
            model = torch.load(self.model_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load classification model: {e}")
            raise
    
    def _load_identification_model(self) -> Any:
        """Load identification model (e.g., face recognition, person re-identification)."""
        try:
            model = torch.load(self.model_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load identification model: {e}")
            raise
    
    def _load_segmentation_model(self) -> Any:
        """Load segmentation model (e.g., U-Net, Mask R-CNN)."""
        try:
            model = torch.load(self.model_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise
    
    def _load_pose_model(self) -> Any:
        """Load pose estimation model (e.g., OpenPose, MediaPipe)."""
        try:
            import mediapipe
            model = mediapipe.solutions.pose.Pose()
            return model
        except ImportError:
            logger.warning("MediaPipe not available, using generic loader")
            return self._load_generic_model()
    
    def _load_generic_model(self) -> Any:
        """Generic model loader as fallback."""
        try:
            model = torch.load(self.model_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load generic model: {e}")
            raise
    
    def predict(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Perform inference using the loaded model.
        
        Args:
            input_data: Input data for inference
            **kwargs: Additional parameters for inference
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_id} is not loaded")
        
        start_time = time.time()
        
        try:
            result = self._predict_by_type(input_data, **kwargs)
            
            # Update statistics
            inference_time = time.time() - start_time
            with self._lock:
                self.inference_count += 1
                self.total_inference_time += inference_time
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for model {self.model_id}: {e}")
            raise
    
    def _predict_by_type(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Perform inference based on model type."""
        if self.model_type == ModelType.DETECTION:
            return self._predict_detection(input_data, **kwargs)
        elif self.model_type == ModelType.CLASSIFICATION:
            return self._predict_classification(input_data, **kwargs)
        elif self.model_type == ModelType.IDENTIFICATION:
            return self._predict_identification(input_data, **kwargs)
        elif self.model_type == ModelType.SEGMENTATION:
            return self._predict_segmentation(input_data, **kwargs)
        elif self.model_type == ModelType.POSE:
            return self._predict_pose(input_data, **kwargs)
        else:
            raise ValueError(f"Unsupported model type for prediction: {self.model_type}")
    
    def _predict_detection(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Perform detection inference."""
        if hasattr(self.model, 'predict'):
            results = self.model.predict(input_data, **kwargs)
            return {
                'boxes': results.boxes.xyxy if hasattr(results, 'boxes') else [],
                'scores': results.boxes.conf if hasattr(results, 'boxes') else [],
                'classes': results.boxes.cls if hasattr(results, 'boxes') else [],
                'raw_results': results
            }
        else:
            # Generic inference
            with torch.no_grad():
                outputs = self.model(input_data)
                return {'output': outputs}
    
    def _predict_classification(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Perform classification inference."""
        with torch.no_grad():
            # Assume input is a batch of images
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float()
            else:
                input_tensor = input_data
            
            input_tensor = input_tensor.to(self.device)
            outputs = self.model(input_tensor)
            
            if outputs.dim() > 1:
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                
                return {
                    'predicted_classes': predicted_classes.cpu().numpy(),
                    'probabilities': probabilities.cpu().numpy(),
                    'scores': probabilities.max(dim=1)[0].cpu().numpy()
                }
            else:
                return {'output': outputs}
    
    def _predict_identification(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Perform identification inference."""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float()
            else:
                input_tensor = input_data
            
            input_tensor = input_tensor.to(self.device)
            features = self.model(input_tensor)
            
            return {
                'features': features.cpu().numpy(),
                'feature_norm': torch.norm(features, dim=1).cpu().numpy()
            }
    
    def _predict_segmentation(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Perform segmentation inference."""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float()
            else:
                input_tensor = input_data
            
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            
            return {'segmentation_mask': output.cpu().numpy()}
    
    def _predict_pose(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Perform pose estimation inference."""
        if hasattr(self.model, 'process'):
            # MediaPipe model
            results = self.model.process(input_data)
            landmarks = []
            
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
            
            return {
                'landmarks': landmarks,
                'visibility': [landmark.visibility for landmark in results.pose_landmarks.landmark] if results.pose_landmarks else []
            }
        else:
            # Generic pose model
            with torch.no_grad():
                output = self.model(input_data)
                return {'pose_output': output.cpu().numpy()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        with self._lock:
            avg_inference_time = (
                self.total_inference_time / self.inference_count 
                if self.inference_count > 0 else 0
            )
            
            return {
                'model_id': self.model_id,
                'is_loaded': self.is_loaded,
                'load_time': self.load_time,
                'inference_count': self.inference_count,
                'avg_inference_time': avg_inference_time,
                'total_inference_time': self.total_inference_time,
                'device': str(self.device),
                'model_type': self.model_type.value
            }
    
    def unload_model(self) -> bool:
        """
        Unload the model to free memory.
        
        Returns:
            bool: True if unloading successful, False otherwise
        """
        with self._lock:
            if self.is_loaded:
                try:
                    if hasattr(self.model, 'close'):
                        self.model.close()
                    
                    del self.model
                    self.model = None
                    self.is_loaded = False
                    
                    # Force garbage collection
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info(f"Model {self.model_id} unloaded successfully")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to unload model {self.model_id}: {e}")
                    return False
            return True


class ModelManager:
    """
    Central manager for all model instances.
    
    Features:
    - Dynamic model loading/unloading
    - Memory management
    - Load balancing
    - Thread safety
    - Performance monitoring
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        """Initialize the model manager.
        
        Args:
            registry: ModelRegistry instance to use
        """
        self.registry = registry or ModelRegistry()
        self.model_instances: Dict[str, ModelInstance] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Load management
        self.max_loaded_models = 10
        self.load_strategy = "lru"  # lru, fifo, memory_based
        
        # Memory management
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
        # Statistics
        self.global_stats = {
            'total_predictions': 0,
            'total_models_loaded': 0,
            'active_models': 0
        }
        
        logger.info("ModelManager initialized")
    
    def load_model(self, model_id: str, device: str = "auto", force: bool = False) -> bool:
        """
        Load a model by ID.
        
        Args:
            model_id: Unique identifier for the model
            device: Device to load the model on ('auto', 'cpu', 'cuda', 'mps')
            force: Force reload even if already loaded
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        if model_id in self.model_instances and not force:
            logger.info(f"Model {model_id} is already loaded")
            return True
        
        # Get model info from registry
        model_info = self.registry.get_model_info(model_id)
        if not model_info:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        if not model_info["is_active"]:
            logger.error(f"Model {model_id} is not active")
            return False
        
        # Create model instance
        try:
            config = self._load_model_config(model_info["config_path"])
            model_instance = ModelInstance(
                model_id=model_id,
                model_path=model_info["model_path"],
                config=config,
                model_type=ModelType(model_info["model_type"]),
                device=device
            )
            
            # Load the model
            if model_instance.load_model():
                # Add to instances
                if model_id in self.model_instances:
                    # Remove old instance
                    self.model_instances[model_id].unload_model()
                
                self.model_instances[model_id] = model_instance
                self.global_stats['total_models_loaded'] += 1
                self.global_stats['active_models'] = len(self.model_instances)
                
                logger.info(f"Model {model_id} loaded and registered")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to create model instance for {model_id}: {e}")
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model by ID.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            bool: True if unloading successful, False otherwise
        """
        if model_id in self.model_instances:
            success = self.model_instances[model_id].unload_model()
            if success:
                del self.model_instances[model_id]
                self.global_stats['active_models'] = len(self.model_instances)
            return success
        return True
    
    def get_model(self, model_id: str) -> Optional[ModelInstance]:
        """
        Get a loaded model instance.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            ModelInstance if loaded, None otherwise
        """
        return self.model_instances.get(model_id)
    
    def predict(
        self, 
        model_id: str, 
        input_data: Any, 
        auto_load: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform inference using a model.
        
        Args:
            model_id: Unique identifier for the model
            input_data: Input data for inference
            auto_load: Auto-load model if not loaded
            **kwargs: Additional parameters for inference
            
        Returns:
            Dictionary containing prediction results
        """
        model_instance = self.model_instances.get(model_id)
        
        if not model_instance and auto_load:
            logger.info(f"Auto-loading model {model_id}")
            if not self.load_model(model_id):
                raise RuntimeError(f"Failed to auto-load model {model_id}")
            model_instance = self.model_instances[model_id]
        
        if not model_instance:
            raise ValueError(f"Model {model_id} is not loaded and auto_load is disabled")
        
        # Perform prediction
        result = model_instance.predict(input_data, **kwargs)
        self.global_stats['total_predictions'] += 1
        
        return result
    
    def predict_batch(
        self, 
        model_id: str, 
        input_data_batch: List[Any],
        max_workers: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform batch inference using a model.
        
        Args:
            model_id: Unique identifier for the model
            input_data_batch: List of input data for inference
            max_workers: Maximum number of worker threads
            **kwargs: Additional parameters for inference
            
        Returns:
            List of prediction results
        """
        model_instance = self.model_instances.get(model_id)
        
        if not model_instance and auto_load:
            if not self.load_model(model_id):
                raise RuntimeError(f"Failed to auto-load model {model_id}")
            model_instance = self.model_instances[model_id]
        
        if not model_instance:
            raise ValueError(f"Model {model_id} is not loaded")
        
        # Use thread pool for batch processing
        def predict_single(data):
            return model_instance.predict(data, **kwargs)
        
        results = list(self.thread_pool.map(predict_single, input_data_batch))
        self.global_stats['total_predictions'] += len(input_data_batch)
        
        return results
    
    def list_loaded_models(self) -> List[str]:
        """Get list of loaded model IDs."""
        return list(self.model_instances.keys())
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all loaded models."""
        stats = {
            'global': self.global_stats.copy(),
            'models': {}
        }
        
        for model_id, instance in self.model_instances.items():
            stats['models'][model_id] = instance.get_statistics()
        
        return stats
    
    def get_model_statistics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific model."""
        instance = self.model_instances.get(model_id)
        return instance.get_statistics() if instance else None
    
    def cleanup_memory(self) -> int:
        """
        Cleanup memory by unloading least recently used models.
        
        Returns:
            int: Number of models unloaded
        """
        unloaded_count = 0
        
        if len(self.model_instances) > self.max_loaded_models:
            # Sort models by last inference time (simple LRU)
            model_times = []
            for model_id, instance in self.model_instances.items():
                last_inference_time = instance.total_inference_time / max(instance.inference_count, 1)
                model_times.append((model_id, last_inference_time))
            
            # Sort by last inference time (ascending - least recently used first)
            model_times.sort(key=lambda x: x[1])
            
            # Unload models until under threshold
            for model_id, _ in model_times:
                if len(self.model_instances) <= self.max_loaded_models:
                    break
                
                if self.unload_model(model_id):
                    unloaded_count += 1
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return unloaded_count
    
    def _load_model_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from file."""
        import json
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}


# Global model manager instance
model_manager = ModelManager()