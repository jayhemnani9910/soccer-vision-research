"""
Model Pipeline - Orchestration Layer

This module provides the ModelPipeline class that orchestrates execution across
all four models (RF-DETR, SAM2, SigLIP, ResNet) with support for:
- Sequential and parallel execution
- Data preprocessing and postprocessing
- Memory management
- Error handling and fallback strategies
- Performance monitoring
- Batch processing optimization

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import time
from pathlib import Path
import cv2
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from .results import DetectionResult, IdentificationResult, SegmentationResult

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for model pipeline."""
    SEQUENTIAL = "sequential"  # Run models one after another
    PARALLEL = "parallel"     # Run models simultaneously
    ADAPTIVE = "adaptive"     # Choose mode based on available resources


class PipelineStage(Enum):
    """Stages in the processing pipeline."""
    PREPROCESSING = "preprocessing"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    IDENTIFICATION = "identification"
    CLASSIFICATION = "classification"
    POSTPROCESSING = "postprocessing"
    FUSION = "fusion"


@dataclass
class PipelineConfig:
    """Configuration for model pipeline execution."""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_workers: int = 4
    batch_size: int = 8
    enable_preprocessing_cache: bool = True
    enable_gpu_fallback: bool = True
    timeout_seconds: int = 60
    retry_count: int = 2
    memory_limit_gb: float = 8.0


class ModelPipeline:
    """
    Orchestrates execution across multiple models with intelligent scheduling,
    memory management, and performance optimization.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, device: str = "auto"):
        """
        Initialize ModelPipeline.
        
        Args:
            config: Pipeline configuration
            device: Device for computation
        """
        self.config = config or PipelineConfig()
        self.device = self._get_device(device)
        
        # Threading and execution
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._lock = threading.Lock()
        
        # Caching for preprocessing
        self._preprocessing_cache = {}
        self._cache_lock = threading.Lock()
        
        # Performance tracking
        self.stage_times = {stage.value: [] for stage in PipelineStage}
        self.total_executions = 0
        
        # Memory management
        self.memory_threshold = self.config.memory_limit_gb * 1024**3  # Convert to bytes
        
        logger.info(f"ModelPipeline initialized with {self.config.execution_mode.value} mode")
    
    def _get_device(self, device: str) -> torch.device:
        """Get computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def run_detection(
        self,
        detection_model,
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.5,
        max_detections: int = 50,
        **kwargs
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Execute detection on images using the provided model.
        
        Args:
            detection_model: The detection model to use
            images: Input images
            confidence_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
            max_detections: Maximum number of detections
            **kwargs: Additional model-specific parameters
            
        Returns:
            Detection results
        """
        start_time = time.time()
        
        try:
            # Preprocess input
            processed_images = self._preprocess_images(images)
            
            # Execute detection
            if hasattr(detection_model, 'predict'):
                # RF-DETR style interface
                results = detection_model.predict(
                    processed_images,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                    max_detections=max_detections,
                    **kwargs
                )
            else:
                # Generic model interface
                with torch.no_grad():
                    results = self._run_generic_detection(
                        detection_model, processed_images
                    )
            
            # Postprocess results
            final_results = self._postprocess_detection_results(
                results, images, start_time
            )
            
            # Update timing
            self.stage_times[PipelineStage.DETECTION.value].append(time.time() - start_time)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Detection execution failed: {e}")
            raise
    
    def run_segmentation(
        self,
        segmentation_model,
        images: Union[np.ndarray, List[np.ndarray]],
        prompts: Optional[List[List[Dict]]] = None,
        track_objects: bool = True,
        **kwargs
    ) -> Union[SegmentationResult, List[SegmentationResult]]:
        """
        Execute segmentation on images using the provided model.
        
        Args:
            segmentation_model: The segmentation model to use
            images: Input images as numpy arrays
            prompts: Optional segmentation prompts
            track_objects: Whether to enable object tracking
            **kwargs: Additional model-specific parameters
            
        Returns:
            Segmentation results
        """
        start_time = time.time()
        
        try:
            # Convert to tensors if needed
            tensor_images = self._images_to_tensors(images)
            
            # Execute segmentation
            if hasattr(segmentation_model, 'extract_masks'):
                # SAM2 style interface
                results = self._run_sam2_segmentation(
                    segmentation_model, tensor_images, prompts, track_objects
                )
            else:
                # Generic segmentation interface
                with torch.no_grad():
                    results = self._run_generic_segmentation(
                        segmentation_model, tensor_images
                    )
            
            # Postprocess results
            final_results = self._postprocess_segmentation_results(
                results, images, start_time
            )
            
            # Update timing
            self.stage_times[PipelineStage.SEGMENTATION.value].append(time.time() - start_time)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Segmentation execution failed: {e}")
            raise
    
    def run_identification_siglip(
        self,
        siglip_model,
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        player_candidates: List[str],
        team_context: Optional[str] = None,
        **kwargs
    ) -> Union[IdentificationResult, List[IdentificationResult]]:
        """
        Execute player identification using SigLIP model.
        
        Args:
            siglip_model: The SigLIP model to use
            images: Input images
            player_candidates: List of possible player names
            team_context: Optional team context
            **kwargs: Additional model-specific parameters
            
        Returns:
            Identification results
        """
        start_time = time.time()
        
        try:
            # Preprocess images for SigLIP
            processed_images = self._preprocess_images_for_siglip(images)
            
            # Execute identification
            if hasattr(siglip_model, 'identify_player'):
                # SigLIP interface
                if isinstance(images, list):
                    # Batch processing
                    results = siglip_model.batch_identify_players(
                        processed_images, player_candidates, team_context
                    )
                else:
                    # Single image
                    result = siglip_model.identify_player(
                        processed_images[0] if processed_images else images,
                        player_candidates, team_context
                    )
                    results = [result]
            else:
                # Fallback to generic interface
                results = self._run_generic_identification(
                    siglip_model, processed_images, player_candidates
                )
            
            # Postprocess results
            final_results = self._postprocess_identification_results(
                results, images, start_time
            )
            
            # Update timing
            self.stage_times[PipelineStage.IDENTIFICATION.value].append(time.time() - start_time)
            
            return final_results
            
        except Exception as e:
            logger.error(f"SigLIP identification execution failed: {e}")
            raise
    
    def run_identification_resnet(
        self,
        resnet_model,
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        **kwargs
    ) -> Union[IdentificationResult, List[IdentificationResult]]:
        """
        Execute player identification using ResNet model.
        
        Args:
            resnet_model: The ResNet model to use
            images: Input images
            **kwargs: Additional model-specific parameters
            
        Returns:
            Identification results
        """
        start_time = time.time()
        
        try:
            # Preprocess images for ResNet
            processed_images = self._preprocess_images_for_resnet(images)
            
            # Execute identification
            if hasattr(resnet_model, 'identify_player'):
                # ResNet interface
                if isinstance(images, list):
                    # Batch processing
                    results = []
                    for img in processed_images:
                        if isinstance(img, np.ndarray):
                            # Assume preprocessed numpy array
                            result = resnet_model.identify_player(img)
                        else:
                            result = resnet_model.identify_player(img)
                        results.append(result)
                else:
                    # Single image
                    result = resnet_model.identify_player(processed_images[0])
                    results = [result]
            else:
                # Fallback to generic interface
                results = self._run_generic_identification(
                    resnet_model, processed_images, None
                )
            
            # Postprocess results
            final_results = self._postprocess_identification_results(
                results, images, start_time
            )
            
            # Update timing
            self.stage_times[PipelineStage.IDENTIFICATION.value].append(time.time() - start_time)
            
            return final_results
            
        except Exception as e:
            logger.error(f"ResNet identification execution failed: {e}")
            raise
    
    def run_multi_model_pipeline(
        self,
        models: Dict[str, Any],
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        pipeline_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a complete multi-model pipeline.
        
        Args:
            models: Dictionary of model_name -> model_instance
            images: Input images
            pipeline_config: Configuration for the pipeline
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of results from each model
        """
        start_time = time.time()
        
        try:
            results = {}
            
            # Determine execution mode
            mode = pipeline_config.get('execution_mode', 'sequential')
            
            if mode == 'parallel' and self.config.execution_mode == ExecutionMode.PARALLEL:
                results = self._run_parallel_pipeline(models, images, pipeline_config)
            else:
                results = self._run_sequential_pipeline(models, images, pipeline_config)
            
            # Add metadata
            total_time = time.time() - start_time
            results['metadata'] = {
                'total_time': total_time,
                'execution_mode': mode,
                'num_models': len(models),
                'device': str(self.device)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-model pipeline execution failed: {e}")
            raise
    
    def _run_sequential_pipeline(
        self,
        models: Dict[str, Any],
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run pipeline models sequentially."""
        results = {}
        
        # Detection first
        if 'detection' in models and 'rf_detr' in models:
            logger.info("Running sequential detection...")
            results['detection'] = self.run_detection(
                models['rf_detr'], images, **config.get('detection', {})
            )
        
        # Segmentation
        if 'segmentation' in models and 'sam2' in models:
            logger.info("Running sequential segmentation...")
            results['segmentation'] = self.run_segmentation(
                models['sam2'], images, **config.get('segmentation', {})
            )
        
        # Identification
        if 'identification' in models:
            if 'siglip' in models and 'player_candidates' in config:
                logger.info("Running sequential SigLIP identification...")
                results['identification'] = self.run_identification_siglip(
                    models['siglip'], images, config['player_candidates'],
                    config.get('team_context')
                )
            elif 'resnet' in models:
                logger.info("Running sequential ResNet identification...")
                results['identification'] = self.run_identification_resnet(
                    models['resnet'], images
                )
        
        return results
    
    def _run_parallel_pipeline(
        self,
        models: Dict[str, Any],
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run pipeline models in parallel."""
        results = {}
        futures = {}
        
        # Submit parallel tasks
        if 'detection' in models and 'rf_detr' in models:
            futures['detection'] = self.thread_pool.submit(
                self.run_detection, models['rf_detr'], images,
                **config.get('detection', {})
            )
        
        if 'segmentation' in models and 'sam2' in models:
            futures['segmentation'] = self.thread_pool.submit(
                self.run_segmentation, models['sam2'], images,
                **config.get('segmentation', {})
            )
        
        # Wait for results
        for task_name, future in futures.items():
            try:
                results[task_name] = future.result(timeout=self.config.timeout_seconds)
            except Exception as e:
                logger.error(f"Parallel task {task_name} failed: {e}")
                results[task_name] = None
        
        return results
    
    def _preprocess_images(self, images: Union[str, np.ndarray, List[str], List[np.ndarray]]) -> List[np.ndarray]:
        """Preprocess images for model input."""
        if isinstance(images, (str, np.ndarray)):
            images = [images]
        
        processed_images = []
        for img in images:
            if isinstance(img, str):
                # Load image from path
                image = cv2.imread(img)
                if image is None:
                    raise ValueError(f"Could not load image from path: {img}")
            elif isinstance(img, np.ndarray):
                image = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            processed_images.append(image)
        
        return processed_images
    
    def _preprocess_images_for_siglip(self, images: Union[str, np.ndarray, List[str], List[np.ndarray]]) -> List:
        """Preprocess images specifically for SigLIP model."""
        processed = []
        
        for img in images:
            if isinstance(img, str):
                # SigLIP expects PIL images
                processed.append(Image.open(img).convert('RGB'))
            elif isinstance(img, np.ndarray):
                # Convert numpy array to PIL Image
                if len(img.shape) == 3 and img.shape[2] == 3:
                    processed.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                else:
                    processed.append(Image.fromarray(img))
            else:
                processed.append(img)
        
        return processed
    
    def _preprocess_images_for_resnet(self, images: Union[str, np.ndarray, List[str], List[np.ndarray]]) -> List[np.ndarray]:
        """Preprocess images specifically for ResNet model."""
        return self._preprocess_images(images)
    
    def _images_to_tensors(self, images: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """Convert images to tensors for model input."""
        if isinstance(images, list):
            images = np.stack(images)
        
        # Convert BGR to RGB
        if len(images.shape) == 4 and images.shape[3] == 3:
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        images = images.astype(np.float32) / 255.0
        tensor_images = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        return tensor_images.to(self.device)
    
    def _run_generic_detection(self, model, images: List[np.ndarray]) -> Dict[str, Any]:
        """Run detection with generic model interface."""
        results = []
        
        for img in images:
            # Convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                output = model(img_tensor)
            
            # Process output (simplified)
            results.append({
                'detections': [],
                'raw_output': output
            })
        
        return {'batch_results': results} if len(results) > 1 else results[0]
    
    def _run_sam2_segmentation(
        self,
        model,
        tensor_images: torch.Tensor,
        prompts: Optional[List[List[Dict]]],
        track_objects: bool
    ) -> Dict[str, Any]:
        """Run SAM2-style segmentation."""
        results = []
        
        batch_size = tensor_images.shape[0]
        
        for i in range(batch_size):
            img = tensor_images[i:i+1]  # Keep batch dimension
            
            # Prepare prompts if provided
            img_prompts = prompts[i] if prompts and i < len(prompts) else None
            
            # Run segmentation
            masks = model.extract_masks(img, img_prompts, track_objects)
            
            results.append({
                'masks': {k: v.cpu().numpy() for k, v in masks.items()},
                'frame_id': i
            })
        
        return {'batch_results': results} if len(results) > 1 else results[0]
    
    def _run_generic_segmentation(self, model, tensor_images: torch.Tensor) -> Dict[str, Any]:
        """Run segmentation with generic model interface."""
        with torch.no_grad():
            outputs = model(tensor_images)
        
        return {'segmentation_masks': outputs}
    
    def _run_generic_identification(self, model, images: List, player_candidates: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Run identification with generic model interface."""
        results = []
        
        for img in images:
            try:
                if hasattr(model, 'forward'):
                    # PyTorch model
                    if isinstance(img, np.ndarray):
                        img_tensor = torch.from_numpy(img).float()
                        if len(img_tensor.shape) == 3:
                            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                        img_tensor = img_tensor.to(self.device)
                    
                    with torch.no_grad():
                        output = model(img_tensor)
                    
                    # Simple result processing
                    result = {
                        'predictions': ['unknown'],
                        'confidence': 0.5,
                        'raw_output': output.cpu().numpy() if hasattr(output, 'cpu') else output
                    }
                else:
                    # Generic model
                    result = {
                        'predictions': ['unknown'],
                        'confidence': 0.5,
                        'raw_output': None
                    }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Generic identification failed for image: {e}")
                results.append({
                    'predictions': ['unknown'],
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def _postprocess_detection_results(self, raw_results: Any, original_images: Any, start_time: float) -> Union[DetectionResult, List[DetectionResult]]:
        """Postprocess detection results."""
        # Simplified postprocessing - in real implementation would be more comprehensive
        execution_time = time.time() - start_time
        
        if isinstance(original_images, list):
            results = []
            for i, result in enumerate(raw_results.get('batch_results', [raw_results])):
                detection_result = DetectionResult(
                    image_path=None,  # Would be set from original image info
                    detections=result.get('detections', []),
                    execution_time=execution_time,
                    model_name='rf_detr'
                )
                results.append(detection_result)
            return results
        else:
            return DetectionResult(
                image_path=None,
                detections=raw_results.get('detections', []),
                execution_time=execution_time,
                model_name='rf_detr'
            )
    
    def _postprocess_segmentation_results(self, raw_results: Any, original_images: Any, start_time: float) -> Union[SegmentationResult, List[SegmentationResult]]:
        """Postprocess segmentation results."""
        execution_time = time.time() - start_time
        
        if isinstance(original_images, list):
            results = []
            for i, result in enumerate(raw_results.get('batch_results', [raw_results])):
                segmentation_result = SegmentationResult(
                    frame_id=i,
                    masks=result.get('masks', {}),
                    execution_time=execution_time,
                    model_name='sam2'
                )
                results.append(segmentation_result)
            return results
        else:
            return SegmentationResult(
                frame_id=0,
                masks=raw_results.get('masks', {}),
                execution_time=execution_time,
                model_name='sam2'
            )
    
    def _postprocess_identification_results(self, raw_results: List[Dict[str, Any]], original_images: Any, start_time: float) -> Union[IdentificationResult, List[IdentificationResult]]:
        """Postprocess identification results."""
        execution_time = time.time() - start_time
        
        if isinstance(original_images, list):
            results = []
            for i, result in enumerate(raw_results):
                identification_result = IdentificationResult(
                    image_index=i,
                    player_name=result.get('predictions', ['unknown'])[0],
                    confidence=result.get('confidence', 0.0),
                    predictions=result.get('predictions', []),
                    execution_time=execution_time,
                    model_name='siglip_resnet'
                )
                results.append(identification_result)
            return results
        else:
            return IdentificationResult(
                image_index=0,
                player_name=raw_results[0].get('predictions', ['unknown'])[0] if raw_results else 'unknown',
                confidence=raw_results[0].get('confidence', 0.0) if raw_results else 0.0,
                predictions=raw_results[0].get('predictions', []) if raw_results else [],
                execution_time=execution_time,
                model_name='siglip_resnet'
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        stats = {}
        
        for stage, times in self.stage_times.items():
            if times:
                stats[stage] = {
                    'mean_time': np.mean(times),
                    'total_calls': len(times),
                    'total_time': np.sum(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }
            else:
                stats[stage] = {
                    'mean_time': 0.0,
                    'total_calls': 0,
                    'total_time': 0.0,
                    'min_time': 0.0,
                    'max_time': 0.0
                }
        
        stats['total_executions'] = self.total_executions
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        self.thread_pool.shutdown(wait=True)
        with self._cache_lock:
            self._preprocessing_cache.clear()
        
        logger.info("ModelPipeline cleanup completed")


if __name__ == "__main__":
    # Example usage
    pipeline = ModelPipeline(device="auto")
    
    print("ModelPipeline Performance Stats:")
    print(pipeline.get_performance_stats())
    
    pipeline.cleanup()
    print("ModelPipeline test completed")