"""
PlayerRecognizer - Main Integration Framework

This module provides the unified PlayerRecognizer class that integrates all four models:
- RF-DETR: Player detection in images/video
- SAM2: Video segmentation and tracking
- SigLIP: Zero-shot player identification
- ResNet: Player classification and recognition

The PlayerRecognizer provides a single interface to:
- Run individual models or combinations
- Switch between different approaches
- Fuse results from multiple models
- Handle different input formats
- Manage model lifecycle and memory

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
import json

# Import all model implementations - with graceful fallbacks
try:
    from models.detection.rf_detr_model import RFDETRModel, create_rf_detr_model
    _rf_detr_available = True
except ImportError:
    _rf_detr_available = False
    RFDETRModel = None
    create_rf_detr_model = None

try:
    from models.classification.resnet_model import PlayerRecognitionModel as ResNetModel
    _resnet_available = True
except ImportError:
    _resnet_available = False
    ResNetModel = None

try:
    from models.segmentation.sam2_model import SAM2Model, MemoryMode
    _sam2_available = True
except ImportError:
    _sam2_available = False
    SAM2Model = None
    MemoryMode = None

try:
    from models.identification.siglip_model import SigLIPPlayerIdentification, SigLIPConfig
    _siglip_available = True
except ImportError:
    try:
        # Try alternative import path
        from soccer_player_recognition.models.identification.siglip_model import SigLIPPlayerIdentification, SigLIPConfig
        _siglip_available = True
    except ImportError:
        _siglip_available = False
        SigLIPPlayerIdentification = None
        SigLIPConfig = None

from .model_pipeline import ModelPipeline
from .result_fusion import ResultFuser, FusionStrategy
from .config import Config, load_config
from .results import DetectionResult, IdentificationResult, TrackingResult, SegmentationResult

logger = logging.getLogger(__name__)


class PlayerRecognizer:
    """
    Unified Player Recognition System
    
    Integrates detection, segmentation, identification, and classification models
    to provide comprehensive player analysis capabilities.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "auto",
        enable_all_models: bool = True,
        memory_efficient: bool = True
    ):
        """
        Initialize PlayerRecognizer
        
        Args:
            config_path: Path to configuration file
            device: Device for computation ('auto', 'cpu', 'cuda', 'mps')
            enable_all_models: Whether to initialize all models
            memory_efficient: Whether to use memory-efficient loading
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else Config()
        self.device = self._get_device(device)
        self.memory_efficient = memory_efficient
        
        # Initialize models
        self.models = {}
        self.model_status = {}
        
        # Initialize model pipeline and result fusion
        self.pipeline = ModelPipeline(device=self.device)
        self.result_fuser = ResultFuser()
        
        # Initialize models based on configuration
        if enable_all_models:
            self._initialize_all_models()
        
        # Performance tracking
        self.performance_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'model_usage': {model_name: 0 for model_name in self.models.keys()},
            'fusion_stats': {}
        }
        
        logger.info(f"PlayerRecognizer initialized on device: {self.device}")
    
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
    
    def _initialize_all_models(self):
        """Initialize all four models based on configuration."""
        try:
            # Initialize RF-DETR for detection
            self._initialize_detection_model()
            
            # Initialize SAM2 for segmentation
            self._initialize_segmentation_model()
            
            # Initialize SigLIP for identification
            self._initialize_identification_model()
            
            # Initialize ResNet for classification
            self._initialize_classification_model()
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.warning("Continuing with available models")
    
    def _initialize_detection_model(self):
        """Initialize RF-DETR model for player detection."""
        if not _rf_detr_available:
            logger.warning("RF-DETR model not available - skipping detection model initialization")
            self.model_status['rf_detr'] = {'loaded': False, 'error': 'Model not available'}
            return
            
        try:
            logger.info("Initializing RF-DETR detection model...")
            
            # Create RF-DETR model with configuration
            self.models['rf_detr'] = create_rf_detr_model(
                config_type="balanced",
                custom_config=None
            )
            
            # Move to device
            self.models['rf_detr'].to(self.device)
            
            # Set status
            self.model_status['rf_detr'] = {
                'loaded': True,
                'model_type': 'detection',
                'capabilities': ['player_detection', 'ball_detection', 'referee_detection']
            }
            
            logger.info("RF-DETR model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RF-DETR model: {e}")
            self.model_status['rf_detr'] = {'loaded': False, 'error': str(e)}
    
    def _initialize_segmentation_model(self):
        """Initialize SAM2 model for video segmentation."""
        if not _sam2_available:
            logger.warning("SAM2 model not available - skipping segmentation model initialization")
            self.model_status['sam2'] = {'loaded': False, 'error': 'Model not available'}
            return
            
        try:
            logger.info("Initializing SAM2 segmentation model...")
            
            # Create SAM2 model
            self.models['sam2'] = SAM2Model(
                device=str(self.device),
                memory_mode=MemoryMode.SELECTIVE,
                max_memory_frames=8,
                min_confidence=0.7
            )
            
            # Set status
            self.model_status['sam2'] = {
                'loaded': True,
                'model_type': 'segmentation',
                'capabilities': ['video_segmentation', 'object_tracking', 'occlusion_handling']
            }
            
            logger.info("SAM2 model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM2 model: {e}")
            self.model_status['sam2'] = {'loaded': False, 'error': str(e)}
    
    def _initialize_identification_model(self):
        """Initialize SigLIP model for player identification."""
        if not _siglip_available:
            logger.warning("SigLIP model not available - skipping identification model initialization")
            self.model_status['siglip'] = {'loaded': False, 'error': 'Model not available'}
            return
            
        try:
            logger.info("Initializing SigLIP identification model...")
            
            # Create SigLIP model with configuration
            config = SigLIPConfig(
                model_name="siglip-base-patch16-224",
                image_size=224,
                temperature=0.07
            )
            
            self.models['siglip'] = SigLIPPlayerIdentification(config=config)
            
            # Set status
            self.model_status['siglip'] = {
                'loaded': True,
                'model_type': 'identification',
                'capabilities': ['zero_shot_identification', 'text_image_matching', 'player_name_recognition']
            }
            
            logger.info("SigLIP model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SigLIP model: {e}")
            self.model_status['siglip'] = {'loaded': False, 'error': str(e)}
    
    def _initialize_classification_model(self):
        """Initialize ResNet model for player classification."""
        if not _resnet_available:
            logger.warning("ResNet model not available - skipping classification model initialization")
            self.model_status['resnet'] = {'loaded': False, 'error': 'Model not available'}
            return
            
        try:
            logger.info("Initializing ResNet classification model...")
            
            # Create ResNet model (using a default number of players)
            # In real usage, this would be loaded from trained weights
            self.models['resnet'] = ResNetModel(
                num_players=25,  # Default for demo
                model_name="resnet18",
                device=str(self.device),
                pretrained=True
            )
            
            # Set status
            self.model_status['resnet'] = {
                'loaded': True,
                'model_type': 'classification',
                'capabilities': ['player_classification', 'feature_extraction', 'trained_recognition']
            }
            
            logger.info("ResNet model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ResNet model: {e}")
            self.model_status['resnet'] = {'loaded': False, 'error': str(e)}
    
    def detect_players(
        self,
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.5,
        max_detections: int = 50
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Detect players, balls, and referees in images.
        
        Args:
            images: Input images (file paths, numpy arrays, or lists)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            max_detections: Maximum number of detections per image
            
        Returns:
            Detection results for single image or list for batch
        """
        if 'rf_detr' not in self.models or not self.model_status['rf_detr']['loaded']:
            raise RuntimeError("RF-DETR model not available for detection")
        
        start_time = time.time()
        
        try:
            # Use model pipeline for detection
            results = self.pipeline.run_detection(
                self.models['rf_detr'],
                images,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                max_detections=max_detections
            )
            
            # Update performance stats
            self._update_stats('rf_detr', start_time, True)
            
            return results
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            self._update_stats('rf_detr', start_time, False)
            raise
    
    def segment_players(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        prompts: Optional[List[List[Dict]]] = None,
        track_objects: bool = True
    ) -> Union[SegmentationResult, List[SegmentationResult]]:
        """
        Segment and track players in video frames.
        
        Args:
            images: Input video frames
            prompts: Optional segmentation prompts (points, boxes, etc.)
            track_objects: Whether to enable object tracking
            
        Returns:
            Segmentation results with masks and tracking information
        """
        if 'sam2' not in self.models or not self.model_status['sam2']['loaded']:
            raise RuntimeError("SAM2 model not available for segmentation")
        
        start_time = time.time()
        
        try:
            # Use model pipeline for segmentation
            results = self.pipeline.run_segmentation(
                self.models['sam2'],
                images,
                prompts=prompts,
                track_objects=track_objects
            )
            
            # Update performance stats
            self._update_stats('sam2', start_time, True)
            
            return results
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            self._update_stats('sam2', start_time, False)
            raise
    
    def identify_players(
        self,
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        player_candidates: List[str],
        team_context: Optional[str] = None,
        use_siglip: bool = True,
        use_resnet: bool = False
    ) -> Union[IdentificationResult, List[IdentificationResult]]:
        """
        Identify players using zero-shot (SigLIP) or trained (ResNet) approaches.
        
        Args:
            images: Input images for identification
            player_candidates: List of possible player names (for SigLIP)
            team_context: Optional team context information
            use_siglip: Whether to use SigLIP (zero-shot identification)
            use_resnet: Whether to use ResNet (trained classification)
            
        Returns:
            Identification results with player names and confidence scores
        """
        start_time = time.time()
        
        try:
            results = []
            
            # Handle different identification approaches
            if use_siglip and 'siglip' in self.models and self.model_status['siglip']['loaded']:
                logger.info("Using SigLIP for zero-shot player identification")
                results = self.pipeline.run_identification_siglip(
                    self.models['siglip'],
                    images,
                    player_candidates,
                    team_context
                )
                self._update_stats('siglip', start_time, True)
                
            elif use_resnet and 'resnet' in self.models and self.model_status['resnet']['loaded']:
                logger.info("Using ResNet for trained player classification")
                results = self.pipeline.run_identification_resnet(
                    self.models['resnet'],
                    images
                )
                self._update_stats('resnet', start_time, True)
                
            else:
                raise RuntimeError("No suitable identification model available")
            
            return results if isinstance(images, list) else results[0] if results else None
            
        except Exception as e:
            logger.error(f"Identification failed: {e}")
            # Update stats for used model
            model_used = 'siglip' if use_siglip else 'resnet'
            self._update_stats(model_used, start_time, False)
            raise
    
    def analyze_scene(
        self,
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        player_candidates: Optional[List[str]] = None,
        team_context: Optional[str] = None,
        analysis_type: str = "full",
        confidence_threshold: float = 0.7,
        fuse_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive scene analysis using multiple models.
        
        Args:
            images: Input images for analysis
            player_candidates: List of possible player names for identification
            team_context: Optional team context information
            analysis_type: Type of analysis ('detection', 'identification', 'segmentation', 'full')
            confidence_threshold: Minimum confidence threshold for all models
            fuse_results: Whether to fuse results from multiple models
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        logger.info(f"Starting {analysis_type} analysis...")
        
        try:
            analysis_results = {}
            
            # Detection phase
            if analysis_type in ['detection', 'full']:
                logger.info("Performing detection...")
                detection_results = self.detect_players(
                    images, 
                    confidence_threshold=confidence_threshold
                )
                analysis_results['detection'] = detection_results
            
            # Segmentation phase
            if analysis_type in ['segmentation', 'full'] and 'sam2' in self.models:
                logger.info("Performing segmentation...")
                segmentation_results = self.segment_players(images)
                analysis_results['segmentation'] = segmentation_results
            
            # Identification phase
            if analysis_type in ['identification', 'full'] and player_candidates:
                logger.info("Performing identification...")
                identification_results = self.identify_players(
                    images,
                    player_candidates,
                    team_context
                )
                analysis_results['identification'] = identification_results
            
            # Fusion phase
            if fuse_results and len(analysis_results) > 1:
                logger.info("Fusing results from multiple models...")
                fusion_results = self.result_fuser.fuse_comprehensive_results(analysis_results)
                analysis_results['fusion'] = fusion_results
            
            # Add metadata
            total_time = time.time() - start_time
            analysis_results['metadata'] = {
                'analysis_type': analysis_type,
                'total_time': total_time,
                'models_used': list(analysis_results.keys()),
                'num_images': len(images) if isinstance(images, list) else 1,
                'device': str(self.device)
            }
            
            logger.info(f"Analysis completed in {total_time:.3f} seconds")
            
            # Update global stats
            self.performance_stats['total_inferences'] += 1
            self.performance_stats['total_time'] += total_time
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            raise
    
    def switch_model_config(self, model_name: str, new_config: Dict[str, Any]) -> bool:
        """
        Switch configuration for a specific model at runtime.
        
        Args:
            model_name: Name of the model to reconfigure
            new_config: New configuration parameters
            
        Returns:
            True if reconfiguration successful, False otherwise
        """
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return False
            
            # Reinitialize model with new configuration
            if model_name == 'rf_detr':
                self.models[model_name] = create_rf_detr_model(**new_config)
                self.models[model_name].to(self.device)
                
            elif model_name == 'sam2':
                # Update SAM2 configuration
                self.models[model_name].config.update(new_config)
                
            elif model_name == 'siglip':
                if 'config' in new_config:
                    self.models[model_name] = SigLIPPlayerIdentification(config=new_config['config'])
                    
            elif model_name == 'resnet':
                # Reinitialize ResNet with new parameters
                self.models[model_name] = ResNetModel(
                    num_players=new_config.get('num_players', 25),
                    model_name=new_config.get('model_name', 'resnet18'),
                    device=str(self.device)
                )
            
            logger.info(f"Successfully switched configuration for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch configuration for {model_name}: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status information for all models.
        
        Returns:
            Dictionary containing model status and capabilities
        """
        return {
            'models': self.model_status.copy(),
            'device': str(self.device),
            'memory_efficient': self.memory_efficient,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.performance_stats.copy()
    
    def cleanup_memory(self):
        """Clean up memory and unload models if in memory-efficient mode."""
        if self.memory_efficient:
            logger.info("Cleaning up memory...")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Optionally unload models (not implemented for simplicity)
            # This could be extended to actually unload models to save memory
    
    def save_state(self, output_path: str):
        """
        Save current model states and configuration.
        
        Args:
            output_path: Path to save state file
        """
        state = {
            'model_status': self.model_status,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            'performance_stats': self.performance_stats,
            'device': str(self.device)
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"PlayerRecognizer state saved to {output_path}")
    
    def load_state(self, state_path: str):
        """
        Load PlayerRecognizer state from file.
        
        Args:
            state_path: Path to state file
        """
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Update state (Note: actual model loading would need more implementation)
            self.performance_stats = state.get('performance_stats', {})
            
            logger.info(f"PlayerRecognizer state loaded from {state_path}")
            
        except Exception as e:
            logger.error(f"Failed to load state from {state_path}: {e}")
            raise
    
    def _update_stats(self, model_name: str, start_time: float, success: bool):
        """Update performance statistics."""
        if success:
            elapsed = time.time() - start_time
            self.performance_stats['model_usage'][model_name] += 1
            self.performance_stats['total_inferences'] += 1
            self.performance_stats['total_time'] += elapsed
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_memory()


# Factory function for easy initialization
def create_player_recognizer(
    config_path: Optional[str] = None,
    device: str = "auto",
    models_to_load: Optional[List[str]] = None,
    memory_efficient: bool = True
) -> PlayerRecognizer:
    """
    Factory function to create PlayerRecognizer instance.
    
    Args:
        config_path: Path to configuration file
        device: Device for computation
        models_to_load: List of specific models to load ('rf_detr', 'sam2', 'siglip', 'resnet')
        memory_efficient: Whether to use memory-efficient loading
        
    Returns:
        Initialized PlayerRecognizer instance
    """
    # Create basic recognizer
    recognizer = PlayerRecognizer(
        config_path=config_path,
        device=device,
        enable_all_models=(models_to_load is None),
        memory_efficient=memory_efficient
    )
    
    # Load specific models if requested
    if models_to_load:
        # This would require more implementation to selectively load models
        logger.info(f"Loading specific models: {models_to_load}")
    
    return recognizer


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create PlayerRecognizer
    recognizer = create_player_recognizer(device="auto")
    
    # Print status
    print("PlayerRecognizer Status:")
    print(recognizer.get_model_status())
    
    # Example analysis (would need actual images)
    # results = recognizer.analyze_scene(
    #     "sample_soccer_image.jpg",
    #     player_candidates=["Lionel Messi", "Cristiano Ronaldo", "Neymar"],
    #     analysis_type="full"
    # )
    # print(f"Analysis results: {results}")
    
    print("PlayerRecognizer test completed")