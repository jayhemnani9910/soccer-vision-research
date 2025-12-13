"""
RF-DETR Model Implementation for Soccer Player Detection

This module implements the RF-DETR model specifically designed for soccer player detection
including players, goalkeepers, referees, and balls.
"""

import torch
import torch.nn as nn
import torchvision
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import logging
from pathlib import Path

# Import path fix for relative imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from models.detection.rf_detr_config import RFDETRConfig, RFDETRSoccerConfigs
    from utils.rf_detr_utils import RFDETRPreprocessor, RFDETRPostprocessor
except ImportError:
    # Fallback for when running from different directory
    from soccer_player_recognition.models.detection.rf_detr_config import RFDETRConfig, RFDETRSoccerConfigs
    from soccer_player_recognition.utils.rf_detr_utils import RFDETRPreprocessor, RFDETRPostprocessor


class RFDETRBackbone(nn.Module):
    """ResNet backbone for RF-DETR feature extraction."""
    
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True):
        """
        Initialize backbone network.
        
        Args:
            model_name: Backbone model name
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained ResNet
        if model_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=pretrained)
        elif model_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone model: {model_name}")
        
        # Remove avgpool and fc layers to keep only features
        self.backbone_features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Feature pyramid network components
        self.num_features = backbone.fc.in_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        features = self.backbone_features(x)
        return features


class RFDETRTransformer(nn.Module):
    """Transformer encoder-decoder for RF-DETR."""
    
    def __init__(self, 
                 d_model: int = 256, 
                 nhead: int = 8, 
                 num_feature_levels: int = 3,
                 dec_n_points: int = 4, 
                 enc_n_points: int = 4):
        """
        Initialize transformer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_feature_levels: Number of feature levels
            dec_n_points: Number of decoder points
            enc_n_points: Number of encoder points
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Learnable object queries
        self.object_queries = nn.Parameter(torch.randn(100, d_model))
        
        # Output projections
        self.class_proj = nn.Linear(d_model, 4)  # 4 soccer classes
        self.bbox_proj = nn.Linear(d_model, 4)   # 4 coordinates
        
        # Positional encodings
        self.pos_encodings = nn.Parameter(torch.randn(num_feature_levels, d_model))
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer.
        
        Args:
            features: Backbone features
            
        Returns:
            Tuple of (class_logits, bbox_predictions)
        """
        batch_size = features.shape[0]
        
        # Flatten features
        flattened_features = features.view(features.size(0), self.d_model, -1).transpose(1, 2)
        
        # Add positional encodings
        pos_enc = self.pos_encodings.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Encoder pass
        encoded_features = self.encoder(flattened_features + pos_enc)
        
        # Object queries
        queries = self.object_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Decoder pass
        decoded_features = self.decoder(queries, encoded_features)
        
        # Project to class and bbox predictions
        class_logits = self.class_proj(decoded_features)
        bbox_pred = self.bbox_proj(decoded_features).sigmoid()
        
        return class_logits, bbox_pred


class RFDETRModel(nn.Module):
    """RF-DETR model for soccer player detection."""
    
    def __init__(self, config: RFDETRConfig):
        """
        Initialize RF-DETR model.
        
        Args:
            config: RF-DETR configuration object
        """
        super().__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        self.device = torch.device(config.model_device)
        
        # Build model components
        self.backbone = RFDETRBackbone(
            model_name=config.backbone_model_name, 
            pretrained=True
        )
        
        self.transformer = RFDETRTransformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_feature_levels=config.num_feature_levels,
            dec_n_points=config.dec_n_points,
            enc_n_points=config.enc_n_points
        )
        
        # Feature dimension adapter
        self.feature_adapter = nn.Conv2d(
            self.backbone.num_features, 
            config.d_model, 
            kernel_size=1
        )
        
        # Initialize model
        self._init_model()
        
        # Setup preprocessing and postprocessing
        self.preprocessor = RFDETRPreprocessor(config)
        self.postprocessor = RFDETRPostprocessor(config)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RF-DETR model with {self.config.class_names}")
    
    def _init_model(self):
        """Initialize model weights."""
        # Initialize transformer components
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize bias for class predictions
        nn.init.constant_(self.transformer.class_proj.bias, -2.0)
        nn.init.constant_(self.transformer.bbox_proj.bias, 0.0)
    
    def load_pretrained_weights(self, model_path: Optional[str] = None):
        """
        Load pretrained weights.
        
        Args:
            model_path: Path to pretrained weights file
        """
        if model_path is None:
            model_path = self.config.pretrained_model_path
        
        model_path = Path(model_path)
        
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['state_dict'])
                else:
                    self.load_state_dict(checkpoint)
                
                self.logger.info(f"Loaded pretrained weights from {model_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load pretrained weights from {model_path}: {e}")
                self.logger.info("Continuing with random initialization")
        else:
            self.logger.warning(f"Pretrained weights not found at {model_path}")
            self.logger.info("Using randomly initialized model")
    
    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        return super().to(device)
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model.
        
        Args:
            images: Input images tensor [batch_size, C, H, W]
            
        Returns:
            Dictionary containing predictions and intermediate outputs
        """
        # Extract features from backbone
        features = self.backbone(images)  # [B, C, H, W]
        
        # Adapt feature dimensions
        adapted_features = self.feature_adapter(features)  # [B, d_model, H, W]
        
        # Transformer for object detection
        class_logits, bbox_pred = self.transformer(adapted_features)
        
        return {
            'class_logits': class_logits,  # [B, num_queries, num_classes]
            'bbox_pred': bbox_pred,        # [B, num_queries, 4]
            'features': features
        }
    
    def predict(self, 
                images: Union[np.ndarray, torch.Tensor, List[str]], 
                confidence_threshold: Optional[float] = None,
                nms_threshold: Optional[float] = None,
                max_detections: Optional[int] = None,
                return_confidence: bool = True) -> Dict:
        """
        Perform detection on input images.
        
        Args:
            images: Input images (numpy array, tensor, or file paths)
            confidence_threshold: Minimum confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
            max_detections: Maximum number of detections to return
            return_confidence: Whether to return confidence scores
            
        Returns:
            Detection results dictionary
        """
        # Update thresholds if specified
        if confidence_threshold is not None:
            self.config.score_threshold = confidence_threshold
        if nms_threshold is not None:
            self.config.nms_threshold = nms_threshold
        if max_detections is not None:
            self.config.max_detections = max_detections
        
        # Handle different input types
        if isinstance(images, list) and isinstance(images[0], str):
            # Input is list of file paths
            batch_tensor, image_info_list, original_images = self.preprocessor.preprocess_batch_from_paths(images)
        elif isinstance(images, np.ndarray):
            # Input is numpy array
            if len(images.shape) == 3:
                images = images[np.newaxis, ...]  # Add batch dimension
            
            batch_tensor = self.preprocessor.preprocess_batch(images.tolist())
            image_info_list = [self.preprocessor.get_image_info(img) for img in images]
            original_images = images.tolist()
        elif isinstance(images, torch.Tensor):
            batch_tensor = images.to(self.device)
            image_info_list = [{} for _ in range(batch_tensor.shape[0])]
            original_images = []
        else:
            raise ValueError("Invalid input type. Expected numpy array, tensor, or list of file paths")
        
        # Move to device
        batch_tensor = batch_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            predictions = self.forward(batch_tensor)
        
        # Process predictions
        results = []
        batch_size = batch_tensor.shape[0]
        
        for i in range(batch_size):
            # Extract predictions for current image
            class_logits_i = predictions['class_logits'][i]  # [num_queries, num_classes]
            bbox_pred_i = predictions['bbox_pred'][i]        # [num_queries, 4]
            
            # Convert to detection format
            detection_results = self._convert_predictions_to_detections(
                class_logits_i, 
                bbox_pred_i, 
                image_info_list[i]
            )
            
            results.append(detection_results)
        
        # Format final output
        if batch_size == 1:
            return results[0]
        else:
            return {
                'batch_results': results,
                'batch_size': batch_size
            }
    
    def _convert_predictions_to_detections(self, 
                                         class_logits: torch.Tensor,
                                         bbox_pred: torch.Tensor,
                                         image_info: Dict) -> Dict:
        """
        Convert model predictions to detection results.
        
        Args:
            class_logits: Class prediction logits
            bbox_pred: Bounding box predictions
            image_info: Image information for coordinate scaling
            
        Returns:
            Detection results dictionary
        """
        # Get class predictions
        class_probs = torch.softmax(class_logits, dim=-1)
        scores, class_ids = torch.max(class_probs, dim=-1)
        
        # Get bounding boxes (normalized coordinates)
        bboxes = bbox_pred  # [num_queries, 4]
        
        # Filter by confidence threshold
        valid_indices = scores > self.config.score_threshold
        
        if not valid_indices.any():
            return {
                'detections': [],
                'total_detections': 0,
                'classes_detected': {},
                'max_confidence': 0.0,
                'min_confidence': 1.0
            }
        
        # Apply NMS
        valid_bboxes = bboxes[valid_indices]
        valid_scores = scores[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        
        # Simple NMS implementation
        keep_indices = self._simple_nms(valid_bboxes, valid_scores, valid_class_ids)
        
        # Process final detections
        detections = []
        if len(keep_indices) > 0:
            final_bboxes = valid_bboxes[keep_indices]
            final_scores = valid_scores[keep_indices]
            final_class_ids = valid_class_ids[keep_indices]
            
            for bbox, score, class_id in zip(final_bboxes, final_scores, final_class_ids):
                # Convert to original image coordinates
                scaled_bbox = self._scale_bbox_to_original(bbox, image_info)
                
                detection = {
                    'bbox': scaled_bbox,
                    'confidence': score.item(),
                    'class_id': class_id.item(),
                    'class_name': self.config.class_names[class_id.item()]
                }
                detections.append(detection)
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Limit max detections
        if len(detections) > self.config.max_detections:
            detections = detections[:self.config.max_detections]
        
        # Generate summary
        classes_detected = {}
        max_confidence = 0.0
        min_confidence = 1.0
        
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in classes_detected:
                classes_detected[class_name] = 0
            classes_detected[class_name] += 1
            
            max_confidence = max(max_confidence, detection['confidence'])
            min_confidence = min(min_confidence, detection['confidence'])
        
        if not detections:
            min_confidence = 0.0
        
        return {
            'detections': detections,
            'total_detections': len(detections),
            'classes_detected': classes_detected,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence
        }
    
    def _simple_nms(self, boxes: torch.Tensor, scores: torch.Tensor, class_ids: torch.Tensor, 
                   threshold: float = None) -> torch.Tensor:
        """
        Simple Non-Maximum Suppression implementation.
        
        Args:
            boxes: Bounding boxes
            scores: Confidence scores
            class_ids: Class IDs
            threshold: IoU threshold
            
        Returns:
            Indices of kept detections
        """
        if threshold is None:
            threshold = self.config.nms_threshold
        
        keep_indices = []
        used_mask = torch.zeros(len(boxes), dtype=torch.bool)
        
        # Sort by scores
        _, sorted_indices = torch.sort(scores, descending=True)
        
        for i in sorted_indices:
            if used_mask[i]:
                continue
            
            # Keep current detection
            keep_indices.append(i)
            
            if len(keep_indices) >= self.config.max_detections:
                break
            
            # Mark overlapping detections
            for j in sorted_indices:
                if i != j and not used_mask[j]:
                    # Calculate IoU (simplified)
                    iou = self._calculate_simple_iou(boxes[i], boxes[j])
                    if iou > threshold:
                        used_mask[j] = True
        
        return torch.tensor(keep_indices, dtype=torch.long)
    
    def _calculate_simple_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate simple IoU between two boxes."""
        # Convert to corner format and calculate intersection
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return float(intersection / max(union, 1e-6))
    
    def _scale_bbox_to_original(self, bbox: torch.Tensor, image_info: Dict) -> List[float]:
        """Scale bounding box to original image coordinates."""
        if not image_info:
            return bbox.tolist()
        
        original_width = image_info.get('original_width', 640)
        original_height = image_info.get('original_height', 640)
        
        # Convert from normalized coordinates to original
        x1 = bbox[0] * original_width
        y1 = bbox[1] * original_height
        x2 = bbox[2] * original_width
        y2 = bbox[3] * original_height
        
        return [x1.item(), y1.item(), x2.item(), y2.item()]
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'RF-DETR',
            'config': {
                'backbone': self.config.backbone_model_name,
                'd_model': self.config.d_model,
                'num_heads': self.config.nhead,
                'num_classes': self.config.num_classes,
                'input_size': self.config.input_size,
                'device': str(self.device)
            },
            'class_names': self.config.class_names,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# Factory functions for easy model creation
def create_rf_detr_model(config_type: str = "balanced", custom_config: Optional[RFDETRConfig] = None) -> RFDETRModel:
    """
    Create RF-DETR model with specified configuration.
    
    Args:
        config_type: Configuration type ('balanced', 'real_time', 'high_accuracy', 'training')
        custom_config: Custom configuration object
        
    Returns:
        Initialized RF-DETR model
    """
    if custom_config is not None:
        config = custom_config
    elif config_type == "balanced":
        config = RFDETRSoccerConfigs.BALANCED
    elif config_type == "real_time":
        config = RFDETRSoccerConfigs.REAL_TIME
    elif config_type == "high_accuracy":
        config = RFDETRSoccerConfigs.HIGH_ACCURACY
    elif config_type == "training":
        config = RFDETRSoccerConfigs.TRAINING
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    model = RFDETRModel(config)
    return model


def load_rf_detr_model(model_path: str, config_type: str = "balanced", 
                      config: Optional[RFDETRConfig] = None) -> RFDETRModel:
    """
    Load RF-DETR model from weights file.
    
    Args:
        model_path: Path to model weights
        config_type: Configuration type
        config: Custom configuration
        
    Returns:
        Loaded RF-DETR model
    """
    model = create_rf_detr_model(config_type, config)
    model.load_pretrained_weights(model_path)
    return model