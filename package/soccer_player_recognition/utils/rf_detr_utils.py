"""
RF-DETR Utility Functions for Soccer Player Detection

This module provides preprocessing and postprocessing utilities for the RF-DETR model
optimized for soccer player detection tasks.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

# Import path fix for relative imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.detection.rf_detr_config import RFDETRConfig
except ImportError:
    # Fallback for when running from different directory
    from soccer_player_recognition.models.detection.rf_detr_config import RFDETRConfig


class RFDETRPreprocessor:
    """Preprocessing utilities for RF-DETR model input."""
    
    def __init__(self, config: RFDETRConfig):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: RFDETR configuration object
        """
        self.config = config
        self.input_size = config.input_size
        
        # Define transform pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image for RF-DETR inference.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            Preprocessed image tensor suitable for model input
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.input_size)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        return image_tensor
    
    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            Batch of preprocessed image tensors
        """
        processed_images = []
        for image in images:
            processed_image = self.preprocess_image(image)
            if len(processed_image.shape) == 4:  # Has batch dimension
                processed_image = processed_image.squeeze(0)  # Remove batch dim
            processed_images.append(processed_image)
        
        # Stack all images into a batch
        batch_tensor = torch.stack(processed_images, dim=0)
        return batch_tensor
    
    def preprocess_batch_from_paths(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[Dict], List[np.ndarray]]:
        """
        Preprocess a batch of images from file paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (batch_tensor, image_info_list, original_images)
        """
        original_images = []
        image_info_list = []
        
        for path in image_paths:
            original_image = cv2.imread(path)
            if original_image is None:
                raise ValueError(f"Could not load image from {path}")
            original_images.append(original_image)
            
            image_info = self.get_image_info(original_image)
            image_info_list.append(image_info)
        
        batch_tensor = self.preprocess_batch(original_images)
        return batch_tensor, image_info_list, original_images
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, float]:
        """
        Get original image information for postprocessing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with original image dimensions and scaling factors
        """
        original_height, original_width = image.shape[:2]
        target_width, target_height = self.input_size
        
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        return {
            'original_width': original_width,
            'original_height': original_height,
            'target_width': target_width,
            'target_height': target_height,
            'scale_x': scale_x,
            'scale_y': scale_y
        }


class RFDETRPostprocessor:
    """Postprocessing utilities for RF-DETR model outputs."""
    
    def __init__(self, config: RFDETRConfig):
        """
        Initialize postprocessor with configuration.
        
        Args:
            config: RFDETR configuration object
        """
        self.config = config
        self.class_names = config.class_names
        self.num_classes = config.num_classes
        self.score_threshold = config.score_threshold
        self.nms_threshold = config.nms_threshold
        self.max_detections = config.max_detections
        self.output_format = config.output_format
    
    def process_predictions(self, 
                          predictions: torch.Tensor, 
                          image_info: Dict[str, float]) -> List[Dict]:
        """
        Process model predictions into detection results.
        
        Args:
            predictions: Model predictions tensor
            image_info: Original image information from preprocessor
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Extract bounding boxes, scores, and class predictions
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]  # confidence scores
        class_ids = predictions[:, 5].long()  # class predictions
        
        # Filter by confidence threshold
        valid_indices = scores > self.score_threshold
        
        if not valid_indices.any():
            return detections
        
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices]
        
        # Apply Non-Maximum Suppression
        keep_indices = self._apply_nms(boxes, scores, class_ids)
        
        # Process final detections
        for idx in keep_indices:
            box = boxes[idx].cpu().numpy()
            score = scores[idx].cpu().item()
            class_id = class_ids[idx].cpu().item()
            
            # Convert coordinates back to original image space
            if self.output_format == 'xyxy':
                box = self._scale_coordinates(box, image_info)
            elif self.output_format == 'xywh':
                box = self._convert_xyxy_to_xywh(box, image_info)
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
            
            detection = {
                'bbox': box,  # [x1, y1, x2, y2] or [x, y, w, h]
                'confidence': score,
                'class_id': class_id,
                'class_name': class_name
            }
            
            detections.append(detection)
        
        # Sort by confidence and limit max detections
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        if len(detections) > self.max_detections:
            detections = detections[:self.max_detections]
        
        return detections
    
    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor, class_ids: torch.Tensor) -> List[int]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            boxes: Bounding boxes tensor
            scores: Confidence scores tensor
            class_ids: Class IDs tensor
            
        Returns:
            List of indices to keep after NMS
        """
        keep_indices = []
        
        # Group by class for class-specific NMS
        for class_id in torch.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            if len(class_boxes) == 0:
                continue
            
            # Apply NMS
            class_keep = self._torch_nms(class_boxes, class_scores)
            keep_indices.extend(class_mask.nonzero()[class_keep].tolist())
        
        return keep_indices
    
    def _torch_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Implement Non-Maximum Suppression using PyTorch.
        
        Args:
            boxes: Bounding boxes tensor (N, 4)
            scores: Confidence scores tensor (N,)
            
        Returns:
            Indices of boxes to keep after NMS
        """
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou(boxes, boxes)
        
        # Sort by confidence
        _, sorted_indices = torch.sort(scores, descending=True)
        
        keep_indices = []
        while len(sorted_indices) > 0:
            # Keep the highest scoring box
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx.item())
            
            if len(sorted_indices) == 1:
                break
            
            # Remove the current box
            sorted_indices = sorted_indices[1:]
            
            # Calculate IoUs with remaining boxes
            current_iou = iou_matrix[current_idx][sorted_indices]
            
            # Remove boxes with high IoU
            keep_mask = current_iou <= self.nms_threshold
            sorted_indices = sorted_indices[keep_mask]
        
        return torch.tensor(keep_indices, dtype=torch.long)
    
    def _calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Intersection over Union (IoU) between two sets of boxes.
        
        Args:
            boxes1: First set of boxes (N, 4)
            boxes2: Second set of boxes (M, 4)
            
        Returns:
            IoU matrix (N, M)
        """
        # Calculate intersection area
        inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union area
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1.unsqueeze(1) + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        return iou
    
    def _scale_coordinates(self, box: np.ndarray, image_info: Dict[str, float]) -> List[float]:
        """
        Scale coordinates back to original image space.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            image_info: Image information from preprocessor
            
        Returns:
            Scaled bounding box coordinates
        """
        scale_x = image_info['original_width'] / image_info['target_width']
        scale_y = image_info['original_height'] / image_info['target_height']
        
        scaled_box = [
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y
        ]
        
        return scaled_box
    
    def _convert_xyxy_to_xywh(self, box: np.ndarray, image_info: Dict[str, float]) -> List[float]:
        """
        Convert xyxy format to xywh format and scale to original image.
        
        Args:
            box: Bounding box in xyxy format
            image_info: Image information from preprocessor
            
        Returns:
            Bounding box in xywh format
        """
        xyxy_scaled = self._scale_coordinates(box, image_info)
        
        x = xyxy_scaled[0]
        y = xyxy_scaled[1]
        w = xyxy_scaled[2] - xyxy_scaled[0]
        h = xyxy_scaled[3] - xyxy_scaled[1]
        
        return [x, y, w, h]
    
    def format_detection_output(self, detections: List[Dict], format_type: str = 'xyxy') -> Dict:
        """
        Format detection results for output.
        
        Args:
            detections: List of detection dictionaries
            format_type: Output format ('xyxy' or 'xywh')
            
        Returns:
            Formatted output dictionary
        """
        output = {
            'detections': [],
            'summary': {
                'total_detections': len(detections),
                'classes_detected': {},
                'max_confidence': 0.0,
                'min_confidence': 1.0
            }
        }
        
        for detection in detections:
            bbox = detection['bbox']
            
            # Convert format if needed
            if format_type == 'xywh' and len(bbox) == 4:
                x, y, w, h = bbox
                if len(detection['bbox']) == 4:  # xyxy format
                    x1, y1, x2, y2 = bbox
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                bbox = [x, y, w, h]
            
            formatted_detection = {
                'bbox': bbox,
                'confidence': detection['confidence'],
                'class_id': detection['class_id'],
                'class_name': detection['class_name']
            }
            
            output['detections'].append(formatted_detection)
            
            # Update summary statistics
            class_name = detection['class_name']
            if class_name not in output['summary']['classes_detected']:
                output['summary']['classes_detected'][class_name] = 0
            output['summary']['classes_detected'][class_name] += 1
            
            output['summary']['max_confidence'] = max(output['summary']['max_confidence'], detection['confidence'])
            output['summary']['min_confidence'] = min(output['summary']['min_confidence'], detection['confidence'])
        
        return output


# Utility functions for common operations
def preprocess_for_soccer_detection(image: np.ndarray, config: RFDETRConfig) -> Tuple[torch.Tensor, Dict]:
    """
    Preprocess image for soccer player detection.
    
    Args:
        image: Input image as numpy array
        config: RF-DETR configuration
        
    Returns:
        Tuple of (preprocessed_tensor, image_info)
    """
    preprocessor = RFDETRPreprocessor(config)
    image_tensor = preprocessor.preprocess_image(image)
    image_info = preprocessor.get_image_info(image)
    return image_tensor, image_info


def postprocess_soccer_detection(predictions: torch.Tensor, image_info: Dict, config: RFDETRConfig) -> Dict:
    """
    Postprocess soccer detection predictions.
    
    Args:
        predictions: Model predictions
        image_info: Image information from preprocessor
        config: RF-DETR configuration
        
    Returns:
        Formatted detection results
    """
    postprocessor = RFDETRPostprocessor(config)
    detections = postprocessor.process_predictions(predictions, image_info)
    return postprocessor.format_detection_output(detections)


def load_and_preprocess_image(image_path: str, config: RFDETRConfig) -> Tuple[torch.Tensor, Dict, np.ndarray]:
    """
    Load and preprocess image from file path.
    
    Args:
        image_path: Path to image file
        config: RF-DETR configuration
        
    Returns:
        Tuple of (preprocessed_tensor, image_info, original_image)
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    preprocessor = RFDETRPreprocessor(config)
    image_tensor = preprocessor.preprocess_image(original_image)
    image_info = preprocessor.get_image_info(original_image)
    
    return image_tensor, image_info, original_image


def batch_process_images(image_paths: List[str], config: RFDETRConfig) -> Tuple[torch.Tensor, List[Dict], List[np.ndarray]]:
    """
    Load and preprocess multiple images.
    
    Args:
        image_paths: List of image file paths
        config: RF-DETR configuration
        
    Returns:
        Tuple of (batch_tensor, image_info_list, original_images)
    """
    original_images = []
    image_info_list = []
    
    for path in image_paths:
        original_image = cv2.imread(path)
        if original_image is None:
            raise ValueError(f"Could not load image from {path}")
        original_images.append(original_image)
        
        preprocessor = RFDETRPreprocessor(config)
        image_info = preprocessor.get_image_info(original_image)
        image_info_list.append(image_info)
    
    preprocessor = RFDETRPreprocessor(config)
    batch_tensor = preprocessor.preprocess_batch(original_images)
    
    return batch_tensor, image_info_list, original_images