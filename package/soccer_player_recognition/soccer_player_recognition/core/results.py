"""
Results Data Structures

This module defines standardized result classes for all model outputs:
- DetectionResult: Object detection results from RF-DETR
- IdentificationResult: Player identification results from SigLIP/ResNet  
- SegmentationResult: Video segmentation results from SAM2
- TrackingResult: Temporal tracking and consistency results

Each result class includes:
- Standardized output format
- Performance metrics and timing
- Model metadata and confidence scores
- Support for batch processing
- Serialization capabilities

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import json


@dataclass
class DetectionResult:
    """
    Standardized detection result from RF-DETR or similar detection models.
    
    Contains:
    - Bounding boxes with confidence scores
    - Class predictions and metadata
    - Execution timing and model information
    - Support for batch processing
    """
    
    # Input information
    image_path: Optional[str] = None
    image_id: Optional[int] = None
    image_shape: Optional[tuple] = None
    
    # Detection results
    detections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    execution_time: float = 0.0
    model_name: str = "rf_detr"
    model_version: str = "1.0.0"
    
    # Quality metrics
    total_detections: int = 0
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    min_confidence: float = 1.0
    
    # Processing metadata
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    device: str = "cpu"
    
    # Quality indicators
    has_players: bool = False
    has_ball: bool = False
    has_referees: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Update detection count
        self.total_detections = len(self.detections)
        
        # Calculate confidence statistics
        if self.detections:
            confidences = [det.get('confidence', 0.0) for det in self.detections]
            self.avg_confidence = float(np.mean(confidences))
            self.max_confidence = float(np.max(confidences))
            self.min_confidence = float(np.min(confidences))
            
            # Check for specific object types
            for det in self.detections:
                class_name = det.get('class_name', '').lower()
                if 'player' in class_name:
                    self.has_players = True
                elif 'ball' in class_name:
                    self.has_ball = True
                elif 'referee' in class_name:
                    self.has_referees = True
    
    def get_players_only(self) -> List[Dict[str, Any]]:
        """Get only player detections."""
        return [det for det in self.detections 
                if 'player' in det.get('class_name', '').lower()]
    
    def get_high_confidence_detections(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Get detections above confidence threshold."""
        return [det for det in self.detections 
                if det.get('confidence', 0.0) >= threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'image_path': self.image_path,
            'image_id': self.image_id,
            'image_shape': self.image_shape,
            'detections': self.detections,
            'execution_time': self.execution_time,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'total_detections': self.total_detections,
            'avg_confidence': self.avg_confidence,
            'max_confidence': self.max_confidence,
            'min_confidence': self.min_confidence,
            'timestamp': self.timestamp,
            'device': self.device,
            'has_players': self.has_players,
            'has_ball': self.has_ball,
            'has_referees': self.has_referees
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class IdentificationResult:
    """
    Standardized identification result from SigLIP or ResNet models.
    
    Contains:
    - Player identification with confidence scores
    - Multiple prediction candidates
    - Execution timing and model information
    - Support for different identification approaches
    """
    
    # Input information
    image_index: Optional[int] = None
    image_path: Optional[str] = None
    crop_region: Optional[List[float]] = None  # [x1, y1, x2, y2] for detected region
    
    # Identification results
    player_name: str = "unknown"
    confidence: float = 0.0
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Identification metadata
    method: str = "zero_shot"  # zero_shot, trained_classification, ensemble
    player_candidates: List[str] = field(default_factory=list)
    team_context: Optional[str] = None
    
    # Performance metrics
    execution_time: float = 0.0
    model_name: str = "siglip"
    model_version: str = "1.0.0"
    
    # Quality metrics
    top_k_accuracy: float = 0.0  # For top-k accuracy evaluation
    entropy: float = 0.0  # Prediction uncertainty
    
    # Processing metadata
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    device: str = "cpu"
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Calculate entropy for uncertainty measurement
        if self.predictions:
            probs = [pred.get('confidence', 0.0) for pred in self.predictions]
            if probs:
                probs = np.array(probs)
                probs = probs / probs.sum() if probs.sum() > 0 else probs
                # Calculate entropy (measure of uncertainty)
                entropy_vals = -probs * np.log(probs + 1e-10)
                self.entropy = float(entropy_vals.sum())
        
        # Calculate top-k accuracy if we have ground truth candidates
        if self.player_candidates and self.predictions:
            self.top_k_accuracy = self._calculate_top_k_accuracy()
    
    def _calculate_top_k_accuracy(self, k: int = 3) -> float:
        """Calculate top-k accuracy given player candidates."""
        if not self.predictions:
            return 0.0
        
        # Get top-k predictions
        top_predictions = [pred.get('player', 'unknown') for pred in self.predictions[:k]]
        
        # Check if any top prediction is in candidates
        correct_predictions = sum(1 for pred in top_predictions 
                                if pred in self.player_candidates)
        
        return correct_predictions / min(k, len(top_predictions))
    
    def get_top_predictions(self, k: int = 3) -> List[Dict[str, Any]]:
        """Get top-k predictions."""
        return self.predictions[:k] if self.predictions else []
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if identification has high confidence."""
        return self.confidence >= threshold
    
    def get_alternative_candidates(self, threshold: float = 0.1) -> List[str]:
        """Get alternative player candidates with significant confidence."""
        if not self.predictions:
            return []
        
        candidates = []
        for pred in self.predictions:
            if pred.get('player') != self.player_name:
                if pred.get('confidence', 0.0) >= threshold:
                    candidates.append(pred['player'])
        
        return candidates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'image_index': self.image_index,
            'image_path': self.image_path,
            'crop_region': self.crop_region,
            'player_name': self.player_name,
            'confidence': self.confidence,
            'predictions': self.predictions,
            'method': self.method,
            'player_candidates': self.player_candidates,
            'team_context': self.team_context,
            'execution_time': self.execution_time,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'top_k_accuracy': self.top_k_accuracy,
            'entropy': self.entropy,
            'timestamp': self.timestamp,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentificationResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SegmentationResult:
    """
    Standardized segmentation result from SAM2 or similar segmentation models.
    
    Contains:
    - Segmentation masks for detected objects
    - Temporal tracking information
    - Confidence scores and object metadata
    - Support for video sequences
    """
    
    # Input information
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None
    
    # Segmentation results
    masks: Dict[str, Any] = field(default_factory=dict)  # object_id -> mask data
    
    # Performance metrics
    execution_time: float = 0.0
    model_name: str = "sam2"
    model_version: str = "1.0.0"
    
    # Quality metrics
    total_masks: int = 0
    avg_mask_confidence: float = 0.0
    mask_coverage_ratio: float = 0.0  # Percentage of image covered by masks
    
    # Tracking metadata
    object_tracks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    temporal_consistency_score: float = 0.0
    
    # Processing metadata
    device: str = "cpu"
    memory_usage: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Update mask count
        self.total_masks = len(self.masks)
        
        # Calculate average confidence and coverage
        if self.masks:
            # Calculate average confidence
            confidences = []
            total_area = 0
            
            for obj_id, mask_data in self.masks.items():
                if isinstance(mask_data, np.ndarray):
                    # Calculate mask statistics
                    mask = mask_data
                    if len(mask.shape) == 2:  # Binary mask
                        confidence = float(mask.mean())
                        confidences.append(confidence)
                        total_area += mask.sum()
                    elif len(mask.shape) == 3 and mask.shape[2] == 1:  # Single channel
                        confidence = float(mask[:, :, 0].mean())
                        confidences.append(confidence)
                        total_area += mask[:, :, 0].sum()
                elif isinstance(mask_data, dict) and 'confidence' in mask_data:
                    confidences.append(mask_data['confidence'])
            
            if confidences:
                self.avg_mask_confidence = float(np.mean(confidences))
            
            # Calculate coverage ratio (simplified)
            # In real implementation, would use actual image dimensions
            if total_area > 0:
                estimated_image_area = 640 * 480  # Placeholder
                self.mask_coverage_ratio = float(total_area / estimated_image_area)
        
        # Extract temporal consistency information
        if self.object_tracks:
            consistency_scores = [track.get('consistency_score', 0.0) 
                                for track in self.object_tracks.values()]
            if consistency_scores:
                self.temporal_consistency_score = float(np.mean(consistency_scores))
    
    def get_object_mask(self, object_id: str) -> Optional[Any]:
        """Get mask for specific object."""
        return self.masks.get(object_id)
    
    def get_high_confidence_masks(self, threshold: float = 0.6) -> Dict[str, Any]:
        """Get masks above confidence threshold."""
        filtered_masks = {}
        
        for obj_id, mask_data in self.masks.items():
            if isinstance(mask_data, np.ndarray):
                confidence = float(mask_data.mean()) if len(mask_data.shape) == 2 else 0.0
                if confidence >= threshold:
                    filtered_masks[obj_id] = mask_data
            elif isinstance(mask_data, dict) and 'confidence' in mask_data:
                if mask_data['confidence'] >= threshold:
                    filtered_masks[obj_id] = mask_data
        
        return filtered_masks
    
    def update_tracking_info(self, object_id: str, track_info: Dict[str, Any]):
        """Update tracking information for an object."""
        if object_id not in self.object_tracks:
            self.object_tracks[object_id] = {}
        self.object_tracks[object_id].update(track_info)
    
    def get_tracked_objects(self) -> List[str]:
        """Get list of currently tracked objects."""
        return list(self.object_tracks.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Handle numpy arrays for JSON serialization
        serializable_masks = {}
        for obj_id, mask_data in self.masks.items():
            if isinstance(mask_data, np.ndarray):
                serializable_masks[obj_id] = {
                    'mask': mask_data.tolist(),
                    'shape': mask_data.shape,
                    'confidence': float(mask_data.mean())
                }
            else:
                serializable_masks[obj_id] = mask_data
        
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'masks': serializable_masks,
            'execution_time': self.execution_time,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'total_masks': self.total_masks,
            'avg_mask_confidence': self.avg_mask_confidence,
            'mask_coverage_ratio': self.mask_coverage_ratio,
            'object_tracks': self.object_tracks,
            'temporal_consistency_score': self.temporal_consistency_score,
            'device': self.device,
            'memory_usage': self.memory_usage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SegmentationResult':
        """Create from dictionary."""
        # Restore numpy arrays
        masks = {}
        for obj_id, mask_data in data.get('masks', {}).items():
            if isinstance(mask_data, dict) and 'mask' in mask_data:
                mask = np.array(mask_data['mask'])
                if 'shape' in mask_data and mask_data['shape'] != mask.shape:
                    mask = mask.reshape(mask_data['shape'])
                masks[obj_id] = mask
            else:
                masks[obj_id] = mask_data
        
        data['masks'] = masks
        return cls(**data)


@dataclass
class TrackingResult:
    """
    Standardized tracking result for temporal consistency and object tracking.
    
    Contains:
    - Object tracks across multiple frames
    - Track quality metrics
    - Temporal consistency scores
    - Occlusion handling results
    """
    
    # Tracking metadata
    track_id: str
    object_class: str = "player"
    
    # Track information
    track_history: List[Dict[str, Any]] = field(default_factory=list)
    current_bbox: Optional[List[float]] = None
    current_mask: Optional[Any] = None
    
    # Quality metrics
    track_confidence: float = 0.0
    temporal_consistency: float = 0.0
    track_length: int = 0
    
    # Occlusion handling
    occlusion_events: List[Dict[str, Any]] = field(default_factory=list)
    recovery_count: int = 0
    
    # Performance metrics
    total_execution_time: float = 0.0
    avg_frame_time: float = 0.0
    
    # Metadata
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    model_name: str = "sam2"
    
    def __post_init__(self):
        """Post-initialization processing."""
        self.track_length = len(self.track_history)
        
        if self.track_history:
            # Calculate temporal consistency
            self.temporal_consistency = self._calculate_temporal_consistency()
            
            # Set frame range
            frames = [track.get('frame_id', 0) for track in self.track_history]
            self.start_frame = min(frames) if frames else None
            self.end_frame = max(frames) if frames else None
            
            # Set current state
            if self.track_history:
                latest_track = self.track_history[-1]
                self.current_bbox = latest_track.get('bbox')
                self.current_mask = latest_track.get('mask')
        
        # Calculate average frame time
        if self.total_execution_time > 0 and self.track_length > 0:
            self.avg_frame_time = self.total_execution_time / self.track_length
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency score for the track."""
        if len(self.track_history) < 2:
            return 1.0
        
        # Calculate bbox consistency (simplified)
        consistencies = []
        for i in range(1, len(self.track_history)):
            prev_bbox = self.track_history[i-1].get('bbox')
            curr_bbox = self.track_history[i].get('bbox')
            
            if prev_bbox and curr_bbox:
                # Calculate IoU between consecutive frames
                iou = self._calculate_iou(prev_bbox, curr_bbox)
                consistencies.append(iou)
        
        return float(np.mean(consistencies)) if consistencies else 0.0
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def add_frame_track(self, frame_id: int, track_data: Dict[str, Any]):
        """Add track information for a new frame."""
        track_data['frame_id'] = frame_id
        self.track_history.append(track_data)
        
        # Update current state
        self.current_bbox = track_data.get('bbox')
        self.current_mask = track_data.get('mask')
        
        # Update track length
        self.track_length = len(self.track_history)
        
        # Update temporal consistency
        self.temporal_consistency = self._calculate_temporal_consistency()
    
    def get_track_quality_score(self) -> float:
        """Get overall track quality score."""
        if self.track_length == 0:
            return 0.0
        
        # Combine multiple factors
        length_score = min(self.track_length / 30.0, 1.0)  # Normalize to ~30 frames
        confidence_score = self.track_confidence
        consistency_score = self.temporal_consistency
        
        # Weighted combination
        quality_score = (0.3 * length_score + 
                        0.4 * confidence_score + 
                        0.3 * consistency_score)
        
        return quality_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Handle numpy arrays
        serializable_history = []
        for track in self.track_history:
            serializable_track = track.copy()
            
            if 'mask' in serializable_track and isinstance(serializable_track['mask'], np.ndarray):
                mask = serializable_track['mask']
                serializable_track['mask'] = {
                    'data': mask.tolist(),
                    'shape': mask.shape
                }
            
            serializable_history.append(serializable_track)
        
        return {
            'track_id': self.track_id,
            'object_class': self.object_class,
            'track_history': serializable_history,
            'current_bbox': self.current_bbox,
            'current_mask': self.current_mask.tolist() if isinstance(self.current_mask, np.ndarray) else self.current_mask,
            'track_confidence': self.track_confidence,
            'temporal_consistency': self.temporal_consistency,
            'track_length': self.track_length,
            'occlusion_events': self.occlusion_events,
            'recovery_count': self.recovery_count,
            'total_execution_time': self.total_execution_time,
            'avg_frame_time': self.avg_frame_time,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'model_name': self.model_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackingResult':
        """Create from dictionary."""
        # Restore numpy arrays
        if 'current_mask' in data and data['current_mask']:
            if isinstance(data['current_mask'], dict):
                mask = np.array(data['current_mask']['data'])
                if 'shape' in data['current_mask']:
                    mask = mask.reshape(data['current_mask']['shape'])
                data['current_mask'] = mask
        
        # Restore history masks
        for track in data.get('track_history', []):
            if 'mask' in track and isinstance(track['mask'], dict):
                mask = np.array(track['mask']['data'])
                if 'shape' in track['mask']:
                    mask = mask.reshape(track['mask']['shape'])
                track['mask'] = mask
        
        return cls(**data)


# Utility functions for batch processing
def batch_detection_results(results: List[DetectionResult]) -> Dict[str, Any]:
    """Aggregate multiple detection results."""
    if not results:
        return {}
    
    total_detections = sum(r.total_detections for r in results)
    avg_confidence = np.mean([r.avg_confidence for r in results])
    
    return {
        'total_images': len(results),
        'total_detections': total_detections,
        'avg_confidence': avg_confidence,
        'avg_execution_time': np.mean([r.execution_time for r in results]),
        'has_players': any(r.has_players for r in results),
        'has_ball': any(r.has_ball for r in results),
        'has_referees': any(r.has_referees for r in results)
    }


def batch_identification_results(results: List[IdentificationResult]) -> Dict[str, Any]:
    """Aggregate multiple identification results."""
    if not results:
        return {}
    
    avg_confidence = np.mean([r.confidence for r in results])
    high_confidence_count = sum(1 for r in results if r.is_high_confidence())
    
    return {
        'total_identifications': len(results),
        'avg_confidence': avg_confidence,
        'high_confidence_rate': high_confidence_count / len(results),
        'avg_execution_time': np.mean([r.execution_time for r in results]),
        'unique_players': len(set(r.player_name for r in results if r.player_name != 'unknown')),
        'avg_entropy': np.mean([r.entropy for r in results])
    }


if __name__ == "__main__":
    # Example usage
    print("Testing Result Classes...")
    
    # Create sample detection result
    detection = DetectionResult(
        image_path="test.jpg",
        detections=[
            {'bbox': [100, 100, 200, 300], 'confidence': 0.9, 'class_name': 'player'},
            {'bbox': [300, 200, 400, 350], 'confidence': 0.8, 'class_name': 'ball'}
        ]
    )
    
    print("Detection Result:")
    print(f"  Players detected: {detection.has_players}")
    print(f"  Ball detected: {detection.has_ball}")
    print(f"  Average confidence: {detection.avg_confidence:.3f}")
    
    # Create sample identification result
    identification = IdentificationResult(
        player_name="Lionel Messi",
        confidence=0.95,
        predictions=[
            {'player': 'Lionel Messi', 'confidence': 0.95},
            {'player': 'Cristiano Ronaldo', 'confidence': 0.03},
            {'player': 'Neymar', 'confidence': 0.02}
        ]
    )
    
    print("\nIdentification Result:")
    print(f"  Player: {identification.player_name}")
    print(f"  Confidence: {identification.confidence:.3f}")
    print(f"  High confidence: {identification.is_high_confidence()}")
    
    print("\nResult classes test completed")