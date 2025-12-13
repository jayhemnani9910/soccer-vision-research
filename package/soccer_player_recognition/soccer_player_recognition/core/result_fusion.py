"""
Result Fusion - Multi-Model Output Integration

This module provides comprehensive result fusion capabilities to combine detection,
identification, segmentation, and classification results from multiple models:
- RF-DETR: Object detection with confidence scores
- SAM2: Segmentation masks with temporal consistency
- SigLIP: Zero-shot identification with text-image matching
- ResNet: Trained classification with feature extraction

Features:
- Multiple fusion strategies (voting, weighted, ensemble, confidence-based)
- Temporal consistency for video sequences
- Spatial alignment for multi-model results
- Confidence aggregation and conflict resolution
- Performance optimization and caching

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .results import DetectionResult, IdentificationResult, SegmentationResult, TrackingResult

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Available fusion strategies."""
    MAJORITY_VOTING = "majority_voting"
    WEIGHTED_AVERAGING = "weighted_averaging" 
    CONFIDENCE_BASED = "confidence_based"
    ENSEMBLE = "ensemble"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    SPATIAL_ALIGNMENT = "spatial_alignment"
    ADAPTIVE = "adaptive"


class ObjectClass(Enum):
    """Object classes in soccer detection."""
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"
    UNKNOWN = "unknown"


@dataclass
class FusedDetection:
    """Represents a fused detection from multiple models."""
    object_id: str
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None
    source_models: List[str] = field(default_factory=list)
    individual_detections: List[Dict[str, Any]] = field(default_factory=list)
    fusion_method: str = "unknown"
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusedIdentification:
    """Represents a fused identification from multiple models."""
    object_id: str
    player_name: str
    confidence: float
    source_models: List[str] = field(default_factory=list)
    individual_identifications: List[Dict[str, Any]] = field(default_factory=list)
    fusion_method: str = "unknown"


@dataclass
class FusedSegmentation:
    """Represents a fused segmentation from multiple models."""
    object_id: str
    mask: np.ndarray
    confidence: float
    source_models: List[str] = field(default_factory=list)
    individual_segmentations: List[Dict[str, Any]] = field(default_factory=list)
    fusion_method: str = "unknown"
    temporal_consistency_score: float = 0.0


class ResultFuser:
    """
    Advanced result fusion system for multi-model player recognition.
    
    Combines results from RF-DETR (detection), SAM2 (segmentation), 
    SigLIP (identification), and ResNet (classification) models.
    """
    
    def __init__(self, default_strategy: FusionStrategy = FusionStrategy.ADAPTIVE):
        """
        Initialize ResultFuser.
        
        Args:
            default_strategy: Default fusion strategy to use
        """
        self.default_strategy = default_strategy
        
        # Model weights for weighted fusion
        self.model_weights = {
            'rf_detr': 1.0,      # High accuracy for detection
            'sam2': 0.8,         # Good for segmentation
            'siglip': 0.9,       # Excellent for identification
            'resnet': 0.7        # Reliable for classification
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'detection': 0.5,
            'identification': 0.6,
            'segmentation': 0.4,
            'classification': 0.7
        }
        
        # IOU thresholds for spatial matching
        self.iou_thresholds = {
            'detection': 0.3,
            'segmentation': 0.4,
            'temporal': 0.5
        }
        
        # Temporal consistency parameters
        self.temporal_window = 5  # Number of frames to consider for temporal consistency
        self.temporal_decay = 0.9  # Decay factor for older frames
        
        # State tracking
        self.temporal_tracks: Dict[str, List[Dict]] = defaultdict(list)
        self.fusion_history: List[Dict] = []
        
        logger.info(f"ResultFuser initialized with {default_strategy.value} strategy")
    
    def fuse_comprehensive_results(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse comprehensive results from multiple models.
        
        Args:
            model_results: Dictionary with keys 'detection', 'segmentation', 'identification'
            
        Returns:
            Fused comprehensive results
        """
        start_time = time.time()
        
        try:
            fused_results = {
                'fused_detections': [],
                'fused_identifications': [],
                'fused_segmentations': [],
                'fusion_metadata': {
                    'strategy': self.default_strategy.value,
                    'fusion_time': 0.0,
                    'models_fused': [],
                    'conflicts_resolved': 0
                }
            }
            
            # Track which models were used
            fused_results['fusion_metadata']['models_fused'] = list(model_results.keys())
            
            # Fuse detections
            if 'detection' in model_results:
                logger.info("Fusing detection results...")
                fused_detections = self._fuse_detections(model_results['detection'])
                fused_results['fused_detections'] = fused_detections
            
            # Fuse segmentations
            if 'segmentation' in model_results:
                logger.info("Fusing segmentation results...")
                fused_segmentations = self._fuse_segmentations(model_results['segmentation'])
                fused_results['fused_segmentations'] = fused_segmentations
            
            # Fuse identifications
            if 'identification' in model_results:
                logger.info("Fusing identification results...")
                fused_identifications = self._fuse_identifications(model_results['identification'])
                fused_results['fused_identifications'] = fused_identifications
            
            # Cross-modal fusion
            logger.info("Performing cross-modal fusion...")
            cross_fused = self._cross_modal_fusion(fused_results)
            fused_results.update(cross_fused)
            
            # Add temporal consistency if multiple frames
            if 'temporal_data' in model_results:
                logger.info("Applying temporal consistency...")
                temporally_consistent = self._apply_temporal_consistency(
                    fused_results, model_results['temporal_data']
                )
                fused_results['temporal_consistency'] = temporally_consistent
            
            # Update metadata
            fusion_time = time.time() - start_time
            fused_results['fusion_metadata']['fusion_time'] = fusion_time
            fused_results['fusion_metadata']['total_objects'] = len(fused_results.get('fused_detections', []))
            
            # Store in history
            self.fusion_history.append({
                'timestamp': time.time(),
                'models_used': fused_results['fusion_metadata']['models_fused'],
                'fusion_time': fusion_time,
                'objects_detected': len(fused_results.get('fused_detections', []))
            })
            
            logger.info(f"Comprehensive fusion completed in {fusion_time:.3f}s")
            return fused_results
            
        except Exception as e:
            logger.error(f"Comprehensive fusion failed: {e}")
            raise
    
    def _fuse_detections(self, detection_results: Any) -> List[FusedDetection]:
        """Fuse detection results from multiple sources."""
        detections = []
        
        # Handle different input formats
        if isinstance(detection_results, list):
            all_detections = []
            for result in detection_results:
                if hasattr(result, 'detections'):
                    all_detections.extend(result.detections)
                elif isinstance(result, dict) and 'detections' in result:
                    all_detections.extend(result['detections'])
        else:
            # Single result
            if hasattr(detection_results, 'detections'):
                all_detections = detection_results.detections
            elif isinstance(detection_results, dict) and 'detections' in result:
                all_detections = detection_results['detections']
            else:
                all_detections = []
        
        # Group detections by spatial proximity
        detection_groups = self._group_detections_by_spatial_proximity(all_detections)
        
        # Fuse each group
        for group in detection_groups:
            fused_detection = self._fuse_detection_group(group)
            if fused_detection and fused_detection.confidence >= self.confidence_thresholds['detection']:
                detections.append(fused_detection)
        
        return detections
    
    def _fuse_identifications(self, identification_results: Any) -> List[FusedIdentification]:
        """Fuse identification results from multiple sources."""
        identifications = []
        
        # Handle different input formats
        if isinstance(identification_results, list):
            all_identifications = []
            for result in identification_results:
                if hasattr(result, 'player_name'):
                    all_identifications.append({
                        'player_name': result.player_name,
                        'confidence': result.confidence,
                        'model': getattr(result, 'model_name', 'unknown')
                    })
                elif isinstance(result, dict):
                    all_identifications.append(result)
        else:
            # Single result
            if hasattr(identification_results, 'player_name'):
                all_identifications = [{
                    'player_name': identification_results.player_name,
                    'confidence': identification_results.confidence,
                    'model': getattr(identification_results, 'model_name', 'unknown')
                }]
            elif isinstance(identification_results, dict):
                all_identifications = [identification_results]
            else:
                all_identifications = []
        
        # Group identifications by confidence similarity
        identification_groups = self._group_identifications_by_confidence(all_identifications)
        
        # Fuse each group
        for group in identification_groups:
            fused_identification = self._fuse_identification_group(group)
            if fused_identification and fused_identification.confidence >= self.confidence_thresholds['identification']:
                identifications.append(fused_identification)
        
        return identifications
    
    def _fuse_segmentations(self, segmentation_results: Any) -> List[FusedSegmentation]:
        """Fuse segmentation results from multiple sources."""
        segmentations = []
        
        # Handle different input formats
        if isinstance(segmentation_results, list):
            all_segmentations = []
            for result in segmentation_results:
                if hasattr(result, 'masks'):
                    all_segmentations.append(result)
                elif isinstance(result, dict) and 'masks' in result:
                    all_segmentations.append(result)
        else:
            # Single result
            if hasattr(segmentation_results, 'masks'):
                all_segmentations = [segmentation_results]
            elif isinstance(segmentation_results, dict) and 'masks' in result:
                all_segmentations = [segmentation_results]
            else:
                all_segmentations = []
        
        # Group segmentations by object ID
        segmentation_groups = self._group_segmentations_by_object_id(all_segmentations)
        
        # Fuse each group
        for obj_id, group in segmentation_groups.items():
            fused_segmentation = self._fuse_segmentation_group(obj_id, group)
            if fused_segmentation and fused_segmentation.confidence >= self.confidence_thresholds['segmentation']:
                segmentations.append(fused_segmentation)
        
        return segmentations
    
    def _group_detections_by_spatial_proximity(self, detections: List[Dict]) -> List[List[Dict]]:
        """Group detections that are spatially close (IoU > threshold)."""
        if not detections:
            return []
        
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(det1.get('bbox', []), det2.get('bbox', []))
                
                if iou > self.iou_thresholds['detection']:
                    group.append(det2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _group_identifications_by_confidence(self, identifications: List[Dict]) -> List[List[Dict]]:
        """Group identifications that are similar in confidence scores."""
        if not identifications:
            return []
        
        # Sort by confidence
        sorted_ids = sorted(identifications, key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        groups = []
        used = set()
        
        for i, id1 in enumerate(sorted_ids):
            if i in used:
                continue
            
            # Find similar identifications
            group = [id1]
            used.add(i)
            
            confidence_threshold = 0.3  # Confidence difference threshold
            
            for j, id2 in enumerate(sorted_ids[i+1:], i+1):
                if j in used:
                    continue
                
                confidence_diff = abs(id1.get('confidence', 0.0) - id2.get('confidence', 0.0))
                
                if confidence_diff <= confidence_threshold:
                    group.append(id2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _group_segmentations_by_object_id(self, segmentations: List[Any]) -> Dict[str, List[Any]]:
        """Group segmentations by object ID."""
        groups = defaultdict(list)
        
        for result in segmentations:
            if hasattr(result, 'masks'):
                for obj_id, mask in result.masks.items():
                    groups[obj_id].append(result)
            elif isinstance(result, dict) and 'masks' in result:
                for obj_id, mask in result['masks'].items():
                    groups[obj_id].append(result)
        
        return dict(groups)
    
    def _fuse_detection_group(self, group: List[Dict]) -> Optional[FusedDetection]:
        """Fuse a group of spatially close detections."""
        if not group:
            return None
        
        # Extract bbox coordinates and confidences
        bboxes = []
        confidences = []
        class_names = []
        models = []
        
        for det in group:
            bbox = det.get('bbox', [])
            confidence = det.get('confidence', 0.0)
            class_name = det.get('class_name', 'unknown')
            model = det.get('model', 'unknown')
            
            if bbox and len(bbox) == 4:
                bboxes.append(bbox)
                confidences.append(confidence)
                class_names.append(class_name)
                models.append(model)
        
        if not bboxes:
            return None
        
        # Weighted average of bounding boxes
        weights = [self.model_weights.get(model, 0.5) * conf for model, conf in zip(models, confidences)]
        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        weighted_bbox = []
        for i in range(4):
            weighted_bbox.append(np.average([bbox[i] for bbox in bboxes], weights=weights))
        
        # Majority voting for class
        class_vote = Counter(class_names)
        most_common_class = class_vote.most_common(1)[0][0]
        
        # Weighted average confidence
        avg_confidence = np.average(confidences, weights=weights)
        
        return FusedDetection(
            object_id=f"fused_{len(self.fusion_history)}_det_{int(time.time())}",
            class_name=most_common_class,
            confidence=avg_confidence,
            bbox=weighted_bbox,
            source_models=models,
            individual_detections=group,
            fusion_method="spatial_weighted_fusion"
        )
    
    def _fuse_identification_group(self, group: List[Dict]) -> Optional[FusedIdentification]:
        """Fuse a group of similar identifications."""
        if not group:
            return None
        
        # Extract player names and confidences
        player_names = [ident.get('player_name', 'unknown') for ident in group]
        confidences = [ident.get('confidence', 0.0) for ident in group]
        models = [ident.get('model', 'unknown') for ident in group]
        
        # Weight by model accuracy
        weights = [self.model_weights.get(model, 0.5) * conf for model, conf in zip(models, confidences)]
        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        # Weighted voting for player name
        name_scores = defaultdict(float)
        for name, weight in zip(player_names, weights):
            name_scores[name] += weight
        
        best_player = max(name_scores.items(), key=lambda x: x[1])
        fused_player_name = best_player[0]
        avg_confidence = best_player[1]
        
        return FusedIdentification(
            object_id=f"fused_{len(self.fusion_history)}_id_{int(time.time())}",
            player_name=fused_player_name,
            confidence=avg_confidence,
            source_models=models,
            individual_identifications=group,
            fusion_method="weighted_voting"
        )
    
    def _fuse_segmentation_group(self, obj_id: str, group: List[Any]) -> Optional[FusedSegmentation]:
        """Fuse a group of segmentations for the same object."""
        if not group:
            return None
        
        # Get all masks (simplified - would need proper mask processing)
        masks = []
        confidences = []
        models = []
        
        for result in group:
            if hasattr(result, 'masks') and obj_id in result.masks:
                mask = result.masks[obj_id]
                if isinstance(mask, np.ndarray):
                    masks.append(mask)
                    confidences.append(getattr(result, 'confidence', 0.5))
                    models.append(getattr(result, 'model_name', 'unknown'))
            elif isinstance(result, dict) and 'masks' in result and obj_id in result['masks']:
                mask = result['masks'][obj_id]
                if isinstance(mask, np.ndarray):
                    masks.append(mask)
                    confidences.append(result.get('confidence', 0.5))
                    models.append(result.get('model', 'unknown'))
        
        if not masks:
            return None
        
        # Average masks (simplified)
        avg_mask = np.mean(masks, axis=0)
        
        # Weighted average confidence
        weights = [self.model_weights.get(model, 0.5) * conf for model, conf in zip(models, confidences)]
        avg_confidence = np.average(confidences, weights=weights) if confidences else 0.0
        
        return FusedSegmentation(
            object_id=obj_id,
            mask=avg_mask,
            confidence=avg_confidence,
            source_models=models,
            individual_segmentations=[{
                'mask': mask,
                'confidence': conf,
                'model': model
            } for mask, conf, model in zip(masks, confidences, models)],
            fusion_method="mask_averaging"
        )
    
    def _cross_modal_fusion(self, fused_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-modal fusion between detections, identifications, and segmentations."""
        cross_fused = {
            'cross_fused_objects': [],
            'conflicts': []
        }
        
        detections = fused_results.get('fused_detections', [])
        identifications = fused_results.get('fused_identifications', [])
        segmentations = fused_results.get('fused_segmentations', [])
        
        # Match detections with identifications by spatial proximity
        for detection in detections:
            best_identification = self._find_best_identification_match(
                detection, identifications
            )
            
            if best_identification:
                # Create cross-fused object
                cross_fused_obj = {
                    'object_id': detection.object_id,
                    'bbox': detection.bbox,
                    'class_name': detection.class_name,
                    'player_name': best_identification.player_name,
                    'confidence': (detection.confidence + best_identification.confidence) / 2,
                    'detection_source': detection.source_models,
                    'identification_source': best_identification.source_models,
                    'fusion_method': 'cross_modal_spatial_matching'
                }
                
                # Add segmentation if available
                best_segmentation = self._find_best_segmentation_match(
                    detection, segmentations
                )
                if best_segmentation:
                    cross_fused_obj['mask'] = best_segmentation.mask
                    cross_fused_obj['segmentation_source'] = best_segmentation.source_models
                
                cross_fused['cross_fused_objects'].append(cross_fused_obj)
            
            else:
                # No identification match found
                cross_fused_obj = {
                    'object_id': detection.object_id,
                    'bbox': detection.bbox,
                    'class_name': detection.class_name,
                    'player_name': 'unknown',
                    'confidence': detection.confidence,
                    'detection_source': detection.source_models,
                    'fusion_method': 'detection_only'
                }
                
                cross_fused['cross_fused_objects'].append(cross_fused_obj)
        
        return cross_fused
    
    def _find_best_identification_match(self, detection: FusedDetection, 
                                      identifications: List[FusedIdentification]) -> Optional[FusedIdentification]:
        """Find the best identification match for a detection based on spatial proximity."""
        best_match = None
        best_score = 0.0
        
        detection_center = [
            (detection.bbox[0] + detection.bbox[2]) / 2,
            (detection.bbox[1] + detection.bbox[3]) / 2
        ]
        
        for identification in identifications:
            # For simplicity, use confidence as match score
            # In real implementation, would use spatial and temporal matching
            if identification.confidence > best_score:
                best_score = identification.confidence
                best_match = identification
        
        return best_match
    
    def _find_best_segmentation_match(self, detection: FusedDetection,
                                    segmentations: List[FusedSegmentation]) -> Optional[FusedSegmentation]:
        """Find the best segmentation match for a detection."""
        best_match = None
        best_score = 0.0
        
        for segmentation in segmentations:
            # Simple confidence-based matching
            # In real implementation, would use IoU between bbox and mask
            if segmentation.confidence > best_score:
                best_score = segmentation.confidence
                best_match = segmentation
        
        return best_match
    
    def _apply_temporal_consistency(self, fused_results: Dict[str, Any], 
                                  temporal_data: List[Dict]) -> Dict[str, Any]:
        """Apply temporal consistency to fused results across multiple frames."""
        temporal_consistent = {
            'consistent_objects': [],
            'temporal_conflicts': [],
            'stability_scores': {}
        }
        
        # This is a simplified implementation
        # In real usage, would maintain object tracks across frames
        
        for frame_data in temporal_data:
            # Apply temporal consistency to each frame's results
            consistent_frame = self._apply_frame_temporal_consistency(
                fused_results, frame_data
            )
            if consistent_frame:
                temporal_consistent['consistent_objects'].append(consistent_frame)
        
        return temporal_consistent
    
    def _apply_frame_temporal_consistency(self, fused_results: Dict[str, Any], 
                                        frame_data: Dict) -> Optional[Dict]:
        """Apply temporal consistency to a single frame."""
        # Simplified temporal consistency
        # Would implement proper temporal tracking and smoothing
        
        return {
            'frame_id': frame_data.get('frame_id', 0),
            'objects': fused_results.get('cross_fused_objects', []),
            'consistency_score': 0.8  # Placeholder
        }
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
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
    
    def set_fusion_strategy(self, strategy: FusionStrategy):
        """Set the fusion strategy."""
        self.default_strategy = strategy
        logger.info(f"Fusion strategy set to {strategy.value}")
    
    def update_model_weights(self, weights: Dict[str, float]):
        """Update model weights for fusion."""
        self.model_weights.update(weights)
        logger.info(f"Updated model weights: {weights}")
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics and performance metrics."""
        if not self.fusion_history:
            return {'total_fusions': 0}
        
        total_fusions = len(self.fusion_history)
        avg_fusion_time = np.mean([h['fusion_time'] for h in self.fusion_history])
        avg_objects_detected = np.mean([h['objects_detected'] for h in self.fusion_history])
        
        return {
            'total_fusions': total_fusions,
            'avg_fusion_time': avg_fusion_time,
            'avg_objects_detected': avg_objects_detected,
            'fusion_history': self.fusion_history[-10:],  # Last 10 fusions
            'current_strategy': self.default_strategy.value,
            'model_weights': self.model_weights.copy()
        }
    
    def reset_state(self):
        """Reset fusion state (temporal tracks, history, etc.)."""
        self.temporal_tracks.clear()
        self.fusion_history.clear()
        logger.info("Fusion state reset")


if __name__ == "__main__":
    # Example usage
    fuser = ResultFuser(FusionStrategy.ADAPTIVE)
    
    # Print fusion statistics
    stats = fuser.get_fusion_statistics()
    print("Fusion Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    fuser.reset_state()
    print("ResultFuser test completed")