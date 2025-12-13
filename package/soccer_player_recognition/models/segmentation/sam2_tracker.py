"""
SAM2 Multi-Object Tracker

This module provides multi-object tracking functionality for SAM2, handling
object association across frames, trajectory management, and tracking performance
evaluation for video sequences.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from scipy.optimize import linear_sum_assignment

from .sam2_model import SAM2Model, FrameData, TrackingState, MemoryMode

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Individual track object"""
    track_id: str
    history: deque  # Recent masks and positions
    last_seen: int
    confidence_history: List[float]
    bbox_history: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    is_active: bool = True
    age: int = 0
    total_visible_count: int = 0
    consecutive_invisible_count: int = 0


@dataclass
class Detection:
    """Detection from current frame"""
    object_id: str
    mask: torch.Tensor
    bbox: Tuple[int, int, int, int]
    confidence: float
    features: Optional[torch.Tensor] = None


@dataclass
class TrackingConfig:
    """Configuration for tracking parameters"""
    max_disappeared: int = 30  # Max frames before track deletion
    max_distance: float = 50.0  # Max center distance for matching
    min_confidence: float = 0.6  # Min confidence for valid detection
    iou_threshold: float = 0.3  # IoU threshold for matching
    appearance_threshold: float = 0.7  # Appearance similarity threshold
    bbox_smooth_factor: float = 0.5  # Smoothing factor for bbox prediction


class SAM2Tracker:
    """
    SAM2 Multi-Object Tracker for video sequences.
    
    Features:
    - Multi-object tracking with unique IDs
    - IoU and appearance-based matching
    - Trajectory management and smoothing
    - Track confidence and quality assessment
    - Performance metrics and evaluation
    """
    
    def __init__(
        self,
        sam2_model: SAM2Model,
        config: Optional[TrackingConfig] = None,
        enable_trajectory_smoothing: bool = True,
        enable_track_merging: bool = False,
    ):
        """
        Initialize SAM2 tracker
        
        Args:
            sam2_model: Pre-initialized SAM2 model
            config: Tracking configuration parameters
            enable_trajectory_smoothing: Whether to smooth trajectories
            enable_track_merging: Whether to merge similar tracks
        """
        self.sam2_model = sam2_model
        self.config = config or TrackingConfig()
        
        # Track management
        self.active_tracks: Dict[str, Track] = {}
        self.next_track_id: int = 1
        self.frame_count: int = 0
        
        # Performance metrics
        self.metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks_created': 0,
            'total_tracks_lost': 0,
            'tracking_accuracy': [],
            'precision_scores': [],
            'recall_scores': []
        }
        
        # Optional features
        self.enable_smoothing = enable_trajectory_smoothing
        self.enable_merging = enable_track_merging
        
        logger.info("SAM2 Tracker initialized")
    
    def track_frame(
        self,
        image: torch.Tensor,
        frame_id: int,
        prompts: Optional[List[Dict]] = None,
        ground_truth: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process single frame for tracking
        
        Args:
            image: Input image tensor
            frame_id: Frame identifier
            prompts: Optional SAM2 prompts
            ground_truth: Optional ground truth annotations
            
        Returns:
            Dictionary with tracking results
        """
        self.frame_count = frame_id
        
        # Get detections from SAM2
        detections = self._get_detections(image, prompts)
        
        # Update tracks
        track_results = self._update_tracks(detections, frame_id)
        
        # Update metrics
        self._update_metrics(track_results, ground_truth)
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        return track_results
    
    def _get_detections(self, image: torch.Tensor, prompts: Optional[List[Dict]]) -> List[Detection]:
        """
        Get object detections from SAM2 model
        
        Args:
            image: Input image tensor
            prompts: Optional SAM2 prompts
            
        Returns:
            List of Detection objects
        """
        # Extract masks using SAM2
        masks = self.sam2_model.extract_masks(image, prompts)
        
        detections = []
        B, C, H, W = image.shape
        
        for obj_id, mask in masks.items():
            # Convert mask to binary
            binary_mask = (mask > self.config.min_confidence).float()
            
            # Compute bounding box from mask
            bbox = self._mask_to_bbox(binary_mask.squeeze(0))
            
            # Extract features for appearance matching
            features = self._extract_features(image, mask)
            
            # Compute confidence
            confidence = float(mask.max().item())
            
            if confidence >= self.config.min_confidence:
                detection = Detection(
                    object_id=obj_id,
                    mask=mask,
                    bbox=bbox,
                    confidence=confidence,
                    features=features
                )
                detections.append(detection)
        
        self.metrics['total_detections'] += len(detections)
        return detections
    
    def _mask_to_bbox(self, mask: torch.Tensor) -> Tuple[int, int, int, int]:
        """Convert binary mask to bounding box"""
        mask_np = mask.cpu().numpy()
        y_indices, x_indices = np.where(mask_np > 0.5)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            return (0, 0, 1, 1)  # Default small bbox
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
    
    def _extract_features(self, image: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract appearance features for tracking"""
        try:
            # Get image features
            features = self.sam2_model.encode_frame(image)
            
            # Mask the features
            mask_expanded = F.interpolate(mask, size=features.shape[-2:], mode='bilinear')
            masked_features = features * mask_expanded
            
            # Global average pooling
            pooled_features = F.adaptive_avg_pool2d(masked_features, (1, 1))
            pooled_features = pooled_features.flatten(1)
            
            return pooled_features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def _update_tracks(self, detections: List[Detection], frame_id: int) -> Dict[str, Any]:
        """
        Update tracking with new detections
        
        Args:
            detections: List of detections in current frame
            frame_id: Current frame ID
            
        Returns:
            Dictionary with tracking results
        """
        if not self.active_tracks and detections:
            # Initialize tracks for first detections
            return self._initialize_tracks(detections, frame_id)
        
        # Match detections to existing tracks
        matches, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        track_results = {'matches': {}, 'new_tracks': [], 'lost_tracks': []}
        
        for detection_idx, track_idx in matches:
            detection = detections[detection_idx]
            track_id = list(self.active_tracks.keys())[track_idx]
            
            self._update_track(track_id, detection, frame_id)
            track_results['matches'][track_id] = detection
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = list(self.active_tracks.keys())[track_idx]
            track = self.active_tracks[track_id]
            track.consecutive_invisible_count += 1
            track.last_seen = frame_id
            track_results['lost_tracks'].append(track_id)
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            new_track_id = self._create_track(detection, frame_id)
            track_results['new_tracks'].append(new_track_id)
        
        return track_results
    
    def _initialize_tracks(self, detections: List[Detection], frame_id: int) -> Dict[str, Any]:
        """Initialize tracks for first frame"""
        track_results = {'matches': {}, 'new_tracks': [], 'lost_tracks': []}
        
        for detection in detections:
            track_id = self._create_track(detection, frame_id)
            track_results['new_tracks'].append(track_id)
        
        return track_results
    
    def _match_detections_to_tracks(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU and appearance similarity
        
        Args:
            detections: List of detections
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if not detections or not self.active_tracks:
            return [], list(range(len(detections))), list(range(len(self.active_tracks)))
        
        # Build cost matrix
        track_ids = list(self.active_tracks.keys())
        cost_matrix = self._build_cost_matrix(detections, track_ids)
        
        # Solve assignment problem
        detection_indices = list(range(len(detections)))
        track_indices = list(range(len(track_ids)))
        
        if len(cost_matrix) > 0:
            detection_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter valid matches
        matches = []
        unmatched_detections = set(detection_indices)
        unmatched_tracks = set(track_indices)
        
        for det_idx, track_idx in zip(detection_indices, track_indices):
            if track_idx < len(cost_matrix) and det_idx < len(cost_matrix[track_idx]):
                cost = cost_matrix[track_idx, det_idx]
                if cost < self.config.max_distance:
                    matches.append((det_idx, track_idx))
                    unmatched_detections.discard(det_idx)
                    unmatched_tracks.discard(track_idx)
        
        return matches, list(unmatched_detections), list(unmatched_tracks)
    
    def _build_cost_matrix(self, detections: List[Detection], track_ids: List[str]) -> np.ndarray:
        """
        Build cost matrix for assignment problem
        
        Args:
            detections: Current frame detections
            track_ids: Active track IDs
            
        Returns:
            Cost matrix [num_tracks, num_detections]
        """
        num_tracks = len(track_ids)
        num_detections = len(detections)
        cost_matrix = np.full((num_tracks, num_detections), np.inf)
        
        for i, track_id in enumerate(track_ids):
            track = self.active_tracks[track_id]
            
            for j, detection in enumerate(detections):
                # Compute cost components
                bbox_cost = self._bbox_cost(track, detection)
                appearance_cost = self._appearance_cost(track, detection)
                confidence_cost = self._confidence_cost(track, detection)
                
                # Weighted combination
                total_cost = (0.4 * bbox_cost + 
                            0.4 * appearance_cost + 
                            0.2 * confidence_cost)
                
                cost_matrix[i, j] = total_cost
        
        return cost_matrix
    
    def _bbox_cost(self, track: Track, detection: Detection) -> float:
        """Compute bounding box matching cost"""
        if not track.bbox_history:
            return 1.0
        
        last_bbox = track.bbox_history[-1]
        det_bbox = detection.bbox
        
        # Compute IoU
        iou = self._compute_iou(last_bbox, det_bbox)
        
        # Convert IoU to cost (higher IoU = lower cost)
        return 1.0 - iou
    
    def _appearance_cost(self, track: Track, detection: Detection) -> float:
        """Compute appearance similarity cost"""
        if not track.history or detection.features is None:
            return 0.5  # Default cost when features unavailable
        
        # Use most recent features for comparison
        last_detection = track.history[-1]
        if last_detection.features is not None:
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                last_detection.features.flatten(),
                detection.features.flatten(),
                dim=0
            ).item()
            
            # Convert similarity to cost
            return 1.0 - max(0.0, similarity)
        
        return 0.5
    
    def _confidence_cost(self, track: Track, detection: Detection) -> float:
        """Compute confidence-based cost"""
        if not track.confidence_history:
            return 1.0 - detection.confidence
        
        # Prefer higher confidence detections
        avg_confidence = np.mean(track.confidence_history[-5:])  # Last 5 frames
        confidence_diff = abs(avg_confidence - detection.confidence)
        
        return confidence_diff
    
    def _compute_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union of two bounding boxes"""
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        # Calculate intersection
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _create_track(self, detection: Detection, frame_id: int) -> str:
        """
        Create new track for detection
        
        Args:
            detection: Detection object
            frame_id: Current frame ID
            
        Returns:
            New track ID
        """
        track_id = f"track_{self.next_track_id}"
        self.next_track_id += 1
        
        # Initialize track
        track = Track(
            track_id=track_id,
            history=deque(maxlen=10),  # Keep last 10 frames
            last_seen=frame_id,
            confidence_history=[detection.confidence],
            bbox_history=[detection.bbox]
        )
        
        # Add current detection to history
        track.history.append(detection)
        
        self.active_tracks[track_id] = track
        self.metrics['total_tracks_created'] += 1
        
        return track_id
    
    def _update_track(self, track_id: str, detection: Detection, frame_id: int) -> None:
        """Update existing track with new detection"""
        track = self.active_tracks[track_id]
        
        # Update track properties
        track.last_seen = frame_id
        track.age += 1
        track.total_visible_count += 1
        track.consecutive_invisible_count = 0
        
        # Update history
        track.history.append(detection)
        track.confidence_history.append(detection.confidence)
        track.bbox_history.append(detection.bbox)
        
        # Keep history size manageable
        if len(track.confidence_history) > 20:
            track.confidence_history = track.confidence_history[-10:]
        if len(track.bbox_history) > 20:
            track.bbox_history = track.bbox_history[-10:]
    
    def _cleanup_tracks(self) -> None:
        """Remove tracks that have disappeared for too long"""
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            if (track.consecutive_invisible_count > self.config.max_disappeared or
                track.last_seen < self.frame_count - self.config.max_disappeared):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            self.metrics['total_tracks_lost'] += 1
    
    def _update_metrics(self, track_results: Dict[str, Any], ground_truth: Optional[List[Dict]]) -> None:
        """Update tracking performance metrics"""
        self.metrics['total_frames'] += 1
        
        # Add implementation for metric computation based on ground truth
        # This would typically include tracking precision, recall, IDF1, etc.
    
    def get_tracking_results(self) -> Dict[str, Any]:
        """
        Get current tracking results
        
        Returns:
            Dictionary with current tracking state
        """
        results = {
            'frame_id': self.frame_count,
            'num_active_tracks': len(self.active_tracks),
            'tracks': {},
            'metrics': self.metrics.copy()
        }
        
        for track_id, track in self.active_tracks.items():
            track_info = {
                'track_id': track_id,
                'age': track.age,
                'confidence': np.mean(track.confidence_history[-5:]) if track.confidence_history else 0.0,
                'bbox': track.bbox_history[-1] if track.bbox_history else None,
                'is_active': track.is_active,
                'last_seen': track.last_seen,
                'total_visible_count': track.total_visible_count,
                'consecutive_invisible_count': track.consecutive_invisible_count
            }
            results['tracks'][track_id] = track_info
        
        return results
    
    def get_trajectory(self, track_id: str, num_frames: int = 10) -> Optional[List[Dict]]:
        """
        Get trajectory for specific track
        
        Args:
            track_id: Track identifier
            num_frames: Number of recent frames to return
            
        Returns:
            List of trajectory points with frame info
        """
        if track_id not in self.active_tracks:
            return None
        
        track = self.active_tracks[track_id]
        trajectory = []
        
        # Get recent history
        history_items = list(track.history)[-num_frames:]
        
        for detection in history_items:
            trajectory.append({
                'frame_id': track.last_seen - len(track.history) + len(trajectory),
                'bbox': detection.bbox,
                'confidence': detection.confidence,
                'center': (detection.bbox[0] + detection.bbox[2]//2, 
                          detection.bbox[1] + detection.bbox[3]//2)
            })
        
        return trajectory
    
    def reset(self) -> None:
        """Reset tracker state"""
        self.active_tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        
        # Reset metrics
        self.metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks_created': 0,
            'total_tracks_lost': 0,
            'tracking_accuracy': [],
            'precision_scores': [],
            'recall_scores': []
        }