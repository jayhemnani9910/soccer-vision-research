"""
SAM2 Utils - Mask Generation and Tracking Utilities

This module provides utility functions for SAM2 video segmentation and tracking,
including mask generation, processing, visualization, evaluation metrics, and
data handling utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from scipy import ndimage
from sklearn.metrics import jaccard_score
import logging

logger = logging.getLogger(__name__)


class MaskProcessor:
    """Utility class for mask processing and generation"""
    
    @staticmethod
    def generate_random_masks(
        image_shape: Tuple[int, int],
        num_masks: int,
        min_area: float = 0.01,
        max_area: float = 0.3
    ) -> torch.Tensor:
        """
        Generate random masks for testing
        
        Args:
            image_shape: (height, width) of target image
            num_masks: Number of masks to generate
            min_area: Minimum mask area ratio
            max_area: Maximum mask area ratio
            
        Returns:
            Tensor of shape [num_masks, H, W]
        """
        H, W = image_shape
        masks = []
        
        for _ in range(num_masks):
            # Random center point
            center_x = np.random.randint(W // 4, 3 * W // 4)
            center_y = np.random.randint(H // 4, 3 * H // 4)
            
            # Random size
            area = np.random.uniform(min_area, max_area) * H * W
            radius = np.sqrt(area / np.pi)
            
            # Create circular mask
            y, x = np.ogrid[:H, :W]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Add some noise
            noise = np.random.random((H, W)) < 0.02
            mask = mask | noise
            
            masks.append(mask.astype(np.float32))
        
        return torch.from_numpy(np.stack(masks))
    
    @staticmethod
    def refine_mask(
        mask: torch.Tensor,
        iterations: int = 2,
        kernel_size: int = 3
    ) -> torch.Tensor:
        """
        Refine mask using morphological operations
        
        Args:
            mask: Input binary mask [1, H, W] or [H, W]
            iterations: Number of morphological iterations
            kernel_size: Size of morphological kernel
            
        Returns:
            Refined mask
        """
        mask_np = mask.cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)
        
        # Convert to uint8
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply morphological operations
        mask_refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        # Convert back to float tensor
        mask_refined = mask_refined.astype(np.float32) / 255.0
        mask_refined = torch.from_numpy(mask_refined)
        
        if mask.dim() == 3:
            mask_refined = mask_refined.unsqueeze(0)
        
        return mask_refined
    
    @staticmethod
    def mask_to_polygon(mask: torch.Tensor) -> List[List[int]]:
        """
        Convert binary mask to polygon coordinates
        
        Args:
            mask: Binary mask tensor [H, W]
            
        Returns:
            List of polygon coordinate lists
        """
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to list of points
            polygon = approx.flatten().tolist()
            if len(polygon) >= 6:  # At least 3 points (6 coordinates)
                polygons.append(polygon)
        
        return polygons
    
    @staticmethod
    def interpolate_masks(
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        num_intermediate: int = 2
    ) -> List[torch.Tensor]:
        """
        Interpolate between two masks
        
        Args:
            mask1: First mask tensor
            mask2: Second mask tensor
            num_intermediate: Number of intermediate masks
            
        Returns:
            List of interpolated masks
        """
        masks = [mask1]
        
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            interpolated = (1 - alpha) * mask1 + alpha * mask2
            interpolated = torch.clamp(interpolated, 0, 1)
            masks.append(interpolated)
        
        masks.append(mask2)
        return masks
    
    @staticmethod
    def smooth_mask(mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Apply Gaussian smoothing to mask
        
        Args:
            mask: Input mask tensor
            sigma: Gaussian sigma value
            
        Returns:
            Smoothed mask
        """
        # Apply gaussian blur
        mask_np = mask.cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)
        
        smoothed = cv2.GaussianBlur(mask_np, (0, 0), sigma)
        
        return torch.from_numpy(smoothed.astype(np.float32)).unsqueeze(0) if mask.dim() == 3 else torch.from_numpy(smoothed.astype(np.float32))


class TrackingEvaluator:
    """Utility class for tracking evaluation metrics"""
    
    @staticmethod
    def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """Compute Intersection over Union between two masks"""
        mask1_np = mask1.cpu().numpy().astype(bool)
        mask2_np = mask2.cpu().numpy().astype(bool)
        
        intersection = np.logical_and(mask1_np, mask2_np).sum()
        union = np.logical_or(mask1_np, mask2_np).sum()
        
        if union == 0:
            return 1.0  # Both masks are empty
        
        return intersection / union
    
    @staticmethod
    def compute_dice_coefficient(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """Compute Dice coefficient between two masks"""
        mask1_np = mask1.cpu().numpy().astype(bool)
        mask2_np = mask2.cpu().numpy().astype(bool)
        
        intersection = np.logical_and(mask1_np, mask2_np).sum()
        total_area = mask1_np.sum() + mask2_np.sum()
        
        if total_area == 0:
            return 1.0  # Both masks are empty
        
        return 2.0 * intersection / total_area
    
    @staticmethod
    def compute_tracking_accuracy(
        predicted_tracks: Dict[str, List[int]],
        ground_truth_tracks: Dict[str, List[int]],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute tracking accuracy metrics
        
        Args:
            predicted_tracks: Predicted track frame sequences
            ground_truth_tracks: Ground truth track frame sequences
            iou_threshold: IoU threshold for valid matches
            
        Returns:
            Dictionary with accuracy metrics
        """
        metrics = {}
        
        # Compute precision, recall, F1
        all_predicted = set()
        all_gt = set()
        
        for track_id, frames in predicted_tracks.items():
            all_predicted.update(frames)
        
        for track_id, frames in ground_truth_tracks.items():
            all_gt.update(frames)
        
        true_positives = len(all_predicted & all_gt)
        false_positives = len(all_predicted - all_gt)
        false_negatives = len(all_gt - all_predicted)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score
        
        return metrics
    
    @staticmethod
    def compute_trajectory_smoothness(trajectory: List[Dict]) -> float:
        """
        Compute trajectory smoothness metric
        
        Args:
            trajectory: List of trajectory points with 'center' coordinates
            
        Returns:
            Smoothness score (lower is smoother)
        """
        if len(trajectory) < 3:
            return 0.0
        
        smoothness_scores = []
        
        for i in range(1, len(trajectory) - 1):
            prev_center = np.array(trajectory[i-1]['center'])
            curr_center = np.array(trajectory[i]['center'])
            next_center = np.array(trajectory[i+1]['center'])
            
            # Compute acceleration-like metric
            v1 = curr_center - prev_center
            v2 = next_center - curr_center
            
            acceleration = v2 - v1
            smoothness = np.linalg.norm(acceleration)
            smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.0
    
    @staticmethod
    def evaluate_tracking_sequence(
        predictions: Dict[int, Dict],
        ground_truth: Dict[int, Dict],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate tracking performance over sequence
        
        Args:
            predictions: Frame-wise predictions {frame_id: {track_id: mask}}
            ground_truth: Frame-wise ground truth {frame_id: {track_id: mask}}
            iou_threshold: IoU threshold for valid matches
            
        Returns:
            Dictionary with evaluation metrics
        """
        ious = []
        track_lengths = []
        
        for frame_id in sorted(ground_truth.keys()):
            if frame_id not in predictions:
                continue
            
            pred_frame = predictions[frame_id]
            gt_frame = ground_truth[frame_id]
            
            # Compute IoU for each track
            for track_id, gt_mask in gt_frame.items():
                if track_id in pred_frame:
                    pred_mask = pred_frame[track_id]
                    iou = TrackingEvaluator.compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
        
        if not ious:
            return {'mean_iou': 0.0, 'iou_std': 0.0, 'coverage': 0.0}
        
        metrics = {
            'mean_iou': np.mean(ious),
            'iou_std': np.std(ious),
            'coverage': len(ious) / sum(len(gt_frame) for gt_frame in ground_truth.values())
        }
        
        return metrics


class VideoProcessor:
    """Utility class for video processing and frame handling"""
    
    @staticmethod
    def load_video_frames(
        video_path: Union[str, Path],
        num_frames: Optional[int] = None,
        start_frame: int = 0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load frames from video file
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to load
            start_frame: Starting frame index
            
        Returns:
            Tuple of (frames_array, video_info)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps
        }
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for i in range(frame_count if num_frames is None else min(num_frames, frame_count - start_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames could be loaded from video")
        
        return np.stack(frames), video_info
    
    @staticmethod
    def save_masks_to_video(
        frames: np.ndarray,
        masks_dict: Dict[str, torch.Tensor],
        output_path: Union[str, Path],
        fps: float = 30.0,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> None:
        """
        Save frames with masks overlaid as video
        
        Args:
            frames: Input frames array [N, H, W, 3]
            masks_dict: Dictionary of {track_id: mask_tensor}
            output_path: Output video path
            fps: Output video FPS
            colors: Color mapping for each track
        """
        if colors is None:
            colors = {f'track_{i}': tuple(np.random.randint(0, 256, 3)) for i in range(len(masks_dict))}
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        H, W = frames.shape[1:3]
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        
        for i, frame in enumerate(frames):
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Apply masks
            for track_id, mask in masks_dict.items():
                if i < len(mask):
                    mask_np = (mask[i].cpu().numpy() > 0.5).astype(np.uint8)
                    color = colors.get(track_id, (255, 255, 255))
                    
                    # Apply colored mask
                    colored_mask = np.zeros_like(frame_bgr)
                    colored_mask[:, :] = color
                    
                    # Blend mask with frame
                    alpha = 0.3
                    frame_bgr = cv2.addWeighted(frame_bgr, 1-alpha, colored_mask, alpha, 0)
                    
                    # Draw contours
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame_bgr, contours, -1, color, 2)
            
            out.write(frame_bgr)
        
        out.release()


class VisualizationUtils:
    """Utility class for visualization and plotting"""
    
    @staticmethod
    def plot_tracking_results(
        frames: np.ndarray,
        tracking_results: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot tracking results over multiple frames
        
        Args:
            frames: Input frames array
            tracking_results: Tracking results from tracker
            save_path: Optional path to save plot
            figsize: Figure size
        """
        num_frames = min(6, len(frames))  # Show up to 6 frames
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        tracks = tracking_results.get('tracks', {})
        
        for i in range(num_frames):
            ax = axes[i]
            
            # Show frame
            frame_idx = i * len(frames) // num_frames
            ax.imshow(frames[frame_idx])
            ax.set_title(f'Frame {frame_idx}')
            ax.axis('off')
            
            # Draw track bounding boxes
            for track_id, track_info in tracks.items():
                if track_info.get('bbox'):
                    bbox = track_info['bbox']
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add track ID
                    ax.text(bbox[0], bbox[1] - 10, track_id, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                           color='white', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_trajectory_analysis(trajectories: Dict[str, List[Dict]]) -> None:
        """
        Plot trajectory analysis for multiple objects
        
        Args:
            trajectories: Dictionary of {track_id: trajectory_points}
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        
        for i, (track_id, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) < 2:
                continue
            
            centers = [point['center'] for point in trajectory]
            centers = np.array(centers)
            
            ax1.plot(centers[:, 0], centers[:, 1], 'o-', 
                    color=colors[i], label=track_id, alpha=0.7)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Object Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot confidence over time
        for i, (track_id, trajectory) in enumerate(trajectories.items()):
            frame_ids = [point['frame_id'] for point in trajectory]
            confidences = [point['confidence'] for point in trajectory]
            
            ax2.plot(frame_ids, confidences, 'o-', 
                    color=colors[i], label=track_id, alpha=0.7)
        
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Track Confidence Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_segmentation_overlay(
        image: np.ndarray,
        masks: Dict[str, torch.Tensor],
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Create segmentation overlay on image
        
        Args:
            image: Input image [H, W, 3]
            masks: Dictionary of {id: mask_tensor}
            alpha: Transparency of masks
            
        Returns:
            Overlay image
        """
        overlay = image.copy()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
        
        for i, (mask_id, mask) in enumerate(masks.items()):
            mask_np = (mask.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
            color = (colors[i][:3] * 255).astype(np.uint8)
            
            # Apply color to mask regions
            colored_mask = np.zeros_like(overlay)
            colored_mask[mask_np] = color
            
            # Blend with original
            overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
        
        return overlay


class DataLoader:
    """Utility class for loading and preprocessing data"""
    
    @staticmethod
    def load_annotation_data(annotation_path: Union[str, Path]) -> Dict[int, Dict]:
        """
        Load annotation data from JSON file
        
        Args:
            annotation_path: Path to annotation file
            
        Returns:
            Dictionary mapping frame_id to annotations
        """
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        annotations = {}
        for frame_data in data.get('frames', []):
            frame_id = frame_data['frame_id']
            annotations[frame_id] = {}
            
            for obj in frame_data.get('objects', []):
                track_id = obj['track_id']
                annotations[frame_id][track_id] = {
                    'bbox': obj['bbox'],
                    'mask': np.array(obj['mask']),
                    'attributes': obj.get('attributes', {})
                }
        
        return annotations
    
    @staticmethod
    def preprocess_frame(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Preprocess frame for model input
        
        Args:
            image: Input image array [H, W, 3]
            target_size: Optional target size (H, W)
            
        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        # Resize if needed
        if target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    @staticmethod
    def create_prompts_from_annotations(
        annotations: Dict[int, Dict],
        frame_id: int,
        prompt_type: str = 'points'
    ) -> List[Dict]:
        """
        Create SAM2 prompts from ground truth annotations
        
        Args:
            annotations: Annotation dictionary
            frame_id: Current frame ID
            prompt_type: Type of prompts ('points', 'boxes', 'both')
            
        Returns:
            List of prompt dictionaries
        """
        if frame_id not in annotations:
            return []
        
        frame_annotations = annotations[frame_id]
        prompts = []
        
        for track_id, obj_data in frame_annotations.items():
            bbox = obj_data['bbox']
            
            prompt = {'id': track_id}
            
            if prompt_type in ['points', 'both']:
                # Create point prompts at bbox center
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
                prompt['point_coords'] = torch.tensor([[center_x, center_y]])
                prompt['point_labels'] = torch.tensor([1])  # Positive point
            
            if prompt_type in ['boxes', 'both']:
                # Create box prompt
                prompt['box_coords'] = torch.tensor(bbox)
            
            prompts.append(prompt)
        
        return prompts


def save_results_to_json(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save results to JSON file"""
    # Convert numpy/torch types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    converted_results = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=2)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    config_path = Path(config_path)
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # For YAML files, you would need PyYAML
        # For now, return empty dict
        logger.warning("YAML loading not implemented. Please use JSON config.")
        return {}


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)