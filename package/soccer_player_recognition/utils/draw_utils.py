"""
Drawing and visualization utilities for soccer player recognition.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import torch


class Visualizer:
    """Visualization utilities for soccer player recognition."""
    
    def __init__(self, colors: Optional[Dict[str, Tuple[int, int, int]]] = None):
        """
        Initialize visualizer.
        
        Args:
            colors: Custom color mapping for classes
        """
        self.default_colors = {
            'player': (0, 255, 0),      # Green
            'ball': (255, 0, 0),        # Blue
            'referee': (255, 165, 0),   # Orange
            'team1': (255, 0, 255),     # Magenta
            'team2': (0, 255, 255),     # Cyan
            'unknown': (128, 128, 128)  # Gray
        }
        
        if colors:
            self.default_colors.update(colors)
    
    def draw_bounding_box(self, 
                         image: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         label: str = "",
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2,
                         font_scale: float = 0.5,
                         font_thickness: int = 1) -> np.ndarray:
        """
        Draw bounding box on image.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            label: Label text
            color: Box color (B, G, R)
            thickness: Box line thickness
            font_scale: Font scale for label
            font_thickness: Font thickness for label
            
        Returns:
            Image with bounding box
        """
        x, y, w, h = bbox
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label
        if label:
            # Calculate label background size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale, font_thickness)[0]
            
            # Draw label background
            cv2.rectangle(image, (x, y - label_size[1] - 5), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       (255, 255, 255), font_thickness)
        
        return image
    
    def draw_multiple_boxes(self,
                           image: np.ndarray,
                           boxes: List[Tuple[int, int, int, int]],
                           labels: Optional[List[str]] = None,
                           colors: Optional[List[Tuple[int, int, int]]] = None,
                           **kwargs) -> np.ndarray:
        """
        Draw multiple bounding boxes on image.
        
        Args:
            image: Input image
            boxes: List of bounding boxes (x, y, w, h)
            labels: List of labels
            colors: List of colors
            **kwargs: Additional arguments for draw_bounding_box
            
        Returns:
            Image with all bounding boxes
        """
        result_image = image.copy()
        
        if labels is None:
            labels = ["" for _ in boxes]
        
        if colors is None:
            colors = [self.default_colors.get('unknown') for _ in boxes]
        
        for i, (box, label, color) in enumerate(zip(boxes, labels, colors)):
            result_image = self.draw_bounding_box(
                result_image, box, label, color, **kwargs
            )
        
        return result_image
    
    def draw_keypoints(self,
                      image: np.ndarray,
                      keypoints: List[Tuple[int, int]],
                      connections: Optional[List[Tuple[int, int]]] = None,
                      point_color: Tuple[int, int, int] = (0, 0, 255),
                      line_color: Tuple[int, int, int] = (255, 0, 0),
                      point_radius: int = 3,
                      thickness: int = 2) -> np.ndarray:
        """
        Draw keypoints and connections on image.
        
        Args:
            image: Input image
            keypoints: List of keypoint coordinates (x, y)
            connections: List of connections between keypoints
            point_color: Color for keypoints
            line_color: Color for connections
            point_radius: Radius of keypoint circles
            thickness: Line thickness
            
        Returns:
            Image with keypoints
        """
        result_image = image.copy()
        
        # Draw connections first
        if connections:
            for start_idx, end_idx in connections:
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    all(0 <= x < image.shape[1] and 0 <= y < image.shape[0] 
                        for x, y in [keypoints[start_idx], keypoints[end_idx]])):
                    cv2.line(result_image, keypoints[start_idx], keypoints[end_idx],
                           line_color, thickness)
        
        # Draw keypoints
        for x, y in keypoints:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(result_image, (int(x), int(y)), point_radius, 
                          point_color, -1)
        
        return result_image
    
    def draw_skeleton(self,
                     image: np.ndarray,
                     keypoints: List[Tuple[int, int]],
                     skeleton: List[List[int]],
                     point_color: Tuple[int, int, int] = (0, 255, 0),
                     line_color: Tuple[int, int, int] = (0, 0, 255),
                     point_radius: int = 4,
                     thickness: int = 2) -> np.ndarray:
        """
        Draw human skeleton on image.
        
        Args:
            image: Input image
            keypoints: List of keypoint coordinates
            skeleton: List of [parent_idx, child_idx] pairs
            point_color: Color for joints
            line_color: Color for bones
            point_radius: Radius of joint circles
            thickness: Line thickness
            
        Returns:
            Image with skeleton
        """
        return self.draw_keypoints(image, keypoints, skeleton, point_color, 
                                 line_color, point_radius, thickness)
    
    def draw_track(self,
                  image: np.ndarray,
                  track_history: List[Tuple[int, int]],
                  color: Tuple[int, int, int] = (255, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        Draw tracking trail on image.
        
        Args:
            image: Input image
            track_history: List of positions over time
            color: Color of the trail
            thickness: Line thickness
            
        Returns:
            Image with tracking trail
        """
        result_image = image.copy()
        
        if len(track_history) < 2:
            return result_image
        
        # Draw connecting lines
        for i in range(1, len(track_history)):
            cv2.line(result_image, track_history[i-1], track_history[i],
                   color, thickness)
        
        # Draw start and end points
        cv2.circle(result_image, track_history[0], 5, (0, 255, 0), -1)  # Green for start
        cv2.circle(result_image, track_history[-1], 5, (0, 0, 255), -1)  # Red for end
        
        return result_image
    
    def draw_field_lines(self, 
                        image: np.ndarray,
                        field_type: str = 'soccer') -> np.ndarray:
        """
        Draw soccer field lines on image.
        
        Args:
            image: Input image
            field_type: Type of field (currently only 'soccer' supported)
            
        Returns:
            Image with field lines
        """
        result_image = image.copy()
        h, w = image.shape[:2]
        
        # Field dimensions (proportional)
        field_color = (255, 255, 255)  # White
        line_thickness = max(2, min(h, w) // 200)
        
        if field_type == 'soccer':
            # Draw outer rectangle
            margin = min(h, w) // 8
            cv2.rectangle(result_image, (margin, margin), 
                         (w - margin, h - margin), field_color, line_thickness)
            
            # Draw center line
            cv2.line(result_image, (w // 2, margin), (w // 2, h - margin),
                    field_color, line_thickness)
            
            # Draw center circle
            center = (w // 2, h // 2)
            radius = min(h, w) // 8
            cv2.circle(result_image, center, radius, field_color, line_thickness)
            
            # Draw center dot
            cv2.circle(result_image, center, 2, field_color, -1)
            
            # Draw penalty areas
            penalty_area_height = h // 3
            penalty_area_width = w // 5
            penalty_margin = margin
            
            # Left penalty area
            cv2.rectangle(result_image, (margin, h // 2 - penalty_area_height // 2),
                         (margin + penalty_area_width, h // 2 + penalty_area_height // 2),
                         field_color, line_thickness)
            
            # Right penalty area
            cv2.rectangle(result_image, (w - margin - penalty_area_width, 
                                       h // 2 - penalty_area_height // 2),
                         (w - margin, h // 2 + penalty_area_height // 2),
                         field_color, line_thickness)
        
        return result_image
    
    def create_visualization_grid(self,
                                 images: List[np.ndarray],
                                 titles: Optional[List[str]] = None,
                                 layout: Optional[Tuple[int, int]] = None,
                                 figsize: Tuple[int, int] = (15, 10)) -> np.ndarray:
        """
        Create a grid visualization of multiple images.
        
        Args:
            images: List of images
            titles: Optional titles for each image
            layout: Grid layout (rows, cols)
            figsize: Figure size
            
        Returns:
            Combined image grid
        """
        n_images = len(images)
        
        if layout is None:
            # Auto-determine layout
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = layout
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Hide axes and plot images
        for i, ax in enumerate(axes):
            if i < n_images:
                # Convert BGR to RGB for matplotlib
                rgb_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                ax.imshow(rgb_image)
                
                if titles and i < len(titles):
                    ax.set_title(titles[i])
            ax.axis('off')
        
        # Hide extra subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        grid_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        grid_image = grid_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return grid_image
    
    def add_text_overlay(self,
                        image: np.ndarray,
                        text: str,
                        position: Tuple[int, int],
                        font_scale: float = 1.0,
                        color: Tuple[int, int, int] = (255, 255, 255),
                        thickness: int = 2,
                        background: bool = True) -> np.ndarray:
        """
        Add text overlay to image.
        
        Args:
            image: Input image
            text: Text to add
            position: Text position (x, y)
            font_scale: Font scale
            color: Text color
            thickness: Text thickness
            background: Whether to add background rectangle
            
        Returns:
            Image with text overlay
        """
        result_image = image.copy()
        
        # Get text size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                   font_scale, thickness)[0]
        
        # Add background if requested
        if background:
            background_margin = 5
            background_rect = [position[0] - background_margin,
                             position[1] - text_size[1] - background_margin,
                             position[0] + text_size[0] + background_margin,
                             position[1] + background_margin]
            
            cv2.rectangle(result_image, 
                         (background_rect[0], background_rect[1]),
                         (background_rect[2], background_rect[3]),
                         (0, 0, 0), -1)
        
        # Add text
        cv2.putText(result_image, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return result_image
    
    def visualize_detection_results(self,
                                   image: np.ndarray,
                                   detections: List[Dict[str, Any]],
                                   field_lines: bool = False) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            field_lines: Whether to draw field lines
            
        Returns:
            Image with visualized detections
        """
        result_image = image.copy()
        
        # Draw field lines if requested
        if field_lines:
            result_image = self.draw_field_lines(result_image)
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            label = detection.get('label', 'object')
            confidence = detection.get('confidence', 1.0)
            
            # Get color based on label
            color = self.default_colors.get(label, self.default_colors['unknown'])
            
            # Create label with confidence
            confidence_str = f"{confidence:.2f}"
            display_label = f"{label} {confidence_str}"
            
            # Draw bounding box
            result_image = self.draw_bounding_box(
                result_image, bbox, display_label, color
            )
        
        return result_image


def create_confusion_matrix_visualization(confusion_matrix: np.ndarray,
                                        class_names: List[str],
                                        title: str = "Confusion Matrix",
                                        save_path: Optional[str] = None) -> np.ndarray:
    """
    Create confusion matrix visualization.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Confusion matrix visualization image
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add class names
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Convert to numpy array
    fig = plt.gcf()
    fig.canvas.draw()
    confusion_matrix_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    confusion_matrix_img = confusion_matrix_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return confusion_matrix_img


def plot_training_metrics(metrics_history: Dict[str, List[float]],
                        save_path: Optional[str] = None,
                        title: str = "Training Metrics") -> np.ndarray:
    """
    Plot training metrics over epochs.
    
    Args:
        metrics_history: Dictionary of metric names and their values over epochs
        save_path: Optional path to save the plot
        title: Plot title
        
    Returns:
        Training metrics visualization image
    """
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(next(iter(metrics_history.values()))) + 1)
    
    # Plot each metric
    for metric_name, values in metrics_history.items():
        plt.plot(epochs, values, marker='o', label=metric_name)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Convert to numpy array
    fig = plt.gcf()
    fig.canvas.draw()
    metrics_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    metrics_img = metrics_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return metrics_img


def create_comparison_visualization(original: np.ndarray,
                                  processed: np.ndarray,
                                  title_original: str = "Original",
                                  title_processed: str = "Processed",
                                  save_path: Optional[str] = None) -> np.ndarray:
    """
    Create side-by-side comparison visualization.
    
    Args:
        original: Original image
        processed: Processed image
        title_original: Title for original image
        title_processed: Title for processed image
        save_path: Optional path to save the plot
        
    Returns:
        Comparison visualization image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(original_rgb)
    ax1.set_title(title_original)
    ax1.axis('off')
    
    ax2.imshow(processed_rgb)
    ax2.set_title(title_processed)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    comparison_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    comparison_img = comparison_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return comparison_img