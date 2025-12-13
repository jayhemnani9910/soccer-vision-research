"""
RF-DETR Soccer Player Detection Demo

This demo showcases the RF-DETR model for soccer player detection,
including players, goalkeepers, referees, and balls.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import logging
import argparse

# Import RF-DETR components
from models.detection.rf_detr_model import create_rf_detr_model, load_rf_detr_model
from models.detection.rf_detr_config import RFDETRSoccerConfigs
from utils.rf_detr_utils import load_and_preprocess_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFDETRDemo:
    """Demo class for RF-DETR soccer player detection."""
    
    def __init__(self, config_type: str = "balanced", model_path: str = None):
        """
        Initialize RF-DETR demo.
        
        Args:
            config_type: Configuration type ('balanced', 'real_time', 'high_accuracy', 'training')
            model_path: Path to pretrained model weights (optional)
        """
        self.config_type = config_type
        self.model_path = model_path
        
        # Create model
        logger.info(f"Creating RF-DETR model with {config_type} configuration...")
        self.model = create_rf_detr_model(config_type)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pretrained weights from {model_path}")
            self.model.load_pretrained_weights(model_path)
        else:
            logger.info("Using randomly initialized model weights")
        
        # Move to appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"RF-DETR model initialized on {self.device}")
    
    def detect_on_image(self, image_path: str, 
                       confidence_threshold: float = None,
                       nms_threshold: float = None,
                       save_results: bool = True) -> dict:
        """
        Perform detection on a single image.
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence threshold
            nms_threshold: NMS threshold
            save_results: Whether to save annotated image
            
        Returns:
            Detection results dictionary
        """
        # Load and preprocess image
        image_tensor, image_info, original_image = load_and_preprocess_image(
            image_path, self.model.config
        )
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Perform detection
        results = self.model.predict(
            image_tensor, 
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        
        # Save annotated image if requested
        if save_results:
            annotated_image = self._draw_annotations(original_image, results)
            output_path = self._get_output_path(image_path, "detection")
            cv2.imwrite(output_path, annotated_image)
            logger.info(f"Annotated image saved to {output_path}")
        
        return results
    
    def detect_on_batch(self, image_paths: list, 
                       confidence_threshold: float = None,
                       nms_threshold: float = None,
                       save_results: bool = True) -> list:
        """
        Perform detection on a batch of images.
        
        Args:
            image_paths: List of image file paths
            confidence_threshold: Minimum confidence threshold
            nms_threshold: NMS threshold
            save_results: Whether to save annotated images
            
        Returns:
            List of detection results
        """
        batch_results = []
        
        for image_path in image_paths:
            logger.info(f"Processing image: {image_path}")
            
            try:
                results = self.detect_on_image(
                    image_path, 
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                    save_results=save_results
                )
                batch_results.append(results)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                batch_results.append(None)
        
        return batch_results
    
    def detect_on_video(self, video_path: str, 
                       output_path: str = None,
                       confidence_threshold: float = None,
                       nms_threshold: float = None,
                       save_video: bool = True) -> dict:
        """
        Perform detection on video frames.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            confidence_threshold: Minimum confidence threshold
            nms_threshold: NMS threshold
            save_video: Whether to save output video
            
        Returns:
            Dictionary with video processing summary
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        output_video = None
        if save_video:
            if output_path is None:
                output_path = self._get_output_path(video_path, "video")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video frames
        frame_count = 0
        detection_counts = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform detection on current frame
                results = self.model.predict(
                    frame,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold
                )
                
                # Annotate frame
                annotated_frame = self._draw_annotations(frame, results)
                
                # Save frame if output video is set
                if output_video:
                    output_video.write(annotated_frame)
                
                # Log progress and statistics
                frame_count += 1
                detection_counts.append(results.get('total_detections', 0))
                
                if frame_count % 10 == 0:
                    logger.info(f"Processed frame {frame_count}/{total_frames}")
        
        finally:
            cap.release()
            if output_video:
                output_video.release()
        
        # Generate summary
        summary = {
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'fps': fps,
            'detection_stats': {
                'total_detections': sum(detection_counts),
                'avg_detections_per_frame': np.mean(detection_counts) if detection_counts else 0,
                'max_detections': max(detection_counts) if detection_counts else 0,
                'min_detections': min(detection_counts) if detection_counts else 0
            }
        }
        
        logger.info(f"Video processing complete. Processed {frame_count} frames.")
        return summary
    
    def _draw_annotations(self, image: np.ndarray, results: dict) -> np.ndarray:
        """
        Draw detection annotations on image.
        
        Args:
            image: Input image
            results: Detection results
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        detections = results.get('detections', [])
        
        # Define colors for each class
        colors = {
            'player': (0, 255, 0),      # Green
            'goalkeeper': (255, 0, 0),  # Blue
            'referee': (255, 255, 0),   # Yellow
            'ball': (0, 0, 255)         # Red
        }
        
        for detection in detections:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Skip background class
            if class_id == 0 or class_name == 'background':
                continue
            
            # Get color for this class
            color = colors.get(class_name, (128, 128, 128))  # Default gray
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw summary in corner
        summary_text = f"Detections: {results.get('total_detections', 0)}"
        cv2.putText(annotated, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return annotated
    
    def _get_output_path(self, input_path: str, suffix: str) -> str:
        """Generate output path for results."""
        input_path = Path(input_path)
        output_dir = Path("outputs/detection")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_name = f"{input_path.stem}_{suffix}_{self.config_type}{input_path.suffix}"
        return str(output_dir / output_name)
    
    def print_results(self, results: dict, image_path: str = ""):
        """
        Print detection results in a formatted way.
        
        Args:
            results: Detection results dictionary
            image_path: Path to processed image
        """
        print("\n" + "="*60)
        print(f"DETECTION RESULTS - {image_path}")
        print("="*60)
        
        detections = results.get('detections', [])
        total_detections = results.get('total_detections', 0)
        
        print(f"Total Detections: {total_detections}")
        print(f"Classes Detected: {results.get('classes_detected', {})}")
        print(f"Confidence Range: {results.get('min_confidence', 0):.3f} - {results.get('max_confidence', 0):.3f}")
        print("\nDetailed Detections:")
        print("-"*60)
        
        for i, detection in enumerate(detections, 1):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            print(f"{i:2d}. {class_name:10s} | "
                  f"Conf: {confidence:.3f} | "
                  f"BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        
        print("="*60)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return self.model.get_model_info()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="RF-DETR Soccer Player Detection Demo")
    
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image, video, or directory")
    parser.add_argument("--config", type=str, default="balanced",
                       choices=["balanced", "real_time", "high_accuracy", "training"],
                       help="Model configuration type")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pretrained model weights")
    parser.add_argument("--confidence", type=float, default=None,
                       help="Confidence threshold (overrides config)")
    parser.add_argument("--nms", type=float, default=None,
                       help="NMS threshold (overrides config)")
    parser.add_argument("--output_dir", type=str, default="outputs/detection",
                       help="Output directory for results")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save annotated images/videos")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = RFDETRDemo(config_type=args.config, model_path=args.model_path)
    
    # Print model info
    model_info = demo.get_model_info()
    print("\nModel Information:")
    print(f"Type: {model_info['model_type']}")
    print(f"Backbone: {model_info['config']['backbone']}")
    print(f"Classes: {model_info['class_names']}")
    print(f"Input Size: {model_info['config']['input_size']}")
    print(f"Device: {model_info['config']['device']}")
    print()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    try:
        if input_path.is_file():
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # Single image
                print(f"Processing single image: {args.input}")
                results = demo.detect_on_image(
                    args.input,
                    confidence_threshold=args.confidence,
                    nms_threshold=args.nms,
                    save_results=not args.no_save
                )
                demo.print_results(results, str(input_path))
                
            elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                # Video file
                print(f"Processing video: {args.input}")
                summary = demo.detect_on_video(
                    args.input,
                    confidence_threshold=args.confidence,
                    nms_threshold=args.nms,
                    save_video=not args.no_save
                )
                print(f"\nVideo processing complete:")
                print(f"Frames processed: {summary['processed_frames']}/{summary['total_frames']}")
                print(f"Total detections: {summary['detection_stats']['total_detections']}")
                print(f"Avg detections/frame: {summary['detection_stats']['avg_detections_per_frame']:.1f}")
                
            else:
                print(f"Unsupported file format: {input_path.suffix}")
                return
        
        elif input_path.is_dir():
            # Directory of images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_paths = [f for f in input_path.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            if not image_paths:
                print(f"No image files found in directory: {args.input}")
                return
            
            print(f"Processing {len(image_paths)} images from directory: {args.input}")
            results_list = demo.detect_on_batch(
                [str(p) for p in image_paths],
                confidence_threshold=args.confidence,
                nms_threshold=args.nms,
                save_results=not args.no_save
            )
            
            # Print summary
            total_detections = sum(r.get('total_detections', 0) for r in results_list if r is not None)
            print(f"\nBatch processing complete:")
            print(f"Images processed: {len([r for r in results_list if r is not None])}/{len(image_paths)}")
            print(f"Total detections: {total_detections}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()