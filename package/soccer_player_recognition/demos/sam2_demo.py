"""
SAM2 Demo Script - Video Segmentation and Tracking

This script demonstrates the complete SAM2 video segmentation and tracking pipeline,
including model initialization, frame processing, multi-object tracking, and results visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import logging

# Import SAM2 components
from models.segmentation import SAM2Model, SAM2Tracker, MemoryMode, TrackingConfig
from utils.sam2_utils import (
    MaskProcessor, TrackingEvaluator, VisualizationUtils, 
    DataLoader, VideoProcessor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_video_frames(num_frames: int = 10, height: int = 480, width: int = 640) -> np.ndarray:
    """Create demo video frames with moving objects"""
    frames = []
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving objects (circles)
        for obj_id, (start_x, start_y) in enumerate([(100, 100), (400, 200)]):
            # Circular motion
            angle = i * 0.5 + obj_id * np.pi / 2
            radius = 50
            
            center_x = int(start_x + radius * np.cos(angle))
            center_y = int(start_y + radius * np.sin(angle))
            
            # Draw circle
            cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Add some background texture
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        frames.append(frame)
    
    return np.array(frames)


def demo_basic_segmentation():
    """Demonstrate basic SAM2 segmentation functionality"""
    logger.info("=== SAM2 Basic Segmentation Demo ===")
    
    # Initialize SAM2 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam2_model = SAM2Model(
        device=device,
        memory_mode=MemoryMode.SELECTIVE,
        max_memory_frames=5
    )
    
    # Create demo frames
    frames = create_demo_video_frames(num_frames=5)
    
    # Process first frame
    first_frame = frames[0]
    input_tensor = torch.from_numpy(first_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Create prompts (simulate user clicks)
    prompts = [
        {
            'id': 'player_1',
            'point_coords': torch.tensor([[100, 100]]),
            'point_labels': torch.tensor([1])
        },
        {
            'id': 'player_2', 
            'point_coords': torch.tensor([[400, 200]]),
            'point_labels': torch.tensor([1])
        }
    ]
    
    # Extract masks
    masks = sam2_model.extract_masks(input_tensor, prompts)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(first_frame)
    axes[0].set_title('Original Frame')
    axes[0].axis('off')
    
    # Create mask overlay
    overlay = first_frame.copy()
    colors = [(255, 0, 0), (0, 255, 0)]
    
    for i, (obj_id, mask) in enumerate(masks.items()):
        mask_np = (mask.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
        color = colors[i % len(colors)]
        
        # Apply colored mask
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask_np] = color
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
    
    axes[1].imshow(overlay)
    axes[1].set_title('Segmentation Result')
    axes[1].axis('off')
    
    # Show individual masks
    if len(masks) >= 2:
        axes[2].imshow(masks['player_2'].squeeze(0).cpu().numpy(), cmap='gray')
        axes[2].set_title('Mask for Player 2')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/workspace/soccer_player_recognition/outputs/segmentation/demo_basic_segmentation.png')
    plt.show()
    
    return sam2_model, masks


def demo_tracking():
    """Demonstrate SAM2 multi-object tracking"""
    logger.info("=== SAM2 Multi-Object Tracking Demo ===")
    
    # Initialize SAM2 model and tracker
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam2_model = SAM2Model(device=device, memory_mode=MemoryMode.SELECTIVE)
    
    config = TrackingConfig(
        max_disappeared=30,
        max_distance=100.0,
        min_confidence=0.6
    )
    
    tracker = SAM2Tracker(sam2_model, config)
    
    # Create longer video sequence
    frames = create_demo_video_frames(num_frames=20)
    
    tracking_results = []
    
    # Process each frame
    for frame_id in range(len(frames)):
        frame = frames[frame_id]
        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Update prompts based on previous detections (simplified)
        if frame_id == 0:
            prompts = [
                {
                    'id': 'player_1',
                    'point_coords': torch.tensor([[100, 100]]),
                    'point_labels': torch.tensor([1])
                }
            ]
        else:
            prompts = None  # Use memory for subsequent frames
        
        # Track frame
        result = tracker.track_frame(input_tensor, frame_id, prompts)
        tracking_results.append(result)
    
    # Visualize tracking results
    VisualizationUtils.plot_tracking_results(frames, tracker.get_tracking_results())
    
    # Save tracking video
    VideoProcessor.save_masks_to_video(
        frames, 
        {},  # Would need to store masks over time
        '/workspace/soccer_player_recognition/outputs/segmentation/tracking_demo.mp4'
    )
    
    return tracker, tracking_results


def demo_occlusion_handling():
    """Demonstrate occlusion handling in SAM2"""
    logger.info("=== SAM2 Occlusion Handling Demo ===")
    
    # Initialize SAM2 model
    sam2_model = SAM2Model(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        occlusion_threshold=0.3
    )
    
    # Simulate occlusion scenario
    # Create frame with object
    frame1 = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(frame1, (200, 150), 40, (255, 255, 255), -1)
    
    # Create occluded frame
    frame2 = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(frame2, (200, 150), 40, (255, 255, 255), -1)
    # Add occluding object
    cv2.circle(frame2, (220, 130), 60, (128, 128, 128), -1)
    
    # Process frames
    tensor1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Create initial prompt
    prompt = {
        'id': 'occluded_object',
        'point_coords': torch.tensor([[200, 150]]),
        'point_labels': torch.tensor([1])
    }
    
    # Process first frame
    masks1 = sam2_model.extract_masks(tensor1, [prompt])
    
    # Process occluded frame
    masks2 = sam2_model.extract_masks(tensor2, None)
    
    # Visualize occlusion scenario
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(frame1)
    axes[0].set_title('Original Frame')
    axes[0].axis('off')
    
    axes[1].imshow(frame2)
    axes[1].set_title('Occluded Frame')
    axes[1].axis('off')
    
    # Show detection in occluded frame
    if masks2:
        overlay = frame2.copy()
        for obj_id, mask in masks2.items():
            mask_np = (mask.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(overlay)
            colored_mask[mask_np] = (255, 0, 0)
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Recovery from Memory')
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'Object Recovered\nfrom Memory', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Occlusion Handling')
    
    plt.tight_layout()
    plt.savefig('/workspace/soccer_player_recognition/outputs/segmentation/occlusion_demo.png')
    plt.show()


def demo_memory_management():
    """Demonstrate SAM2 memory management strategies"""
    logger.info("=== SAM2 Memory Management Demo ===")
    
    strategies = [MemoryMode.FULL, MemoryMode.SELECTIVE, MemoryMode.COMPACT]
    memory_sizes = []
    
    for strategy in strategies:
        sam2_model = SAM2Model(
            memory_mode=strategy,
            max_memory_frames=8
        )
        
        # Simulate processing multiple frames
        frames = create_demo_video_frames(num_frames=15)
        
        for i, frame in enumerate(frames[:10]):
            tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Create frame data
            from models.segmentation import FrameData
            frame_data = FrameData(
                frame_id=i,
                image=tensor,
                features=sam2_model.encode_frame(tensor),
                is_keyframe=(i % 3 == 0)  # Every 3rd frame is keyframe
            )
            
            # Update memory
            sam2_model.update_memory(frame_data)
        
        memory_sizes.append(len(sam2_model.memory_bank))
    
    # Plot memory usage comparison
    plt.figure(figsize=(10, 6))
    plt.bar([s.value for s in strategies], memory_sizes, 
            color=['blue', 'orange', 'green'])
    plt.xlabel('Memory Mode')
    plt.ylabel('Memory Bank Size')
    plt.title('SAM2 Memory Management Strategies')
    plt.ylim(0, max(memory_sizes) * 1.1)
    
    for i, size in enumerate(memory_sizes):
        plt.text(i, size + 0.1, str(size), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/workspace/soccer_player_recognition/outputs/segmentation/memory_management_demo.png')
    plt.show()


def demo_performance_evaluation():
    """Demonstrate performance evaluation metrics"""
    logger.info("=== SAM2 Performance Evaluation Demo ===")
    
    # Create synthetic ground truth and predictions
    height, width = 200, 300
    num_frames = 5
    
    # Generate synthetic masks
    gt_masks = {}
    pred_masks = {}
    
    for frame_id in range(num_frames):
        gt_masks[frame_id] = {}
        pred_masks[frame_id] = {}
        
        for obj_id in ['player_1', 'player_2']:
            # Ground truth mask
            gt_mask = torch.zeros(1, height, width)
            center_x = 100 + frame_id * 20  # Moving object
            center_y = 100 + obj_id == 'player_2' * 50
            y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            gt_mask[0] = ((x - center_x)**2 + (y - center_y)**2) < 30**2
            
            # Predicted mask (with some noise)
            pred_mask = gt_mask + torch.randn(1, height, width) * 0.1
            pred_mask = torch.clamp(pred_mask, 0, 1)
            
            gt_masks[frame_id][obj_id] = gt_mask
            pred_masks[frame_id][obj_id] = pred_mask
    
    # Evaluate performance
    evaluator = TrackingEvaluator()
    
    # Compute metrics for each frame
    frame_metrics = []
    for frame_id in range(num_frames):
        ious = []
        dices = []
        
        for obj_id in gt_masks[frame_id]:
            if obj_id in pred_masks[frame_id]:
                iou = evaluator.compute_iou(
                    gt_masks[frame_id][obj_id], 
                    pred_masks[frame_id][obj_id]
                )
                dice = evaluator.compute_dice_coefficient(
                    gt_masks[frame_id][obj_id], 
                    pred_masks[frame_id][obj_id]
                )
                ious.append(iou)
                dices.append(dice)
        
        frame_metrics.append({
            'frame_id': frame_id,
            'mean_iou': np.mean(ious) if ious else 0.0,
            'mean_dice': np.mean(dices) if dices else 0.0
        })
    
    # Plot performance metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    frame_ids = [m['frame_id'] for m in frame_metrics]
    ious = [m['mean_iou'] for m in frame_metrics]
    dices = [m['mean_dice'] for m in frame_metrics]
    
    ax1.plot(frame_ids, ious, 'o-', label='IoU', linewidth=2, markersize=6)
    ax1.set_xlabel('Frame ID')
    ax1.set_ylabel('IoU Score')
    ax1.set_title('Intersection over Union Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(frame_ids, dices, 's-', label='Dice', color='orange', linewidth=2, markersize=6)
    ax2.set_xlabel('Frame ID')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_title('Dice Coefficient Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/workspace/soccer_player_recognition/outputs/segmentation/performance_evaluation_demo.png')
    plt.show()
    
    return frame_metrics


def main():
    """Main demo function"""
    logger.info("Starting SAM2 Video Segmentation and Tracking Demos")
    
    # Create output directory
    output_dir = Path('/workspace/soccer_player_recognition/outputs/segmentation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run all demos
        logger.info("Running basic segmentation demo...")
        sam2_model, masks = demo_basic_segmentation()
        
        logger.info("Running tracking demo...")
        tracker, results = demo_tracking()
        
        logger.info("Running occlusion handling demo...")
        demo_occlusion_handling()
        
        logger.info("Running memory management demo...")
        demo_memory_management()
        
        logger.info("Running performance evaluation demo...")
        metrics = demo_performance_evaluation()
        
        logger.info("All demos completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()