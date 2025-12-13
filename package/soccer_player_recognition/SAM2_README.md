# SAM2 Video Segmentation and Tracking Implementation

This repository contains a complete implementation of SAM2 (Segment Anything Model 2) for video object segmentation and tracking. The implementation provides frame-by-frame segmentation, multi-object tracking, occlusion handling, and memory management for video sequences.

## 🎯 Features

### Core SAM2Model
- **Frame-by-frame segmentation** with temporal consistency
- **Prompt-based mask generation** supporting points, boxes, and other prompts
- **Multi-object segmentation** with confidence scoring
- **Memory management** with three modes: FULL, SELECTIVE, and COMPACT
- **Occlusion detection and recovery** using temporal memory
- **Automatic keyframe selection** for efficient processing

### SAM2Tracker Multi-Object Tracking
- **Multi-object tracking** with unique track IDs across frames
- **IoU and appearance-based matching** for detection association
- **Trajectory management** with configurable tracking parameters
- **Track confidence assessment** with quality metrics
- **Track state management** (active, disappeared, occluded states)
- **Performance metrics collection** and evaluation capabilities

### SAM2 Utils
- **Mask processing** with morphological refinement operations
- **Tracking evaluation metrics** (IoU, Dice coefficient, precision, recall, F1-score)
- **Video frame loading and processing** utilities
- **Visualization tools** for segmentation and tracking results
- **Data loading and preprocessing** helpers for various formats
- **Results export** to JSON and video formats

## 📁 Project Structure

```
soccer_player_recognition/
├── models/segmentation/
│   ├── __init__.py              # SAM2 module exports
│   ├── sam2_model.py            # Core SAM2 model implementation
│   └── sam2_tracker.py          # Multi-object tracking system
├── utils/
│   └── sam2_utils.py            # Utilities for mask generation and tracking
├── demos/
│   └── sam2_demo.py             # Comprehensive demo script
├── sam2_requirements.txt        # Python dependencies
└── test_sam2_implementation.py  # Structure verification test
```

## 🚀 Installation

### Requirements
- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- OpenCV 4.5+
- NumPy 1.21+
- Matplotlib 3.5+
- SciPy 1.7+
- scikit-learn 1.0+

### Install Dependencies
```bash
# Install from requirements file
pip install -r sam2_requirements.txt

# Or install minimum requirements
pip install torch torchvision opencv-python numpy matplotlib scipy scikit-learn
```

## 📖 Usage

### Basic Usage

```python
from models.segmentation import SAM2Model, SAM2Tracker, MemoryMode, TrackingConfig
from utils.sam2_utils import DataLoader, VisualizationUtils
import torch

# Initialize SAM2 model
sam2_model = SAM2Model(
    device='cuda',
    memory_mode=MemoryMode.SELECTIVE,
    max_memory_frames=8,
    min_confidence=0.7
)

# Initialize tracker with custom config
config = TrackingConfig(
    max_disappeared=30,
    max_distance=100.0,
    min_confidence=0.6,
    iou_threshold=0.3
)
tracker = SAM2Tracker(sam2_model, config)

# Process video frames
for frame_id, frame in enumerate(video_frames):
    # Convert frame to tensor
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # For first frame, provide prompts
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
    result = tracker.track_frame(frame_tensor, frame_id, prompts)
    
    # Get current tracking state
    tracking_state = tracker.get_tracking_results()
    
    print(f"Frame {frame_id}: {len(tracking_state['tracks'])} active tracks")
```

### Advanced Features

#### Custom Memory Management
```python
# Different memory modes for various scenarios
modes = {
    'high_accuracy': MemoryMode.FULL,      # Best accuracy, highest memory
    'balanced': MemoryMode.SELECTIVE,      # Balanced accuracy and memory
    'memory_efficient': MemoryMode.COMPACT) # Lowest memory usage
}

sam2_model = SAM2Model(memory_mode=modes['balanced'])
```

#### Occlusion Handling
```python
# SAM2 automatically handles occlusions using temporal memory
# When an object is occluded, the model tries to recover it from memory

# Manual occlusion recovery
mask = sam2_model._recover_from_memory('object_id')
if mask is not None:
    print("Successfully recovered object from memory")
```

#### Performance Evaluation
```python
from utils.sam2_utils import TrackingEvaluator

evaluator = TrackingEvaluator()

# Compute evaluation metrics
metrics = evaluator.evaluate_tracking_sequence(
    predictions=predicted_tracks,
    ground_truth=ground_truth_tracks,
    iou_threshold=0.5
)

print(f"Mean IoU: {metrics['mean_iou']:.3f}")
print(f"Coverage: {metrics['coverage']:.3f}")
```

#### Visualization
```python
from utils.sam2_utils import VisualizationUtils

# Plot tracking results over multiple frames
VisualizationUtils.plot_tracking_results(frames, tracking_results, 'output.png')

# Create segmentation overlay
overlay = VisualizationUtils.create_segmentation_overlay(frame, masks)

# Plot trajectory analysis
VisualizationUtils.plot_trajectory_analysis(trajectories)
```

## 🎬 Demo Script

Run the comprehensive demo to see all features:

```bash
python demos/sam2_demo.py
```

The demo includes:
- Basic segmentation demonstration
- Multi-object tracking in video sequences
- Occlusion handling scenarios  
- Memory management strategy comparison
- Performance evaluation metrics visualization
- Complete pipeline integration examples

## ⚙️ Configuration

### SAM2Model Parameters

```python
sam2_model = SAM2Model(
    model_path=None,                    # Path to pre-trained weights
    device='cuda',                      # 'cuda' or 'cpu'
    memory_mode=MemoryMode.SELECTIVE,   # Memory management strategy
    max_memory_frames=8,               # Maximum frames to store
    min_confidence=0.7,                # Confidence threshold
    occlusion_threshold=0.5,           # Occlusion detection threshold
    keyframe_threshold=0.8,            # Keyframe selection threshold
    learnable_prototypes=True          # Use learnable prototypes
)
```

### TrackingConfig Parameters

```python
config = TrackingConfig(
    max_disappeared=30,                # Max frames before track deletion
    max_distance=100.0,                # Max center distance for matching
    min_confidence=0.6,                # Min confidence for valid detection
    iou_threshold=0.3,                 # IoU threshold for matching
    appearance_threshold=0.7,          # Appearance similarity threshold
    bbox_smooth_factor=0.5             # Bounding box smoothing factor
)
```

## 📊 Memory Management Modes

### MemoryMode.FULL
- Stores all previous frames in memory
- Best accuracy for long sequences
- Highest memory usage

### MemoryMode.SELECTIVE (Recommended)
- Stores only key frames (visually distinct frames)
- Balanced between accuracy and memory efficiency
- Good for most applications

### MemoryMode.COMPACT
- Stores only essential frames (periodic sampling)
- Lowest memory usage
- Suitable for resource-constrained environments

## 🎯 Key Features Explained

### 1. Frame-by-Frame Segmentation
- Each frame is encoded using the image encoder
- Prompts guide the segmentation process
- Temporal consistency maintained through memory

### 2. Multi-Object Tracking
- Unique track IDs assigned to each object
- IoU and appearance-based matching between frames
- Configurable parameters for different scenarios

### 3. Occlusion Handling
- Automatic detection of occluded objects
- Recovery using temporal memory
- Configurable occlusion thresholds

### 4. Performance Optimization
- Automatic keyframe selection reduces computation
- Memory-efficient storage strategies
- Batch processing capabilities

## 📈 Performance Metrics

The implementation provides comprehensive tracking evaluation:

- **IoU (Intersection over Union)**: Measures overlap accuracy
- **Dice Coefficient**: Measures segmentation quality
- **Precision/Recall**: Classification performance
- **F1 Score**: Harmonic mean of precision and recall
- **Trajectory Smoothness**: Motion consistency metric

## 🔧 Testing

Verify the implementation structure:

```bash
python test_sam2_implementation.py
```

This will check:
- File structure completeness
- Code syntax validity
- Import structure
- Feature completeness

## 📝 Implementation Notes

### Architecture Overview
The SAM2 implementation follows a modular design:

1. **SAM2Model**: Core segmentation model with memory management
2. **SAM2Tracker**: Multi-object tracking with data association
3. **Utils**: Supporting utilities for processing and visualization

### Key Classes

#### SAM2Model
- `FrameData`: Frame information storage
- `TrackingState`: Object tracking state
- `MemoryMode`: Memory management strategies

#### SAM2Tracker  
- `Track`: Individual object track
- `Detection`: Detection from current frame
- `TrackingConfig`: Tracking parameters

#### Utils
- `MaskProcessor`: Mask generation and refinement
- `TrackingEvaluator`: Performance metrics
- `VisualizationUtils`: Results visualization
- `VideoProcessor`: Video handling utilities

## 🤝 Contributing

This implementation provides a solid foundation for video segmentation and tracking. Key areas for enhancement:

1. **Model Architecture**: Integration with actual SAM2 pre-trained models
2. **Performance Optimization**: GPU memory optimization for large videos
3. **Real-time Processing**: Frame rate optimization for real-time applications
4. **Advanced Features**: Handling of complex occlusion scenarios
5. **Evaluation**: Integration with standard video tracking benchmarks

## 📄 License

This implementation is provided for educational and research purposes. Please ensure compliance with any relevant licenses when using pre-trained models or datasets.

## 🙏 Acknowledgments

This implementation is inspired by the SAM2 (Segment Anything Model 2) paper and provides a comprehensive framework for video object segmentation and tracking research and development.