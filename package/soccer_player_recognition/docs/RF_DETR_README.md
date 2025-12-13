# RF-DETR Soccer Player Detection Implementation

This directory contains a complete implementation of the RF-DETR (Real-Time DEtection Transformer) model specifically designed for soccer player detection. The implementation supports detection of players, goalkeepers, referees, and balls in soccer images and videos.

## Features

- **Soccer-Specific Detection**: Optimized for 4 classes - players, goalkeepers, referees, and balls
- **Flexible Configuration**: Multiple pre-configured settings for different use cases
- **Real-time Processing**: Optimized for real-time inference on video streams
- **High Accuracy**: Configurable for high-accuracy offline analysis
- **Batch Processing**: Support for processing multiple images efficiently
- **Video Processing**: End-to-end video detection with output video generation
- **Comprehensive Utilities**: Preprocessing, postprocessing, and visualization tools

## Architecture Overview

The implementation consists of three main components:

### 1. Model Configuration (`models/detection/rf_detr_config.py`)
- `RFDETRConfig`: Main configuration class with soccer-specific parameters
- `RFDETRSoccerConfigs`: Predefined configurations for different scenarios
  - `BALANCED`: General-purpose configuration
  - `REAL_TIME`: Optimized for real-time processing
  - `HIGH_ACCURACY`: Maximum accuracy for offline analysis
  - `TRAINING`: Configuration for model training

### 2. Model Implementation (`models/detection/rf_detr_model.py`)
- `RFDETRModel`: Main model class with inference capabilities
- `RFDETRBackbone`: ResNet backbone for feature extraction
- `RFDETRTransformer`: Transformer encoder-decoder for object detection

### 3. Utility Functions (`utils/rf_detr_utils.py`)
- `RFDETRPreprocessor`: Image preprocessing and normalization
- `RFDETRPostprocessor`: Prediction postprocessing and NMS
- Utility functions for batch processing and common operations

## Quick Start

### Basic Usage

```python
import torch
from models.detection.rf_detr_model import create_rf_detr_model
import cv2

# Create model with balanced configuration
model = create_rf_detr_model("balanced")

# Load image
image = cv2.imread("soccer_image.jpg")

# Perform detection
results = model.predict(image, confidence_threshold=0.7)

# Print results
print(f"Detections: {results['total_detections']}")
for detection in results['detections']:
    print(f"{detection['class_name']}: {detection['confidence']:.2f}")
```

### Load Pretrained Weights

```python
from models.detection.rf_detr_model import load_rf_detr_model

# Load model with pretrained weights
model = load_rf_detr_model(
    model_path="weights/rf_detr_soccer.pth",
    config_type="balanced"
)

# Perform detection
results = model.predict("image.jpg")
```

### Custom Configuration

```python
from models.detection.rf_detr_config import RFDETRConfig
from models.detection.rf_detr_model import RFDETRModel

# Create custom configuration
config = RFDETRConfig(
    input_size=(800, 800),
    score_threshold=0.8,
    nms_threshold=0.4,
    max_detections=200
)

# Create model with custom config
model = RFDETRModel(config)
```

## Demo Usage

The included demo script provides a comprehensive interface for testing the RF-DETR model:

### Command Line Interface

```bash
# Process single image
python demos/rf_detr_demo.py --input soccer_image.jpg --config balanced

# Process video
python demos/rf_detr_demo.py --input soccer_video.mp4 --config real_time

# Process directory of images
python demos/rf_detr_demo.py --input soccer_images/ --config balanced

# Custom thresholds
python demos/rf_detr_demo.py --input image.jpg --confidence 0.8 --nms 0.4

# High accuracy mode
python demos/rf_detr_demo.py --input image.jpg --config high_accuracy
```

### Programmatic Usage

```python
from demos.rf_detr_demo import RFDETRDemo

# Create demo instance
demo = RFDETRDemo(config_type="real_time", model_path="weights/rf_detr_soccer.pth")

# Detect on single image
results = demo.detect_on_image("soccer_image.jpg", save_results=True)

# Detect on batch of images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_results = demo.detect_on_batch(image_paths, save_results=True)

# Detect on video
video_summary = demo.detect_on_video(
    "soccer_video.mp4", 
    output_path="annotated_video.mp4",
    save_video=True
)

# Print formatted results
demo.print_results(results, "soccer_image.jpg")
```

## Configuration Options

### Model Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `input_size` | Model input size (width, height) | (640, 640) | (416, 416), (640, 640), (800, 800) |
| `score_threshold` | Confidence threshold | 0.7 | 0.3 - 0.9 |
| `nms_threshold` | Non-maximum suppression threshold | 0.5 | 0.3 - 0.8 |
| `max_detections` | Maximum detections per image | 100 | 50 - 500 |
| `d_model` | Model dimension | 256 | 256, 512 |
| `nhead` | Number of attention heads | 8 | 8, 16 |

### Soccer Classes

The model detects the following 4 classes:

1. **Player** (ID: 1): Regular soccer players
2. **Goalkeeper** (ID: 2): Goalkeepers
3. **Referee** (ID: 3): Referees and umpires
4. **Ball** (ID: 4): Soccer ball

### Output Format

Detection results are returned as dictionaries with the following structure:

```python
{
    'detections': [
        {
            'bbox': [x1, y1, x2, y2],  # or [x, y, w, h] if xywh format
            'confidence': 0.95,
            'class_id': 1,
            'class_name': 'player'
        }
    ],
    'total_detections': 1,
    'classes_detected': {'player': 1},
    'max_confidence': 0.95,
    'min_confidence': 0.95
}
```

## Batch Processing

### Process Multiple Images

```python
from utils.rf_detr_utils import batch_process_images
from models.detection.rf_detr_model import create_rf_detr_model

# Create model
model = create_rf_detr_model("balanced")

# Load and preprocess batch
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_tensor, image_info_list, original_images = batch_process_images(
    image_paths, model.config
)

# Perform batch inference
batch_results = model.predict(batch_tensor, return_confidence=True)

# Process individual results
for i, results in enumerate(batch_results['batch_results']):
    print(f"Image {i+1}: {results['total_detections']} detections")
```

### Process Video Frames

```python
from utils.video_utils import load_video_frames
import cv2

# Load video frames
video_path = "soccer_video.mp4"
frames = load_video_frames(video_path, max_frames=100)  # Load first 100 frames

# Process frames
results_list = []
for frame in frames:
    results = model.predict(frame)
    results_list.append(results)
    print(f"Frame {len(results_list)}: {results['total_detections']} detections")

# Save annotated video
output_path = "annotated_video.mp4"
# Implementation depends on your video writing preferences
```

## Performance Optimization

### Real-time Processing

For real-time applications, use the `real_time` configuration:

```python
# Real-time optimized model
model = create_rf_detr_model("real_time")

# Lower resolution for faster processing
config = RFDETRConfig(input_size=(416, 416))
model = RFDETRModel(config)
```

### GPU Acceleration

```python
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model automatically uses available device
model = create_rf_detr_model("balanced")

# For manual device management
model.to(device)
```

### Memory Optimization

```python
# For large images or batches
config = RFDETRConfig(
    input_size=(416, 416),  # Smaller input size
    max_detections=50,      # Limit detections
    score_threshold=0.7     # Higher threshold for fewer detections
)

model = RFDETRModel(config)
```

## Visualization

### Draw Annotations

```python
import cv2
from demos.rf_detr_demo import RFDETRDemo

# Create demo instance
demo = RFDETRDemo("balanced")

# Detect and get annotated image
results = demo.detect_on_image("soccer_image.jpg", save_results=False)
annotated_image = demo._draw_annotations(original_image, results)

# Save annotated image
cv2.imwrite("annotated_image.jpg", annotated_image)
```

### Custom Visualization

```python
import cv2
import numpy as np

def draw_custom_annotations(image, results):
    """Custom annotation drawing function."""
    annotated = image.copy()
    
    # Define class colors
    colors = {
        'player': (0, 255, 0),
        'goalkeeper': (255, 0, 0),
        'referee': (255, 255, 0),
        'ball': (0, 0, 255)
    }
    
    for detection in results['detections']:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Skip low confidence
        if confidence < 0.5:
            continue
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        color = colors.get(class_name, (128, 128, 128))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}"
        cv2.putText(annotated, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return annotated

# Use custom visualization
results = model.predict("image.jpg")
custom_annotated = draw_custom_annotations(original_image, results)
```

## Error Handling

### Common Issues and Solutions

```python
# Handle missing weights
try:
    model.load_pretrained_weights("weights/rf_detr_soccer.pth")
except FileNotFoundError:
    print("Pretrained weights not found, using random initialization")

# Handle invalid images
try:
    results = model.predict("invalid_image.jpg")
except ValueError as e:
    print(f"Invalid image: {e}")

# Handle CUDA out of memory
if torch.cuda.is_available():
    try:
        results = model.predict(image)
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear GPU cache
            torch.cuda.empty_cache()
            # Use CPU instead
            model.to(torch.device("cpu"))
            results = model.predict(image)
        else:
            raise e
```

## Training Integration

For training the model on your soccer dataset:

```python
from models.detection.rf_detr_config import RFDETRSoccerConfigs
from models.detection.rf_detr_model import RFDETRModel

# Use training configuration
config = RFDETRSoccerConfigs.TRAINING
model = RFDETRModel(config)

# Training loop example (pseudo-code)
for epoch in range(config.num_epochs):
    for batch in train_loader:
        images, targets = batch
        
        # Forward pass
        predictions = model(images)
        
        # Compute loss (implement your loss function)
        loss = compute_detection_loss(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save checkpoint
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
```

## File Structure

```
soccer_player_recognition/
├── models/
│   └── detection/
│       ├── __init__.py
│       ├── rf_detr_model.py          # Main model implementation
│       └── rf_detr_config.py         # Configuration classes
├── utils/
│   ├── __init__.py
│   └── rf_detr_utils.py              # Pre/postprocessing utilities
├── demos/
│   └── rf_detr_demo.py               # Demo script and examples
└── outputs/
    └── detection/                    # Output directory for results
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+
- NumPy 1.20+
- Torchvision 0.11+

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update documentation for any new features

## License

This implementation follows the same license as the main soccer player recognition project.

---

For more detailed information about the model architecture and implementation details, refer to the source code documentation and comments in each module.