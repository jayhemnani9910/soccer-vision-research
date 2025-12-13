# Soccer Player Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced computer vision system for soccer player recognition, tracking, and analysis using deep learning techniques. This system provides comprehensive capabilities for player detection, identification, classification, and real-time tracking from video streams.

## Features

- **🎯 Player Detection**: Advanced object detection using YOLOv8 and other state-of-the-art models
- **🏷️ Player Identification**: Deep learning-based player recognition using facial features and jersey analysis
- **📊 Player Classification**: Automatic categorization by team, position, and role
- **🔄 Real-time Tracking**: Multi-object tracking with stable ID assignment across frames
- **📈 Performance Analytics**: Detailed statistics and performance metrics
- **🎬 Video Analysis**: Support for live streams, recorded videos, and image sequences
- **🔧 Configurable**: Highly customizable configurations for different use cases

## System Architecture

```
soccer_player_recognition/
├── config/           # Configuration files
├── data/            # Dataset and sample data
├── demos/           # Demo applications and scripts
├── docs/            # Documentation
├── models/          # Pre-trained models and checkpoints
├── outputs/         # Generated outputs
│   ├── classification/
│   ├── detection/
│   ├── identification/
│   └── segmentation/
├── tests/           # Unit and integration tests
├── utils/           # Utility functions and helpers
└── soccer_player_recognition/  # Main package
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM for model inference

### Quick Install

1. **Clone the repository**
   ```bash
   git clone https://github.com/example/soccer-player-recognition.git
   cd soccer-player-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU support** (CUDA):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

### Optional Extras

Install additional components based on your needs:

```bash
# Development tools
pip install -e .[dev]

# GPU acceleration
pip install -e .[gpu]

# Player tracking features
pip install -e .[tracking]

# Visualization and monitoring
pip install -e .[visualization]
```

## Quick Start

### Command Line Interface

```bash
# Run player detection on a video
soccer-detect --input video.mp4 --output results/ --model yolov8n

# Run complete recognition pipeline
soccer-recognition --input match.mp4 --mode full --output results/

# Interactive identification
soccer-identify --input image.jpg --gallery player_images/
```

### Python API

```python
from soccer_player_recognition import PlayerRecognizer

# Initialize the recognizer
recognizer = PlayerRecognizer(
    detection_model='yolov8n',
    identification_model='resnet50',
    device='cuda'
)

# Process video
results = recognizer.process_video(
    input_path='match.mp4',
    output_path='results/',
    save_visualizations=True
)

# Access results
print(f"Detected {len(results.detections)} players")
print(f"Identified {len(results.identifications)} unique players")
```

### Jupyter Notebook

```python
# Load sample data and run analysis
from soccer_player_recognition.demos.demo_analysis import DemoAnalysis

demo = DemoAnalysis()
demo.load_sample_data()
results = demo.run_full_pipeline()
demo.visualize_results()
```

## Configuration

The system uses YAML configuration files for flexible customization:

```yaml
# config/detection_config.yaml
model:
  name: "yolov8n"
  confidence_threshold: 0.5
  iou_threshold: 0.45

preprocessing:
  input_size: [640, 640]
  normalize: true
  augment: true

# config/identification_config.yaml
identification:
  method: "face_recognition"
  gallery_path: "data/player_gallery"
  threshold: 0.6
  use_ensemble: true
```

## Data Preparation

### Dataset Structure

```
data/
├── raw/
│   ├── images/
│   ├── videos/
│   └── annotations/
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── player_gallery/
    ├── player_001/
    ├── player_002/
    └── ...
```

### Creating Player Gallery

```python
from soccer_player_recognition.utils.gallery_builder import GalleryBuilder

builder = GalleryBuilder()
builder.add_player_images('player_001', ['img1.jpg', 'img2.jpg'])
builder.build_gallery('data/player_gallery')
```

## Model Training

### Detection Models

```bash
# Train custom YOLO model
python scripts/train_detection.py \
    --data data/soccer_dataset.yaml \
    --model yolov8n \
    --epochs 100 \
    --batch-size 16
```

### Identification Models

```bash
# Train player identification model
python scripts/train_identification.py \
    --data data/player_annotations.json \
    --model resnet50 \
    --freeze_backbone \
    --epochs 50
```

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Detection**: mAP, Precision, Recall, F1-Score
- **Identification**: Accuracy, Top-K Accuracy, ROC-AUC
- **Tracking**: IDF1, MOTA, MOTP, IDF1
- **Classification**: Class-wise accuracy, confusion matrix

## API Reference

### Core Classes

- `PlayerRecognizer`: Main recognition system
- `DetectionEngine`: Player detection module
- `IdentificationEngine`: Player identification module
- `TrackingEngine`: Multi-object tracking
- `ClassificationEngine`: Player classification
- `VideoProcessor`: Video processing utilities

### Key Methods

```python
# Detection
results = recognizer.detect_players(image, confidence=0.5)

# Identification
player_ids = recognizer.identify_players(detections, gallery)

# Tracking
tracks = recognizer.track_players(detections, frame_id)

# Classification
positions = recognizer.classify_players(detections)
```

## Examples

### Basic Usage

```python
import cv2
from soccer_player_recognition import PlayerRecognizer

# Load image
image = cv2.imread('soccer_field.jpg')

# Initialize recognizer
recognizer = PlayerRecognizer()

# Detect players
detections = recognizer.detect_players(image)

# Visualize results
for detection in detections:
    bbox = detection.bbox
    label = detection.label
    confidence = detection.confidence
    cv2.rectangle(image, bbox, (0, 255, 0), 2)
    cv2.putText(image, f'{label} {confidence:.2f}', 
                (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2)

cv2.imshow('Detection Results', image)
```

### Batch Processing

```python
from pathlib import Path
from soccer_player_recognition import BatchProcessor

processor = BatchProcessor()
results = processor.process_directory(
    input_dir='data/videos/',
    output_dir='results/',
    recursive=True
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/example/soccer-player-recognition.git

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Code formatting
black soccer_player_recognition/
flake8 soccer_player_recognition/
```

### Adding New Models

1. Create model class in appropriate module
2. Implement required interface methods
3. Add configuration options
4. Update documentation and tests

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size
- Use smaller model (yolov8n vs yolov8x)
- Enable gradient checkpointing

**Low Detection Accuracy**
- Check input image quality
- Adjust confidence thresholds
- Verify model compatibility

**Slow Inference**
- Enable mixed precision training
- Use GPU acceleration
- Optimize preprocessing pipeline

### Performance Optimization

```python
# Enable optimizations
recognizer.enable_tensorrt()  # For NVIDIA GPUs
recognizer.enable_onnx()      # For cross-platform deployment
recognizer.enable_quantization()  # For mobile deployment
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 team for object detection framework
- Hugging Face for transformer models
- OpenCV community for computer vision tools
- PyTorch team for deep learning framework

## Citation

```bibtex
@software{soccer_player_recognition,
  title={Soccer Player Recognition System},
  author={AI Development Team},
  year={2024},
  url={https://github.com/example/soccer-player-recognition}
}
```

## Contact

- **Email**: ai@example.com
- **Issues**: [GitHub Issues](https://github.com/example/soccer-player-recognition/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/soccer-player-recognition/discussions)

---

**Made with ⚽ by the AI Development Team**