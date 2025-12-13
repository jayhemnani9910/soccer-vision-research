# API Reference

## Overview

This document provides comprehensive API documentation for the Soccer Player Recognition System, including all classes, methods, and utilities.

## Core Classes

### PlayerRecognizer

The main class that integrates all models for comprehensive player recognition.

```python
class PlayerRecognizer:
    def __init__(
        self, 
        config_path: Optional[str] = None,
        device: str = "auto",
        enable_all_models: bool = True,
        memory_efficient: bool = True
    )
```

#### Constructor Parameters

- `config_path` (Optional[str]): Path to configuration file. If None, uses default config.
- `device` (str): Device for computation. Options: 'auto', 'cpu', 'cuda', 'mps'. Default: 'auto'.
- `enable_all_models` (bool): Whether to initialize all available models. Default: True.
- `memory_efficient` (bool): Whether to use memory-efficient loading. Default: True.

#### Methods

##### detect_players()

```python
def detect_players(
    self,
    images: Union[str, np.ndarray, List[str], List[np.ndarray]],
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.5,
    max_detections: int = 50
) -> Union[DetectionResult, List[DetectionResult]]
```

Detects players, balls, and referees in images.

**Parameters:**
- `images`: Input images (file paths, numpy arrays, or lists)
- `confidence_threshold` (float): Minimum confidence for detections. Range: [0.0, 1.0]. Default: 0.7
- `nms_threshold` (float): Non-maximum suppression threshold. Range: [0.0, 1.0]. Default: 0.5
- `max_detections` (int): Maximum number of detections per image. Default: 50

**Returns:**
- `DetectionResult` or `List[DetectionResult]`: Detection results for single image or batch

**Raises:**
- `RuntimeError`: If RF-DETR model is not available

##### segment_players()

```python
def segment_players(
    self,
    images: Union[np.ndarray, List[np.ndarray]],
    prompts: Optional[List[List[Dict]]] = None,
    track_objects: bool = True
) -> Union[SegmentationResult, List[SegmentationResult]]
```

Segments and tracks players in video frames.

**Parameters:**
- `images`: Input video frames as numpy arrays
- `prompts` (Optional[List[List[Dict]]]): Optional segmentation prompts (points, boxes, etc.)
- `track_objects` (bool): Whether to enable object tracking. Default: True

**Returns:**
- `SegmentationResult` or `List[SegmentationResult]`: Segmentation results with masks and tracking

**Raises:**
- `RuntimeError`: If SAM2 model is not available

##### identify_players()

```python
def identify_players(
    self,
    images: Union[str, np.ndarray, List[str], List[np.ndarray]],
    player_candidates: List[str],
    team_context: Optional[str] = None,
    use_siglip: bool = True,
    use_resnet: bool = False
) -> Union[IdentificationResult, List[IdentificationResult]]
```

Identifies players using zero-shot (SigLIP) or trained (ResNet) approaches.

**Parameters:**
- `images`: Input images for identification
- `player_candidates` (List[str]): List of possible player names (for SigLIP)
- `team_context` (Optional[str]): Optional team context information
- `use_siglip` (bool): Whether to use SigLIP (zero-shot identification). Default: True
- `use_resnet` (bool): Whether to use ResNet (trained classification). Default: False

**Returns:**
- `IdentificationResult` or `List[IdentificationResult]`: Player names and confidence scores

**Raises:**
- `RuntimeError`: If no suitable identification model is available

##### analyze_scene()

```python
def analyze_scene(
    self,
    images: Union[str, np.ndarray, List[str], List[np.ndarray]],
    player_candidates: Optional[List[str]] = None,
    team_context: Optional[str] = None,
    analysis_type: str = "full",
    confidence_threshold: float = 0.7,
    fuse_results: bool = True
) -> Dict[str, Any]
```

Performs comprehensive scene analysis using multiple models.

**Parameters:**
- `images`: Input images for analysis
- `player_candidates` (Optional[List[str]]): List of possible player names for identification
- `team_context` (Optional[str]): Optional team context information
- `analysis_type` (str): Type of analysis. Options: 'detection', 'identification', 'segmentation', 'full'. Default: 'full'
- `confidence_threshold` (float): Minimum confidence threshold for all models. Default: 0.7
- `fuse_results` (bool): Whether to fuse results from multiple models. Default: True

**Returns:**
- `Dict[str, Any]`: Comprehensive analysis results

##### switch_model_config()

```python
def switch_model_config(self, model_name: str, new_config: Dict[str, Any]) -> bool
```

Switches configuration for a specific model at runtime.

**Parameters:**
- `model_name` (str): Name of the model to reconfigure ('rf_detr', 'sam2', 'siglip', 'resnet')
- `new_config` (Dict[str, Any]): New configuration parameters

**Returns:**
- `bool`: True if reconfiguration successful, False otherwise

##### get_model_status()

```python
def get_model_status(self) -> Dict[str, Any]
```

Gets status information for all models.

**Returns:**
- `Dict[str, Any]`: Model status and capabilities information

##### get_performance_stats()

```python
def get_performance_stats(self) -> Dict[str, Any]
```

Gets performance statistics.

**Returns:**
- `Dict[str, Any]`: Performance metrics and statistics

##### cleanup_memory()

```python
def cleanup_memory(self)
```

Cleans up memory and unloads models if in memory-efficient mode.

##### save_state()

```python
def save_state(self, output_path: str)
```

Saves current model states and configuration.

**Parameters:**
- `output_path` (str): Path to save state file

##### load_state()

```python
def load_state(self, state_path: str)
```

Loads PlayerRecognizer state from file.

**Parameters:**
- `state_path` (str): Path to state file

#### Factory Function

```python
def create_player_recognizer(
    config_path: Optional[str] = None,
    device: str = "auto",
    models_to_load: Optional[List[str]] = None,
    memory_efficient: bool = True
) -> PlayerRecognizer
```

Creates PlayerRecognizer instance with optional selective model loading.

## Model-Specific Engines

### DetectionEngine (RF-DETR Model)

```python
class RFDETRModel:
    def __init__(self, config: Dict[str, Any])
    def detect(self, images: List[np.ndarray]) -> DetectionResult
    def set_threshold(self, threshold: float)
    def get_model_info(self) -> Dict[str, Any]
```

**Capabilities:**
- Player detection with bounding boxes
- Ball and referee detection
- Multi-class object detection
- Real-time inference

### IdentificationEngine (SigLIP Model)

```python
class SigLIPPlayerIdentification:
    def __init__(self, config: SigLIPConfig)
    def identify(self, images: List[np.ndarray], player_names: List[str]) -> IdentificationResult
    def update_gallery(self, player_images: Dict[str, List[str]])
    def get_candidate_scores(self) -> np.ndarray
```

**Capabilities:**
- Zero-shot player identification
- Text-image matching
- Flexible player name recognition
- No training required for new players

### ClassificationEngine (ResNet Model)

```python
class ResNetModel:
    def __init__(self, num_players: int, model_name: str, device: str, pretrained: bool = True)
    def classify(self, images: List[np.ndarray]) -> ClassificationResult
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int)
    def save_checkpoint(self, path: str)
    def load_checkpoint(self, path: str)
```

**Capabilities:**
- Player classification (requires training)
- Feature extraction
- High accuracy for known players
- Fine-tuning support

### SegmentationEngine (SAM2 Model)

```python
class SAM2Model:
    def __init__(self, device: str, memory_mode: MemoryMode, max_memory_frames: int = 8, min_confidence: float = 0.7)
    def segment(self, images: List[np.ndarray], prompts: Optional[List] = None) -> SegmentationResult
    def track(self, images: List[np.ndarray]) -> TrackingResult
    def set_prompts(self, prompts: List[Dict])
```

**Capabilities:**
- Video segmentation with masks
- Object tracking across frames
- Occlusion handling
- Memory-efficient processing

## Data Types

### DetectionResult

```python
@dataclass
class DetectionResult:
    bbox: List[Tuple[int, int, int, int]]  # [x1, y1, x2, y2]
    labels: List[str]
    confidences: List[float]
    image_path: Optional[str] = None
    processing_time: float = 0.0
    num_detections: int = 0
```

### IdentificationResult

```python
@dataclass
class IdentificationResult:
    player_names: List[str]
    confidence_scores: List[float]
    method_used: str  # 'siglip' or 'resnet'
    image_path: Optional[str] = None
    candidate_matches: Optional[Dict[str, float]] = None
```

### SegmentationResult

```python
@dataclass
class SegmentationResult:
    masks: List[np.ndarray]
    object_ids: List[int]
    tracking_ids: List[int]  # For temporal consistency
    confidence_scores: List[float]
    frame_numbers: Optional[List[int]] = None
```

### ClassificationResult

```python
@dataclass
class ClassificationResult:
    predicted_classes: List[str]
    confidence_scores: List[float]
    features: Optional[List[np.ndarray]] = None
    top_k_predictions: Optional[List[List[Tuple[str, float]]]] = None
```

## Utility Functions

### Configuration

```python
def load_config(config_path: str) -> Config
def save_config(config: Config, output_path: str)
def validate_config(config: Config) -> bool
```

### Input/Output

```python
def validate_input(input_path: str) -> bool
def create_output_dirs(output_path: str)
def save_results(results: Dict, output_path: str, format: str = 'json')
def visualize_results(image: np.ndarray, results: Dict, save_path: Optional[str] = None) -> np.ndarray
```

### Performance

```python
def get_device_info() -> Dict[str, Any]
def benchmark_model(model, test_images: List[np.ndarray]) -> Dict[str, float]
def optimize_for_inference(model: nn.Module, use_tensorrt: bool = False) -> nn.Module
```

### Video Processing

```python
class VideoProcessor:
    def __init__(self, input_path: str, output_path: str)
    def extract_frames(self, max_frames: Optional[int] = None) -> List[np.ndarray]
    def process_video(self, recognizer: PlayerRecognizer, **kwargs) -> Dict
    def save_processed_video(self, frames: List[np.ndarray], results: List[Dict])
```

## Configuration Parameters

### System Configuration

```yaml
system:
  device: "cuda"              # Computation device
  device_id: 0                # GPU device ID
  mixed_precision: true       # Enable FP16
  gradient_checkpointing: true # Save memory
  max_memory_fraction: 0.8    # GPU memory usage limit
  compile_model: false        # Use PyTorch compile
```

### Model Configurations

#### RF-DETR (Detection)

```yaml
rf_detr:
  model_path: "models/pretrained/rf_detr/rf_detr_epoch_20.pth"
  input_size: [640, 640]
  confidence_threshold: 0.6
  nms_threshold: 0.5
  class_names: ["player", "ball", "referee", "goalkeeper"]
```

#### SigLIP (Identification)

```yaml
siglip:
  model_path: "models/pretrained/siglip/siglip-vit-so400m-14-e384.pt"
  input_size: [384, 384]
  temperature: 100.0
  batch_size: 32
```

#### ResNet (Classification)

```yaml
resnet:
  model_path: "models/pretrained/resnet/resnet50-0676ba61.pth"
  architecture: "resnet50"
  num_classes: 25
  input_size: [224, 224]
  dropout: 0.5
```

#### SAM2 (Segmentation)

```yaml
sam2:
  model_path: "models/pretrained/sam2/sam2_hiera_large.pt"
  input_size: [1024, 1024]
  confidence_threshold: 0.75
  use_multimask: true
  memory_mode: "selective"
```

## Error Handling

### Common Exceptions

- `RuntimeError`: Model not available or initialization failed
- `ValueError`: Invalid configuration parameters or input format
- `FileNotFoundError`: Model weights or configuration files not found
- `OutOfMemoryError`: GPU memory insufficient
- `ImportError`: Required dependencies not installed

### Error Recovery

```python
try:
    recognizer = PlayerRecognizer()
    results = recognizer.analyze_scene("image.jpg")
except RuntimeError as e:
    if "CUDA" in str(e):
        # Fall back to CPU
        recognizer = PlayerRecognizer(device="cpu")
    else:
        logger.error(f"Model error: {e}")
        raise
```

## Performance Considerations

### Memory Management

- Use `memory_efficient=True` for systems with limited RAM
- Call `cleanup_memory()` after processing large datasets
- Monitor GPU memory usage with `get_performance_stats()`

### Inference Speed

- Batch process multiple images when possible
- Use appropriate batch sizes for your hardware
- Enable mixed precision for faster inference on modern GPUs
- Consider model quantization for production deployment

### Accuracy vs Speed Trade-offs

- Lower confidence thresholds improve recall but may increase false positives
- Higher NMS thresholds reduce duplicate detections but may merge close objects
- Use ensemble methods for maximum accuracy but expect slower processing

## Examples

### Basic Usage

```python
from soccer_player_recognition import PlayerRecognizer

# Initialize with default settings
recognizer = PlayerRecognizer()

# Detect players in image
detection_results = recognizer.detect_players("soccer_field.jpg")
print(f"Detected {detection_results.num_detections} players")

# Full analysis with identification
results = recognizer.analyze_scene(
    "match_video.mp4",
    player_candidates=["Lionel Messi", "Cristiano Ronaldo", "Neymar"],
    analysis_type="full"
)
```

### Batch Processing

```python
import glob
from soccer_player_recognition import PlayerRecognizer

recognizer = PlayerRecognizer(memory_efficient=True)

# Process multiple images
image_paths = glob.glob("images/*.jpg")
all_results = []

for image_path in image_paths:
    try:
        results = recognizer.analyze_scene(image_path)
        all_results.append(results)
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

recognizer.cleanup_memory()
```

### Custom Configuration

```python
from soccer_player_recognition import PlayerRecognizer

# Use custom configuration
custom_config = {
    "rf_detr": {
        "confidence_threshold": 0.8,
        "nms_threshold": 0.4
    },
    "siglip": {
        "temperature": 50.0
    }
}

recognizer = PlayerRecognizer()
recognizer.switch_model_config("rf_detr", custom_config["rf_detr"])
```

## Version Information

- **Version**: 1.0.0
- **Python Compatibility**: 3.8+
- **PyTorch Compatibility**: 2.0+
- **GPU Support**: CUDA 11.8+, ROCm 5.4+

## Changelog

### v1.0.0
- Initial release with full API
- Support for RF-DETR, SAM2, SigLIP, and ResNet models
- Comprehensive error handling and logging
- Multi-platform deployment support