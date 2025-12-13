# Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues when using the Soccer Player Recognition System. It covers installation problems, runtime errors, performance issues, and model-specific troubleshooting.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Performance Issues](#performance-issues)
4. [Model-Specific Problems](#model-specific-problems)
5. [Configuration Issues](#configuration-issues)
6. [Data and Input Problems](#data-and-input-problems)
7. [Memory Issues](#memory-issues)
8. [Output and Visualization Issues](#output-and-visualization-issues)
9. [Debugging Tools](#debugging-tools)
10. [FAQ](#faq)

## Installation Issues

### CUDA/GPU Installation Problems

#### Problem: CUDA version mismatch
```
CUDA version mismatch: Expected 11.8, found 12.0
```

**Solution:**
```bash
# Check CUDA version
nvcc --version

# Install correct PyTorch version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Problem: CUDA out of memory during installation
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
```

**Solutions:**
1. **Reduce batch size:**
   ```python
   recognizer = PlayerRecognizer()
   recognizer._initialize_detection_model()
   # Add to detection config:
   detection_config = {
       "batch_size": 1,  # Instead of 2
       "memory_efficient": True
   }
   ```

2. **Use CPU installation:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Enable gradient checkpointing:**
   ```python
   # Enable in configuration
   config = {
       "system": {
           "gradient_checkpointing": True,
           "mixed_precision": True
       }
   }
   ```

### Dependency Conflicts

#### Problem: Package version conflicts
```
ERROR: pip's dependency resolver does not currently work with packages that
come from different versions of the same package.
```

**Solutions:**
1. **Create fresh virtual environment:**
   ```bash
   python -m venv soccer_env
   source soccer_env/bin/activate  # Linux/Mac
   soccer_env\Scripts\activate     # Windows
   
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

2. **Use conda environment:**
   ```bash
   conda create -n soccer python=3.9
   conda activate soccer
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

3. **Install with specific versions:**
   ```bash
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
   pip install ultralytics==8.0.0
   pip install transformers==4.30.0
   ```

### Missing System Dependencies

#### Problem: OpenCV installation fails
```
opencv-python requires a different Python: 3.11.0 not in '>=3.7'
```

**Solutions:**
1. **Install from conda:**
   ```bash
   conda install opencv
   ```

2. **Install headless version:**
   ```bash
   pip install opencv-python-headless==4.8.0.74
   ```

3. **Install system packages (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
   ```

#### Problem: Missing FFmpeg
```
ImportError: No module named 'imageio_ffmpeg'
```

**Solution:**
```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows (using conda):
conda install -c conda-forge ffmpeg

# Install Python wrapper
pip install imageio[ffmpeg]
```

## Runtime Errors

### Model Loading Errors

#### Problem: Model weights not found
```
FileNotFoundError: Model weights not found at models/pretrained/rf_detr/rf_detr_epoch_20.pth
```

**Solutions:**
1. **Download missing models:**
   ```bash
   python scripts/download_models.py
   
   # Or manually download:
   wget -O models/pretrained/rf_detr/rf_detr_epoch_20.pth \
     https://example.com/models/rf_detr_epoch_20.pth
   ```

2. **Check model directory structure:**
   ```bash
   ls -la models/pretrained/
   # Should contain:
   # rf_detr/
   # sam2/
   # siglip/
   # resnet/
   ```

3. **Update configuration paths:**
   ```yaml
   # config/model_config.yaml
   models:
     rf_detr:
       model_path: "/absolute/path/to/your/models/rf_detr_epoch_20.pth"
   ```

#### Problem: Invalid model weights
```
RuntimeError: Error(s) in loading state_dict for RFDETRModel:
    Missing key(s): 'backbone.0.weight', 'backbone.0.bias'
```

**Solutions:**
1. **Check model version compatibility:**
   ```python
   # Verify model version
   model_info = torch.load("model.pth", map_location='cpu')
   print(model_info.keys() if isinstance(model_info, dict) else model_info)
   ```

2. **Re-download compatible model:**
   ```bash
   # Check model compatibility
   python -c "import torch; print(torch.__version__)"
   
   # Download compatible version
   wget -O models/model.pth https://github.com/username/repo/releases/download/v1.0/model.pt
   ```

3. **Use model conversion tool:**
   ```python
   # Convert model format
   from soccer_player_recognition.utils import convert_model
   
   convert_model(
       input_path="old_model.pth",
       output_path="new_model.pth",
       source_framework="pytorch1.8",
       target_framework="pytorch2.0"
   )
   ```

### Import Errors

#### Problem: Module not found
```
ImportError: cannot import name 'PlayerRecognizer' from 'soccer_player_recognition'
```

**Solutions:**
1. **Check installation:**
   ```bash
   pip install -e .  # Install in development mode
   
   # Verify package is installed
   python -c "import soccer_player_recognition; print('OK')"
   ```

2. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   
   # Add current directory if needed
   sys.path.insert(0, '/path/to/soccer_player_recognition')
   ```

3. **Check __init__.py files:**
   ```bash
   # Ensure all __init__.py files exist
   find soccer_player_recognition -name "__init__.py"
   ```

#### Problem: CUDA extension compilation fails
```
error: Microsoft Visual C++ 14.0 is required
```

**Solutions:**
1. **Install Visual Studio Build Tools (Windows):**
   - Download Visual Studio Build Tools
   - Select "C++ build tools" workload
   - Restart system

2. **Use pre-compiled wheels:**
   ```bash
   pip install --only-binary=all package_name
   ```

3. **Install from conda-forge:**
   ```bash
   conda install -c conda-forge package_name
   ```

## Performance Issues

### Slow Inference

#### Problem: Very slow processing speed
```
FPS: 2.5 (Expected: 30+)
```

**Diagnosis:**
```python
from soccer_player_recognition.utils import get_performance_stats

# Get current performance stats
stats = recognizer.get_performance_stats()
print(f"Average processing time: {stats['avg_time']:.3f}s")
print(f"Models loaded: {stats['models_loaded']}")
print(f"Device: {stats['device']}")
```

**Solutions:**
1. **Enable GPU acceleration:**
   ```python
   recognizer = PlayerRecognizer(device="cuda")
   
   # Verify GPU is available
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU name: {torch.cuda.get_device_name(0)}")
   ```

2. **Optimize batch processing:**
   ```python
   # Process multiple images at once
   images = [img1, img2, img3, img4]
   results = recognizer.detect_players(images, batch_size=4)
   
   # Instead of processing individually
   # for img in images: recognizer.detect_players(img)
   ```

3. **Reduce input resolution:**
   ```python
   # Lower resolution for speed
   config = {
       "rf_detr": {
           "input_size": [320, 320]  # Instead of [640, 640]
       }
   }
   ```

4. **Use memory-efficient mode:**
   ```python
   recognizer = PlayerRecognizer(memory_efficient=True)
   ```

### Memory Leaks

#### Problem: Out of memory errors after processing multiple images
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
1. **Enable memory cleanup:**
   ```python
   recognizer = PlayerRecognizer(memory_efficient=True)
   
   # Manual cleanup after processing
   del results
   recognizer.cleanup_memory()
   torch.cuda.empty_cache()
   ```

2. **Use context manager:**
   ```python
   with PlayerRecognizer() as recognizer:
       results = recognizer.analyze_scene("image.jpg")
   # Memory automatically cleaned up
   ```

3. **Process smaller batches:**
   ```python
   # Instead of processing 100 images at once
   batch_size = 10
   for i in range(0, len(images), batch_size):
       batch = images[i:i+batch_size]
       results = recognizer.detect_players(batch)
       del results  # Explicit cleanup
   ```

4. **Monitor memory usage:**
   ```python
   import psutil
   import GPUtil
   
   def monitor_memory():
       # System memory
       memory = psutil.virtual_memory()
       print(f"System RAM: {memory.percent}% used")
       
       # GPU memory
       if torch.cuda.is_available():
           gpu = GPUtil.getGPUs()[0]
           print(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
   
   monitor_memory()
   ```

### Low Accuracy

#### Problem: Poor detection results
```
Detected 2 players (Expected: 11 players in a soccer team)
```

**Diagnosis:**
```python
# Check confidence thresholds
config = recognizer.get_model_status()
print(f"Detection threshold: {config['models']['rf_detr']['confidence_threshold']}")

# Test with lower threshold
results = recognizer.detect_players(
    "image.jpg",
    confidence_threshold=0.3  # Lower than default 0.7
)
print(f"With lower threshold: {results.num_detections} detections")
```

**Solutions:**
1. **Adjust confidence thresholds:**
   ```python
   # High sensitivity (more detections)
   results = recognizer.detect_players(
       image,
       confidence_threshold=0.3,
       nms_threshold=0.3
   )
   
   # Conservative (fewer but more accurate)
   results = recognizer.detect_players(
       image,
       confidence_threshold=0.9,
       nms_threshold=0.7
   )
   ```

2. **Check image quality:**
   ```python
   import cv2
   import numpy as np
   
   # Load and analyze image
   image = cv2.imread("image.jpg")
   height, width = image.shape[:2]
   print(f"Image size: {width}x{height}")
   
   # Check image brightness
   brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
   print(f"Average brightness: {brightness:.1f} (Good range: 50-200)")
   
   if brightness < 50:
       print("Image is too dark - improve lighting")
   elif brightness > 200:
       print("Image is overexposed - reduce lighting")
   ```

3. **Improve preprocessing:**
   ```python
   # Enhance image quality
   def enhance_image(image):
       # Convert to HSV
       hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       
       # Increase saturation
       hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
       
       # Convert back to BGR
       enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
       
       return enhanced
   
   enhanced_image = enhance_image(image)
   results = recognizer.detect_players(enhanced_image)
   ```

## Model-Specific Problems

### RF-DETR Issues

#### Problem: No detections on soccer images
```
DetectionResult(num_detections=0)
```

**Solutions:**
1. **Verify model is loaded:**
   ```python
   status = recognizer.get_model_status()
   if not status['models']['rf_detr']['loaded']:
       print("RF-DETR model not loaded")
       # Reinitialize
       recognizer._initialize_detection_model()
   ```

2. **Check class mappings:**
   ```python
   # Default classes: player, ball, referee, goalkeeper
   # Verify your images contain these objects
   
   # Use human-readable class names
   CLASS_NAMES = {0: 'player', 1: 'ball', 2: 'referee', 3: 'goalkeeper'}
   
   for i, (bbox, label, conf) in enumerate(zip(
       results.bbox, results.labels, results.confidences
   )):
       class_name = CLASS_NAMES.get(label, f"unknown_{label}")
       print(f"Detection {i}: {class_name} (confidence: {conf:.3f})")
   ```

3. **Adjust for soccer-specific scenarios:**
   ```python
   # Soccer-specific configuration
   soccer_config = {
       "confidence_threshold": 0.4,  # Lower for distant players
       "nms_threshold": 0.4,         # Looser NMS for crowded scenes
       "max_detections": 25          # Allow more detections
   }
   
   results = recognizer.detect_players(
       "soccer_field.jpg",
       **soccer_config
   )
   ```

#### Problem: Wrong object classes detected
```
Detected: "person" instead of "player"
```

**Solutions:**
1. **Fine-tune on soccer data:**
   ```bash
   python scripts/train_detection.py \
       --data data/soccer_dataset.yaml \
       --model rf_detr \
       --num_epochs 50 \
       --batch_size 8
   ```

2. **Use custom class mapping:**
   ```python
   # Map detected classes to soccer-specific terms
   def map_classes(detection_results):
       class_mapping = {
           'person': 'player',
           'sports ball': 'ball',
           'human': 'player'
       }
       
       mapped_results = []
       for detection in detection_results:
           if detection.label in class_mapping:
               detection.label = class_mapping[detection.label]
           mapped_results.append(detection)
       
       return mapped_results
   ```

### SAM2 Issues

#### Problem: Segmentation masks are inaccurate
```
SegmentationResult with poor mask quality
```

**Solutions:**
1. **Provide better prompts:**
   ```python
   # Use multiple prompt types
   prompts = [
       # Point prompts on player center
       {"type": "point", "x": 320, "y": 240, "label": 1},
       
       # Box prompt for better guidance
       {"type": "box", "x1": 280, "y1": 200, "x2": 360, "y2": 280},
       
       # Negative point to exclude background
       {"type": "point", "x": 400, "y": 100, "label": 0}
   ]
   
   results = model.segment_with_prompts(image, prompts)
   ```

2. **Use higher resolution:**
   ```python
   model = SAM2Model(
       input_size=[1024, 1024],  # Higher resolution
       memory_mode=MemoryMode.SELECTIVE
   )
   ```

3. **Enable multimask output:**
   ```python
   model = SAM2Model(
       use_multimask=True,
       min_confidence=0.8  # Higher confidence threshold
   )
   ```

#### Problem: Tracking across frames is inconsistent
```
Tracking IDs change unexpectedly between frames
```

**Solutions:**
1. **Increase memory frames:**
   ```python
   model = SAM2Model(
       max_memory_frames=12,  # More frames for better tracking
       memory_mode=MemoryMode.RECALL
   )
   ```

2. **Use stable reference points:**
   ```python
   # Always start tracking from first frame with good quality
   def get_first_frame_prompts(image):
       # Detect players first
       detections = recognizer.detect_players(image)
       
       # Create prompts from detections
       prompts = []
       for bbox, conf in zip(detections.bbox, detections.confidences):
           if conf > 0.8:  # Only high-confidence detections
               x1, y1, x2, y2 = bbox
               prompts.append({
                   "type": "box",
                   "x1": x1, "y1": y1, "x2": x2, "y2": y2
               })
       
       return prompts
   ```

### SigLIP Issues

#### Problem: Poor identification accuracy
```
Identified: "Unknown player" with low confidence
```

**Solutions:**
1. **Improve player descriptions:**
   ```python
   # Good description (detailed)
   good_desc = (
       "Lionel Messi in blue and red striped Barcelona jersey number 10, "
       "short dark hair, Argentine player"
   )
   
   # Bad description (vague)
   bad_desc = "Messi"
   ```

2. **Add more reference images:**
   ```python
   # Add multiple angles and conditions
   gallery_builder.add_player_images("messi", [
       "messi_front.jpg",
       "messi_side.jpg", 
       "messi_action.jpg",
       "messi_celebration.jpg",
       "messi_closeup.jpg"
   ])
   ```

3. **Adjust temperature parameter:**
   ```python
   # Lower temperature for more confident predictions
   config = SigLIPConfig(temperature=50.0)  # Default is 100.0
   
   # Higher temperature for more diverse predictions  
   config = SigLIPConfig(temperature=200.0)
   ```

4. **Use ensemble identification:**
   ```python
   # Combine multiple identification methods
   siglip_result = recognizer.identify_players(
       image, player_candidates, use_siglip=True
   )
   
   resnet_result = recognizer.identify_players(
       image, player_candidates, use_resnet=True
   )
   
   # Fuse results
   final_result = fuse_identification_results(
       [siglip_result, resnet_result],
       weights=[0.6, 0.4]
   )
   ```

### ResNet Issues

#### Problem: Model predicts wrong player
```
Actual: "Messi" | Predicted: "Ronaldo" | Confidence: 0.95
```

**Solutions:**
1. **Check training data balance:**
   ```python
   import os
   from collections import Counter
   
   # Check class distribution
   data_path = "data/players/train/"
   class_counts = Counter()
   
   for player_dir in os.listdir(data_path):
       if os.path.isdir(os.path.join(data_path, player_dir)):
           image_count = len(os.listdir(os.path.join(data_path, player_dir)))
           class_counts[player_dir] = image_count
   
   print("Class distribution:")
   for player, count in class_counts.items():
       print(f"{player}: {count} images")
   
   # Ensure balanced dataset
   min_count = min(class_counts.values())
   print(f"Consider balancing to ~{min_count} images per class")
   ```

2. **Improve data quality:**
   ```python
   # Remove mislabeled images
   def clean_dataset(data_path):
       for player_dir in os.listdir(data_path):
           player_path = os.path.join(data_path, player_dir)
           if os.path.isdir(player_path):
               # Remove images with wrong resolution
               for img_file in os.listdir(player_path):
                   if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                       img_path = os.path.join(player_path, img_file)
                       img = cv2.imread(img_path)
                       if img is None or img.shape[:2] < (224, 224):
                           os.remove(img_path)
                           print(f"Removed: {img_path}")
   
   clean_dataset("data/players/train/")
   ```

3. **Fine-tune model:**
   ```python
   # Use transfer learning
   model = ResNetModel(
       architecture="resnet50",
       num_classes=25,
       pretrained=True  # Start from ImageNet weights
   )
   
   # Freeze early layers
   for param in model.parameters():
       param.requires_grad = False
   
   # Unfreeze final layers
   for param in model.fc.parameters():
       param.requires_grad = True
   
   # Train with lower learning rate
   optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)
   ```

## Configuration Issues

### Invalid Configuration File

#### Problem: Configuration file parsing error
```
yaml.scanner.ScannerError: mapping values are not allowed in this context
```

**Solutions:**
1. **Validate YAML syntax:**
   ```bash
   # Install YAML validator
   pip install pyyaml
   
   # Test parsing
   python -c "
   import yaml
   with open('config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   print('Configuration valid')
   "
   ```

2. **Fix common YAML errors:**
   ```yaml
   # Wrong (tab characters)
   system:
       device: cuda
   
   # Correct (spaces)
   system:
     device: cuda
   
   # Wrong (missing quotes around special characters)
   name: "Team A vs Team B"
   
   # Wrong (colon in value)
   description: "Match: Team A vs Team B"
   ```

3. **Use configuration template:**
   ```python
   from soccer_player_recognition.utils.config_generator import generate_template
   
   # Generate configuration template
   generate_template("my_config.yaml")
   
   # Edit the generated file
   # Then load it
   recognizer = PlayerRecognizer(config_path="my_config.yaml")
   ```

### Device Configuration Problems

#### Problem: Device not available
```
RuntimeError: CUDA device not available
```

**Solutions:**
1. **Check available devices:**
   ```python
   import torch
   
   print("Available devices:")
   if torch.cuda.is_available():
       print(f"CUDA devices: {torch.cuda.device_count()}")
       for i in range(torch.cuda.device_count()):
           print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
   else:
       print("CUDA not available - will use CPU")
   
   # Use available device
   if torch.cuda.is_available():
       device = "cuda:0"
   else:
       device = "cpu"
   
   recognizer = PlayerRecognizer(device=device)
   ```

2. **Handle multiple GPUs:**
   ```python
   # Use specific GPU
   recognizer = PlayerRecognizer(device="cuda:1")  # Use second GPU
   
   # Or let system choose
   recognizer = PlayerRecognizer(device="auto")
   ```

3. **CPU fallback:**
   ```python
   try:
       recognizer = PlayerRecognizer(device="cuda")
   except RuntimeError as e:
       print(f"CUDA error: {e}")
       print("Falling back to CPU...")
       recognizer = PlayerRecognizer(device="cpu")
   ```

## Data and Input Problems

### Invalid Input Formats

#### Problem: Unsupported file format
```
ValueError: Unsupported image format: .webp
```

**Solutions:**
1. **Convert image format:**
   ```python
   from PIL import Image
   
   # Convert WebP to JPEG
   image = Image.open("image.webp")
   image.save("image.jpg", "JPEG")
   
   # Or use OpenCV
   import cv2
   image = cv2.imread("image.webp")
   cv2.imwrite("image.jpg", image)
   ```

2. **Add custom image loader:**
   ```python
   import imageio
   import numpy as np
   
   def load_custom_image(path):
       # Support for various formats
       if path.lower().endswith('.webp'):
           return imageio.imread(path)
       else:
           return cv2.imread(path)
   
   # Use custom loader
   image = load_custom_image("image.webp")
   results = recognizer.detect_players(image)
   ```

3. **Install additional dependencies:**
   ```bash
   pip install imageio imageio-ffmpeg
   pip install pillow-heif  # For HEIC/HEIF formats
   ```

### Video Processing Issues

#### Problem: Video file cannot be opened
```
cv2.error: OpenCV(4.8.0) .* videoio(FFMPEG): tag 'h264' is not supported
```

**Solutions:**
1. **Install FFmpeg codecs:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg libavcodec-extra
   
   # macOS
   brew install ffmpeg
   
   # Windows (conda)
   conda install -c conda-forge ffmpeg
   ```

2. **Convert video format:**
   ```bash
   # Convert to standard format
   ffmpeg -i input_video.avi -c:v libx264 -c:a aac output_video.mp4
   ```

3. **Extract frames for processing:**
   ```python
   import cv2
   
   def extract_frames(video_path, max_frames=None):
       cap = cv2.VideoCapture(video_path)
       frames = []
       
       frame_count = 0
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           frames.append(frame)
           frame_count += 1
           
           if max_frames and frame_count >= max_frames:
               break
       
       cap.release()
       return frames
   
   # Extract and process frames
   frames = extract_frames("video.mp4", max_frames=100)
   results = recognizer.analyze_scene(frames)
   ```

### Image Quality Issues

#### Problem: Blurry or low-resolution images
```
Low detection accuracy on poor quality images
```

**Solutions:**
1. **Image enhancement:**
   ```python
   import cv2
   import numpy as np
   
   def enhance_image_quality(image):
       # Denoise
       denoised = cv2.bilateralFilter(image, 9, 75, 75)
       
       # Sharpen
       kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
       sharpened = cv2.filter2D(denoised, -1, kernel)
       
       # Enhance contrast
       lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
       l, a, b = cv2.split(lab)
       
       # Apply CLAHE
       clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
       l = clahe.apply(l)
       
       enhanced = cv2.merge([l, a, b])
       enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
       
       return enhanced
   
   # Enhance image before processing
   enhanced_image = enhance_image_quality(image)
   results = recognizer.detect_players(enhanced_image)
   ```

2. **Skip poor quality images:**
   ```python
   def assess_image_quality(image):
       # Check resolution
       height, width = image.shape[:2]
       if min(height, width) < 224:
           return False, "Resolution too low"
       
       # Check sharpness (Laplacian variance)
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
       if sharpness < 100:
           return False, "Image too blurry"
       
       return True, "Good quality"
   
   # Check and filter images
   image = cv2.imread("image.jpg")
   is_good, reason = assess_image_quality(image)
   
   if is_good:
       results = recognizer.detect_players(image)
   else:
       print(f"Skipping poor quality image: {reason}")
   ```

## Memory Issues

### GPU Memory Problems

#### Problem: CUDA out of memory during inference
```
RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB
```

**Solutions:**
1. **Reduce batch size:**
   ```python
   # Process one image at a time
   results = recognizer.detect_players(images[0])
   
   # Or use smaller batch size
   for i in range(0, len(images), batch_size):
       batch = images[i:i+batch_size]
       results_batch = recognizer.detect_players(batch, batch_size=len(batch))
   ```

2. **Enable memory efficient loading:**
   ```python
   recognizer = PlayerRecognizer(memory_efficient=True)
   
   # Clear cache after each batch
   recognizer.cleanup_memory()
   torch.cuda.empty_cache()
   ```

3. **Use gradient checkpointing:**
   ```python
   config = {
       "system": {
           "gradient_checkpointing": True,
           "mixed_precision": True
       }
   }
   ```

4. **Monitor memory usage:**
   ```python
   def monitor_gpu_memory():
       if torch.cuda.is_available():
           allocated = torch.cuda.memory_allocated() / 1024**3
           reserved = torch.cuda.memory_reserved() / 1024**3
           print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
   
   monitor_gpu_memory()
   ```

### System Memory Issues

#### Problem: High RAM usage during batch processing
```
Process killed due to memory limit
```

**Solutions:**
1. **Process smaller batches:**
   ```python
   # Instead of processing 1000 images
   batch_size = 10
   for i in range(0, len(images), batch_size):
       batch = images[i:i+batch_size]
       results = recognizer.analyze_scene(batch)
       
       # Save results immediately
       save_batch_results(results, output_dir, i)
       
       # Clear memory
       del results, batch
   ```

2. **Use memory mapping:**
   ```python
   import mmap
   
   def process_with_memory_mapping(image_paths):
       results = []
       
       for path in image_paths:
           # Load image without keeping in memory
           with open(path, 'rb') as f:
               with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                   image = cv2.imdecode(np.frombuffer(mm.read(), dtype=np.uint8), -1)
                   
                   # Process image
                   result = recognizer.analyze_scene(image)
                   results.append(result)
       
       return results
   ```

3. **Stream processing:**
   ```python
   def stream_process_images(image_stream):
       results = []
       
       for image in image_stream:
           result = recognizer.analyze_scene(image)
           results.append(result)
           
           # Process and discard immediately
           yield result
           
           # Clear memory
           del result
       
       return results
   ```

## Output and Visualization Issues

### Incorrect Visualization

#### Problem: Bounding boxes are misaligned
```
Bounding boxes appear in wrong positions
```

**Solutions:**
1. **Check coordinate systems:**
   ```python
   # Ensure consistent coordinate system
   # OpenCV uses (x, y) with origin at top-left
   x, y, w, h = bbox
   
   # Convert to pixel coordinates
   x1, y1 = int(x), int(y)
   x2, y2 = int(x + w), int(y + h)
   
   # Draw bounding box
   cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
   ```

2. **Verify image dimensions:**
   ```python
   # Ensure detection and visualization use same image
   original_image = cv2.imread("image.jpg")
   height, width = original_image.shape[:2]
   
   print(f"Original image size: {width}x{height}")
   
   # Resize for detection if needed
   if width > 1024 or height > 1024:
       scale = 1024 / max(width, height)
       resized = cv2.resize(original_image, None, fx=scale, fy=scale)
   else:
       resized = original_image
       scale = 1.0
   
   # Run detection
   results = recognizer.detect_players(resized)
   
   # Scale coordinates back to original size
   for bbox in results.bbox:
       bbox[0] /= scale  # x1
       bbox[1] /= scale  # y1  
       bbox[2] /= scale  # x2
       bbox[3] /= scale  # y2
   ```

3. **Use visualization utilities:**
   ```python
   from soccer_player_recognition.utils import visualize_results
   
   # Use built-in visualization
   annotated_image = visualize_results(
       original_image, 
       results,
       save_path="output/annotated.jpg",
       show_labels=True,
       show_confidence=True
   )
   ```

### File Output Problems

#### Problem: Output files not saved
```
No output files generated
```

**Solutions:**
1. **Check output directory permissions:**
   ```python
   import os
   
   output_dir = "output/"
   
   # Create directory if it doesn't exist
   os.makedirs(output_dir, exist_ok=True)
   
   # Check write permissions
   if os.access(output_dir, os.W_OK):
       print("Output directory is writable")
   else:
       print("Cannot write to output directory")
       output_dir = "/tmp/output/"  # Use temp directory
       os.makedirs(output_dir, exist_ok=True)
   ```

2. **Use absolute paths:**
   ```python
   # Use absolute paths
   import os
   from pathlib import Path
   
   output_dir = Path(output_dir).absolute()
   output_path = output_dir / "results.json"
   
   # Ensure directory exists
   output_dir.mkdir(parents=True, exist_ok=True)
   
   # Save with full path
   save_results(results, str(output_path))
   ```

3. **Handle file name conflicts:**
   ```python
   from pathlib import Path
   
   def get_unique_filename(base_path):
       path = Path(base_path)
       counter = 1
      
       while path.exists():
           stem = path.stem
           suffix = path.suffix
           parent = path.parent
           path = parent / f"{stem}_{counter}{suffix}"
           counter += 1
      
       return str(path)
   
   # Generate unique filename
   output_path = get_unique_filename("output/results.json")
   save_results(results, output_path)
   ```

## Debugging Tools

### Performance Profiler

```python
from soccer_player_recognition.utils import PerformanceProfiler

# Initialize profiler
profiler = PerformanceProfiler()

# Profile specific operation
with profiler.profile("detection"):
    results = recognizer.detect_players(image)

# Get detailed performance metrics
metrics = profiler.get_metrics()
print(f"Total time: {metrics['total_time']:.3f}s")
print(f"Memory usage: {metrics['memory_peak_mb']:.1f}MB")
print(f"FLOPs: {metrics['flops']:,}")
```

### Model State Inspector

```python
from soccer_player_recognition.utils import inspect_model

# Inspect model state
state = inspect_model(recognizer.models['rf_detr'])
print(f"Model loaded: {state['loaded']}")
print(f"Device: {state['device']}")
print(f"Parameters: {state['num_parameters']:,}")
print(f"Trainable parameters: {state['trainable_parameters']:,}")

# Check for NaN values
if state['has_nan']:
    print("Warning: Model contains NaN values")
```

### Configuration Validator

```python
from soccer_player_recognition.utils.config_validator import validate_config

# Validate configuration
is_valid, errors = validate_config(config)

if is_valid:
    print("Configuration is valid")
else:
    print("Configuration errors:")
    for error in errors:
        print(f"- {error}")
```

### Logging Configuration

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Get model-specific logger
logger = logging.getLogger('soccer_player_recognition.models')
logger.setLevel(logging.DEBUG)
```

## FAQ

### General Questions

**Q: What minimum system requirements are needed?**
A: 
- CPU: Intel i5-8400 or AMD Ryzen 5 2600 (6+ cores recommended)
- GPU: NVIDIA GTX 1060 or better (8GB VRAM recommended)
- RAM: 16GB minimum, 32GB recommended for batch processing
- Storage: 50GB free space for models and datasets

**Q: Can I use this system without a GPU?**
A: Yes, but performance will be significantly slower. CPU-only inference may take 10-100x longer depending on the model.

**Q: How do I add new players to the identification system?**
A: 
1. Collect high-quality reference images of the player
2. Use SigLIP for zero-shot identification (no training required)
3. Or train a custom ResNet model for the player

**Q: What's the difference between detection and identification?**
A: 
- Detection: Finding where players are located in an image
- Identification: Recognizing which specific player it is

### Performance Questions

**Q: How can I improve processing speed?**
A: 
- Use GPU acceleration
- Reduce input resolution
- Process images in batches
- Enable mixed precision
- Use memory-efficient mode

**Q: How accurate are the models?**
A: 
- RF-DETR: ~89% mAP@0.5 for player detection
- SAM2: ~87% mIoU for video segmentation
- SigLIP: ~84% top-1 accuracy for known players
- ResNet: ~94% accuracy for trained player classes

### Troubleshooting Questions

**Q: My model gives different results each time I run it. Is this normal?**
A: Some variability is normal due to:
- Non-deterministic operations
- Different batch processing order
- Memory management

For consistent results, set random seeds and disable certain optimizations.

**Q: The system works well on some images but fails on others. Why?**
A: This is typically due to:
- Image quality (lighting, resolution, blur)
- Camera angle or perspective
- Player occlusions
- Unusual field conditions

Try image enhancement or adjusting confidence thresholds.

**Q: Can I use this system for other sports?**
A: The models can be adapted for other sports, but would require:
- Retraining on sport-specific data
- Adjusting class definitions
- Fine-tuning parameters for different object sizes

### Support

If you're still experiencing issues:

1. **Check the logs** for detailed error messages
2. **Verify your configuration** using the validator tools
3. **Test with sample data** to isolate the problem
4. **Check system requirements** and dependencies
5. **Review this troubleshooting guide** for similar issues

For additional help:
- Create an issue on GitHub with detailed error logs
- Include your system specifications
- Provide sample input that causes the problem
- Share your configuration file (remove sensitive data)

Remember to always include:
- Operating system and Python version
- CUDA/PyTorch versions
- Complete error traceback
- Input image/video samples (if applicable)