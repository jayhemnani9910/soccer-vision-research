# Usage Tutorials

## Table of Contents

1. [Quick Start Tutorial](#quick-start-tutorial)
2. [Basic Detection Tutorial](#basic-detection-tutorial)
3. [Player Identification Tutorial](#player-identification-tutorial)
4. [Video Processing Tutorial](#video-processing-tutorial)
5. [Advanced Configuration Tutorial](#advanced-configuration-tutorial)
6. [Batch Processing Tutorial](#batch-processing-tutorial)
7. [Custom Model Training Tutorial](#custom-model-training-tutorial)
8. [Performance Monitoring Tutorial](#performance-monitoring-tutorial)

## Quick Start Tutorial

### Installation and Setup

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd soccer-player-recognition
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Download Pre-trained Models**
   ```bash
   python scripts/download_models.py
   ```

3. **Verify Installation**
   ```python
   from soccer_player_recognition import PlayerRecognizer
   print("Installation successful!")
   ```

### Your First Recognition

```python
from soccer_player_recognition import PlayerRecognizer

# Initialize the recognizer
recognizer = PlayerRecognizer()

# Process an image
results = recognizer.analyze_scene("path/to/soccer/image.jpg")

# Print results
print(f"Detected {results['detection'].num_detections} players")
print(f"Total processing time: {results['metadata']['total_time']:.3f} seconds")
```

## Basic Detection Tutorial

### Understanding Detection Results

```python
import cv2
import numpy as np
from soccer_player_recognition import PlayerRecognizer

# Initialize recognizer
recognizer = PlayerRecognizer()

# Load image
image_path = "sample_images/match_scene.jpg"
results = recognizer.detect_players(image_path)

# Access detection results
detections = results
print(f"Number of detections: {detections.num_detections}")

# Iterate through each detection
for i, (bbox, label, confidence) in enumerate(zip(
    detections.bbox, 
    detections.labels, 
    detections.confidences
)):
    x1, y1, x2, y2 = bbox
    print(f"Detection {i+1}: {label} (confidence: {confidence:.3f})")
    print(f"  Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
```

### Adjusting Detection Parameters

```python
# High sensitivity (more detections, lower confidence threshold)
results = recognizer.detect_players(
    image_path,
    confidence_threshold=0.3,  # Lower threshold
    nms_threshold=0.3,         # More aggressive NMS
    max_detections=100         # More allowed detections
)

# Low sensitivity (fewer detections, higher confidence threshold)
results = recognizer.detect_players(
    image_path,
    confidence_threshold=0.9,  # Higher threshold
    nms_threshold=0.7,         # Less aggressive NMS
    max_detections=20          # Fewer allowed detections
)
```

### Visualizing Detection Results

```python
import cv2
import numpy as np
from soccer_player_recognition.utils import visualize_results

# Load image
image = cv2.imread(image_path)

# Run detection
results = recognizer.detect_players(image_path)

# Visualize results
annotated_image = visualize_results(image, results, save_path="output/annotated.jpg")

# Display with OpenCV
cv2.imshow("Detection Results", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Player Identification Tutorial

### Zero-Shot Identification with SigLIP

```python
from soccer_player_recognition import PlayerRecognizer

# Initialize recognizer
recognizer = PlayerRecognizer()

# Define player candidates
player_candidates = [
    "Lionel Messi in blue jersey",
    "Cristiano Ronaldo in red jersey",
    "Neymar Jr in yellow jersey",
    "Robert Lewandowski in red jersey",
    "Kevin De Bruyne in blue jersey"
]

# Process image for identification
results = recognizer.identify_players(
    "image_with_players.jpg",
    player_candidates=player_candidates,
    use_siglip=True  # Use zero-shot identification
)

# Access identification results
print(f"Identified players:")
for name, confidence in zip(
    results.player_names, 
    results.confidence_scores
):
    print(f"  {name}: {confidence:.3f}")
```

### Team Context Identification

```python
# Example 1: Barcelona players
barcelona_candidates = [
    "Lionel Messi in Barcelona jersey",
    "Ansu Fati in Barcelona jersey",
    "Pedri in Barcelona jersey",
    "Frenkie de Jong in Barcelona jersey"
]

barcelona_results = recognizer.identify_players(
    "barcelona_match.jpg",
    player_candidates=barcelona_candidates,
    team_context="FC Barcelona",
    use_siglip=True
)

# Example 2: Real Madrid players
real_madrid_candidates = [
    "Karim Benzema in Real Madrid jersey",
    "Vinicius Jr in Real Madrid jersey",
    "Luka Modric in Real Madrid jersey",
    "Toni Kroos in Real Madrid jersey"
]

madrid_results = recognizer.identify_players(
    "real_madrid_match.jpg",
    player_candidates=real_madrid_candidates,
    team_context="Real Madrid",
    use_siglip=True
)
```

### Building a Player Gallery

```python
from soccer_player_recognition.utils.gallery_builder import GalleryBuilder

# Create gallery builder
builder = GalleryBuilder()

# Add player images to gallery
players = {
    "messi": ["images/messi_1.jpg", "images/messi_2.jpg", "images/messi_3.jpg"],
    "ronaldo": ["images/ronaldo_1.jpg", "images/ronaldo_2.jpg"],
    "neymar": ["images/neymar_1.jpg", "images/neymar_2.jpg", "images/neymar_3.jpg", "images/neymar_4.jpg"]
}

for player_name, image_paths in players.items():
    builder.add_player_images(player_name, image_paths)

# Build and save gallery
builder.build_gallery("data/player_gallery")
print("Player gallery created successfully!")
```

### Using ResNet for Trained Identification

```python
# After training a ResNet model with your player dataset
results = recognizer.identify_players(
    "training_image.jpg",
    player_candidates=[],  # Not needed for ResNet
    use_siglip=False,      # Use ResNet instead
    use_resnet=True
)

print(f"Predicted player classes: {results.predicted_classes}")
print(f"Confidence scores: {results.confidence_scores}")
```

## Video Processing Tutorial

### Processing Video Files

```python
from soccer_player_recognition import VideoProcessor

# Initialize video processor
processor = VideoProcessor(
    input_path="match_video.mp4",
    output_path="output/processed_match.mp4"
)

# Initialize recognizer
recognizer = PlayerRecognizer()

# Process video
results = processor.process_video(
    recognizer,
    save_visualizations=True,
    save_detections=True,
    output_fps=30
)

print(f"Video processing completed!")
print(f"Total frames processed: {results['total_frames']}")
print(f"Average FPS: {results['average_fps']:.2f}")
print(f"Total players detected: {results['total_detections']}")
```

### Real-time Video Stream Processing

```python
import cv2
import time
from soccer_player_recognition import PlayerRecognizer

# Initialize recognizer
recognizer = PlayerRecognizer()

# Open video capture
cap = cv2.VideoCapture(0)  # Webcam (or provide video file path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Process video stream
frame_count = 0
total_start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process every nth frame for speed (e.g., every 5th frame)
    if frame_count % 5 == 0:
        start_time = time.time()
        
        try:
            results = recognizer.analyze_scene(frame)
            annotated_frame = visualize_results(frame, results)
            
            # Display FPS
            process_time = time.time() - start_time
            fps_current = 1.0 / process_time
            cv2.putText(annotated_frame, f"FPS: {fps_current:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Real-time Soccer Recognition", annotated_frame)
            
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            cv2.imshow("Real-time Soccer Recognition", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
recognizer.cleanup_memory()

total_time = time.time() - total_start_time
print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
```

### Extracting Specific Moments

```python
from soccer_player_recognition import VideoProcessor

# Initialize processor
processor = VideoProcessor("long_match.mp4", "output/")

# Extract frames at specific timestamps
timestamps = [120, 300, 480, 720]  # seconds
frame_numbers = [int(ts * fps) for ts in timestamps]

# Extract frames
extracted_frames = processor.extract_frames(
    frame_numbers=frame_numbers,
    max_frames=None  # Extract only specified frames
)

print(f"Extracted {len(extracted_frames)} frames")

# Process each extracted frame
for i, frame in enumerate(extracted_frames):
    timestamp = timestamps[i]
    results = recognizer.analyze_scene(frame)
    
    print(f"Frame at {timestamp}s: {results['detection'].num_detections} players detected")
    
    # Save annotated frame
    annotated_frame = visualize_results(frame, results)
    cv2.imwrite(f"output/frame_{timestamp}s.jpg", annotated_frame)
```

## Advanced Configuration Tutorial

### Custom Model Configuration

```python
from soccer_player_recognition import PlayerRecognizer

# Custom configuration
custom_config = {
    "models": {
        "rf_detr": {
            "confidence_threshold": 0.8,
            "nms_threshold": 0.4,
            "max_detections": 25
        },
        "siglip": {
            "temperature": 50.0,  # More confident predictions
            "batch_size": 16      # Smaller batch for memory
        },
        "sam2": {
            "confidence_threshold": 0.8,
            "max_memory_frames": 12  # More frames for better tracking
        },
        "resnet": {
            "num_classes": 30,    # More player classes
            "dropout": 0.3        # Less regularization
        }
    },
    "system": {
        "device": "cuda",
        "mixed_precision": True,
        "gradient_checkpointing": True
    }
}

# Save custom config
import yaml
with open("custom_config.yaml", "w") as f:
    yaml.dump(custom_config, f)

# Initialize with custom config
recognizer = PlayerRecognizer(config_path="custom_config.yaml")
```

### Runtime Model Switching

```python
# Switch to high-speed detection mode
fast_config = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.3,
    "max_detections": 15
}

success = recognizer.switch_model_config("rf_detr", fast_config)
print(f"Fast mode enabled: {success}")

# Switch to high-accuracy mode
accurate_config = {
    "confidence_threshold": 0.9,
    "nms_threshold": 0.7,
    "max_detections": 50
}

success = recognizer.switch_model_config("rf_detr", accurate_config)
print(f"Accuracy mode enabled: {success}")

# Switch identification method
siglip_config = {
    "model_name": "siglip-base-patch16-384",
    "temperature": 100.0,
    "batch_size": 32
}

success = recognizer.switch_model_config("siglip", siglip_config)
print(f"Switched to SigLIP: {success}")
```

### Ensemble Configuration

```yaml
# ensemble_config.yaml
ensemble:
  enabled: true
  voting_strategy: "soft"
  consensus_threshold: 0.7
  weights:
    rf_detr: 0.4     # Higher weight for detection
    siglip: 0.3      # Good for identification
    sam2: 0.2        # Backup for segmentation
    resnet: 0.1      # Limited use in ensemble

  threshold_calibration:
    method: "balanced"  # balanced, precision, recall
    target_tpr: 0.95   # Target true positive rate
```

```python
# Use ensemble configuration
recognizer = PlayerRecognizer(config_path="ensemble_config.yaml")

# Run ensemble analysis
results = recognizer.analyze_scene(
    "complex_scene.jpg",
    analysis_type="full",
    fuse_results=True
)

print(f"Ensemble confidence: {results['fusion'].ensemble_confidence}")
print(f"Consensus reached: {results['fusion'].consensus_achieved}")
```

## Batch Processing Tutorial

### Processing Image Directories

```python
import os
import glob
from pathlib import Path
from soccer_player_recognition import PlayerRecognizer

# Initialize recognizer
recognizer = PlayerRecognizer(memory_efficient=True)

# Process all images in a directory
input_dir = "data/soccer_images/"
output_dir = "output/processed_images/"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Find all image files
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

print(f"Found {len(image_paths)} images to process")

# Process images in batches
batch_size = 5
all_results = []

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch_images = []
    
    # Load batch images
    for path in batch_paths:
        image = cv2.imread(path)
        if image is not None:
            batch_images.append(image)
    
    if batch_images:
        # Process batch
        try:
            batch_results = recognizer.analyze_scene(batch_images)
            
            # Save results for each image
            for j, (path, result) in enumerate(zip(batch_paths, batch_results)):
                filename = Path(path).stem
                
                # Save annotated image
                if 'detection' in result:
                    annotated = visualize_results(batch_images[j], result['detection'])
                    output_path = os.path.join(output_dir, f"{filename}_annotated.jpg")
                    cv2.imwrite(output_path, annotated)
                
                # Save JSON results
                result_path = os.path.join(output_dir, f"{filename}_results.json")
                with open(result_path, 'w') as f:
                    json.dump(serialize_results(result), f, indent=2)
                
                all_results.append({
                    'image_path': path,
                    'output_path': output_path,
                    'num_detections': result['detection'].num_detections if 'detection' in result else 0
                })
                
            print(f"Processed batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")

# Cleanup
recognizer.cleanup_memory()

# Generate summary report
generate_batch_report(all_results, output_dir)
```

### Processing Video Directories

```python
# Process multiple videos
video_dir = "data/match_videos/"
output_base_dir = "output/match_analysis/"

# Find all video files
video_extensions = ["*.mp4", "*.avi", "*.mov"]
video_paths = []
for ext in video_extensions:
    video_paths.extend(glob.glob(os.path.join(video_dir, ext)))

print(f"Found {len(video_paths)} videos to process")

for video_path in video_paths:
    try:
        video_name = Path(video_path).stem
        output_dir = os.path.join(output_base_dir, video_name)
        
        # Initialize video processor
        processor = VideoProcessor(video_path, output_dir)
        
        # Process video
        results = processor.process_video(
            recognizer,
            save_detections=True,
            save_segmentations=True,
            save_identifications=True,
            output_format="mp4"
        )
        
        print(f"Processed {video_name}: {results['average_fps']:.2f} FPS")
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
```

### Parallel Processing

```python
import concurrent.futures
import multiprocessing
from soccer_player_recognition import PlayerRecognizer

def process_image_wrapper(image_path):
    """Wrapper function for parallel processing"""
    try:
        recognizer = PlayerRecognizer()
        results = recognizer.analyze_scene(image_path)
        recognizer.cleanup_memory()
        return {
            'path': image_path,
            'success': True,
            'results': results
        }
    except Exception as e:
        return {
            'path': image_path,
            'success': False,
            'error': str(e)
        }

# Process images in parallel
image_paths = glob.glob("images/*.jpg")

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_image_wrapper, path): path for path in image_paths}
    
    results = []
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.append(result)
        if result['success']:
            print(f"✓ Processed {result['path']}")
        else:
            print(f"✗ Failed {result['path']}: {result['error']}")

print(f"Completed processing {len(results)} images")
```

## Custom Model Training Tutorial

### Training Detection Model

```python
from soccer_player_recognition.models.detection import RFDETRTrainer
from torch.utils.data import DataLoader

# Initialize trainer
trainer = RFDETRTrainer(
    model_config={
        "input_size": [640, 640],
        "num_classes": 4,  # player, ball, referee, goalkeeper
        "pretrained": True
    }
)

# Load custom dataset
train_loader = DataLoader(
    dataset=create_soccer_dataset("data/train/", annotation_format="yolo"),
    batch_size=8,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    dataset=create_soccer_dataset("data/val/", annotation_format="yolo"),
    batch_size=8,
    shuffle=False,
    num_workers=4
)

# Train model
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=0.001,
    save_path="models/custom_rf_detr.pth"
)

# Evaluate on test set
test_results = trainer.evaluate("data/test/")
print(f"Test mAP: {test_results['mAP']:.3f}")
```

### Training Identification Model

```python
from soccer_player_recognition.models.identification import SigLIPTrainer
from torch.utils.data import DataLoader

# For SigLIP (zero-shot, minimal training needed)
# Create player text embeddings
player_descriptions = [
    "Lionel Messi in blue and red striped jersey number 10",
    "Cristiano Ronaldo in red jersey number 7",
    "Neymar Jr in yellow jersey number 10",
    # ... more player descriptions
]

trainer = SigLIPTrainer()
trainer.create_player_embeddings(player_descriptions)
trainer.save_embeddings("models/player_embeddings.pkl")

# For ResNet (requires training)
from soccer_player_recognition.models.classification import ResNetTrainer

# Create dataset with player labels
train_loader = DataLoader(
    dataset=create_player_dataset("data/players/train/", num_players=25),
    batch_size=32,
    shuffle=True,
    num_workers=8
)

val_loader = DataLoader(
    dataset=create_player_dataset("data/players/val/", num_players=25),
    batch_size=32,
    shuffle=False,
    num_workers=8
)

# Initialize trainer
trainer = ResNetTrainer(
    architecture="resnet50",
    num_classes=25,
    pretrained=True
)

# Train model
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=0.001,
    save_path="models/custom_resnet.pth"
)
```

### Training Segmentation Model

```python
from soccer_player_recognition.models.segmentation import SAM2Trainer

# Initialize trainer
trainer = SAM2Trainer(
    model_config={
        "image_size": [1024, 1024],
        "memory_mode": "selective",
        "max_memory_frames": 8
    }
)

# Load video segmentation dataset
train_videos = create_video_dataset("data/videos/train/")
val_videos = create_video_dataset("data/videos/val/")

# Train SAM2 model
trainer.train(
    train_videos=train_videos,
    val_videos=val_videos,
    num_epochs=20,
    save_path="models/custom_sam2.pth"
)

# Fine-tune on specific players
player_prompts = create_player_prompts("data/player_masks/")
trainer.fine_tune(player_prompts, num_epochs=5)
```

## Performance Monitoring Tutorial

### Real-time Performance Tracking

```python
from soccer_player_recognition.utils.performance_monitor import PerformanceMonitor

# Initialize performance monitor
monitor = PerformanceMonitor(
    log_file="performance.log",
    log_interval=10  # Log every 10 inferences
)

# Initialize recognizer
recognizer = PlayerRecognizer()

# Process images with monitoring
image_paths = glob.glob("test_images/*.jpg")
processing_times = []

for i, image_path in enumerate(image_paths):
    start_time = time.time()
    
    try:
        results = recognizer.analyze_scene(image_path)
        processing_time = time.time() - start_time
        
        processing_times.append(processing_time)
        
        # Log performance
        monitor.log_inference(
            image_path=image_path,
            processing_time=processing_time,
            num_detections=results['detection'].num_detections,
            models_used=list(results.keys())
        )
        
        # Print progress
        if (i + 1) % 10 == 0:
            avg_time = np.mean(processing_times[-10:])
            print(f"Processed {i+1}/{len(image_paths)} images. "
                  f"Recent avg time: {avg_time:.3f}s")
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Generate performance report
report = monitor.generate_report()
print(f"Performance Report:")
print(f"Average processing time: {report['avg_time']:.3f}s")
print(f"FPS: {report['fps']:.1f}")
print(f"Memory usage: {report['memory_usage']:.1f} MB")
```

### GPU Memory Monitoring

```python
import psutil
import GPUtil
from soccer_player_recognition.utils import get_device_info

# Monitor system resources
def monitor_resources():
    """Monitor CPU, GPU, and memory usage"""
    # CPU and memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # GPU (if available)
    gpu_info = None
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            gpu_info = {
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': gpu.memoryUsed / gpu.memoryTotal * 100,
                'gpu_utilization': gpu.load * 100
            }
    except:
        pass
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3),
        'gpu_info': gpu_info
    }

# Monitor during processing
recognizer = PlayerRecognizer()

print("Starting resource monitoring...")
resource_logs = []

for image_path in image_paths:
    # Monitor before processing
    before = monitor_resources()
    
    # Process image
    results = recognizer.analyze_scene(image_path)
    
    # Monitor after processing
    after = monitor_resources()
    
    resource_logs.append({
        'image': image_path,
        'before': before,
        'after': after,
        'results': results
    })
    
    # Print current status
    print(f"CPU: {after['cpu_percent']:.1f}%, "
          f"Memory: {after['memory_percent']:.1f}%", end="")
    
    if after['gpu_info']:
        print(f", GPU: {after['gpu_info']['memory_percent']:.1f}%", end="")
    print()

# Analyze resource usage patterns
analyze_resource_usage(resource_logs)
```

### Benchmarking Different Configurations

```python
def benchmark_configurations():
    """Benchmark different model configurations"""
    
    test_images = load_test_images("benchmark_images/", num_images=50)
    
    configurations = {
        "fast": {
            "rf_detr": {"confidence_threshold": 0.4, "nms_threshold": 0.3},
            "siglip": {"batch_size": 64}
        },
        "balanced": {
            "rf_detr": {"confidence_threshold": 0.6, "nms_threshold": 0.5},
            "siglip": {"batch_size": 32}
        },
        "accurate": {
            "rf_detr": {"confidence_threshold": 0.8, "nms_threshold": 0.7},
            "siglip": {"batch_size": 16}
        }
    }
    
    benchmark_results = {}
    
    for config_name, config in configurations.items():
        print(f"Benchmarking {config_name} configuration...")
        
        # Create recognizer with specific config
        recognizer = PlayerRecognizer()
        
        # Apply configuration
        for model_name, model_config in config.items():
            recognizer.switch_model_config(model_name, model_config)
        
        # Benchmark
        start_time = time.time()
        results = []
        
        for image in test_images:
            try:
                result = recognizer.analyze_scene(image)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
        
        total_time = time.time() - start_time
        
        benchmark_results[config_name] = {
            'total_time': total_time,
            'avg_time_per_image': total_time / len(test_images),
            'fps': len(test_images) / total_time,
            'success_rate': len(results) / len(test_images),
            'avg_detections': np.mean([r['detection'].num_detections for r in results])
        }
        
        print(f"  Average time: {benchmark_results[config_name]['avg_time_per_image']:.3f}s")
        print(f"  FPS: {benchmark_results[config_name]['fps']:.1f}")
    
    # Generate comparison report
    generate_benchmark_report(benchmark_results)
    
    return benchmark_results

# Run benchmark
results = benchmark_configurations()
```

### Export Performance Metrics

```python
# Export detailed performance metrics
def export_performance_data():
    """Export performance data to various formats"""
    
    # Get performance statistics
    stats = recognizer.get_performance_stats()
    
    # Export to JSON
    with open("performance_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Export to CSV (flattened)
    import pandas as pd
    
    # Flatten nested dictionaries
    flat_stats = flatten_dict(stats)
    df = pd.DataFrame([flat_stats])
    df.to_csv("performance_stats.csv", index=False)
    
    # Generate detailed report
    report = generate_detailed_performance_report(stats)
    
    with open("performance_report.html", 'w') as f:
        f.write(report)
    
    print("Performance data exported successfully!")

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Export data
export_performance_data()
```

## Next Steps

Now that you've completed these tutorials, you can:

1. **Explore Advanced Features**: Experiment with ensemble methods and custom configurations
2. **Optimize Performance**: Apply the optimization strategies from the performance tutorial
3. **Deploy to Production**: Follow the deployment guide for production setup
4. **Troubleshoot Issues**: Reference the troubleshooting guide for common problems
5. **Contribute**: Help improve the system by adding new features or fixing bugs

## Getting Help

- **Documentation**: Check the API reference and model guide
- **Issues**: Report bugs or request features on GitHub
- **Community**: Join discussions and share your implementations
- **Examples**: Explore more examples in the demos/ directory

Happy coding! ⚽