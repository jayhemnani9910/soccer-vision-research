"""
Demo script showing how to use the model management utilities.

This script demonstrates:
1. Registering models in the registry
2. Loading and managing models with the model manager
3. Tracking performance metrics with the performance monitor
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_registry import ModelRegistry, ModelType
from models.model_manager import ModelManager
from utils.performance_monitor import PerformanceMonitor


def create_demo_config():
    """Create a demo model configuration."""
    config = {
        "model_name": "demo_yolo",
        "version": "1.0",
        "input_size": [640, 640, 3],
        "num_classes": 80,
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "device": "auto",
        "batch_size": 1,
        "description": "Demo YOLO model for object detection"
    }
    
    # Save config to file
    config_path = "soccer_player_recognition/models/demo_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def create_demo_model():
    """Create a simple demo model file."""
    model_path = "soccer_player_recognition/models/demo_model.pt"
    
    # Create a simple mock model
    class DemoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create and save the model
    model = DemoModel()
    torch.save(model.state_dict(), model_path)
    
    return model_path


def demo_model_registry():
    """Demonstrate model registry functionality."""
    print("\n" + "="*50)
    print("DEMO: Model Registry")
    print("="*50)
    
    # Create a new registry for the demo
    registry = ModelRegistry("soccer_player_recognition/models/demo_registry.json")
    
    # Create demo files
    config_path = create_demo_config()
    model_path = create_demo_model()
    
    # Register different types of models
    models_to_register = [
        {
            "model_id": "yolo_detection_v1",
            "model_type": ModelType.DETECTION,
            "model_path": model_path,
            "config_path": config_path,
            "metadata": {
                "framework": "pytorch",
                "accuracy": 0.85,
                "speed_fps": 30,
                "gpu_required": True
            }
        },
        {
            "model_id": "resnet_classifier_v1",
            "model_type": ModelType.CLASSIFICATION,
            "model_path": model_path,
            "config_path": config_path,
            "metadata": {
                "framework": "pytorch",
                "num_classes": 1000,
                "top1_accuracy": 0.76,
                "top5_accuracy": 0.93
            }
        },
        {
            "model_id": "person_reid_v1",
            "model_type": ModelType.IDENTIFICATION,
            "model_path": model_path,
            "config_path": config_path,
            "metadata": {
                "framework": "pytorch",
                "feature_dim": 512,
                "gallery_size": 1000
            }
        }
    ]
    
    # Register all models
    for model_info in models_to_register:
        success = registry.register_model(**model_info)
        print(f"Registered {model_info['model_id']}: {'✓' if success else '✗'}")
    
    # Get models by type
    print(f"\nDetection models: {len(registry.get_models_by_type(ModelType.DETECTION))}")
    print(f"Classification models: {len(registry.get_models_by_type(ModelType.CLASSIFICATION))}")
    print(f"Identification models: {len(registry.get_models_by_type(ModelType.IDENTIFICATION))}")
    
    # Show all registered models
    print(f"\nAll registered models:")
    for model_id, info in registry.list_all_models().items():
        print(f"  - {model_id} ({info['model_type']})")
    
    # Validate registry
    errors = registry.validate_registry()
    if errors:
        print(f"\nRegistry validation errors: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ Registry validation passed")


def demo_model_manager():
    """Demonstrate model manager functionality."""
    print("\n" + "="*50)
    print("DEMO: Model Manager")
    print("="*50)
    
    # Create model manager with existing registry
    registry = ModelRegistry("soccer_player_recognition/models/demo_registry.json")
    model_manager = ModelManager(registry)
    
    # List available models
    available_models = registry.get_active_models()
    print(f"Available models: {len(available_models)}")
    
    if available_models:
        # Load first available model
        model_id = available_models[0]['model_id']
        print(f"\nLoading model: {model_id}")
        
        # Load the model
        success = model_manager.load_model(model_id, device="cpu")  # Use CPU for demo
        print(f"Model loaded: {'✓' if success else '✗'}")
        
        # Check loaded models
        loaded_models = model_manager.list_loaded_models()
        print(f"Currently loaded models: {loaded_models}")
        
        # Get model statistics
        if loaded_models:
            stats = model_manager.get_model_statistics(loaded_models[0])
            print(f"\nModel statistics:")
            print(f"  - Model ID: {stats.get('model_id', 'N/A')}")
            print(f"  - Is Loaded: {stats.get('is_loaded', False)}")
            print(f"  - Device: {stats.get('device', 'N/A')}")
            print(f"  - Inference Count: {stats.get('inference_count', 0)}")
            print(f"  - Average Inference Time: {stats.get('avg_inference_time', 0):.4f}s")
        
        # Perform a demo prediction
        if loaded_models:
            print(f"\nPerforming demo prediction...")
            try:
                # Create dummy input data
                dummy_input = torch.randn(1, 10)
                
                result = model_manager.predict(model_id, dummy_input)
                print("✓ Prediction completed")
                print(f"  Result type: {type(result)}")
                print(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
            except Exception as e:
                print(f"✗ Prediction failed: {e}")
        
        # Unload model
        print(f"\nUnloading model: {model_id}")
        success = model_manager.unload_model(model_id)
        print(f"Model unloaded: {'✓' if success else '✗'}")


def demo_performance_monitor():
    """Demonstrate performance monitor functionality."""
    print("\n" + "="*50)
    print("DEMO: Performance Monitor")
    print("="*50)
    
    # Create performance monitor
    monitor = PerformanceMonitor("soccer_player_recognition/outputs/performance")
    
    # Simulate some inference operations
    model_id = "demo_model"
    num_inferences = 10
    
    print(f"Simulating {num_inferences} inferences...")
    
    for i in range(num_inferences):
        # Start inference timing
        start_time = monitor.start_inference(model_id)
        
        # Simulate inference time (random between 0.01 and 0.1 seconds)
        import time
        import random
        inference_time = random.uniform(0.01, 0.1)
        time.sleep(inference_time)
        
        # End inference and record metrics
        actual_time = monitor.end_inference(model_id, start_time)
        
        # Occasionally simulate accuracy calculations
        if i % 3 == 0:
            # Simulate predictions and ground truth
            predictions = np.random.randint(0, 2, 100)
            ground_truth = np.random.randint(0, 2, 100)
            
            accuracy = monitor.calculate_accuracy(predictions, ground_truth, model_id)
            precision, recall = monitor.calculate_precision_recall(predictions, ground_truth, model_id)
    
    # Calculate and display FPS
    fps = monitor.calculate_fps(model_id)
    print(f"Calculated FPS: {fps:.2f}")
    
    # Monitor resources
    resources = monitor.monitor_resources(model_id)
    print(f"Resource usage:")
    print(f"  - CPU: {resources.get('cpu_percent', 0):.1f}%")
    print(f"  - Memory: {resources.get('memory_mb', 0):.1f} MB")
    if 'gpu_utilization' in resources:
        print(f"  - GPU: {resources.get('gpu_utilization', 0):.1f}%")
    
    # Generate performance report
    report = monitor.get_performance_report(model_id)
    if report:
        print(f"\nPerformance Report:")
        print(f"  - Total Inferences: {report.total_inferences}")
        print(f"  - Average Inference Time: {report.avg_inference_time:.4f}s")
        print(f"  - FPS: {report.fps:.2f}")
        print(f"  - Accuracy Metrics:")
        for metric, value in report.accuracy_metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    # Save performance report
    json_path = monitor.save_performance_report(model_id, "json")
    csv_path = monitor.save_performance_report(model_id, "csv")
    txt_path = monitor.save_performance_report(model_id, "txt")
    
    print(f"\nSaved performance reports:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")
    print(f"  - TXT: {txt_path}")


def main():
    """Main demo function."""
    print("🚀 Model Management Utilities Demo")
    print("="*60)
    
    try:
        # Run all demos
        demo_model_registry()
        demo_model_manager()
        demo_performance_monitor()
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("  • Model registry for centralized model management")
        print("  • Model manager for dynamic loading and inference")
        print("  • Performance monitor for tracking metrics and resource usage")
        print("\nCheck the following directories for outputs:")
        print("  - soccer_player_recognition/models/demo_registry.json")
        print("  - soccer_player_recognition/models/demo_config.json")
        print("  - soccer_player_recognition/models/demo_model.pt")
        print("  - soccer_player_recognition/outputs/performance/")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()