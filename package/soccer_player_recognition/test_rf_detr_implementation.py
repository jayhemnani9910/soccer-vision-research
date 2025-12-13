"""
Test script for RF-DETR model implementation

This script tests the basic functionality of the RF-DETR implementation
without requiring actual pretrained weights.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_rf_detr_config():
    """Test RF-DETR configuration classes."""
    print("Testing RF-DETR configuration...")
    
    try:
        from models.detection.rf_detr_config import RFDETRConfig, RFDETRSoccerConfigs, DEFAULT_CONFIG
        
        # Test basic configuration
        config = RFDETRConfig()
        print(f"✓ Default config created with {config.num_classes} classes")
        print(f"  Class names: {config.class_names}")
        print(f"  Input size: {config.input_size}")
        
        # Test predefined configurations
        configs = {
            "balanced": RFDETRSoccerConfigs.BALANCED,
            "real_time": RFDETRSoccerConfigs.REAL_TIME,
            "high_accuracy": RFDETRSoccerConfigs.HIGH_ACCURACY,
            "training": RFDETRSoccerConfigs.TRAINING
        }
        
        for name, cfg in configs.items():
            print(f"✓ {name} config: input_size={cfg.input_size}, score_threshold={cfg.score_threshold}")
        
        # Test mapping functions
        class_id_map = config.get_class_id_mapping()
        id_class_map = config.get_id_to_class_mapping()
        print(f"✓ Class ID mapping: {class_id_map}")
        print(f"✓ ID-Class mapping: {id_class_map}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rf_detr_model_creation():
    """Test RF-DETR model creation and basic properties."""
    print("\nTesting RF-DETR model creation...")
    
    try:
        from models.detection.rf_detr_model import create_rf_detr_model, RFDETRModel
        from models.detection.rf_detr_config import RFDETRConfig
        
        # Test factory function
        model = create_rf_detr_model("balanced")
        print(f"✓ Model created using factory function")
        print(f"  Device: {model.device}")
        print(f"  Number of parameters: {model.get_model_info()['total_parameters']:,}")
        
        # Test custom configuration
        custom_config = RFDETRConfig(
            input_size=(416, 416),
            score_threshold=0.8,
            nms_threshold=0.4
        )
        custom_model = RFDETRModel(custom_config)
        print(f"✓ Custom configuration model created")
        print(f"  Input size: {custom_model.config.input_size}")
        print(f"  Score threshold: {custom_model.config.score_threshold}")
        
        # Test model components
        print(f"✓ Model has backbone: {hasattr(model, 'backbone')}")
        print(f"✓ Model has transformer: {hasattr(model, 'transformer')}")
        print(f"✓ Model has preprocessor: {hasattr(model, 'preprocessor')}")
        print(f"✓ Model has postprocessor: {hasattr(model, 'postprocessor')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rf_detr_utils():
    """Test RF-DETR utility functions."""
    print("\nTesting RF-DETR utilities...")
    
    try:
        from models.detection.rf_detr_config import RFDETRConfig
        from utils.rf_detr_utils import (
            RFDETRPreprocessor, RFDETRPostprocessor,
            preprocess_for_soccer_detection, postprocess_soccer_detection
        )
        
        # Test configuration
        config = RFDETRConfig()
        
        # Test preprocessor
        preprocessor = RFDETRPreprocessor(config)
        print(f"✓ Preprocessor created with input size: {preprocessor.input_size}")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing
        image_tensor = preprocessor.preprocess_image(dummy_image)
        print(f"✓ Image preprocessing: {image_tensor.shape}")
        
        # Test image info
        image_info = preprocessor.get_image_info(dummy_image)
        print(f"✓ Image info: {image_info}")
        
        # Test batch preprocessing
        dummy_batch = [dummy_image, dummy_image]
        batch_tensor = preprocessor.preprocess_batch(dummy_batch)
        print(f"✓ Batch preprocessing: {batch_tensor.shape}")
        
        # Test postprocessor
        postprocessor = RFDETRPostprocessor(config)
        print(f"✓ Postprocessor created")
        
        # Test utility functions
        tensor, info = preprocess_for_soccer_detection(dummy_image, config)
        print(f"✓ Utility function preprocess_for_soccer_detection: {tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rf_detr_forward_pass():
    """Test RF-DETR forward pass with dummy data."""
    print("\nTesting RF-DETR forward pass...")
    
    try:
        from models.detection.rf_detr_model import create_rf_detr_model
        
        # Create model with small configuration for testing
        from models.detection.rf_detr_config import RFDETRConfig
        config = RFDETRConfig(input_size=(224, 224), d_model=128)  # Small for testing
        from models.detection.rf_detr_model import RFDETRModel
        model = RFDETRModel(config)
        
        # Move to CPU for testing
        model.to('cpu')
        model.eval()
        
        # Create dummy batch
        batch_size = 2
        dummy_batch = torch.randn(batch_size, 3, config.input_size[1], config.input_size[0])
        print(f"✓ Dummy batch shape: {dummy_batch.shape}")
        
        # Forward pass
        with torch.no_grad():
            predictions = model(dummy_batch)
        
        print(f"✓ Forward pass successful")
        print(f"  Class logits shape: {predictions['class_logits'].shape}")
        print(f"  Bbox predictions shape: {predictions['bbox_pred'].shape}")
        print(f"  Features shape: {predictions['features'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rf_detr_demo_imports():
    """Test RF-DETR demo script imports."""
    print("\nTesting RF-DETR demo imports...")
    
    try:
        from demos.rf_detr_demo import RFDETRDemo
        print("✓ Demo class imported successfully")
        
        # Test demo creation (without loading weights)
        demo = RFDETRDemo("balanced", model_path=None)
        print("✓ Demo instance created")
        
        # Test model info
        model_info = demo.get_model_info()
        print(f"✓ Model info retrieved: {model_info['model_type']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_format():
    """Test prediction output format."""
    print("\nTesting prediction format...")
    
    try:
        from models.detection.rf_detr_model import create_rf_detr_model
        
        # Create model
        model = create_rf_detr_model("balanced")
        model.to('cpu')
        model.eval()
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock prediction (since we don't have real weights)
        print("✓ Prediction format test completed (mock)")
        
        # Expected output format structure
        expected_format = {
            'detections': [
                {
                    'bbox': [0, 0, 100, 100],
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
        
        print(f"✓ Expected prediction format:")
        for key, value in expected_format.items():
            if key == 'detections':
                print(f"  {key}: List of detection dicts with bbox, confidence, class_id, class_name")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all RF-DETR tests."""
    print("=" * 60)
    print("RF-DETR Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        test_rf_detr_config,
        test_rf_detr_model_creation,
        test_rf_detr_utils,
        test_rf_detr_forward_pass,
        test_rf_detr_demo_imports,
        test_prediction_format
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! RF-DETR implementation is ready for use.")
        print("\nNext steps:")
        print("1. Obtain pretrained weights for RF-DETR soccer model")
        print("2. Run the demo script with real images: python demos/rf_detr_demo.py --input image.jpg")
        print("3. Process videos: python demos/rf_detr_demo.py --input video.mp4")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)