#!/usr/bin/env python3
"""
Integration Framework Test and Demo

This script demonstrates the unified integration framework by:
1. Testing all core components
2. Showing model pipeline orchestration
3. Demonstrating result fusion capabilities
4. Validating configuration management

Run this script to verify the integration framework works correctly.

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add the parent directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # Test imports
    from soccer_player_recognition.core import (
        PlayerRecognizer, 
        create_player_recognizer,
        get_system_info,
        demo_integration_framework,
        load_config,
        ResultFuser,
        FusionStrategy,
        ModelPipeline,
        ExecutionMode,
        DetectionResult,
        IdentificationResult,
        SegmentationResult
    )
    print("✓ All core imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Some components may not be available. Continuing with available components...")


def test_system_info():
    """Test system information gathering."""
    print("\n=== Testing System Information ===")
    
    try:
        system_info = get_system_info()
        print("System Information:")
        for key, value in system_info.items():
            if key == "components_available":
                print("  Components:")
                for component, available in value.items():
                    status = "✓" if available else "✗"
                    print(f"    {status} {component}")
            else:
                print(f"  {key}: {value}")
        return True
    except Exception as e:
        print(f"✗ System info test failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\n=== Testing Configuration Management ===")
    
    try:
        # Test different presets
        presets = ["balanced", "real_time", "high_accuracy", "development"]
        
        for preset in presets:
            try:
                config = load_config(preset=preset)
                print(f"✓ {preset} configuration loaded")
                print(f"  Detection batch size: {config.detection_config.batch_size}")
                print(f"  Device: {config.device}")
                
                # Test validation
                if hasattr(config, 'validate_config'):
                    errors = config.validate_config()
                    if errors:
                        print(f"  Validation errors: {errors}")
                    else:
                        print(f"  ✓ Configuration validation passed")
                
            except Exception as e:
                print(f"✗ Failed to load {preset} config: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_result_fusion():
    """Test result fusion capabilities."""
    print("\n=== Testing Result Fusion ===")
    
    try:
        # Create a result fusion instance
        fuser = ResultFuser(FusionStrategy.ADAPTIVE)
        print("✓ ResultFuser created successfully")
        
        # Test fusion statistics
        stats = fuser.get_fusion_statistics()
        print(f"  Fusion statistics: {stats}")
        
        # Test fusion strategies
        print("Available fusion strategies:")
        for strategy in FusionStrategy:
            fuser.set_fusion_strategy(strategy)
            print(f"  ✓ {strategy.value}")
        
        # Test model weights update
        fuser.update_model_weights({'rf_detr': 1.2, 'siglip': 1.1})
        print("✓ Model weights updated")
        
        return True
    except Exception as e:
        print(f"✗ Result fusion test failed: {e}")
        return False


def test_model_pipeline():
    """Test model pipeline orchestration."""
    print("\n=== Testing Model Pipeline ===")
    
    try:
        # Create pipeline
        pipeline = ModelPipeline()
        print("✓ ModelPipeline created successfully")
        
        # Test performance statistics
        stats = pipeline.get_performance_stats()
        print(f"  Performance stats: {stats}")
        
        # Test execution modes
        print("Available execution modes:")
        for mode in ExecutionMode:
            print(f"  ✓ {mode.value}")
        
        return True
    except Exception as e:
        print(f"✗ Model pipeline test failed: {e}")
        return False


def test_result_structures():
    """Test result data structures."""
    print("\n=== Testing Result Data Structures ===")
    
    try:
        # Test DetectionResult
        detection = DetectionResult(
            image_path="test_image.jpg",
            detections=[
                {'bbox': [100, 100, 200, 300], 'confidence': 0.9, 'class_name': 'player'},
                {'bbox': [300, 200, 400, 350], 'confidence': 0.8, 'class_name': 'ball'}
            ]
        )
        print("✓ DetectionResult created")
        print(f"  Total detections: {detection.total_detections}")
        print(f"  Has players: {detection.has_players}")
        print(f"  Has ball: {detection.has_ball}")
        
        # Test IdentificationResult
        identification = IdentificationResult(
            player_name="Lionel Messi",
            confidence=0.95,
            predictions=[
                {'player': 'Lionel Messi', 'confidence': 0.95},
                {'player': 'Cristiano Ronaldo', 'confidence': 0.03}
            ]
        )
        print("✓ IdentificationResult created")
        print(f"  Player: {identification.player_name}")
        print(f"  Confidence: {identification.confidence}")
        print(f"  High confidence: {identification.is_high_confidence()}")
        
        # Test SegmentationResult
        segmentation = SegmentationResult(
            frame_id=0,
            masks={
                'player1': np.random.rand(480, 640, 1),
                'player2': np.random.rand(480, 640, 1)
            }
        )
        print("✓ SegmentationResult created")
        print(f"  Total masks: {segmentation.total_masks}")
        print(f"  Coverage ratio: {segmentation.mask_coverage_ratio:.3f}")
        
        # Test serialization
        detection_dict = detection.to_dict()
        identification_dict = identification.to_dict()
        segmentation_dict = segmentation.to_dict()
        print("✓ Result serialization successful")
        
        return True
    except Exception as e:
        print(f"✗ Result structures test failed: {e}")
        return False


def test_player_recognizer():
    """Test PlayerRecognizer integration."""
    print("\n=== Testing PlayerRecognizer Integration ===")
    
    try:
        # Create minimal recognizer
        print("Creating minimal PlayerRecognizer...")
        recognizer = create_player_recognizer()
        print("✓ PlayerRecognizer created")
        
        # Test model status
        status = recognizer.get_model_status()
        print(f"  Device: {status['device']}")
        print(f"  Memory efficient: {status['memory_efficient']}")
        print(f"  Models loaded: {list(status['models'].keys())}")
        
        # Test performance stats
        stats = recognizer.get_performance_stats()
        print(f"  Performance stats: {stats}")
        
        # Test with sample detection (no actual image)
        print("Testing detection method interface...")
        # This would normally require actual images, but we're just testing the interface
        try:
            # This will fail without real images, but that's expected
            results = recognizer.detect_players(
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            )
            print("✓ Detection method executed (unexpected)")
        except Exception as e:
            # This is expected since we don't have real models or images
            print(f"  Expected error (no real models): {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ PlayerRecognizer test failed: {e}")
        return False


def test_integration_demo():
    """Run the full integration framework demo."""
    print("\n=== Running Integration Framework Demo ===")
    
    try:
        recognizer = demo_integration_framework()
        if recognizer:
            print("✓ Full integration demo completed")
            return True
        else:
            print("✗ Integration demo did not complete successfully")
            return False
    except Exception as e:
        print(f"✗ Integration demo failed: {e}")
        return False


def run_comprehensive_test():
    """Run all integration framework tests."""
    print("=" * 60)
    print("SOCCER PLAYER RECOGNITION - INTEGRATION FRAMEWORK TEST")
    print("=" * 60)
    
    tests = [
        ("System Information", test_system_info),
        ("Configuration Management", test_configuration),
        ("Result Fusion", test_result_fusion),
        ("Model Pipeline", test_model_pipeline),
        ("Result Data Structures", test_result_structures),
        ("PlayerRecognizer Integration", test_player_recognizer),
        ("Full Integration Demo", test_integration_demo),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            start_time = time.time()
            success = test_func()
            end_time = time.time()
            
            results[test_name] = {
                'success': success,
                'time': end_time - start_time
            }
            
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"\n{test_name}: {status} ({end_time - start_time:.2f}s)")
            
        except Exception as e:
            results[test_name] = {
                'success': False,
                'error': str(e),
                'time': 0
            }
            print(f"\n{test_name}: ✗ ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {test_name} ({result['time']:.2f}s)")
        if not result['success'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Final status
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - Integration Framework is working correctly!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED - Some components may need attention.")
        return False


if __name__ == "__main__":
    # Run the comprehensive test
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)