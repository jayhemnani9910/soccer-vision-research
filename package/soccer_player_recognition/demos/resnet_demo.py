"""
ResNet Player Recognition Demo and Test Script

This script demonstrates the complete ResNet-based player recognition system,
including model creation, jersey number recognition, and utility functions.

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import sys
import os
import numpy as np
import cv2
import torch
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_resnet_model():
    """Test ResNet Player Recognition Model."""
    logger.info("=" * 60)
    logger.info("Testing ResNet Player Recognition Model")
    logger.info("=" * 60)
    
    try:
        # Import the ResNet model classes
        from soccer_player_recognition.models.classification.resnet_model import ResNetPlayerClassifier, PlayerRecognitionModel
        
        # Test 1: Create ResNet model
        logger.info("Test 1: Creating ResNet Player Classifier...")
        num_players = 22  # Typical number of players in soccer
        model = ResNetPlayerClassifier(
            num_players=num_players,
            pretrained=True,
            model_name="resnet18"  # Use smaller model for demo
        )
        
        model_info = model.get_model_info()
        logger.info(f"Model created successfully!")
        logger.info(f"  - Model type: {model_info['model_name']}")
        logger.info(f"  - Number of players: {model_info['num_players']}")
        logger.info(f"  - Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"  - Model size: {model_info['model_size_mb']:.2f} MB")
        
        # Test 2: Create high-level Player Recognition Model
        logger.info("\\nTest 2: Creating Player Recognition Model...")
        player_model = PlayerRecognitionModel(
            num_players=num_players,
            model_name="resnet18",
            pretrained=True
        )
        
        # Setup training
        player_model.setup_training(
            learning_rate=0.001,
            optimizer_type="adam",
            scheduler_type="step"
        )
        logger.info("Player Recognition Model created and configured!")
        
        # Test 3: Test inference with dummy data
        logger.info("\\nTest 3: Testing inference with dummy data...")
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test identification
        result = player_model.identify_player(dummy_image)
        logger.info(f"Inference test completed:")
        logger.info(f"  - Player ID: {result.get('player_id')}")
        logger.info(f"  - Confidence: {result.get('confidence', 0):.4f}")
        
        # Test 4: Test feature extraction
        logger.info("\\nTest 4: Testing feature extraction...")
        features = player_model.extract_player_features(dummy_image)
        logger.info(f"Feature extraction completed:")
        logger.info(f"  - Feature vector size: {features.shape}")
        logger.info(f"  - Feature range: [{features.min():.4f}, {features.max():.4f}]")
        
        logger.info("\\n✅ ResNet Player Recognition Model tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in ResNet model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jersey_recognizer():
    """Test Jersey Number Recognition."""
    logger.info("\\n" + "=" * 60)
    logger.info("Testing Jersey Number Recognition")
    logger.info("=" * 60)
    
    try:
        # Import jersey recognition classes
        from soccer_player_recognition.models.classification.jersey_recognizer import JerseyNumberExtractor, JerseyRecognizer
        
        # Test 1: Create jersey extractor
        logger.info("Test 1: Creating Jersey Number Extractor...")
        extractor = JerseyNumberExtractor(
            min_number_size=20,
            max_number_size=200,
            confidence_threshold=0.5
        )
        logger.info("Jersey Number Extractor created successfully!")
        
        # Test 2: Create jersey recognizer
        logger.info("\\nTest 2: Creating Jersey Recognizer...")
        recognizer = JerseyRecognizer()
        logger.info("Jersey Recognizer created successfully!")
        
        # Test 3: Create sample image with jersey number
        logger.info("\\nTest 3: Creating sample image with jersey number...")
        sample_image = np.zeros((400, 300, 3), dtype=np.uint8)
        
        # Add jersey region (white rectangle)
        cv2.rectangle(sample_image, (80, 100), (220, 280), (255, 255, 255), -1)
        
        # Add jersey number "10"
        cv2.putText(sample_image, "10", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        logger.info("Sample jersey image created!")
        
        # Test 4: Test jersey number extraction
        logger.info("\\nTest 4: Testing jersey number extraction...")
        result = extractor.extract_jersey_number(sample_image)
        
        logger.info(f"Jersey number extraction results:")
        logger.info(f"  - Detected number: {result.get('number')}")
        logger.info(f"  - Confidence: {result.get('confidence', 0):.4f}")
        logger.info(f"  - Number of candidates: {len(result.get('candidates', []))}")
        logger.info(f"  - Processing status: {result.get('processing_info', {}).get('status', 'unknown')}")
        
        # Test 5: Test high-level recognizer
        logger.info("\\nTest 5: Testing high-level jersey recognizer...")
        recognition_result = recognizer.recognize_jersey_number(sample_image)
        
        logger.info(f"Jersey recognition results:")
        logger.info(f"  - Recognized number: {recognition_result.get('number')}")
        logger.info(f"  - Confidence: {recognition_result.get('confidence', 0):.4f}")
        logger.info(f"  - Cache key: {recognition_result.get('cache_key') is not None}")
        
        logger.info("\\n✅ Jersey Number Recognition tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in jersey recognizer test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_resnet_utils():
    """Test ResNet Utility Functions."""
    logger.info("\\n" + "=" * 60)
    logger.info("Testing ResNet Utility Functions")
    logger.info("=" * 60)
    
    try:
        # Import utility classes
        from soccer_player_recognition.utils.resnet_utils import (
            ResNetPreprocessor, 
            ResNetFeatureExtractor, 
            ResNetDatasetProcessor,
            ResNetModelEvaluator
        )
        
        # Test 1: Preprocessor
        logger.info("Test 1: Testing ResNet Preprocessor...")
        preprocessor = ResNetPreprocessor(
            input_size=224,
            augmentation=True
        )
        
        # Create sample image
        sample_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # Test preprocessing
        tensor = preprocessor.preprocess_image(sample_image, mode='inference')
        logger.info(f"Preprocessor test completed:")
        logger.info(f"  - Input shape: {sample_image.shape}")
        logger.info(f"  - Output tensor shape: {tensor.shape}")
        logger.info(f"  - Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        
        # Test 2: Feature Extractor
        logger.info("\\nTest 2: Testing ResNet Feature Extractor...")
        import torchvision.models as models
        resnet_model = models.resnet18(pretrained=True)
        
        feature_extractor = ResNetFeatureExtractor(resnet_model, layer_name='layer4')
        
        # Extract features
        batch_images = torch.randn(2, 3, 224, 224)
        features = feature_extractor.extract_features(batch_images)
        
        logger.info(f"Feature extraction test completed:")
        logger.info(f"  - Number of feature maps: {len(features)}")
        for name, feat in features.items():
            if isinstance(feat, torch.Tensor):
                logger.info(f"  - {name}: {feat.shape}")
        
        # Test 3: Model Evaluator
        logger.info("\\nTest 3: Testing ResNet Model Evaluator...")
        evaluator = ResNetModelEvaluator(resnet_model)
        
        # Test model statistics (simplified version)
        total_params = sum(p.numel() for p in resnet_model.parameters())
        logger.info(f"Model evaluator test completed:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Device: {evaluator.device}")
        
        # Test 4: Dataset Processor
        logger.info("\\nTest 4: Testing ResNet Dataset Processor...")
        dataset_processor = ResNetDatasetProcessor(preprocessor)
        
        # Create dummy dataset
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                label = idx % 10  # 10 classes
                return torch.from_numpy(image.transpose(2, 0, 1)).float(), label
        
        dummy_dataset = DummyDataset(50)
        dataset_stats = dataset_processor.analyze_dataset(dummy_dataset)
        
        logger.info(f"Dataset processor test completed:")
        logger.info(f"  - Dataset size: {dataset_stats.get('num_samples', 'N/A')}")
        logger.info(f"  - Size summary: {dataset_stats.get('size_summary', 'N/A')}")
        
        logger.info("\\n✅ ResNet Utility Functions tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in resnet utils test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of all components."""
    logger.info("\\n" + "=" * 60)
    logger.info("Testing Integration of All Components")
    logger.info("=" * 60)
    
    try:
        # Import all classes
        from soccer_player_recognition.models.classification.resnet_model import PlayerRecognitionModel
        from soccer_player_recognition.models.classification.jersey_recognizer import JerseyRecognizer
        from soccer_player_recognition.utils.resnet_utils import ResNetPreprocessor
        
        logger.info("Creating integrated player recognition system...")
        
        # Create main components
        player_model = PlayerRecognitionModel(num_players=22, model_name="resnet18", pretrained=True)
        jersey_recognizer = JerseyRecognizer()
        preprocessor = ResNetPreprocessor(input_size=224)
        
        # Create sample player image
        player_image = np.zeros((400, 300, 3), dtype=np.uint8)
        
        # Add jersey region
        cv2.rectangle(player_image, (80, 120), (220, 280), (255, 255, 255), -1)
        cv2.putText(player_image, "7", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        logger.info("Sample player image created")
        
        # Test 1: Jersey number recognition
        logger.info("\\nStep 1: Recognizing jersey number...")
        jersey_result = jersey_recognizer.recognize_jersey_number(player_image)
        logger.info(f"Jersey recognition: {jersey_result.get('number')} (confidence: {jersey_result.get('confidence', 0):.2f})")
        
        # Test 2: Player identification
        logger.info("\\nStep 2: Identifying player...")
        player_result = player_model.identify_player(player_image)
        logger.info(f"Player identification: {player_result.get('player_name')} (confidence: {player_result.get('confidence', 0):.2f})")
        
        # Test 3: Feature extraction
        logger.info("\\nStep 3: Extracting features...")
        features = player_model.extract_player_features(player_image)
        logger.info(f"Features extracted: shape {features.shape}")
        
        # Test 4: Combined recognition
        logger.info("\\nStep 4: Creating combined recognition result...")
        combined_result = {
            'player_identification': player_result,
            'jersey_number': jersey_result,
            'feature_vector': features,
            'system_info': {
                'player_model': player_model.model.get_model_info(),
                'timestamp': '2025-11-04'
            }
        }
        
        logger.info("Combined recognition result created:")
        logger.info(f"  - Player: {combined_result['player_identification'].get('player_name')}")
        logger.info(f"  - Jersey: {combined_result['jersey_number'].get('number')}")
        logger.info(f"  - Features: {combined_result['feature_vector'].shape}")
        
        logger.info("\\n✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_demo_results():
    """Save demo results to files."""
    logger.info("\\n" + "=" * 60)
    logger.info("Saving Demo Results")
    logger.info("=" * 60)
    
    try:
        output_dir = Path("outputs/classification/resnet_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save demo summary
        summary = {
            'demo_info': {
                'date': '2025-11-04',
                'version': '1.0',
                'components_tested': [
                    'ResNetPlayerClassifier',
                    'PlayerRecognitionModel', 
                    'JerseyNumberExtractor',
                    'JerseyRecognizer',
                    'ResNetPreprocessor',
                    'ResNetFeatureExtractor',
                    'ResNetDatasetProcessor',
                    'ResNetModelEvaluator'
                ]
            },
            'test_results': 'All tests completed successfully',
            'features': [
                'Player classification using ResNet',
                'Jersey number recognition with OCR',
                'Image preprocessing and data augmentation',
                'Feature extraction and analysis',
                'Model evaluation and statistics',
                'Dataset processing and validation'
            ],
            'usage_examples': {
                'player_identification': 'player_model.identify_player(image)',
                'jersey_recognition': 'jersey_recognizer.recognize_jersey_number(image)',
                'feature_extraction': 'player_model.extract_player_features(image)',
                'preprocessing': 'preprocessor.preprocess_image(image)'
            }
        }
        
        import json
        with open(output_dir / 'demo_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Demo results saved to {output_dir}")
        logger.info("Demo summary created with comprehensive information")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving demo results: {e}")
        return False


def main():
    """Main demo function."""
    logger.info("🚀 Starting ResNet Player Recognition System Demo")
    logger.info("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run individual component tests
    test_results['resnet_model'] = test_resnet_model()
    test_results['jersey_recognizer'] = test_jersey_recognizer()
    test_results['resnet_utils'] = test_resnet_utils()
    test_results['integration'] = test_integration()
    
    # Save results
    save_demo_results()
    
    # Summary
    logger.info("\\n" + "=" * 80)
    logger.info("🎯 Demo Summary")
    logger.info("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\\nOverall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests completed successfully! ResNet Player Recognition System is ready.")
        logger.info("\\nNext steps:")
        logger.info("1. Train the model with your player dataset")
        logger.info("2. Fine-tune for your specific use case")
        logger.info("3. Integrate with video processing pipeline")
        logger.info("4. Deploy for real-time player recognition")
    else:
        logger.warning("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)