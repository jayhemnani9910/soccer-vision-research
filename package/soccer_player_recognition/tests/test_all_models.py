"""
Comprehensive Model Testing for Soccer Player Recognition System

This module provides comprehensive testing for all models including:
- RF-DETR (Detection)
- SAM2 (Segmentation) 
- SigLIP (Classification)
- ResNet (Classification)
- MediaPipe (Pose Estimation)
"""

import unittest
import torch
import numpy as np
import cv2
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from unittest.mock import Mock, patch, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from models.model_manager import ModelManager, ModelInstance, model_manager
    from models.model_registry import ModelRegistry, ModelType
    from utils.config_loader import get_config
    from utils.performance_monitor import PerformanceMonitor
except ImportError as e:
    logger.warning(f"Could not import all modules: {e}")


class TestModelInstances(unittest.TestCase):
    """Test individual model instances."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = self._create_test_image()
        
        # Create mock model paths
        self.mock_model_paths = {
            'detection': os.path.join(self.temp_dir, 'mock_detection.pt'),
            'classification': os.path.join(self.temp_dir, 'mock_classification.pt'),
            'identification': os.path.join(self.temp_dir, 'mock_identification.pt'),
            'segmentation': os.path.join(self.temp_dir, 'mock_segmentation.pt')
        }
        
        # Create mock model files
        for path in self.mock_model_paths.values():
            self._create_mock_model_file(path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_image(self) -> str:
        """Create a test image file."""
        image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        # Create a simple test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(image_path, test_image)
        return image_path
    
    def _create_mock_model_file(self, path: str):
        """Create a mock model file."""
        # Create a simple mock model
        mock_model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
        torch.save(mock_model, path)
    
    def test_model_instance_creation(self):
        """Test ModelInstance creation."""
        config = {'batch_size': 32, 'device': 'cpu'}
        
        instance = ModelInstance(
            model_id='test_model',
            model_path=self.mock_model_paths['detection'],
            config=config,
            model_type=ModelType.DETECTION,
            device='cpu'
        )
        
        self.assertEqual(instance.model_id, 'test_model')
        self.assertEqual(instance.model_type, ModelType.DETECTION)
        self.assertFalse(instance.is_loaded)
        self.assertEqual(instance.device.type, 'cpu')
    
    def test_model_loading(self):
        """Test model loading functionality."""
        config = {'batch_size': 32}
        
        instance = ModelInstance(
            model_id='test_classification',
            model_path=self.mock_model_paths['classification'],
            config=config,
            model_type=ModelType.CLASSIFICATION,
            device='cpu'
        )
        
        # Test loading
        success = instance.load_model()
        self.assertTrue(success)
        self.assertTrue(instance.is_loaded)
        self.assertIsNotNone(instance.model)
        self.assertGreater(instance.load_time, 0)
    
    def test_model_unloading(self):
        """Test model unloading functionality."""
        config = {}
        
        instance = ModelInstance(
            model_id='test_segmentation',
            model_path=self.mock_model_paths['segmentation'],
            config=config,
            model_type=ModelType.SEGMENTATION,
            device='cpu'
        )
        
        # Load and then unload
        instance.load_model()
        self.assertTrue(instance.is_loaded)
        
        success = instance.unload_model()
        self.assertTrue(success)
        self.assertFalse(instance.is_loaded)
    
    def test_detection_prediction(self):
        """Test detection model prediction."""
        instance = ModelInstance(
            model_id='test_detection',
            model_path=self.mock_model_paths['detection'],
            config={},
            model_type=ModelType.DETECTION,
            device='cpu'
        )
        
        instance.load_model()
        
        # Test with mock input
        test_input = torch.randn(1, 3, 640, 640)
        result = instance.predict(test_input)
        
        self.assertIsInstance(result, dict)
        # Check for expected keys in detection results
        self.assertTrue(any(key in result for key in ['boxes', 'scores', 'classes', 'output']))
    
    def test_classification_prediction(self):
        """Test classification model prediction."""
        instance = ModelInstance(
            model_id='test_classification',
            model_path=self.mock_model_paths['classification'],
            config={},
            model_type=ModelType.CLASSIFICATION,
            device='cpu'
        )
        
        instance.load_model()
        
        # Test with mock input
        test_input = torch.randn(1, 784)
        result = instance.predict(test_input)
        
        self.assertIsInstance(result, dict)
        # Classification should return predicted classes or output
        self.assertTrue('predicted_classes' in result or 'output' in result)
    
    def test_identification_prediction(self):
        """Test identification model prediction."""
        instance = ModelInstance(
            model_id='test_identification',
            model_path=self.mock_model_paths['identification'],
            config={},
            model_type=ModelType.IDENTIFICATION,
            device='cpu'
        )
        
        instance.load_model()
        
        # Test with mock input
        test_input = torch.randn(1, 512)
        result = instance.predict(test_input)
        
        self.assertIsInstance(result, dict)
        # Identification should return features
        self.assertTrue('features' in result)
    
    def test_segmentation_prediction(self):
        """Test segmentation model prediction."""
        instance = ModelInstance(
            model_id='test_segmentation',
            model_path=self.mock_model_paths['segmentation'],
            config={},
            model_type=ModelType.SEGMENTATION,
            device='cpu'
        )
        
        instance.load_model()
        
        # Test with mock input
        test_input = torch.randn(1, 3, 256, 256)
        result = instance.predict(test_input)
        
        self.assertIsInstance(result, dict)
        # Segmentation should return segmentation mask
        self.assertTrue('segmentation_mask' in result)
    
    def test_model_statistics(self):
        """Test model statistics tracking."""
        instance = ModelInstance(
            model_id='test_stats',
            model_path=self.mock_model_paths['classification'],
            config={},
            model_type=ModelType.CLASSIFICATION,
            device='cpu'
        )
        
        instance.load_model()
        
        # Perform multiple predictions to generate statistics
        test_input = torch.randn(1, 784)
        for _ in range(5):
            instance.predict(test_input)
        
        stats = instance.get_statistics()
        
        self.assertEqual(stats['model_id'], 'test_stats')
        self.assertTrue(stats['is_loaded'])
        self.assertEqual(stats['inference_count'], 5)
        self.assertGreater(stats['avg_inference_time'], 0)
        self.assertIsNotNone(stats['load_time'])
    
    def test_device_detection(self):
        """Test automatic device detection."""
        # Test CPU detection
        instance = ModelInstance(
            model_id='test_device',
            model_path=self.mock_model_paths['detection'],
            config={},
            model_type=ModelType.DETECTION,
            device='cpu'
        )
        self.assertEqual(instance.device.type, 'cpu')
        
        # Test auto detection
        instance_auto = ModelInstance(
            model_id='test_device_auto',
            model_path=self.mock_model_paths['detection'],
            config={},
            model_type=ModelType.DETECTION,
            device='auto'
        )
        # Should be either cpu, cuda, or mps
        self.assertIn(instance_auto.device.type, ['cpu', 'cuda', 'mps'])
    
    def test_thread_safety(self):
        """Test thread safety of model instance."""
        import threading
        import queue
        
        instance = ModelInstance(
            model_id='test_thread_safety',
            model_path=self.mock_model_paths['classification'],
            config={},
            model_type=ModelType.CLASSIFICATION,
            device='cpu'
        )
        
        instance.load_model()
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def predict_in_thread():
            try:
                test_input = torch.randn(1, 784)
                result = instance.predict(test_input)
                results_queue.put(result)
            except Exception as e:
                errors_queue.put(e)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=predict_in_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(errors_queue.qsize(), 0, "No errors should occur in threading")
        self.assertEqual(results_queue.qsize(), 5)


class TestModelManager(unittest.TestCase):
    """Test ModelManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model files
        self.mock_model_paths = {
            'rf_detr': os.path.join(self.temp_dir, 'rf_detr.pt'),
            'sam2': os.path.join(self.temp_dir, 'sam2.pt'),
            'resnet': os.path.join(self.temp_dir, 'resnet.pt'),
            'siglip': os.path.join(self.temp_dir, 'siglip.pt')
        }
        
        for path in self.mock_model_paths.values():
            self._create_mock_model_file(path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_model_file(self, path: str):
        """Create a mock model file."""
        mock_model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )
        torch.save(mock_model, path)
    
    @patch('models.model_manager.ModelRegistry')
    def test_model_manager_initialization(self, mock_registry):
        """Test ModelManager initialization."""
        mock_registry_instance = Mock()
        mock_registry.return_value = mock_registry_instance
        
        manager = ModelManager(registry=mock_registry_instance)
        
        self.assertEqual(manager.registry, mock_registry_instance)
        self.assertEqual(len(manager.model_instances), 0)
        self.assertEqual(manager.max_loaded_models, 10)
        self.assertEqual(manager.memory_threshold, 0.8)
    
    def test_model_loading_and_unloading(self):
        """Test loading and unloading models through manager."""
        # Create a mock registry
        registry = ModelRegistry()
        
        # Add mock models to registry
        registry.register_model(
            model_id='test_detector',
            model_path=self.mock_model_paths['rf_detr'],
            model_type=ModelType.DETECTION,
            config={'batch_size': 16},
            is_active=True
        )
        
        manager = ModelManager(registry=registry)
        
        # Test loading
        success = manager.load_model('test_detector', device='cpu')
        self.assertTrue(success)
        self.assertIn('test_detector', manager.list_loaded_models())
        
        # Test unloading
        success = manager.unload_model('test_detector')
        self.assertTrue(success)
        self.assertNotIn('test_detector', manager.list_loaded_models())
    
    def test_get_model(self):
        """Test getting model instances."""
        registry = ModelRegistry()
        registry.register_model(
            model_id='test_classifier',
            model_path=self.mock_model_paths['resnet'],
            model_type=ModelType.CLASSIFICATION,
            config={},
            is_active=True
        )
        
        manager = ModelManager(registry=registry)
        manager.load_model('test_classifier', device='cpu')
        
        # Test getting loaded model
        instance = manager.get_model('test_classifier')
        self.assertIsNotNone(instance)
        self.assertEqual(instance.model_id, 'test_classifier')
        
        # Test getting unloaded model
        instance = manager.get_model('nonexistent_model')
        self.assertIsNone(instance)
    
    def test_predict_with_auto_load(self):
        """Test prediction with auto-loading."""
        registry = ModelRegistry()
        registry.register_model(
            model_id='test_auto_load',
            model_path=self.mock_model_paths['siglip'],
            model_type=ModelType.CLASSIFICATION,
            config={},
            is_active=True
        )
        
        manager = ModelManager(registry=registry)
        
        # Test auto-loading
        test_input = torch.randn(1, 512)
        result = manager.predict('test_auto_load', test_input, auto_load=True)
        
        self.assertIsInstance(result, dict)
        self.assertIn('test_auto_load', manager.list_loaded_models())
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        registry = ModelRegistry()
        registry.register_model(
            model_id='test_batch',
            model_path=self.mock_model_paths['resnet'],
            model_type=ModelType.CLASSIFICATION,
            config={},
            is_active=True
        )
        
        manager = ModelManager(registry=registry)
        
        # Create batch input
        batch_input = [torch.randn(1, 512) for _ in range(3)]
        results = manager.predict_batch('test_batch', batch_input)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, dict)
    
    def test_statistics_tracking(self):
        """Test statistics tracking in manager."""
        registry = ModelRegistry()
        registry.register_model(
            model_id='test_stats',
            model_path=self.mock_model_paths['rf_detr'],
            model_type=ModelType.DETECTION,
            config={},
            is_active=True
        )
        
        manager = ModelManager(registry=registry)
        manager.load_model('test_stats', device='cpu')
        
        # Perform some predictions
        test_input = torch.randn(1, 3, 640, 640)
        for _ in range(3):
            manager.predict('test_stats', test_input)
        
        # Check global statistics
        stats = manager.get_all_statistics()
        self.assertIn('global', stats)
        self.assertIn('models', stats)
        self.assertGreater(stats['global']['total_predictions'], 0)
        
        # Check model-specific statistics
        model_stats = stats['models']['test_stats']
        self.assertEqual(model_stats['inference_count'], 3)
    
    def test_memory_management(self):
        """Test memory cleanup functionality."""
        registry = ModelRegistry()
        
        # Register multiple models
        for i in range(15):  # More than max_loaded_models
            registry.register_model(
                model_id=f'test_model_{i}',
                model_path=self.mock_model_paths['resnet'],
                model_type=ModelType.CLASSIFICATION,
                config={},
                is_active=True
            )
        
        manager = ModelManager(registry=registry)
        manager.max_loaded_models = 5
        
        # Load many models
        for i in range(15):
            manager.load_model(f'test_model_{i}', device='cpu')
        
        # Check that cleanup occurred
        self.assertLessEqual(len(manager.model_instances), manager.max_loaded_models)
        
        # Test manual cleanup
        unloaded_count = manager.cleanup_memory()
        self.assertGreaterEqual(unloaded_count, 0)


class TestModelTypes(unittest.TestCase):
    """Test all model types specified in the project."""
    
    def setUp(self):
        """Set up test fixtures for all model types."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_inputs = self._create_test_inputs()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_inputs(self) -> Dict[str, Any]:
        """Create test inputs for different model types."""
        return {
            'detection': {
                'image_tensor': torch.randn(1, 3, 640, 640),
                'image_path': self._create_test_image('detection')
            },
            'classification': {
                'image_tensor': torch.randn(1, 3, 224, 224),
                'batch_tensor': torch.randn(8, 3, 224, 224),
                'image_path': self._create_test_image('classification')
            },
            'identification': {
                'feature_tensor': torch.randn(1, 512),
                'image_tensor': torch.randn(1, 3, 224, 224)
            },
            'segmentation': {
                'image_tensor': torch.randn(1, 3, 256, 256),
                'mask_tensor': torch.randint(0, 2, (1, 256, 256)).float()
            },
            'pose': {
                'image_tensor': torch.randn(1, 3, 224, 224),
                'image_path': self._create_test_image('pose')
            }
        }
    
    def _create_test_image(self, prefix: str) -> str:
        """Create a test image file."""
        image_path = os.path.join(self.temp_dir, f'test_{prefix}.jpg')
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(image_path, test_image)
        return image_path
    
    def test_rf_detr_detection(self):
        """Test RF-DETR detection model."""
        model_type = ModelType.DETECTION
        test_input = self.test_inputs['detection']['image_tensor']
        
        # Create temporary model file
        model_path = os.path.join(self.temp_dir, 'rf_detr_test.pt')
        mock_model = torch.nn.Sequential(torch.nn.Linear(1, 1))
        torch.save(mock_model, model_path)
        
        instance = ModelInstance(
            model_id='rf_detr_test',
            model_path=model_path,
            config={'confidence_threshold': 0.5},
            model_type=model_type,
            device='cpu'
        )
        
        instance.load_model()
        result = instance.predict(test_input)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(any(key in result for key in ['boxes', 'scores', 'classes', 'output']))
    
    def test_sam2_segmentation(self):
        """Test SAM2 segmentation model."""
        model_type = ModelType.SEGMENTATION
        test_input = self.test_inputs['segmentation']['image_tensor']
        
        # Create temporary model file
        model_path = os.path.join(self.temp_dir, 'sam2_test.pt')
        mock_model = torch.nn.Conv2d(3, 1, 3, padding=1)
        torch.save(mock_model, model_path)
        
        instance = ModelInstance(
            model_id='sam2_test',
            model_path=model_path,
            config={'mask_threshold': 0.5},
            model_type=model_type,
            device='cpu'
        )
        
        instance.load_model()
        result = instance.predict(test_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn('segmentation_mask', result)
    
    def test_resnet_classification(self):
        """Test ResNet classification model."""
        model_type = ModelType.CLASSIFICATION
        test_input = self.test_inputs['classification']['image_tensor']
        
        # Create temporary model file
        model_path = os.path.join(self.temp_dir, 'resnet_test.pt')
        mock_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1000)  # ImageNet classes
        )
        torch.save(mock_model, model_path)
        
        instance = ModelInstance(
            model_id='resnet_test',
            model_path=model_path,
            config={'num_classes': 1000},
            model_type=model_type,
            device='cpu'
        )
        
        instance.load_model()
        result = instance.predict(test_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn('predicted_classes', result)
        self.assertIn('probabilities', result)
    
    def test_siglip_classification(self):
        """Test SigLIP classification model."""
        model_type = ModelType.CLASSIFICATION
        test_input = self.test_inputs['classification']['batch_tensor']
        
        # Create temporary model file
        model_path = os.path.join(self.temp_dir, 'siglip_test.pt')
        mock_model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 100)
        )
        torch.save(mock_model, model_path)
        
        instance = ModelInstance(
            model_id='siglip_test',
            model_path=model_path,
            config={'embedding_dim': 512, 'num_classes': 100},
            model_type=model_type,
            device='cpu'
        )
        
        instance.load_model()
        result = instance.predict(test_input)
        
        self.assertIsInstance(result, dict)
        self.assertTrue('predicted_classes' in result or 'output' in result)
    
    def test_pose_estimation(self):
        """Test MediaPipe pose estimation model."""
        model_type = ModelType.POSE
        test_input = self.test_inputs['pose']['image_path']
        
        # Create temporary model file
        model_path = os.path.join(self.temp_dir, 'pose_test.pt')
        mock_model = torch.nn.Linear(1, 1)
        torch.save(mock_model, model_path)
        
        instance = ModelInstance(
            model_id='pose_test',
            model_path=model_path,
            config={'landmark_count': 33},
            model_type=model_type,
            device='cpu'
        )
        
        instance.load_model()
        
        # Pose estimation should handle different input types
        result = instance.predict(test_input)
        self.assertIsInstance(result, dict)
        
        # If MediaPipe is available, it should return landmarks
        # Otherwise, it should return a generic output
        self.assertTrue('landmarks' in result or 'pose_output' in result)


class TestModelConfiguration(unittest.TestCase):
    """Test model configuration handling."""
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        config = get_config()
        validation_results = config.validate_config()
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn('valid', validation_results)
        self.assertIn('errors', validation_results)
        self.assertIn('warnings', validation_results)
    
    def test_get_model_config(self):
        """Test getting model-specific configurations."""
        config = get_config()
        
        # Test getting config for existing models
        models = ['rf_detr', 'sam2', 'siglip', 'resnet']
        for model_name in models:
            model_config = config.get_model_config(model_name)
            self.assertIsInstance(model_config, dict)
    
    def test_device_configuration(self):
        """Test device configuration handling."""
        config = get_config()
        
        # Test general device config
        device = config.get_device()
        self.assertIn(device, ['cpu', 'cuda', 'mps'])
        
        # Test model-specific device config
        for model_name in ['rf_detr', 'sam2', 'siglip', 'resnet']:
            model_device = config.get_device(model_name)
            self.assertIsInstance(model_device, str)
    
    def test_batch_size_configuration(self):
        """Test batch size configuration."""
        config = get_config()
        
        # Test batch size for different models
        for model_name in ['rf_detr', 'sam2', 'siglip', 'resnet']:
            batch_size = config.get_batch_size(model_name)
            self.assertIsInstance(batch_size, int)
            self.assertGreater(batch_size, 0)


def create_test_suite():
    """Create and return the test suite."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestModelInstances))
    suite.addTest(unittest.makeSuite(TestModelManager))
    suite.addTest(unittest.makeSuite(TestModelTypes))
    suite.addTest(unittest.makeSuite(TestModelConfiguration))
    
    return suite


if __name__ == '__main__':
    # Create test suite
    test_suite = create_test_suite()
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)