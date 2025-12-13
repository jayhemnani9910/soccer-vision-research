"""
Integration Testing Framework for Soccer Player Recognition System

This module provides comprehensive integration testing including:
- End-to-end pipeline testing
- Model interaction and communication
- System integration testing
- Data flow validation
- Configuration integration
- Error handling and recovery
"""

import unittest
import tempfile
import shutil
import json
import yaml
import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from models.model_manager import ModelManager, ModelInstance, model_manager
    from models.model_registry import ModelRegistry, ModelType
    from utils.config_loader import get_config, ConfigLoader
    from utils.performance_monitor import PerformanceMonitor
    from utils.image_utils import ImageUtils
    from utils.video_utils import VideoUtils
    from utils.draw_utils import DrawUtils
    from utils.logger import setup_logger
except ImportError as e:
    logger.warning(f"Could not import all modules: {e}")


class IntegrationTestCase(unittest.TestCase):
    """Base class for integration tests."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, 'test_data')
        self.output_dir = os.path.join(self.temp_dir, 'outputs')
        self.config_dir = os.path.join(self.temp_dir, 'config')
        
        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create test configuration
        self._create_test_configuration()
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_configuration(self):
        """Create test configuration files."""
        # Create YAML config
        config = {
            'models': {
                'test_detector': {
                    'model_path': 'test_detector.pt',
                    'model_type': 'detection',
                    'device': 'cpu',
                    'batch_size': 1,
                    'confidence_threshold': 0.5
                },
                'test_classifier': {
                    'model_path': 'test_classifier.pt',
                    'model_type': 'classification',
                    'device': 'cpu',
                    'batch_size': 1,
                    'num_classes': 1000
                },
                'test_segmenter': {
                    'model_path': 'test_segmenter.pt',
                    'model_type': 'segmentation',
                    'device': 'cpu',
                    'batch_size': 1
                },
                'test_identifier': {
                    'model_path': 'test_identifier.pt',
                    'model_type': 'identification',
                    'device': 'cpu',
                    'batch_size': 1
                }
            },
            'pipeline': {
                'detection_model': 'test_detector',
                'classification_model': 'test_classifier',
                'segmentation_model': 'test_segmenter',
                'identification_model': 'test_identifier',
                'enable_tracking': True,
                'enable_pose_estimation': False
            },
            'output': {
                'detection_output': 'detection/',
                'segmentation_output': 'segmentation/',
                'classification_output': 'classification/',
                'identification_output': 'identification/'
            }
        }
        
        config_file = os.path.join(self.config_dir, 'model_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _create_test_data(self):
        """Create test data files."""
        import numpy as np
        import cv2
        
        # Create test images
        test_images = []
        for i in range(5):
            image_path = os.path.join(self.test_data_dir, f'test_image_{i}.jpg')
            
            # Create a simple test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some simple shapes to make it more realistic
            cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), 2)
            cv2.circle(test_image, (300, 200), 50, (0, 255, 0), 2)
            cv2.putText(test_image, f'Test {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite(image_path, test_image)
            test_images.append(image_path)
        
        self.test_image_paths = test_images
        
        # Create test video metadata
        video_metadata = {
            'fps': 30,
            'duration': 10,  # seconds
            'resolution': (640, 480),
            'total_frames': 300
        }
        
        metadata_file = os.path.join(self.test_data_dir, 'video_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(video_metadata, f)
        
        self.video_metadata = video_metadata


class TestPipelineIntegration(IntegrationTestCase):
    """Test end-to-end pipeline integration."""
    
    def test_complete_detection_pipeline(self):
        """Test complete detection pipeline from input to output."""
        # Create mock models
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Load models
        success = manager.load_model('test_detector', device='cpu')
        self.assertTrue(success, "Failed to load detection model")
        
        # Simulate pipeline processing
        test_image_path = self.test_image_paths[0]
        
        # This would normally be a real image, but we'll use mock input
        import numpy as np
        mock_input = np.random.randn(1, 3, 640, 640)
        
        # Run detection
        result = manager.predict('test_detector', mock_input)
        self.assertIsInstance(result, dict, "Detection result should be a dictionary")
        
        # Verify expected output structure
        expected_keys = ['boxes', 'scores', 'classes']
        has_expected_keys = any(key in result for key in expected_keys)
        self.assertTrue(has_expected_keys, f"Result should contain one of {expected_keys}")
        
        manager.unload_model('test_detector')
    
    def test_complete_classification_pipeline(self):
        """Test complete classification pipeline."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Load model
        success = manager.load_model('test_classifier', device='cpu')
        self.assertTrue(success, "Failed to load classification model")
        
        # Simulate classification
        mock_input = torch.randn(1, 3, 224, 224)
        
        result = manager.predict('test_classifier', mock_input)
        self.assertIsInstance(result, dict, "Classification result should be a dictionary")
        
        # Verify expected output structure
        self.assertTrue(
            'predicted_classes' in result or 'probabilities' in result or 'output' in result,
            "Classification result should contain predicted classes or probabilities"
        )
        
        manager.unload_model('test_classifier')
    
    def test_complete_segmentation_pipeline(self):
        """Test complete segmentation pipeline."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Load model
        success = manager.load_model('test_segmenter', device='cpu')
        self.assertTrue(success, "Failed to load segmentation model")
        
        # Simulate segmentation
        mock_input = torch.randn(1, 3, 256, 256)
        
        result = manager.predict('test_segmenter', mock_input)
        self.assertIsInstance(result, dict, "Segmentation result should be a dictionary")
        
        # Verify expected output structure
        self.assertIn('segmentation_mask', result, "Segmentation result should contain mask")
        
        manager.unload_model('test_segmenter')
    
    def test_multi_model_pipeline(self):
        """Test pipeline with multiple models working together."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Load all models
        models_to_load = ['test_detector', 'test_classifier', 'test_segmenter']
        loaded_models = []
        
        for model_id in models_to_load:
            success = manager.load_model(model_id, device='cpu')
            if success:
                loaded_models.append(model_id)
        
        self.assertGreater(len(loaded_models), 0, "At least one model should load successfully")
        
        # Simulate multi-stage processing
        # Stage 1: Detection
        detection_input = torch.randn(1, 3, 640, 640)
        detection_result = manager.predict('test_detector', detection_input)
        self.assertIsInstance(detection_result, dict)
        
        # Stage 2: Classification (simulated on detection results)
        if 'test_classifier' in loaded_models:
            classification_input = torch.randn(1, 3, 224, 224)
            classification_result = manager.predict('test_classifier', classification_input)
            self.assertIsInstance(classification_result, dict)
        
        # Stage 3: Segmentation (simulated on detection results)
        if 'test_segmenter' in loaded_models:
            segmentation_input = torch.randn(1, 3, 256, 256)
            segmentation_result = manager.predict('test_segmenter', segmentation_input)
            self.assertIsInstance(segmentation_result, dict)
        
        # Cleanup
        for model_id in loaded_models:
            manager.unload_model(model_id)
    
    def test_pipeline_configuration_loading(self):
        """Test pipeline configuration loading and application."""
        # Load configuration
        config = ConfigLoader(config_dir=self.config_dir)
        
        # Verify configuration structure
        pipeline_config = config.get('pipeline', {})
        self.assertIsInstance(pipeline_config, dict)
        
        # Check model assignments
        self.assertIn('detection_model', pipeline_config)
        self.assertIn('classification_model', pipeline_config)
        self.assertEqual(pipeline_config['detection_model'], 'test_detector')
        self.assertEqual(pipeline_config['classification_model'], 'test_classifier')
        
        # Check output configuration
        output_config = config.get('output', {})
        self.assertIsInstance(output_config, dict)
        
        model_configs = config.get('models', {})
        self.assertIn('test_detector', model_configs)
        self.assertEqual(model_configs['test_detector']['model_type'], 'detection')
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        registry = ModelRegistry()
        manager = ModelManager(registry=registry)
        
        # Test with non-existent model
        with self.assertRaises(ValueError):
            result = manager.predict('non_existent_model', torch.randn(1, 3, 224, 224))
        
        # Test with unloaded model and auto_load disabled
        registry.register_model(
            model_id='test_error_handling',
            model_path='dummy_path.pt',
            model_type=ModelType.CLASSIFICATION,
            config={},
            is_active=True
        )
        
        with self.assertRaises(ValueError):
            result = manager.predict('test_error_handling', torch.randn(1, 3, 224, 224), auto_load=False)
    
    def _create_test_registry(self) -> ModelRegistry:
        """Create a test registry with mock models."""
        registry = ModelRegistry()
        
        # Create mock model files and register models
        model_configs = [
            ('test_detector', ModelType.DETECTION, 'test_detector.pt'),
            ('test_classifier', ModelType.CLASSIFICATION, 'test_classifier.pt'),
            ('test_segmenter', ModelType.SEGMENTATION, 'test_segmenter.pt'),
            ('test_identifier', ModelType.IDENTIFICATION, 'test_identifier.pt')
        ]
        
        for model_id, model_type, model_path in model_configs:
            # Create actual model file
            full_path = os.path.join(self.temp_dir, model_path)
            mock_model = self._create_mock_model(model_type)
            torch.save(mock_model, full_path)
            
            # Register model
            registry.register_model(
                model_id=model_id,
                model_path=full_path,
                model_type=model_type,
                config={},
                is_active=True
            )
        
        return registry
    
    def _create_mock_model(self, model_type: ModelType):
        """Create a mock model based on type."""
        if model_type == ModelType.DETECTION:
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 4)  # bounding box coordinates
            )
        elif model_type == ModelType.CLASSIFICATION:
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 1000)
            )
        elif model_type == ModelType.SEGMENTATION:
            return torch.nn.Conv2d(3, 1, 3, padding=1)
        else:  # IDENTIFICATION
            return torch.nn.Sequential(
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128)
            )


class TestDataFlowIntegration(IntegrationTestCase):
    """Test data flow integration between components."""
    
    def test_image_processing_pipeline(self):
        """Test image processing pipeline flow."""
        # This test would typically involve:
        # 1. Loading images
        # 2. Preprocessing
        # 3. Model inference
        # 4. Post-processing
        # 5. Output generation
        
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Load detection model
        manager.load_model('test_detector', device='cpu')
        
        # Simulate image processing pipeline
        for i, image_path in enumerate(self.test_image_paths):
            # In real implementation, this would load and preprocess the image
            mock_processed_input = torch.randn(1, 3, 640, 640)
            
            # Run detection
            result = manager.predict('test_detector', mock_processed_input)
            self.assertIsInstance(result, dict, f"Result should be dict for image {i}")
            
            # Simulate output saving
            output_path = os.path.join(self.output_dir, f'detection_result_{i}.json')
            with open(output_path, 'w') as f:
                json.dump(result, f)
            
            # Verify output file was created
            self.assertTrue(os.path.exists(output_path), f"Output file should exist for image {i}")
        
        manager.unload_model('test_detector')
    
    def test_batch_processing_integration(self):
        """Test batch processing integration."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Load model
        manager.load_model('test_classifier', device='cpu')
        
        # Create batch of inputs
        batch_size = 3
        batch_input = [torch.randn(1, 3, 224, 224) for _ in range(batch_size)]
        
        # Run batch processing
        results = manager.predict_batch('test_classifier', batch_input)
        
        self.assertEqual(len(results), batch_size, "Should return results for all batch items")
        for i, result in enumerate(results):
            self.assertIsInstance(result, dict, f"Result {i} should be a dictionary")
        
        manager.unload_model('test_classifier')
    
    def test_video_processing_pipeline(self):
        """Test video processing pipeline."""
        # This would test frame-by-frame processing
        # For now, we'll simulate it
        
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Load models
        manager.load_model('test_detector', device='cpu')
        
        # Simulate video frames
        total_frames = self.video_metadata['total_frames']
        fps = self.video_metadata['fps']
        
        # Process frames in batches (simulating real-time processing)
        batch_size = 10
        frame_batches = total_frames // batch_size
        
        processing_times = []
        
        for batch_idx in range(frame_batches):
            start_time = time.time()
            
            # Process batch of frames
            batch_input = [torch.randn(1, 3, 640, 640) for _ in range(batch_size)]
            results = manager.predict_batch('test_detector', batch_input)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Verify batch results
            self.assertEqual(len(results), batch_size)
            for result in results:
                self.assertIsInstance(result, dict)
        
        # Check processing performance
        avg_processing_time = sum(processing_times) / len(processing_times)
        expected_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        logger.info(f"Average processing time per batch: {avg_processing_time:.4f}s")
        logger.info(f"Expected processing FPS: {expected_fps:.2f}")
        
        manager.unload_model('test_detector')
    
    def test_data_preprocessing_integration(self):
        """Test data preprocessing integration."""
        # Test image preprocessing pipeline
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        manager.load_model('test_classifier', device='cpu')
        
        # Test different input formats
        input_formats = [
            ("numpy_array", np.random.randn(1, 3, 224, 224)),
            ("tensor", torch.randn(1, 3, 224, 224)),
            ("batch_tensor", torch.randn(4, 3, 224, 224))
        ]
        
        for format_name, test_input in input_formats:
            try:
                result = manager.predict('test_classifier', test_input)
                self.assertIsInstance(result, dict, f"Should handle {format_name} input")
            except Exception as e:
                logger.warning(f"Failed to process {format_name}: {e}")
        
        manager.unload_model('test_classifier')


class TestConfigurationIntegration(IntegrationTestCase):
    """Test configuration system integration."""
    
    def test_config_loader_integration(self):
        """Test configuration loader integration."""
        # Test loading configuration
        config = ConfigLoader(config_dir=self.config_dir)
        
        # Test getting values
        models = config.get('models', {})
        self.assertIsInstance(models, dict)
        
        pipeline = config.get('pipeline', {})
        self.assertIsInstance(pipeline, dict)
        
        # Test model-specific configuration
        detector_config = config.get_model_config('test_detector')
        self.assertIsInstance(detector_config, dict)
        self.assertEqual(detector_config['model_type'], 'detection')
        
        # Test validation
        validation_result = config.validate_config()
        self.assertIsInstance(validation_result, dict)
        self.assertIn('valid', validation_result)
    
    def test_model_registry_integration(self):
        """Test model registry integration."""
        registry = ModelRegistry()
        
        # Test registration
        success = registry.register_model(
            model_id='integration_test_model',
            model_path='/dummy/path/model.pt',
            model_type=ModelType.CLASSIFICATION,
            config={'test_param': 'test_value'},
            is_active=True
        )
        self.assertTrue(success, "Model registration should succeed")
        
        # Test retrieval
        model_info = registry.get_model_info('integration_test_model')
        self.assertIsNotNone(model_info, "Model info should be retrieved")
        self.assertEqual(model_info['model_type'], 'classification')
        self.assertTrue(model_info['is_active'])
        
        # Test deactivation
        success = registry.deactivate_model('integration_test_model')
        self.assertTrue(success, "Model deactivation should succeed")
        
        model_info = registry.get_model_info('integration_test_model')
        self.assertFalse(model_info['is_active'])
    
    def test_environment_configuration(self):
        """Test environment-specific configuration."""
        # Test with different configurations
        test_configs = [
            {
                'name': 'cpu_config',
                'device': 'cpu',
                'batch_size': 1
            },
            {
                'name': 'gpu_config',
                'device': 'cuda',
                'batch_size': 8
            }
        ]
        
        for config_data in test_configs:
            # Create custom config
            custom_config_dir = os.path.join(self.temp_dir, f"config_{config_data['name']}")
            os.makedirs(custom_config_dir, exist_ok=True)
            
            # Update configuration
            config = ConfigLoader(config_dir=self.config_dir)
            config.update_config(f"models.test_detector.device", config_data['device'])
            config.update_config(f"models.test_detector.batch_size", config_data['batch_size'])
            
            # Save updated config
            custom_config_path = os.path.join(custom_config_dir, 'model_config.yaml')
            config.save_config(custom_config_path)
            
            # Verify configuration
            loaded_config = ConfigLoader(config_dir=custom_config_dir)
            detector_config = loaded_config.get_model_config('test_detector')
            
            self.assertEqual(detector_config['device'], config_data['device'])
            self.assertEqual(detector_config['batch_size'], config_data['batch_size'])


class TestErrorHandlingIntegration(IntegrationTestCase):
    """Test error handling and recovery in integrated system."""
    
    def test_model_loading_error_handling(self):
        """Test error handling when model loading fails."""
        registry = ModelRegistry()
        manager = ModelManager(registry=registry)
        
        # Register model with invalid path
        registry.register_model(
            model_id='invalid_model',
            model_path='/non/existent/path/model.pt',
            model_type=ModelType.CLASSIFICATION,
            config={},
            is_active=True
        )
        
        # Loading should fail gracefully
        success = manager.load_model('invalid_model', device='cpu')
        self.assertFalse(success, "Loading should fail for invalid model path")
        
        # Check that model is not in loaded models
        loaded_models = manager.list_loaded_models()
        self.assertNotIn('invalid_model', loaded_models)
    
    def test_prediction_error_handling(self):
        """Test error handling during prediction."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        manager.load_model('test_classifier', device='cpu')
        
        # Test with invalid input
        with self.assertRaises(Exception):
            manager.predict('test_classifier', "invalid_input")
        
        # Test with wrong input shape
        with self.assertRaises(Exception):
            manager.predict('test_classifier', torch.randn(10, 10))  # Wrong shape
        
        manager.unload_model('test_classifier')
    
    def test_resource_cleanup_error_handling(self):
        """Test resource cleanup error handling."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Load model
        manager.load_model('test_detector', device='cpu')
        
        # Test cleanup with closed model
        instance = manager.get_model('test_detector')
        instance.unload_model()
        
        # Try to unload again - should not crash
        success = instance.unload_model()
        self.assertTrue(success, "Second unload should succeed")
    
    def test_concurrent_access_error_handling(self):
        """Test error handling in concurrent scenarios."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        errors = []
        
        def concurrent_prediction(model_id: str, input_data: Any):
            try:
                result = manager.predict(model_id, input_data, auto_load=True)
                return result
            except Exception as e:
                errors.append(e)
                return None
        
        # Load model first
        manager.load_model('test_classifier', device='cpu')
        
        # Run concurrent predictions
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=concurrent_prediction,
                args=('test_classifier', torch.randn(1, 3, 224, 224))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check that no errors occurred
        self.assertEqual(len(errors), 0, f"Concurrent access should not cause errors: {errors}")
        
        manager.unload_model('test_classifier')


class TestPerformanceIntegration(IntegrationTestCase):
    """Test performance integration across components."""
    
    def test_system_performance_integration(self):
        """Test system performance with integrated components."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        # Measure model loading performance
        start_time = time.time()
        loaded_models = []
        
        models_to_load = ['test_detector', 'test_classifier', 'test_segmenter']
        
        for model_id in models_to_load:
            load_start = time.time()
            success = manager.load_model(model_id, device='cpu')
            load_time = time.time() - load_start
            
            if success:
                loaded_models.append(model_id)
                logger.info(f"Loaded {model_id} in {load_time:.4f}s")
        
        total_load_time = time.time() - start_time
        
        # Measure inference performance
        inference_start = time.time()
        
        for model_id in loaded_models:
            # Get appropriate input for model type
            if 'detector' in model_id:
                test_input = torch.randn(1, 3, 640, 640)
            elif 'classifier' in model_id:
                test_input = torch.randn(1, 3, 224, 224)
            else:  # segmenter
                test_input = torch.randn(1, 3, 256, 256)
            
            pred_start = time.time()
            result = manager.predict(model_id, test_input)
            pred_time = time.time() - pred_start
            
            logger.info(f"Inference for {model_id}: {pred_time:.4f}s")
            self.assertIsInstance(result, dict)
        
        total_inference_time = time.time() - inference_start
        
        # Measure cleanup performance
        cleanup_start = time.time()
        for model_id in loaded_models:
            manager.unload_model(model_id)
        cleanup_time = time.time() - cleanup_start
        
        logger.info(f"Total load time: {total_load_time:.4f}s")
        logger.info(f"Total inference time: {total_inference_time:.4f}s")
        logger.info(f"Total cleanup time: {cleanup_time:.4f}s")
        
        # Basic performance assertions
        self.assertLess(total_load_time, 30.0, "Loading should complete within 30 seconds")
        self.assertLess(total_inference_time, 60.0, "Inference should complete within 60 seconds")
        self.assertLess(cleanup_time, 10.0, "Cleanup should complete within 10 seconds")
    
    def test_memory_usage_integration(self):
        """Test memory usage in integrated system."""
        registry = self._create_test_registry()
        manager = ModelManager(registry=registry)
        
        monitor = PerformanceMonitor()
        
        # Measure initial memory
        initial_memory = monitor.get_memory_usage()
        
        # Load models and measure memory usage
        models_loaded = 0
        max_memory = initial_memory
        
        models = ['test_detector', 'test_classifier', 'test_segmenter']
        
        for model_id in models:
            success = manager.load_model(model_id, device='cpu')
            if success:
                models_loaded += 1
                current_memory = monitor.get_memory_usage()
                max_memory = max(max_memory, current_memory)
                logger.info(f"After loading {model_id}: {current_memory:.2f}MB")
        
        # Run some predictions
        for model_id in models[:models_loaded]:
            if 'detector' in model_id:
                test_input = torch.randn(1, 3, 640, 640)
            elif 'classifier' in model_id:
                test_input = torch.randn(1, 3, 224, 224)
            else:
                test_input = torch.randn(1, 3, 256, 256)
            
            result = manager.predict(model_id, test_input)
        
        # Unload models and measure memory
        for model_id in models[:models_loaded]:
            manager.unload_model(model_id)
        
        final_memory = monitor.get_memory_usage()
        memory_growth = final_memory - initial_memory
        
        logger.info(f"Initial memory: {initial_memory:.2f}MB")
        logger.info(f"Peak memory: {max_memory:.2f}MB")
        logger.info(f"Final memory: {final_memory:.2f}MB")
        logger.info(f"Memory growth: {memory_growth:.2f}MB")
        
        # Assert reasonable memory usage (should not grow excessively)
        self.assertLess(memory_growth, 500.0, "Memory growth should be reasonable")
    
    def _create_test_registry(self) -> ModelRegistry:
        """Create a test registry with mock models."""
        registry = ModelRegistry()
        
        # Create mock model files and register models
        model_configs = [
            ('test_detector', ModelType.DETECTION, 'test_detector.pt'),
            ('test_classifier', ModelType.CLASSIFICATION, 'test_classifier.pt'),
            ('test_segmenter', ModelType.SEGMENTATION, 'test_segmenter.pt')
        ]
        
        for model_id, model_type, model_path in model_configs:
            # Create actual model file
            full_path = os.path.join(self.temp_dir, model_path)
            mock_model = self._create_mock_model(model_type)
            torch.save(mock_model, full_path)
            
            # Register model
            registry.register_model(
                model_id=model_id,
                model_path=full_path,
                model_type=model_type,
                config={},
                is_active=True
            )
        
        return registry
    
    def _create_mock_model(self, model_type: ModelType):
        """Create a mock model based on type."""
        if model_type == ModelType.DETECTION:
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(32, 4)
            )
        elif model_type == ModelType.CLASSIFICATION:
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(32, 100)
            )
        else:  # SEGMENTATION
            return torch.nn.Conv2d(3, 1, 3, padding=1)


def create_integration_test_suite():
    """Create and return the integration test suite."""
    suite = unittest.TestSuite()
    
    # Add integration test cases
    suite.addTest(unittest.makeSuite(TestPipelineIntegration))
    suite.addTest(unittest.makeSuite(TestDataFlowIntegration))
    suite.addTest(unittest.makeSuite(TestConfigurationIntegration))
    suite.addTest(unittest.makeSuite(TestErrorHandlingIntegration))
    suite.addTest(unittest.makeSuite(TestPerformanceIntegration))
    
    return suite


if __name__ == '__main__':
    # Create integration test suite
    test_suite = create_integration_test_suite()
    
    # Run tests with verbose output
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    total_time = time.time() - start_time
    
    # Print integration test summary
    print(f"\n{'='*80}")
    print(f"Integration Test Summary:")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All integration tests passed!")
        print("System integration verified successfully.")
    else:
        print("❌ Some integration tests failed!")
        print("System integration issues detected.")
        
    print(f"{'='*80}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)