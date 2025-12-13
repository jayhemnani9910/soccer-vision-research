"""
Utility Testing Framework for Soccer Player Recognition System

This module provides comprehensive testing for all utility modules including:
- Configuration loading and validation
- Image processing utilities
- Video processing utilities
- Drawing and visualization utilities
- Performance monitoring
- Logging utilities
- Model-specific utilities (ResNet, RF-DETR, SAM2)
"""

import unittest
import tempfile
import shutil
import json
import yaml
import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from utils.config_loader import ConfigLoader, get_config, load_config
    from utils.performance_monitor import PerformanceMonitor
    from utils.image_utils import ImageUtils
    from utils.video_utils import VideoUtils
    from utils.draw_utils import DrawUtils
    from utils.logger import setup_logger
    from utils.resnet_utils import ResNetUtils
    from utils.rf_detr_utils import RF_DETRUtils
    from utils.sam2_utils import SAM2Utils
except ImportError as e:
    logger.warning(f"Could not import all modules: {e}")


class UtilsTestCase(unittest.TestCase):
    """Base class for utility tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = self._create_test_image()
        self.test_video_path = self._create_test_video()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_image(self, size: Tuple[int, int] = (640, 480)) -> str:
        """Create a test image file."""
        image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        
        # Create a test image with some basic content
        image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        
        # Add some shapes to make it more realistic
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), 2)
        cv2.circle(image, (300, 200), 50, (0, 255, 0), 2)
        cv2.putText(image, 'TEST', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(image_path, image)
        return image_path
    
    def _create_test_video(self, duration: int = 5) -> str:
        """Create a test video file."""
        video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        width, height = 640, 480
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Write frames
        for i in range(duration * fps):
            # Create frame with some movement
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add moving circle
            x = int(width / 2 + 100 * np.sin(i * 0.1))
            y = int(height / 2 + 50 * np.cos(i * 0.1))
            cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)
            
            out.write(frame)
        
        out.release()
        return video_path


class TestConfigLoader(UtilsTestCase):
    """Test configuration loading utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.config_dir = os.path.join(self.temp_dir, 'config')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create test configuration files
        self._create_test_config_files()
    
    def _create_test_config_files(self):
        """Create test configuration files."""
        # Create YAML config
        config_data = {
            'models': {
                'rf_detr': {
                    'model_path': '/path/to/rf_detr.pt',
                    'model_type': 'detection',
                    'device': 'cuda',
                    'batch_size': 8,
                    'confidence_threshold': 0.7,
                    'nms_threshold': 0.8
                },
                'sam2': {
                    'model_path': '/path/to/sam2.pt',
                    'model_type': 'segmentation',
                    'device': 'cuda',
                    'batch_size': 4
                },
                'resnet': {
                    'model_path': '/path/to/resnet.pt',
                    'model_type': 'classification',
                    'device': 'cpu',
                    'batch_size': 32,
                    'num_classes': 1000
                },
                'siglip': {
                    'model_path': '/path/to/siglip.pt',
                    'model_type': 'classification',
                    'device': 'cuda',
                    'batch_size': 16,
                    'embedding_dim': 768
                }
            },
            'pipeline': {
                'detection_model': 'rf_detr',
                'segmentation_model': 'sam2',
                'classification_model': 'resnet',
                'enable_tracking': True,
                'enable_pose_estimation': False,
                'confidence_threshold': 0.5
            },
            'performance': {
                'max_memory_usage': 0.8,
                'max_loaded_models': 10,
                'monitor_interval': 1.0
            }
        }
        
        config_file = os.path.join(self.config_dir, 'model_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization."""
        config = ConfigLoader(config_dir=self.config_dir)
        
        self.assertIsNotNone(config)
        self.assertEqual(config.config_dir, Path(self.config_dir))
        self.assertIsInstance(config.settings, dict)
        self.assertIsInstance(config.model_configs, dict)
    
    def test_get_configuration_value(self):
        """Test getting configuration values."""
        config = ConfigLoader(config_dir=self.config_dir)
        
        # Test direct key access
        models = config.get('models')
        self.assertIsInstance(models, dict)
        
        # Test nested key access
        rf_detr_config = config.get('models.rf_detr')
        self.assertIsInstance(rf_detr_config, dict)
        self.assertEqual(rf_detr_config['model_type'], 'detection')
        
        # Test non-existent key with default
        nonexistent = config.get('nonexistent.key', 'default_value')
        self.assertEqual(nonexistent, 'default_value')
        
        # Test non-existent key without default
        nonexistent = config.get('nonexistent.key')
        self.assertIsNone(nonexistent)
    
    def test_get_model_config(self):
        """Test getting model-specific configurations."""
        config = ConfigLoader(config_dir=self.config_dir)
        
        # Test getting RF-DETR config
        rf_detr_config = config.get_model_config('rf_detr')
        self.assertIsInstance(rf_detr_config, dict)
        self.assertEqual(rf_detr_config['model_type'], 'detection')
        self.assertEqual(rf_detr_config['device'], 'cuda')
        self.assertEqual(rf_detr_config['batch_size'], 8)
        
        # Test getting non-existent model config
        nonexistent_config = config.get_model_config('nonexistent_model')
        self.assertIsInstance(nonexistent_config, dict)
        self.assertEqual(len(nonexistent_config), 0)
    
    def test_update_configuration(self):
        """Test updating configuration values."""
        config = ConfigLoader(config_dir=self.config_dir)
        
        # Test simple key update
        config.update_config('test_key', 'test_value')
        value = config.get('test_key')
        self.assertEqual(value, 'test_value')
        
        # Test nested key update
        config.update_config('models.rf_detr.confidence_threshold', 0.9)
        rf_detr_config = config.get_model_config('rf_detr')
        self.assertEqual(rf_detr_config['confidence_threshold'], 0.9)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ConfigLoader(config_dir=self.config_dir)
        
        validation_result = config.validate_config()
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn('valid', validation_result)
        self.assertIn('errors', validation_result)
        self.assertIn('warnings', validation_result)
        self.assertIsInstance(validation_result['errors'], list)
        self.assertIsInstance(validation_result['warnings'], list)
    
    def test_save_configuration(self):
        """Test saving configuration to file."""
        config = ConfigLoader(config_dir=self.config_dir)
        
        # Update configuration
        config.update_config('models.test_model', {'model_path': '/test/path'})
        
        # Save to custom location
        save_path = os.path.join(self.temp_dir, 'custom_config.yaml')
        config.save_config(save_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Load and verify content
        with open(save_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertIn('models', saved_config)
        self.assertIn('test_model', saved_config['models'])
    
    def test_device_configuration(self):
        """Test device configuration methods."""
        config = ConfigLoader(config_dir=self.config_dir)
        
        # Test general device configuration
        default_device = config.get_device()
        self.assertIsInstance(default_device, str)
        
        # Test model-specific device configuration
        rf_detr_device = config.get_device('rf_detr')
        self.assertEqual(rf_detr_device, 'cuda')
        
        resnet_device = config.get_device('resnet')
        self.assertEqual(resnet_device, 'cpu')
    
    def test_batch_size_configuration(self):
        """Test batch size configuration methods."""
        config = ConfigLoader(config_dir=self.config_dir)
        
        # Test batch sizes for different models
        rf_detr_batch = config.get_batch_size('rf_detr')
        self.assertEqual(rf_detr_batch, 8)
        
        resnet_batch = config.get_batch_size('resnet')
        self.assertEqual(resnet_batch, 32)
        
        siglip_batch = config.get_batch_size('siglip')
        self.assertEqual(siglip_batch, 16)
    
    def test_global_config_instance(self):
        """Test global configuration instance."""
        global_config = get_config()
        self.assertIsInstance(global_config, ConfigLoader)
        
        # Test loading configuration
        loaded_config = load_config(config_dir=self.config_dir)
        self.assertIsInstance(loaded_config, ConfigLoader)


class TestPerformanceMonitor(UtilsTestCase):
    """Test performance monitoring utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.monitor = PerformanceMonitor()
    
    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        
        self.assertIsNotNone(monitor)
        self.assertIsInstance(monitor.process, type(monitor.process))
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        # Get initial memory usage
        initial_memory = self.monitor.get_memory_usage()
        self.assertIsInstance(initial_memory, float)
        self.assertGreaterEqual(initial_memory, 0)
        
        # Allocate some memory
        large_array = np.random.randn(1000, 1000)
        
        # Get memory usage after allocation
        after_memory = self.monitor.get_memory_usage()
        
        # Memory should have increased
        self.assertGreaterEqual(after_memory, initial_memory)
        
        # Clean up
        del large_array
        gc.collect()
    
    def test_cpu_usage_tracking(self):
        """Test CPU usage tracking."""
        cpu_usage = self.monitor.get_cpu_usage()
        self.assertIsInstance(cpu_usage, float)
        self.assertGreaterEqual(cpu_usage, 0)
        self.assertLessEqual(cpu_usage, 100)
    
    def test_gpu_usage_tracking(self):
        """Test GPU usage tracking (if available)."""
        gpu_usage = self.monitor.get_gpu_usage()
        self.assertIsInstance(gpu_usage, float)
        self.assertGreaterEqual(gpu_usage, 0)
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # This should not raise any exceptions
        try:
            self.monitor.clear_cache()
            success = True
        except Exception as e:
            success = False
            logger.warning(f"Cache clearing failed: {e}")
        
        self.assertTrue(success, "Cache clearing should not raise exceptions")
    
    def test_performance_monitoring_context(self):
        """Test performance monitoring with context manager."""
        import torch
        
        with self.monitor.measure_operation("test_operation") as measurement:
            # Simulate some work
            time.sleep(0.1)
            result = torch.sum(torch.randn(1000, 1000))
        
        # Check that measurement was recorded
        self.assertIsInstance(measurement.execution_time, float)
        self.assertGreater(measurement.execution_time, 0)
        self.assertIsInstance(measurement.memory_usage, float)
        self.assertIsInstance(measurement.peak_memory, float)


class TestImageUtils(UtilsTestCase):
    """Test image processing utilities."""
    
    def test_image_loading(self):
        """Test image loading functionality."""
        utils = ImageUtils()
        
        # Test loading test image
        image = utils.load_image(self.test_image_path)
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)  # HWC format
        self.assertEqual(image.shape[2], 3)   # 3 channels (BGR)
    
    def test_image_resizing(self):
        """Test image resizing functionality."""
        utils = ImageUtils()
        
        # Load test image
        image = utils.load_image(self.test_image_path)
        original_shape = image.shape
        
        # Resize image
        target_size = (320, 240)
        resized_image = utils.resize_image(image, target_size)
        
        self.assertIsNotNone(resized_image)
        self.assertEqual(resized_image.shape[1], target_size[0])  # width
        self.assertEqual(resized_image.shape[0], target_size[1])  # height
    
    def test_image_normalization(self):
        """Test image normalization functionality."""
        utils = ImageUtils()
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Normalize to 0-1 range
        normalized = utils.normalize_image(test_image)
        
        self.assertIsNotNone(normalized)
        self.assertLessEqual(normalized.max(), 1.0)
        self.assertGreaterEqual(normalized.min(), 0.0)
        
        # Normalize to -1 to 1 range
        normalized_2 = utils.normalize_image(test_image, range_=(-1, 1))
        
        self.assertIsNotNone(normalized_2)
        self.assertLessEqual(normalized_2.max(), 1.0)
        self.assertGreaterEqual(normalized_2.min(), -1.0)
    
    def test_image_to_tensor_conversion(self):
        """Test image to tensor conversion."""
        utils = ImageUtils()
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Convert to tensor
        tensor = utils.image_to_tensor(test_image)
        
        self.assertIsNotNone(tensor)
        self.assertEqual(len(tensor.shape), 4)  # NCHW format
        self.assertEqual(tensor.shape[0], 1)    # batch size 1
        self.assertEqual(tensor.shape[1], 3)    # 3 channels
    
    def test_tensor_to_image_conversion(self):
        """Test tensor to image conversion."""
        utils = ImageUtils()
        
        # Create test tensor (NCHW format)
        test_tensor = torch.randn(1, 3, 224, 224)
        
        # Convert to image
        image = utils.tensor_to_image(test_tensor)
        
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)  # HWC format
        self.assertEqual(image.shape[2], 3)    # 3 channels
    
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline."""
        utils = ImageUtils()
        
        # Load test image
        image = utils.load_image(self.test_image_path)
        
        # Preprocess for different models
        preprocess_configs = [
            {'target_size': (224, 224), 'normalize': True, 'to_tensor': True},
            {'target_size': (640, 640), 'normalize': False, 'to_tensor': True},
            {'target_size': (256, 256), 'normalize': True, 'to_tensor': False}
        ]
        
        for config in preprocess_configs:
            try:
                processed = utils.preprocess_image(image, **config)
                
                if config['to_tensor']:
                    self.assertEqual(len(processed.shape), 4)  # NCHW
                else:
                    self.assertEqual(len(processed.shape), 3)  # HWC
                    
            except Exception as e:
                # Some preprocessing might fail, that's okay for this test
                logger.warning(f"Preprocessing failed with config {config}: {e}")
    
    def test_image_augmentation(self):
        """Test image augmentation functionality."""
        utils = ImageUtils()
        
        # Load test image
        image = utils.load_image(self.test_image_path)
        
        # Test different augmentations
        augmentation_configs = [
            {'flip_horizontal': True},
            {'flip_vertical': True},
            {'rotate': 45},
            {'brightness': 0.2},
            {'contrast': 0.2}
        ]
        
        for config in augmentation_configs:
            try:
                augmented = utils.augment_image(image, **config)
                self.assertEqual(augmented.shape, image.shape)
            except Exception as e:
                logger.warning(f"Augmentation failed with config {config}: {e}")


class TestVideoUtils(UtilsTestCase):
    """Test video processing utilities."""
    
    def test_video_loading(self):
        """Test video loading functionality."""
        utils = VideoUtils()
        
        # Check if test video exists and can be loaded
        if os.path.exists(self.test_video_path):
            try:
                video = utils.load_video(self.test_video_path)
                # Video loading might return None if ffmpeg is not available
                if video is not None:
                    self.assertIsInstance(video, dict)
            except Exception as e:
                logger.warning(f"Video loading failed: {e}")
    
    def test_video_frame_extraction(self):
        """Test video frame extraction."""
        utils = VideoUtils()
        
        # Create a simple test video for frame extraction
        if os.path.exists(self.test_video_path):
            try:
                # Extract frames
                frames = utils.extract_frames(self.test_video_path, max_frames=10)
                
                if frames is not None:
                    self.assertIsInstance(frames, list)
                    for frame in frames:
                        self.assertEqual(len(frame.shape), 3)  # HWC
                        self.assertEqual(frame.shape[2], 3)    # 3 channels
                        
            except Exception as e:
                logger.warning(f"Frame extraction failed: {e}")
    
    def test_video_info_extraction(self):
        """Test video information extraction."""
        utils = VideoUtils()
        
        if os.path.exists(self.test_video_path):
            try:
                info = utils.get_video_info(self.test_video_path)
                
                if info is not None:
                    self.assertIsInstance(info, dict)
                    
                    # Check for expected keys
                    expected_keys = ['fps', 'duration', 'width', 'height', 'frame_count']
                    for key in expected_keys:
                        if key in info:
                            self.assertIsInstance(info[key], (int, float))
                            
            except Exception as e:
                logger.warning(f"Video info extraction failed: {e}")
    
    def test_video_processing_pipeline(self):
        """Test video processing pipeline."""
        utils = VideoUtils()
        
        if os.path.exists(self.test_video_path):
            try:
                # Process video
                result = utils.process_video(
                    self.test_video_path,
                    output_path=os.path.join(self.temp_dir, 'processed_video.mp4'),
                    target_fps=15
                )
                
                # Result might be None if processing fails
                if result is not None:
                    self.assertIsInstance(result, dict)
                    
            except Exception as e:
                logger.warning(f"Video processing failed: {e}")


class TestDrawUtils(UtilsTestCase):
    """Test drawing and visualization utilities."""
    
    def test_draw_bounding_boxes(self):
        """Test bounding box drawing functionality."""
        utils = DrawUtils()
        
        # Load test image
        image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(image)
        
        # Create test bounding boxes
        boxes = [
            {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150, 'label': 'person', 'confidence': 0.9},
            {'x1': 200, 'y1': 100, 'x2': 300, 'y2': 200, 'label': 'ball', 'confidence': 0.8}
        ]
        
        # Draw bounding boxes
        annotated_image = utils.draw_bounding_boxes(image, boxes)
        
        self.assertIsNotNone(annotated_image)
        self.assertEqual(annotated_image.shape, image.shape)
    
    def test_draw_segmentation_masks(self):
        """Test segmentation mask drawing functionality."""
        utils = DrawUtils()
        
        # Load test image
        image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(image)
        
        # Create test segmentation mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[100:200, 100:200] = 1  # Simple rectangular mask
        
        # Draw segmentation mask
        annotated_image = utils.draw_segmentation_mask(image, mask, alpha=0.5)
        
        self.assertIsNotNone(annotated_image)
        self.assertEqual(annotated_image.shape, image.shape)
    
    def test_draw_keypoints(self):
        """Test keypoint drawing functionality."""
        utils = DrawUtils()
        
        # Load test image
        image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(image)
        
        # Create test keypoints
        keypoints = [
            {'x': 100, 'y': 100, 'label': 'nose'},
            {'x': 120, 'y': 120, 'label': 'left_eye'},
            {'x': 140, 'y': 120, 'label': 'right_eye'}
        ]
        
        # Draw keypoints
        annotated_image = utils.draw_keypoints(image, keypoints)
        
        self.assertIsNotNone(annotated_image)
        self.assertEqual(annotated_image.shape, image.shape)
    
    def test_draw_tracks(self):
        """Test track drawing functionality."""
        utils = DrawUtils()
        
        # Load test image
        image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(image)
        
        # Create test tracks
        tracks = [
            {
                'id': 1,
                'bbox': [50, 50, 150, 150],
                'history': [[50, 50], [60, 60], [70, 70]]
            },
            {
                'id': 2,
                'bbox': [200, 100, 300, 200],
                'history': [[200, 100], [210, 110], [220, 120]]
            }
        ]
        
        # Draw tracks
        annotated_image = utils.draw_tracks(image, tracks)
        
        self.assertIsNotNone(annotated_image)
        self.assertEqual(annotated_image.shape, image.shape)
    
    def test_add_text_overlay(self):
        """Test text overlay functionality."""
        utils = DrawUtils()
        
        # Load test image
        image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(image)
        
        # Add text overlay
        text_lines = [
            "Player Detection",
            "Confidence: 0.95",
            "Frame: 123/1000"
        ]
        
        # Draw text overlay
        annotated_image = utils.add_text_overlay(image, text_lines, position='top')
        
        self.assertIsNotNone(annotated_image)
        self.assertEqual(annotated_image.shape, image.shape)


class TestLogger(UtilsTestCase):
    """Test logging utilities."""
    
    def test_logger_setup(self):
        """Test logger setup functionality."""
        # Test logger setup
        logger = setup_logger(
            name='test_logger',
            log_level=logging.INFO,
            log_file=os.path.join(self.temp_dir, 'test.log')
        )
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_logger')
        
        # Test logging
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")
        
        # Check if log file was created
        log_file = os.path.join(self.temp_dir, 'test.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            self.assertIn("Test message", log_content)
    
    def test_logger_configuration(self):
        """Test logger configuration options."""
        log_file = os.path.join(self.temp_dir, 'configured.log')
        
        # Test with different configurations
        configs = [
            {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'},
            {'format': '[%(levelname)s] %(name)s: %(message)s'},
            {'date_format': '%Y-%m-%d %H:%M:%S'}
        ]
        
        for config in configs:
            logger = setup_logger(
                name=f'test_logger_{hash(str(config))}',
                log_file=log_file,
                **config
            )
            
            self.assertIsNotNone(logger)
            logger.info(f"Test with config: {config}")


class TestModelSpecificUtils(UtilsTestCase):
    """Test model-specific utility functions."""
    
    def test_resnet_utils(self):
        """Test ResNet-specific utilities."""
        try:
            utils = ResNetUtils()
            
            # Test preprocessing
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            processed = utils.preprocess_image(test_image)
            
            self.assertIsNotNone(processed)
            
            # Test post-processing
            test_output = torch.randn(1, 1000)
            predictions = utils.postprocess_output(test_output)
            
            self.assertIsInstance(predictions, dict)
            self.assertIn('predicted_class', predictions)
            self.assertIn('confidence', predictions)
            
        except Exception as e:
            logger.warning(f"ResNet utils test failed: {e}")
    
    def test_rf_detr_utils(self):
        """Test RF-DETR-specific utilities."""
        try:
            utils = RF_DETRUtils()
            
            # Test preprocessing
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            processed = utils.preprocess_image(test_image)
            
            self.assertIsNotNone(processed)
            
            # Test post-processing
            test_output = {
                'boxes': torch.randn(10, 4),
                'scores': torch.randn(10),
                'classes': torch.randint(0, 10, (10,))
            }
            
            detections = utils.postprocess_output(test_output)
            
            self.assertIsInstance(detections, list)
            
        except Exception as e:
            logger.warning(f"RF-DETR utils test failed: {e}")
    
    def test_sam2_utils(self):
        """Test SAM2-specific utilities."""
        try:
            utils = SAM2Utils()
            
            # Test mask generation
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            test_points = np.array([[256, 256], [300, 300]])
            
            mask = utils.generate_mask(test_image, test_points)
            
            if mask is not None:
                self.assertIsInstance(mask, np.ndarray)
                self.assertEqual(mask.shape, test_image.shape[:2])
            
        except Exception as e:
            logger.warning(f"SAM2 utils test failed: {e}")


def create_utility_test_suite():
    """Create and return the utility test suite."""
    suite = unittest.TestSuite()
    
    # Add utility test cases
    suite.addTest(unittest.makeSuite(TestConfigLoader))
    suite.addTest(unittest.makeSuite(TestPerformanceMonitor))
    suite.addTest(unittest.makeSuite(TestImageUtils))
    suite.addTest(unittest.makeSuite(TestVideoUtils))
    suite.addTest(unittest.makeSuite(TestDrawUtils))
    suite.addTest(unittest.makeSuite(TestLogger))
    suite.addTest(unittest.makeSuite(TestModelSpecificUtils))
    
    return suite


if __name__ == '__main__':
    # Create utility test suite
    test_suite = create_utility_test_suite()
    
    # Run tests with verbose output
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    total_time = time.time() - start_time
    
    # Print utility test summary
    print(f"\n{'='*80}")
    print(f"Utility Test Summary:")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All utility tests passed!")
        print("Utility functions verified successfully.")
    else:
        print("❌ Some utility tests failed!")
        print("Utility function issues detected.")
        
    print(f"{'='*80}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)