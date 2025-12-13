"""
Performance Testing Framework for Soccer Player Recognition System

This module provides comprehensive performance testing including:
- Model loading/unloading benchmarks
- Inference speed measurements
- Memory usage tracking
- Batch processing performance
- GPU/CPU utilization tests
- Throughput analysis
"""

import unittest
import time
import torch
import numpy as np
import psutil
import gc
import threading
import multiprocessing
from typing import Dict, List, Any, Tuple
import tempfile
import os
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from contextlib import contextmanager
import json
from pathlib import Path

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


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    model_id: str
    operation: str
    device: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float = 0.0
    throughput: float = 0.0  # operations per second
    batch_size: int = 1
    input_shape: str = ""
    success: bool = True
    error_message: str = ""


class PerformanceMonitor:
    """Enhanced performance monitoring for testing."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def get_gpu_usage(self) -> float:
        """Get GPU usage if available."""
        if self.gpu_available:
            try:
                return torch.cuda.utilization()
            except:
                return 0.0
        return 0.0
    
    def clear_cache(self):
        """Clear memory caches."""
        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()
        if self.mps_available:
            torch.mps.empty_cache()


class BenchmarkTestCase(unittest.TestCase):
    """Base class for performance benchmark tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
        self.temp_dir = tempfile.mkdtemp()
        self.results: List[PerformanceMetrics] = []
        
    def tearDown(self):
        """Clean up and save results."""
        self._save_benchmark_results()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @contextmanager
    def measure_performance(self, model_id: str, operation: str, device: str, 
                          batch_size: int = 1, input_shape: str = ""):
        """Context manager for measuring performance."""
        start_time = time.perf_counter()
        start_memory = self.monitor.get_memory_usage()
        start_cpu = self.monitor.get_cpu_usage()
        
        try:
            yield
            success = True
            error_message = ""
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            end_memory = self.monitor.get_memory_usage()
            end_cpu = self.monitor.get_cpu_usage()
            
            execution_time = end_time - start_time
            
            # Calculate throughput
            throughput = batch_size / execution_time if execution_time > 0 else 0
            
            # Create metrics
            metrics = PerformanceMetrics(
                model_id=model_id,
                operation=operation,
                device=device,
                execution_time=execution_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=end_cpu - start_cpu,
                batch_size=batch_size,
                input_shape=input_shape,
                throughput=throughput,
                success=success,
                error_message=error_message
            )
            
            self.results.append(metrics)
            logger.info(f"{model_id} - {operation}: {execution_time:.4f}s, "
                       f"Memory: {metrics.memory_usage_mb:.2f}MB, "
                       f"Throughput: {throughput:.2f} ops/s")
    
    def _create_mock_model(self, model_type: ModelType, complexity: str = "simple") -> str:
        """Create a mock model file with specified complexity."""
        model_path = os.path.join(self.temp_dir, f'mock_{model_type.value}_{complexity}.pt')
        
        if complexity == "simple":
            model = torch.nn.Sequential(
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128)
            )
        elif complexity == "medium":
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((4, 4)),
                torch.nn.Flatten(),
                torch.nn.Linear(128 * 16, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 100)
            )
        else:  # complex
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 512, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((7, 7)),
                torch.nn.Flatten(),
                torch.nn.Linear(512 * 49, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1000)
            )
        
        torch.save(model, model_path)
        return model_path
    
    def _create_test_input(self, model_type: ModelType, batch_size: int = 1) -> torch.Tensor:
        """Create test input for different model types."""
        if model_type == ModelType.DETECTION:
            return torch.randn(batch_size, 3, 640, 640)
        elif model_type == ModelType.CLASSIFICATION:
            return torch.randn(batch_size, 3, 224, 224)
        elif model_type == ModelType.SEGMENTATION:
            return torch.randn(batch_size, 3, 256, 256)
        elif model_type == ModelType.IDENTIFICATION:
            return torch.randn(batch_size, 512)
        else:  # POSE
            return torch.randn(batch_size, 3, 224, 224)
    
    def _save_benchmark_results(self):
        """Save benchmark results to JSON file."""
        if not self.results:
            return
        
        results_data = []
        for metric in self.results:
            results_data.append({
                'model_id': metric.model_id,
                'operation': metric.operation,
                'device': metric.device,
                'execution_time': metric.execution_time,
                'memory_usage_mb': metric.memory_usage_mb,
                'cpu_usage_percent': metric.cpu_usage_percent,
                'gpu_usage_percent': metric.gpu_usage_percent,
                'throughput': metric.throughput,
                'batch_size': metric.batch_size,
                'input_shape': metric.input_shape,
                'success': metric.success,
                'error_message': metric.error_message
            })
        
        results_file = os.path.join(self.temp_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")


class TestModelLoadingPerformance(BenchmarkTestCase):
    """Test model loading performance."""
    
    def test_model_loading_speed(self):
        """Test model loading speed for different models."""
        models = [
            (ModelType.DETECTION, "simple"),
            (ModelType.CLASSIFICATION, "medium"),
            (ModelType.SEGMENTATION, "complex"),
            (ModelType.IDENTIFICATION, "simple")
        ]
        
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        
        for model_type, complexity in models:
            for device in devices:
                model_path = self._create_mock_model(model_type, complexity)
                
                with self.measure_performance(
                    f"{model_type.value}_{complexity}",
                    "loading",
                    device,
                    input_shape=str(model_type.value)
                ):
                    instance = ModelInstance(
                        model_id=f"test_{model_type.value}",
                        model_path=model_path,
                        config={},
                        model_type=model_type,
                        device=device
                    )
                    instance.load_model()
    
    def test_model_unloading_speed(self):
        """Test model unloading speed."""
        model_type = ModelType.CLASSIFICATION
        model_path = self._create_mock_model(model_type, "medium")
        
        instance = ModelInstance(
            model_id="test_unload",
            model_path=model_path,
            config={},
            model_type=model_type,
            device="cpu"
        )
        
        instance.load_model()
        
        with self.measure_performance("test_unload", "unloading", "cpu"):
            instance.unload_model()
    
    def test_cold_start_vs_warm_start(self):
        """Test cold start vs warm start performance."""
        model_type = ModelType.CLASSIFICATION
        model_path = self._create_mock_model(model_type, "medium")
        
        # Cold start (first load)
        with self.measure_performance("cold_start", "loading", "cpu", input_shape="cold"):
            instance1 = ModelInstance(
                model_id="cold_start",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            instance1.load_model()
            instance1.unload_model()
        
        # Warm start (reload)
        with self.measure_performance("warm_start", "loading", "cpu", input_shape="warm"):
            instance2 = ModelInstance(
                model_id="warm_start",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            instance2.load_model()


class TestInferencePerformance(BenchmarkTestCase):
    """Test inference performance."""
    
    def test_single_inference_speed(self):
        """Test single inference speed for different model types."""
        models = [
            (ModelType.DETECTION, (1, 3, 640, 640)),
            (ModelType.CLASSIFICATION, (1, 3, 224, 224)),
            (ModelType.SEGMENTATION, (1, 3, 256, 256)),
            (ModelType.IDENTIFICATION, (1, 512))
        ]
        
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        
        for model_type, input_shape in models:
            for device in devices:
                model_path = self._create_mock_model(model_type, "medium")
                
                instance = ModelInstance(
                    model_id=f"test_{model_type.value}",
                    model_path=model_path,
                    config={},
                    model_type=model_type,
                    device=device
                )
                instance.load_model()
                
                test_input = torch.randn(*input_shape)
                
                with self.measure_performance(
                    f"{model_type.value}_single",
                    "inference",
                    device,
                    batch_size=1,
                    input_shape=str(input_shape)
                ):
                    result = instance.predict(test_input)
                
                instance.unload_model()
    
    def test_batch_inference_speed(self):
        """Test batch inference speed with different batch sizes."""
        model_type = ModelType.CLASSIFICATION
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            model_path = self._create_mock_model(model_type, "medium")
            
            instance = ModelInstance(
                model_id=f"test_batch_{batch_size}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            instance.load_model()
            
            test_input = torch.randn(batch_size, 3, 224, 224)
            
            with self.measure_performance(
                f"batch_{batch_size}",
                "batch_inference",
                "cpu",
                batch_size=batch_size,
                input_shape=str(test_input.shape)
            ):
                result = instance.predict(test_input)
            
            instance.unload_model()
    
    def test_continuous_inference_performance(self):
        """Test performance over multiple consecutive inferences."""
        model_type = ModelType.CLASSIFICATION
        num_inferences = 100
        
        model_path = self._create_mock_model(model_type, "medium")
        
        instance = ModelInstance(
            model_id="continuous_test",
            model_path=model_path,
            config={},
            model_type=model_type,
            device="cpu"
        )
        instance.load_model()
        
        test_input = torch.randn(1, 3, 224, 224)
        
        # Measure time for all inferences
        start_time = time.perf_counter()
        for i in range(num_inferences):
            with self.measure_performance(
                f"continuous_{i}",
                "inference",
                "cpu",
                batch_size=1,
                input_shape=str(test_input.shape)
            ):
                result = instance.predict(test_input)
        
        total_time = time.perf_counter() - start_time
        avg_time_per_inference = total_time / num_inferences
        throughput = num_inferences / total_time
        
        logger.info(f"Continuous inference: {avg_time_per_inference:.4f}s per inference, "
                   f"throughput: {throughput:.2f} inferences/s")
        
        instance.unload_model()
    
    def test_memory_efficiency_over_time(self):
        """Test memory efficiency over multiple operations."""
        model_type = ModelType.CLASSIFICATION
        
        for i in range(10):
            model_path = self._create_mock_model(model_type, "simple")
            
            instance = ModelInstance(
                model_id=f"memory_test_{i}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            
            # Measure before loading
            memory_before = self.monitor.get_memory_usage()
            
            instance.load_model()
            test_input = torch.randn(1, 3, 224, 224)
            instance.predict(test_input)
            instance.unload_model()
            
            # Measure after unloading
            memory_after = self.monitor.get_memory_usage()
            memory_diff = memory_after - memory_before
            
            logger.info(f"Iteration {i}: Memory difference: {memory_diff:.2f}MB")
            
            # Assert memory doesn't grow significantly
            self.assertLess(memory_diff, 50, "Memory should not grow significantly")


class TestBatchProcessingPerformance(BenchmarkTestCase):
    """Test batch processing performance."""
    
    def test_batch_processing_vs_individual(self):
        """Compare batch processing vs individual processing."""
        model_type = ModelType.CLASSIFICATION
        batch_sizes = [4, 8, 16]
        num_total_items = 32
        
        model_path = self._create_mock_model(model_type, "medium")
        
        for batch_size in batch_sizes:
            instance = ModelInstance(
                model_id=f"batch_vs_individual_{batch_size}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            instance.load_model()
            
            # Individual processing
            individual_inputs = [torch.randn(1, 3, 224, 224) for _ in range(num_total_items)]
            
            start_time = time.perf_counter()
            for inp in individual_inputs:
                result = instance.predict(inp)
            individual_time = time.perf_counter() - start_time
            
            # Batch processing
            batch_inputs = [torch.randn(batch_size, 3, 224, 224) for _ in range(num_total_items // batch_size)]
            
            start_time = time.perf_counter()
            for batch_inp in batch_inputs:
                result = instance.predict(batch_inp)
            batch_time = time.perf_counter() - start_time
            
            speedup = individual_time / batch_time
            logger.info(f"Batch size {batch_size}: Individual: {individual_time:.4f}s, "
                       f"Batch: {batch_time:.4f}s, Speedup: {speedup:.2f}x")
            
            instance.unload_model()
    
    def test_concurrent_batch_processing(self):
        """Test concurrent batch processing."""
        model_type = ModelType.CLASSIFICATION
        num_concurrent = 4
        batch_size = 8
        num_batches = 5
        
        model_path = self._create_mock_model(model_type, "medium")
        
        def process_batch(batch_id):
            instance = ModelInstance(
                model_id=f"concurrent_{batch_id}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            instance.load_model()
            
            results = []
            for i in range(num_batches):
                test_input = torch.randn(batch_size, 3, 224, 224)
                result = instance.predict(test_input)
                results.append(result)
            
            instance.unload_model()
            return len(results)
        
        # Sequential processing
        start_time = time.perf_counter()
        sequential_results = []
        for i in range(num_concurrent):
            result = process_batch(i)
            sequential_results.append(result)
        sequential_time = time.perf_counter() - start_time
        
        # Concurrent processing
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            concurrent_results = list(executor.map(process_batch, range(num_concurrent)))
        concurrent_time = time.perf_counter() - start_time
        
        speedup = sequential_time / concurrent_time
        logger.info(f"Sequential: {sequential_time:.4f}s, "
                   f"Concurrent: {concurrent_time:.4f}s, Speedup: {speedup:.2f}x")
    
    def test_memory_usage_in_batches(self):
        """Test memory usage with different batch sizes."""
        model_type = ModelType.CLASSIFICATION
        batch_sizes = [1, 4, 8, 16, 32, 64]
        
        memory_usage = {}
        
        for batch_size in batch_sizes:
            model_path = self._create_mock_model(model_type, "medium")
            
            instance = ModelInstance(
                model_id=f"memory_batch_{batch_size}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            
            # Memory before
            memory_before = self.monitor.get_memory_usage()
            
            instance.load_model()
            test_input = torch.randn(batch_size, 3, 224, 224)
            instance.predict(test_input)
            instance.unload_model()
            
            # Memory after
            memory_after = self.monitor.get_memory_usage()
            memory_diff = memory_after - memory_before
            
            memory_usage[batch_size] = memory_diff
            logger.info(f"Batch size {batch_size}: Memory usage: {memory_diff:.2f}MB")
        
        # Assert memory usage scales reasonably
        self.assertLess(memory_usage[64], memory_usage[1] * 100, 
                       "Memory usage should not scale linearly with batch size")


class TestDevicePerformance(BenchmarkTestCase):
    """Test performance across different devices."""
    
    def test_cpu_vs_cuda_performance(self):
        """Compare CPU vs CUDA performance if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model_type = ModelType.CLASSIFICATION
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            model_path = self._create_mock_model(model_type, "medium")
            
            # CPU performance
            cpu_instance = ModelInstance(
                model_id=f"cpu_batch_{batch_size}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            cpu_instance.load_model()
            
            test_input = torch.randn(batch_size, 3, 224, 224)
            
            start_time = time.perf_counter()
            cpu_result = cpu_instance.predict(test_input)
            cpu_time = time.perf_counter() - start_time
            
            cpu_instance.unload_model()
            
            # CUDA performance
            cuda_instance = ModelInstance(
                model_id=f"cuda_batch_{batch_size}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cuda"
            )
            cuda_instance.load_model()
            
            start_time = time.perf_counter()
            cuda_result = cuda_instance.predict(test_input)
            cuda_time = time.perf_counter() - start_time
            
            cuda_instance.unload_model()
            
            speedup = cpu_time / cuda_time
            logger.info(f"Batch {batch_size}: CPU: {cpu_time:.4f}s, "
                       f"CUDA: {cuda_time:.4f}s, Speedup: {speedup:.2f}x")
    
    def test_device_memory_usage(self):
        """Test memory usage across devices."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
        
        model_type = ModelType.CLASSIFICATION
        
        for device in devices:
            model_path = self._create_mock_model(model_type, "medium")
            
            instance = ModelInstance(
                model_id=f"memory_test_{device}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device=device
            )
            
            memory_before = self.monitor.get_memory_usage()
            instance.load_model()
            memory_after_load = self.monitor.get_memory_usage()
            
            test_input = torch.randn(4, 3, 224, 224)
            instance.predict(test_input)
            
            instance.unload_model()
            memory_after_unload = self.monitor.get_memory_usage()
            
            load_memory = memory_after_load - memory_before
            unload_memory = memory_after_unload - memory_after_load
            
            logger.info(f"Device {device}: Load: {load_memory:.2f}MB, "
                       f"Unload: {unload_memory:.2f}MB")


class TestScalabilityPerformance(BenchmarkTestCase):
    """Test system scalability."""
    
    def test_model_manager_scalability(self):
        """Test ModelManager performance with many models."""
        registry = ModelRegistry()
        num_models = 20
        
        # Register many models
        for i in range(num_models):
            model_path = self._create_mock_model(ModelType.CLASSIFICATION, "simple")
            registry.register_model(
                model_id=f"scalability_model_{i}",
                model_path=model_path,
                model_type=ModelType.CLASSIFICATION,
                config={},
                is_active=True
            )
        
        manager = ModelManager(registry=registry)
        
        # Test loading multiple models
        start_time = time.perf_counter()
        loaded_models = []
        for i in range(num_models):
            success = manager.load_model(f"scalability_model_{i}", device="cpu")
            if success:
                loaded_models.append(f"scalability_model_{i}")
        
        total_load_time = time.perf_counter() - start_time
        
        # Test predictions on loaded models
        start_time = time.perf_counter()
        test_input = torch.randn(1, 3, 224, 224)
        for model_id in loaded_models:
            try:
                result = manager.predict(model_id, test_input, auto_load=False)
            except:
                pass  # Some models might fail
        
        total_prediction_time = time.perf_counter() - start_time
        
        logger.info(f"Scalability: {len(loaded_models)} models loaded in {total_load_time:.4f}s")
        logger.info(f"Predictions on {len(loaded_models)} models in {total_prediction_time:.4f}s")
    
    def test_memory_cleanup_performance(self):
        """Test memory cleanup performance."""
        registry = ModelRegistry()
        num_models = 15
        
        # Register models
        for i in range(num_models):
            model_path = self._create_mock_model(ModelType.CLASSIFICATION, "medium")
            registry.register_model(
                model_id=f"cleanup_model_{i}",
                model_path=model_path,
                model_type=ModelType.CLASSIFICATION,
                config={},
                is_active=True
            )
        
        manager = ModelManager(registry=registry)
        manager.max_loaded_models = 5
        
        # Load all models
        for i in range(num_models):
            manager.load_model(f"cleanup_model_{i}", device="cpu")
        
        # Test cleanup
        start_time = time.perf_counter()
        unloaded_count = manager.cleanup_memory()
        cleanup_time = time.perf_counter() - start_time
        
        logger.info(f"Cleanup: {unloaded_count} models unloaded in {cleanup_time:.4f}s")
        self.assertGreater(unloaded_count, 0, "Some models should be unloaded")


class TestPerformanceRegression(BenchmarkTestCase):
    """Test for performance regressions."""
    
    def test_baseline_performance(self):
        """Establish baseline performance metrics."""
        model_type = ModelType.CLASSIFICATION
        model_path = self._create_mock_model(model_type, "medium")
        
        instance = ModelInstance(
            model_id="baseline_test",
            model_path=model_path,
            config={},
            model_type=model_type,
            device="cpu"
        )
        
        # Measure loading time
        with self.measure_performance("baseline", "loading", "cpu"):
            instance.load_model()
        
        # Measure inference time
        test_input = torch.randn(1, 3, 224, 224)
        with self.measure_performance("baseline", "inference", "cpu", batch_size=1):
            result = instance.predict(test_input)
        
        # Save baseline metrics for future comparison
        if self.results:
            baseline_file = os.path.join(self.temp_dir, "baseline_performance.json")
            with open(baseline_file, 'w') as f:
                json.dump([{
                    'operation': r.operation,
                    'execution_time': r.execution_time,
                    'throughput': r.throughput
                } for r in self.results], f, indent=2)
            
            logger.info(f"Baseline metrics saved to {baseline_file}")
    
    def test_performance_consistency(self):
        """Test performance consistency across multiple runs."""
        model_type = ModelType.CLASSIFICATION
        model_path = self._create_mock_model(model_type, "medium")
        
        inference_times = []
        
        for run in range(5):
            instance = ModelInstance(
                model_id=f"consistency_test_{run}",
                model_path=model_path,
                config={},
                model_type=model_type,
                device="cpu"
            )
            
            instance.load_model()
            
            test_input = torch.randn(1, 3, 224, 224)
            
            start_time = time.perf_counter()
            result = instance.predict(test_input)
            inference_time = time.perf_counter() - start_time
            
            inference_times.append(inference_time)
            instance.unload_model()
        
        # Calculate statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        cv = std_time / mean_time  # Coefficient of variation
        
        logger.info(f"Inference times: Mean: {mean_time:.4f}s, "
                   f"Std: {std_time:.4f}s, CV: {cv:.4f}")
        
        # Assert reasonable consistency (CV should be less than 20%)
        self.assertLess(cv, 0.2, "Performance should be consistent across runs")


def create_performance_test_suite():
    """Create and return the performance test suite."""
    suite = unittest.TestSuite()
    
    # Add performance test cases
    suite.addTest(unittest.makeSuite(TestModelLoadingPerformance))
    suite.addTest(unittest.makeSuite(TestInferencePerformance))
    suite.addTest(unittest.makeSuite(TestBatchProcessingPerformance))
    suite.addTest(unittest.makeSuite(TestDevicePerformance))
    suite.addTest(unittest.makeSuite(TestScalabilityPerformance))
    suite.addTest(unittest.makeSuite(TestPerformanceRegression))
    
    return suite


if __name__ == '__main__':
    # Create performance test suite
    test_suite = create_performance_test_suite()
    
    # Run tests with timing
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    total_time = time.time() - start_time
    
    # Print performance summary
    print(f"\n{'='*80}")
    print(f"Performance Test Summary:")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All performance tests passed!")
    else:
        print("❌ Some performance tests failed!")
        
    print(f"{'='*80}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)