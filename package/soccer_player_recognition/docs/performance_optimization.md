# Performance Optimization Guide

## Overview

This guide provides comprehensive strategies for optimizing the Soccer Player Recognition System for maximum performance, covering inference speed, memory usage, accuracy, and resource utilization across different deployment scenarios.

## Table of Contents

1. [Performance Fundamentals](#performance-fundamentals)
2. [Hardware Optimization](#hardware-optimization)
3. [Model Optimization](#model-optimization)
4. [Inference Optimization](#inference-optimization)
5. [Memory Optimization](#memory-optimization)
6. [Batch Processing Optimization](#batch-processing-optimization)
7. [GPU Optimization](#gpu-optimization)
8. [Parallelization Strategies](#parallelization-strategies)
9. [Caching Strategies](#caching-strategies)
10. [Monitoring and Profiling](#monitoring-and-profiling)

## Performance Fundamentals

### Key Performance Metrics

```python
class PerformanceMetrics:
    def __init__(self):
        self.throughput_metrics = {
            'fps': 0,                    # Frames per second
            'images_per_second': 0,      # Images processed per second
            'batch_throughput': 0,       # Batches processed per second
        }
        
        self.latency_metrics = {
            'inference_time': 0,         # Time per inference
            'preprocessing_time': 0,     # Image preprocessing time
            'postprocessing_time': 0,    # Result processing time
            'total_time': 0              # End-to-end processing time
        }
        
        self.resource_metrics = {
            'gpu_utilization': 0,        # GPU usage percentage
            'memory_usage_mb': 0,        # GPU memory usage
            'cpu_utilization': 0,        # CPU usage percentage
            'bandwidth_utilization': 0   # Memory bandwidth usage
        }
        
        self.accuracy_metrics = {
            'detection_accuracy': 0,     # Detection precision/recall
            'identification_accuracy': 0, # ID precision/recall
            'tracking_accuracy': 0,      # Tracking IDF1 score
            'overall_accuracy': 0        # Combined accuracy score
        }
```

### Performance Targets by Use Case

| Use Case | Target FPS | Max Latency | Memory Usage | Accuracy Target |
|----------|------------|-------------|--------------|-----------------|
| Real-time Analysis | 30+ | <33ms | <8GB | >85% |
| Batch Processing | 10+ | <100ms | <16GB | >90% |
| High Accuracy | 5+ | <200ms | <24GB | >95% |
| Mobile/Edge | 5+ | <200ms | <4GB | >80% |

### Performance Profiling

```python
import time
import psutil
import GPUtil
from contextlib import contextmanager
from typing import Dict, Any

class PerformanceProfiler:
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations"""
        # Record start time
        self.start_times[operation_name] = time.time()
        
        # Record initial resource usage
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.virtual_memory().percent
        
        gpu_before = None
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_before = {
                'memory_used': gpu.memoryUsed,
                'memory_util': gpu.memoryUtil,
                'load': gpu.load
            }
        
        try:
            yield self
        finally:
            # Calculate elapsed time
            elapsed = time.time() - self.start_times[operation_name]
            
            # Record final resource usage
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().percent
            
            gpu_after = None
            if torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                gpu_after = {
                    'memory_used': gpu.memoryUsed,
                    'memory_util': gpu.memoryUtil,
                    'load': gpu.load
                }
            
            # Store metrics
            self.metrics[operation_name] = {
                'elapsed_time': elapsed,
                'cpu_usage': {'before': cpu_before, 'after': cpu_after, 'delta': cpu_after - cpu_before},
                'memory_usage': {'before': memory_before, 'after': memory_after, 'delta': memory_after - memory_before},
                'gpu_usage': {'before': gpu_before, 'after': gpu_after}
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = sum(m['elapsed_time'] for m in self.metrics.values())
        
        return {
            'total_time': total_time,
            'operations': list(self.metrics.keys()),
            'operation_times': {k: v['elapsed_time'] for k, v in self.metrics.items()},
            'resource_peak': self._calculate_resource_peaks()
        }
    
    def _calculate_resource_peaks(self) -> Dict[str, Any]:
        """Calculate peak resource usage"""
        peaks = {
            'cpu_peak': 0,
            'memory_peak': 0,
            'gpu_memory_peak': 0,
            'gpu_load_peak': 0
        }
        
        for metrics in self.metrics.values():
            peaks['cpu_peak'] = max(peaks['cpu_peak'], metrics['cpu_usage']['after'])
            peaks['memory_peak'] = max(peaks['memory_peak'], metrics['memory_usage']['after'])
            
            if metrics['gpu_usage']['after']:
                peaks['gpu_memory_peak'] = max(peaks['gpu_memory_peak'], metrics['gpu_usage']['after']['memory_used'])
                peaks['gpu_load_peak'] = max(peaks['gpu_load_peak'], metrics['gpu_usage']['after']['load'])
        
        return peaks

# Usage example
profiler = PerformanceProfiler()

with profiler.profile("full_inference"):
    results = recognizer.analyze_scene(image)

summary = profiler.get_summary()
print(f"Total time: {summary['total_time']:.3f}s")
print(f"Operation breakdown: {summary['operation_times']}")
```

## Hardware Optimization

### GPU Selection Guide

#### NVIDIA GPU Comparison

| GPU Model | Memory | Tensor Cores | FP16 Performance | Recommended Use |
|-----------|--------|--------------|------------------|-----------------|
| RTX 3060 | 12GB | 3rd Gen | 10 TFLOPS | Development/Testing |
| RTX 3080 | 10GB | 3rd Gen | 16 TFLOPS | Real-time Processing |
| RTX 3090 | 24GB | 3rd Gen | 19 TFLOPS | High-throughput |
| RTX 4090 | 24GB | 4th Gen | 40 TFLOPS | Maximum Performance |
| A100 | 40GB | 3rd Gen | 19 TFLOPS | Enterprise/Research |

#### GPU Memory Optimization

```python
class GPUMemoryManager:
    def __init__(self, max_memory_fraction: float = 0.8):
        self.max_memory_fraction = max_memory_fraction
        self.initial_memory = self._get_memory_info()
    
    def _get_memory_info(self):
        """Get current GPU memory info"""
        if not torch.cuda.is_available():
            return None
        
        gpu = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(gpu).total_memory
        reserved = torch.cuda.memory_reserved(gpu)
        allocated = torch.cuda.memory_allocated(gpu)
        free = total - reserved
        
        return {
            'total': total,
            'reserved': reserved,
            'allocated': allocated,
            'free': free
        }
    
    def optimize_for_model(self, model_size_gb: float, buffer_gb: float = 1.0):
        """Optimize GPU settings for specific model"""
        total_memory_gb = self.initial_memory['total'] / (1024**3)
        target_usage = model_size_gb + buffer_gb
        
        if target_usage / total_memory_gb > self.max_memory_fraction:
            print(f"Warning: Model may not fit. Required: {target_usage:.1f}GB, "
                  f"Available: {total_memory_gb * self.max_memory_fraction:.1f}GB")
            
            # Suggest batch size reduction
            safe_model_size = total_memory_gb * self.max_memory_fraction - buffer_gb
            print(f"Consider using smaller model or reducing batch size")
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(self.max_memory_fraction)
        
        return {
            'target_usage_gb': target_usage,
            'available_gb': total_memory_gb * self.max_memory_fraction,
            'memory_efficiency': target_usage / (total_memory_gb * self.max_memory_fraction)
        }
    
    def enable_memory_efficient_attention(self, enable: bool = True):
        """Enable memory efficient attention mechanisms"""
        if enable:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        return {
            'tf32_enabled': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark
        }
```

### CPU Optimization

#### Multi-core Processing

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import threading

class CPUOptimizer:
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        
    def optimize_data_loading(self, dataset_size: int, batch_size: int) -> Dict[str, int]:
        """Optimize data loader settings"""
        # Optimal num_workers based on CPU cores and batch size
        if dataset_size < 1000:
            optimal_workers = min(4, self.num_workers)
        elif dataset_size < 10000:
            optimal_workers = min(8, self.num_workers)
        else:
            optimal_workers = min(16, self.num_workers)
        
        # Adjust for batch size
        if batch_size > 64:
            optimal_workers = max(1, optimal_workers // 2)
        
        return {
            'optimal_num_workers': optimal_workers,
            'pin_memory': True,
            'persistent_workers': dataset_size > 1000,
            'prefetch_factor': min(4, batch_size // 8)
        }
    
    def parallel_preprocessing(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Parallel image preprocessing"""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._preprocess_image, images))
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Single image preprocessing"""
        # Resize, normalize, etc.
        return cv2.resize(image, (640, 640))
```

### Memory Architecture Optimization

#### NUMA-Aware Processing

```python
import numa
from numba import cuda

class NUMAOptimizer:
    def __init__(self):
        self.available_nodes = numa.get_num_available_nodes()
        self.node_memory = self._get_node_memory()
        
    def _get_node_memory(self):
        """Get memory per NUMA node"""
        memory_info = {}
        for node_id in range(self.available_nodes):
            try:
                memory_info[node_id] = numa.node_size(node_id)
            except:
                memory_info[node_id] = 0
        return memory_info
    
    def bind_to_node(self, node_id: int):
        """Bind process to specific NUMA node"""
        try:
            numa.run_on_node(node_id)
            print(f"Bound to NUMA node {node_id}")
            return True
        except:
            print(f"Failed to bind to NUMA node {node_id}")
            return False
    
    def optimize_memory_allocation(self, size_gb: float):
        """Optimize memory allocation across NUMA nodes"""
        size_bytes = int(size_gb * 1024**3)
        optimal_node = max(range(self.available_nodes), 
                          key=lambda n: self.node_memory[n])
        
        print(f"Optimal NUMA node: {optimal_node}")
        print(f"Memory per node: {self.node_memory[optimal_node] / (1024**3):.1f}GB")
        
        return {
            'optimal_node': optimal_node,
            'node_memory_gb': self.node_memory[optimal_node] / (1024**3),
            'allocation_size_gb': size_gb
        }
```

## Model Optimization

### Model Quantization

#### Dynamic Quantization

```python
import torch.quantization as quantization

class ModelQuantizer:
    def __init__(self, model):
        self.model = model
        
    def dynamic_quantize(self) -> torch.nn.Module:
        """Apply dynamic quantization to the model"""
        print("Applying dynamic quantization...")
        
        # Quantize dynamically (only for linear layers)
        quantized_model = quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        print("Dynamic quantization completed")
        return quantized_model
    
    def static_quantize(self, calibration_data: List[torch.Tensor]) -> torch.nn.Module:
        """Apply static quantization with calibration data"""
        print("Applying static quantization...")
        
        # Prepare model for quantization
        quantized_model = quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            inplace=False
        )
        
        # Set quantization config
        quantized_model.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # Prepare for calibration
        quantization.prepare(quantized_model, inplace=True)
        
        # Run calibration
        with torch.no_grad():
            for data in calibration_data[:100]:  # Use first 100 samples
                quantized_model(data.to('cuda'))
        
        # Convert to quantized model
        quantized_model = quantization.convert(quantized_model, inplace=True)
        
        print("Static quantization completed")
        return quantized_model
    
    def benchmark_quantization(self, original_model, quantized_model, test_data):
        """Benchmark quantization performance"""
        def benchmark_model(model, data, num_iterations=100):
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    model(data)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    model(data)
            end_time = time.time()
            
            return {
                'total_time': end_time - start_time,
                'avg_time': (end_time - start_time) / num_iterations,
                'fps': num_iterations / (end_time - start_time)
            }
        
        original_stats = benchmark_model(original_model, test_data)
        quantized_stats = benchmark_model(quantized_model, test_data)
        
        return {
            'original': original_stats,
            'quantized': quantized_stats,
            'speedup': original_stats['avg_time'] / quantized_stats['avg_time'],
            'size_reduction': self._calculate_size_reduction(original_model, quantized_model)
        }
    
    def _calculate_size_reduction(self, original, quantized):
        """Calculate model size reduction"""
        def get_model_size(model):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 * 1024)  # MB
        
        original_size = get_model_size(original)
        quantized_size = get_model_size(quantized)
        
        return {
            'original_mb': original_size,
            'quantized_mb': quantized_size,
            'reduction_ratio': 1 - (quantized_size / original_size)
        }
```

### Model Pruning

#### Structured Pruning

```python
class ModelPruner:
    def __init__(self, model):
        self.model = model
        self.pruning_history = []
    
    def prune_conv_layer(self, layer: torch.nn.Conv2d, sparsity: float = 0.5):
        """Prune convolution layer using L1-norm"""
        # Get weights
        weights = layer.weight.data.clone()
        
        # Calculate L1 norm for each filter
        filter_norms = torch.norm(weights.view(weights.size(0), -1), p=1, dim=1)
        
        # Select filters to prune
        num_filters_to_prune = int(weights.size(0) * sparsity)
        _, indices = torch.topk(filter_norms, num_filters_to_prune, largest=False)
        
        # Create mask
        mask = torch.ones(weights.size(0), device=weights.device)
        mask[indices] = 0
        
        # Apply mask
        layer.weight.data *= mask.view(-1, 1, 1, 1)
        
        # Store pruning info
        self.pruning_history.append({
            'layer': str(layer),
            'sparsity': sparsity,
            'pruned_filters': indices.tolist()
        })
        
        return mask
    
    def global_magnitude_prune(self, sparsity: float = 0.3):
        """Apply global magnitude-based pruning"""
        # Collect all parameters
        parameters = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                parameters.append((name, param.data.clone()))
        
        # Calculate global threshold
        all_weights = torch.cat([p.flatten() for _, p in parameters])
        threshold = torch.quantile(torch.abs(all_weights), sparsity)
        
        # Apply pruning
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                param.data[torch.abs(param.data) < threshold] = 0
        
        return threshold
    
    def fine_tune_after_pruning(self, train_loader, criterion, optimizer, num_epochs=10):
        """Fine-tune model after pruning"""
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                output = self.model(data)
                loss = criterion(output, target)
                
                loss.backward()
                
                # Gradient clipping to avoid exploding gradients after pruning
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Pruning fine-tune Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
```

### Model Compilation

#### TorchScript Compilation

```python
class ModelCompiler:
    def __init__(self, model):
        self.model = model
        
    def compile_to_torchscript(self, example_input, optimize: bool = True):
        """Compile model to TorchScript"""
        self.model.eval()
        
        # Create a TorchScript model
        scripted_model = torch.jit.trace(self.model, example_input)
        
        if optimize:
            # Enable optimizations
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        return scripted_model
    
    def compile_with_optimizations(self, example_input):
        """Compile with various optimizations"""
        self.model.eval()
        
        # Trace model
        traced_model = torch.jit.trace(self.model, example_input)
        
        # Apply optimizations
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        # Benchmark performance
        stats = self._benchmark_performance(optimized_model, example_input)
        
        return optimized_model, stats
    
    def _benchmark_performance(self, model, data, num_iterations=100):
        """Benchmark compiled model performance"""
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(data)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                model(data)
        end_time = time.time()
        
        return {
            'total_time': end_time - start_time,
            'avg_time': (end_time - start_time) / num_iterations,
            'fps': num_iterations / (end_time - start_time)
        }
```

### TensorRT Optimization

```python
class TensorRTOptimizer:
    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        
    def optimize_for_tensorrt(self, 
                             precision: str = 'fp16',
                             workspace_size: int = 1 << 30,
                             max_batch_size: int = 8):
        """Optimize ONNX model for TensorRT"""
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        # Set precision
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            # Would need calibration data here
        
        # Build optimized engine
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(self.onnx_model_path, 'rb') as f:
            if not parser.parse(f.read()):
                raise RuntimeError("Failed to parse ONNX model")
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        return engine
    
    def benchmark_tensorrt(self, engine, test_input, num_iterations=100):
        """Benchmark TensorRT engine"""
        import tensorrt as trt
        
        runtime = trt.Runtime(logger)
        context = engine.create_execution_context()
        
        # Allocate buffers
        input_shape = test_input.shape
        output_shape = (input_shape[0], 4)  # Assuming detection output
        
        d_input = cuda.mem_alloc(test_input.nbytes)
        d_output = cuda.mem_alloc(output_shape[0] * output_shape[1] * 4)
        
        bindings = [int(d_input), int(d_output)]
        
        # Warmup
        for _ in range(10):
            cuda.memcpy_htod(d_input, test_input.contiguous())
            context.execute_v2(bindings)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            cuda.memcpy_htod(d_input, test_input.contiguous())
            context.execute_v2(bindings)
            output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, d_output)
        end_time = time.time()
        
        return {
            'total_time': end_time - start_time,
            'avg_time': (end_time - start_time) / num_iterations,
            'fps': num_iterations / (end_time - start_time),
            'throughput_mb_s': test_input.nbytes * num_iterations / (end_time - start_time) / (1024**2)
        }
```

## Inference Optimization

### Batch Size Optimization

```python
class BatchOptimizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def find_optimal_batch_size(self, 
                               input_shape: tuple,
                               max_memory_gb: float = 8,
                               target_memory_fraction: float = 0.8):
        """Find optimal batch size for given memory constraints"""
        # Calculate memory per sample
        sample_memory = self._estimate_sample_memory(input_shape)
        max_memory_bytes = max_memory_gb * (1024**3) * target_memory_fraction
        
        # Start with conservative batch size
        batch_size = max(1, int(max_memory_bytes // sample_memory))
        
        # Test and adjust
        optimal_batch_size = self._test_batch_size(batch_size, input_shape)
        
        return optimal_batch_size
    
    def _estimate_sample_memory(self, input_shape: tuple) -> int:
        """Estimate memory usage per sample"""
        # Rough estimation for different model types
        if 'detection' in str(self.model.__class__.__name__).lower():
            # Detection models typically use 4-6x input size
            multiplier = 5
        elif 'segmentation' in str(self.model.__class__.__name__).lower():
            # Segmentation models are memory intensive
            multiplier = 10
        else:
            multiplier = 3
        
        sample_size = np.prod(input_shape) * 4  # 4 bytes per float32
        return int(sample_size * multiplier)
    
    def _test_batch_size(self, batch_size: int, input_shape: tuple) -> int:
        """Test if batch size fits in memory"""
        try:
            # Create test batch
            test_input = torch.randn(batch_size, *input_shape).to(self.device)
            
            # Run inference
            with torch.no_grad():
                self.model(test_input)
            
            # Success - return this batch size
            return batch_size
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # Reduce batch size and try again
                return self._test_batch_size(batch_size // 2, input_shape)
            else:
                raise e
    
    def adaptive_batch_processing(self, 
                                 images: List[np.ndarray],
                                 memory_threshold: float = 0.8) -> List[Any]:
        """Process images with adaptive batch sizing"""
        results = []
        batch_size = 1  # Start small
        
        for i in range(0, len(images), batch_size):
            batch_end = min(i + batch_size, len(images))
            batch_images = images[i:batch_end]
            
            try:
                # Try with current batch size
                batch_results = self._process_batch(batch_images)
                results.extend(batch_results)
                
                # Try to increase batch size
                if batch_size < len(images) - i:
                    potential_batch_size = min(batch_size * 2, len(images) - i)
                    if self._batch_size_fits(potential_batch_size):
                        batch_size = potential_batch_size
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # Reduce batch size
                    batch_size = max(1, batch_size // 2)
                    batch_results = self._process_batch(batch_images)
                    results.extend(batch_results)
                else:
                    raise e
        
        return results
    
    def _process_batch(self, images: List[np.ndarray]) -> List[Any]:
        """Process a batch of images"""
        batch_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for img in images
        ]).to(self.device)
        
        with torch.no_grad():
            results = self.model(batch_tensor)
        
        return results
    
    def _batch_size_fits(self, batch_size: int) -> bool:
        """Check if batch size fits in memory"""
        try:
            test_input = torch.randn(batch_size, 3, 640, 640).to(self.device)
            with torch.no_grad():
                self.model(test_input)
            return True
        except RuntimeError:
            return False
```

### Stream Processing

```python
class StreamProcessor:
    def __init__(self, model, buffer_size: int = 30):
        self.model = model
        self.buffer_size = buffer_size
        self.buffer = Queue(maxsize=buffer_size)
        self.processing_thread = None
        self.results_queue = Queue()
        
    def start_stream_processing(self):
        """Start background stream processing"""
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        if not self.buffer.full():
            self.buffer.put(frame)
    
    def get_result(self) -> Optional[Any]:
        """Get processed result if available"""
        try:
            return self.results_queue.get_nowait()
        except:
            return None
    
    def _process_stream(self):
        """Background processing loop"""
        frames_batch = []
        
        while True:
            try:
                # Collect frames for batch processing
                while len(frames_batch) < self.optimal_batch_size and not self.buffer.empty():
                    frame = self.buffer.get_nowait()
                    frames_batch.append(frame)
                
                if frames_batch:
                    # Process batch
                    results = self._process_batch(frames_batch)
                    self.results_queue.put(results)
                    frames_batch = []
                else:
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                    
            except Exception as e:
                print(f"Stream processing error: {e}")
    
    def _process_batch(self, frames: List[np.ndarray]) -> List[Any]:
        """Process batch of frames"""
        if not frames:
            return []
        
        batch_tensor = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            for frame in frames
        ]).to(self.device)
        
        with torch.no_grad():
            results = self.model(batch_tensor)
        
        return results
```

### Mixed Precision Optimization

```python
class MixedPrecisionOptimizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
        
    def enable_mixed_precision(self, enable: bool = True):
        """Enable/disable mixed precision training"""
        if enable:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("Mixed precision enabled")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("Mixed precision disabled")
        
        return enable
    
    def infer_with_mixed_precision(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference with mixed precision"""
        with torch.cuda.amp.autocast():
            results = self.model(inputs)
        
        return results
    
    def benchmark_mixed_precision(self, test_inputs: List[torch.Tensor], num_iterations=100):
        """Benchmark mixed precision vs full precision"""
        # Full precision
        torch.backends.cuda.matmul.allow_tf32 = False
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                for inp in test_inputs:
                    self.model(inp.to(self.device))
        fp_time = time.time() - start_time
        
        # Mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                for inp in test_inputs:
                    self.infer_with_mixed_precision(inp.to(self.device))
        mp_time = time.time() - start_time
        
        return {
            'full_precision_time': fp_time,
            'mixed_precision_time': mp_time,
            'speedup': fp_time / mp_time,
            'memory_reduction': self._measure_memory_reduction()
        }
    
    def _measure_memory_reduction(self) -> Dict[str, float]:
        """Measure memory usage reduction with mixed precision"""
        # Measure full precision memory
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        test_input = torch.randn(8, 3, 640, 640).to(self.device)
        
        with torch.no_grad():
            self.model(test_input)
        
        fp_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        # Measure mixed precision memory
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        
        with torch.no_grad():
            self.infer_with_mixed_precision(test_input)
        
        mp_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        return {
            'fp_memory_gb': fp_memory,
            'mp_memory_gb': mp_memory,
            'reduction_ratio': 1 - (mp_memory / fp_memory),
            'savings_gb': fp_memory - mp_memory
        }
```

## Memory Optimization

### Gradient Checkpointing

```python
class MemoryOptimizer:
    def __init__(self, model):
        self.model = model
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage"""
        # For transformer-based models
        for module in self.model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
        
        print("Gradient checkpointing enabled")
    
    def optimize_attention_memory(self, enable_flash_attention: bool = True):
        """Optimize attention mechanism memory usage"""
        if enable_flash_attention:
            try:
                # Enable Flash Attention if available
                torch.backends.cuda.enable_flash_sdp(True)
                print("Flash Attention enabled")
            except:
                print("Flash Attention not available")
        
        # Use memory efficient attention
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                if hasattr(module, 'use_memory_efficient_attention'):
                    module.use_memory_efficient_attention = True
        
        return enable_flash_attention
    
    def dynamic_memory_management(self):
        """Enable dynamic memory management"""
        # Clear cache after each batch
        def clear_memory():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Register memory cleanup hooks
        for module in self.model.modules():
            if hasattr(module, 'register_backward_hook'):
                module.register_backward_hook(
                    lambda mod, grad_in, grad_out: clear_memory()
                )
        
        return clear_memory
    
    def memory_efficient_forward(self, inputs: torch.Tensor):
        """Run forward pass with memory optimizations"""
        # Enable gradient checkpointing if needed
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            with torch.utils.checkpoint.checkpoint(self.model.forward, inputs):
                return self.model(inputs)
        else:
            return self.model(inputs)
```

### Tensor Management

```python
class TensorManager:
    def __init__(self, max_memory_gb: float = 8):
        self.max_memory_bytes = max_memory_gb * (1024**3)
        self.tensor_cache = {}
        self.cache_size = 0
        
    def cache_tensor(self, key: str, tensor: torch.Tensor):
        """Cache tensor with LRU eviction"""
        tensor_size = tensor.numel() * tensor.element_size()
        
        # Check if we need to evict
        while self.cache_size + tensor_size > self.max_memory_bytes and self.tensor_cache:
            # Evict least recently used
            oldest_key = next(iter(self.tensor_cache))
            old_tensor = self.tensor_cache.pop(old_key)
            self.cache_size -= old_tensor.numel() * old_tensor.element_size()
        
        # Cache new tensor
        self.tensor_cache[key] = tensor
        self.cache_size += tensor_size
    
    def get_cached_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Get cached tensor if available"""
        tensor = self.tensor_cache.get(key)
        if tensor is not None:
            # Move to most recently used position
            self.tensor_cache[key] = self.tensor_cache.pop(key)
        return tensor
    
    def clear_cache(self):
        """Clear tensor cache"""
        self.tensor_cache.clear()
        self.cache_size = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout"""
        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Use channels-last layout for convolutions
        if len(tensor.shape) == 4:  # NCHW -> NHWC
            tensor = tensor.permute(0, 2, 3, 1).contiguous()
            tensor = tensor.permute(0, 3, 1, 2)  # Keep original layout
        
        return tensor
```

## Batch Processing Optimization

### Queue-based Processing

```python
from queue import Queue, Empty
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchProcessor:
    def __init__(self, model, max_workers: int = 4, max_queue_size: int = 100):
        self.model = model
        self.max_workers = max_workers
        self.input_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue()
        self.workers = []
        self.running = False
        
    def start_workers(self):
        """Start processing workers"""
        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop_workers(self):
        """Stop processing workers"""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.workers = []
    
    def _worker_loop(self, worker_id: int):
        """Worker processing loop"""
        while self.running:
            try:
                # Get batch from queue
                batch_items = []
                timeout = 1.0  # Timeout to check running flag
                
                # Collect batch
                for _ in range(self.optimal_batch_size):
                    try:
                        item = self.input_queue.get(timeout=timeout)
                        batch_items.append(item)
                    except Empty:
                        break
                
                if batch_items:
                    # Process batch
                    results = self._process_batch(batch_items)
                    
                    # Put results in result queue
                    for i, result in enumerate(results):
                        self.result_queue.put((batch_items[i][0], result))
                    
                    # Mark batch items as done
                    for item in batch_items:
                        self.input_queue.task_done()
                
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    def process_batch_async(self, items: List[Any]) -> List[Any]:
        """Add items to batch processing queue"""
        results = []
        
        # Add items to queue
        for item in items:
            self.input_queue.put((item, item))  # Use item as both key and value
        
        # Collect results
        for _ in items:
            try:
                key, result = self.result_queue.get(timeout=10)
                results.append(result)
            except Empty:
                print("Result queue timeout")
                break
        
        return results
```

### Dynamic Batch Sizing

```python
class DynamicBatchSizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimal_batch_size = 1
        self.memory_threshold = 0.8
        self.performance_history = []
        
    def adapt_batch_size(self, images: List[np.ndarray]) -> List[Any]:
        """Process images with adaptive batch size"""
        batch_start = 0
        all_results = []
        
        while batch_start < len(images):
            # Find optimal batch size for current memory state
            batch_size = self._find_optimal_batch_size()
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            # Process batch
            batch_results = self._process_batch(batch_images)
            all_results.extend(batch_results)
            
            # Update performance metrics
            self._update_performance_metrics(len(batch_images), batch_results)
            
            batch_start = batch_end
        
        return all_results
    
    def _find_optimal_batch_size(self) -> int:
        """Find optimal batch size based on current performance"""
        if not self.performance_history:
            return 1
        
        # Get recent performance metrics
        recent_metrics = self.performance_history[-10:]
        
        # Calculate average throughput
        avg_throughput = np.mean([m['throughput'] for m in recent_metrics])
        avg_memory = np.mean([m['memory_usage'] for m in recent_metrics])
        
        # Adjust batch size based on memory usage
        if avg_memory > self.memory_threshold:
            # Reduce batch size
            self.optimal_batch_size = max(1, self.optimal_batch_size // 2)
        elif avg_memory < self.memory_threshold * 0.6:
            # Try to increase batch size
            self.optimal_batch_size = min(32, self.optimal_batch_size * 2)
        
        return self.optimal_batch_size
    
    def _update_performance_metrics(self, batch_size: int, results: List[Any]):
        """Update performance tracking"""
        current_time = time.time()
        
        metrics = {
            'timestamp': current_time,
            'batch_size': batch_size,
            'num_results': len(results),
            'throughput': len(results) / (time.time() - current_time),
            'memory_usage': self._get_memory_usage(),
            'gpu_utilization': self._get_gpu_utilization()
        }
        
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage fraction"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return allocated / total
        return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            gpu = GPUtil.getGPUs()[0]
            return gpu.load * 100
        except:
            return 0.0
```

## GPU Optimization

### CUDA Optimization

```python
class CUDAOptimizer:
    def __init__(self):
        self.device_count = torch.cuda.device_count()
        self.current_device = torch.cuda.current_device()
        
    def optimize_cuda_settings(self):
        """Optimize CUDA settings for inference"""
        # Enable TF32 for better performance on RTX 30 series+
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = True
        
        # Enable cuDNN deterministic mode for consistent results
        # torch.backends.cudnn.deterministic = True
        
        # Set memory fraction if needed
        # torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Configure CUDA streams
        self.streams = {
            'compute': torch.cuda.Stream(device=self.current_device),
            'communication': torch.cuda.Stream(device=self.current_device)
        }
        
        return {
            'tf32_enabled': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'device_count': self.device_count,
            'current_device': self.current_device
        }
    
    def multi_gpu_setup(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Set up model for multi-GPU inference"""
        if self.device_count <= 1:
            return [model]
        
        # Data parallel setup
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for inference")
            model = torch.nn.DataParallel(model)
        
        # Model parallel setup (more advanced)
        model_parts = self._parallelize_model(model)
        
        return model_parts
    
    def _parallelize_model(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Split model across multiple GPUs"""
        # This is a simplified example - real implementation would be more complex
        model_parts = []
        
        # Split model by layers
        layers = list(model.modules())
        mid_point = len(layers) // 2
        
        # First half on GPU 0
        part1 = torch.nn.Sequential(*layers[:mid_point]).cuda(0)
        
        # Second half on GPU 1
        part2 = torch.nn.Sequential(*layers[mid_point:]).cuda(1)
        
        model_parts = [part1, part2]
        
        return model_parts
    
    def benchmark_gpu_performance(self, model, test_inputs: List[torch.Tensor]):
        """Benchmark GPU performance across different configurations"""
        results = {}
        
        # Single GPU performance
        single_gpu_results = self._benchmark_single_gpu(model, test_inputs)
        results['single_gpu'] = single_gpu_results
        
        # Multi-GPU performance (if available)
        if self.device_count > 1:
            multi_gpu_model = torch.nn.DataParallel(model)
            multi_gpu_results = self._benchmark_single_gpu(multi_gpu_model, test_inputs)
            results['multi_gpu'] = multi_gpu_results
            
            # Calculate speedup
            results['speedup'] = (single_gpu_results['avg_time'] / 
                                multi_gpu_results['avg_time'])
        
        return results
    
    def _benchmark_single_gpu(self, model, test_inputs):
        """Benchmark single GPU configuration"""
        model.eval()
        torch.cuda.empty_cache()
        
        num_iterations = 50
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                for inp in test_inputs[:2]:
                    model(inp.cuda())
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                for inp in test_inputs:
                    model(inp.cuda())
        end_time = time.time()
        
        # Measure memory
        memory_used = torch.cuda.max_memory_allocated() / (1024**3)
        
        return {
            'total_time': end_time - start_time,
            'avg_time': (end_time - start_time) / num_iterations,
            'fps': num_iterations / (end_time - start_time),
            'memory_used_gb': memory_used,
            'throughput_mb_s': sum(inp.numel() * 4 * num_iterations for inp in test_inputs) / (end_time - start_time) / (1024**2)
        }
```

### CUDA Streams and Events

```python
class CUDAStreamManager:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        torch.cuda.set_device(device_id)
        
        # Create CUDA streams
        self.streams = {
            'main': torch.cuda.Stream(device=device_id),
            'preprocessing': torch.cuda.Stream(device=device_id),
            'inference': torch.cuda.Stream(device=device_id),
            'postprocessing': torch.cuda.Stream(device=device_id)
        }
        
        # Create events
        self.events = {
            'preprocessing_done': torch.cuda.Event(enable_timing=True),
            'inference_done': torch.cuda.Event(enable_timing=True),
            'postprocessing_done': torch.cuda.Event(enable_timing=True)
        }
    
    def overlapping_processing(self, image_batches: List[torch.Tensor]):
        """Overlap preprocessing, inference, and postprocessing"""
        results = []
        
        for i, batch in enumerate(image_batches):
            with torch.cuda.stream(self.streams['main']):
                # Preprocessing
                with torch.cuda.stream(self.streams['preprocessing']):
                    processed_batch = self._preprocess_batch(batch)
                    self.events['preprocessing_done'].record()
                
                # Wait for preprocessing to complete
                torch.cuda.current_stream().wait_event(self.events['preprocessing_done'])
                
                # Inference
                with torch.cuda.stream(self.streams['inference']):
                    inference_results = self._run_inference(processed_batch)
                    self.events['inference_done'].record()
                
                # Wait for inference to complete
                torch.cuda.current_stream().wait_event(self.events['inference_done'])
                
                # Postprocessing
                with torch.cuda.stream(self.streams['postprocessing']):
                    final_results = self._postprocess_results(inference_results)
                    self.events['postprocessing_done'].record()
                
                # Wait for all processing to complete
                torch.cuda.current_stream().wait_event(self.events['postprocessing_done'])
                
                results.append(final_results)
        
        # Synchronize all streams
        torch.cuda.synchronize()
        
        return results
    
    def _preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Preprocess batch on separate stream"""
        # Preprocessing operations
        normalized = batch / 255.0
        resized = torch.nn.functional.interpolate(normalized, size=(640, 640))
        return resized
    
    def _run_inference(self, batch: torch.Tensor) -> torch.Tensor:
        """Run inference on separate stream"""
        with torch.no_grad():
            results = self.model(batch)
        return results
    
    def _postprocess_results(self, results: torch.Tensor) -> Any:
        """Postprocess results on separate stream"""
        # Postprocessing operations
        processed = torch.softmax(results, dim=1)
        return processed
```

## Parallelization Strategies

### Thread vs Process Pool

```python
class ParallelizationManager:
    def __init__(self, model, use_processes: bool = False):
        self.model = model
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
    def parallel_inference(self, 
                          images: List[np.ndarray], 
                          num_workers: int = 4) -> List[Any]:
        """Run parallel inference on multiple CPU cores or threads"""
        with self.executor_class(max_workers=num_workers) as executor:
            if self.use_processes:
                # For CPU-intensive preprocessing with processes
                futures = {
                    executor.submit(self._process_and_infer, image): i 
                    for i, image in enumerate(images)
                }
            else:
                # For I/O-bound operations with threads
                futures = {
                    executor.submit(self._run_inference_cpu, image): i 
                    for i, image in enumerate(images)
                }
            
            # Collect results
            results = [None] * len(images)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Processing failed for image {idx}: {e}")
                    results[idx] = None
        
        return results
    
    def _process_and_infer(self, image: np.ndarray) -> Any:
        """Process image and run inference (for processes)"""
        # Heavy preprocessing
        processed_image = self._heavy_preprocessing(image)
        
        # Run inference (would need model in each process)
        # result = self.model(processed_image)
        
        return processed_image
    
    def _run_inference_cpu(self, image: np.ndarray) -> Any:
        """Run inference on CPU (for threads)"""
        # Lighter operations suitable for threads
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
        else:
            image_tensor = image
        
        return image_tensor
```

### Distributed Processing

```python
class DistributedProcessor:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        
        # Initialize distributed processing
        if world_size > 1:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
    
    def distributed_inference(self, 
                             images: List[np.ndarray],
                             model: torch.nn.Module) -> List[Any]:
        """Run distributed inference across multiple GPUs"""
        if self.world_size == 1:
            # Single GPU inference
            return self._single_gpu_inference(images, model)
        
        # Split images across GPUs
        images_per_gpu = len(images) // self.world_size
        start_idx = self.rank * images_per_gpu
        end_idx = start_idx + images_per_gpu if self.rank < self.world_size - 1 else len(images)
        
        local_images = images[start_idx:end_idx]
        
        # Gather results from all GPUs
        local_results = self._single_gpu_inference(local_images, model)
        
        # Gather all results to rank 0
        if torch.distributed.is_available():
            # Gather local result counts
            local_count = torch.tensor(len(local_results), dtype=torch.long, device='cuda')
            gather_counts = [torch.zeros_like(local_count) for _ in range(self.world_size)]
            torch.distributed.all_gather(gather_counts, local_count)
            
            # Gather local results
            all_results = [None] * sum(count.item() for count in gather_counts)
            
            for i, count in enumerate(gather_counts):
                start = sum(c.item() for c in gather_counts[:i])
                end = start + count.item()
                
                if i == self.rank:
                    # Send local results
                    local_tensors = [torch.tensor(r) for r in local_results]
                    torch.distributed.gather(
                        torch.stack(local_tensors),
                        gather_list=all_results[start:end] if i != 0 else None,
                        dst=0
                    )
                elif i != 0:
                    # Receive results
                    receive_list = [torch.zeros(count.item(), dtype=torch.long, device='cuda') 
                                  for _ in range(count.item())]
                    torch.distributed.gather(
                        torch.tensor([]),
                        gather_list=receive_list,
                        dst=i
                    )
                    all_results[start:end] = [t.item() for t in receive_list]
        
        return all_results if self.rank == 0 else None
    
    def _single_gpu_inference(self, images: List[np.ndarray], model: torch.nn.Module) -> List[Any]:
        """Single GPU inference"""
        model.eval()
        
        results = []
        for image in images:
            with torch.no_grad():
                # Assuming single image processing
                if isinstance(image, np.ndarray):
                    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                    tensor = tensor.unsqueeze(0).cuda()
                
                result = model(tensor)
                results.append(result.cpu().item())
        
        return results
```

## Caching Strategies

### Multi-level Caching

```python
from functools import lru_cache
import redis
import pickle

class MultiLevelCache:
    def __init__(self, 
                 l1_cache_size: int = 1000,  # In-memory LRU cache
                 l2_cache_ttl: int = 3600,   # Redis cache TTL
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        
        # L1 Cache (in-memory)
        self.l1_cache = {}
        self.l1_cache_size = l1_cache_size
        self.l1_access_count = {}
        
        # L2 Cache (Redis)
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
            self.redis_client.ping()
            self.l2_available = True
        except:
            self.redis_client = None
            self.l2_available = False
        
        self.l2_cache_ttl = l2_cache_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        # Try L1 cache first
        if key in self.l1_cache:
            self.l1_access_count[key] = self.l1_access_count.get(key, 0) + 1
            return self.l1_cache[key]
        
        # Try L2 cache if available
        if self.l2_available:
            try:
                serialized_value = self.redis_client.get(f"l2:{key}")
                if serialized_value:
                    value = pickle.loads(serialized_value)
                    
                    # Promote to L1 cache
                    self._promote_to_l1(key, value)
                    return value
            except Exception as e:
                print(f"L2 cache error: {e}")
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in multi-level cache"""
        # Always set in L1 cache
        self._promote_to_l1(key, value)
        
        # Set in L2 cache if available
        if self.l2_available:
            try:
                serialized_value = pickle.dumps(value)
                self.redis_client.setex(
                    f"l2:{key}", 
                    self.l2_cache_ttl, 
                    serialized_value
                )
            except Exception as e:
                print(f"L2 cache write error: {e}")
    
    def _promote_to_l1(self, key: str, value: Any) -> None:
        """Promote item to L1 cache with LRU eviction"""
        # If cache is full, evict least recently used item
        if len(self.l1_cache) >= self.l1_cache_size and key not in self.l1_cache:
            lru_key = min(self.l1_access_count.keys(), 
                         key=lambda k: self.l1_access_count[k])
            del self.l1_cache[lru_key]
            del self.l1_access_count[lru_key]
        
        self.l1_cache[key] = value
        self.l1_access_count[key] = self.l1_access_count.get(key, 0) + 1
    
    def clear_l1(self) -> None:
        """Clear L1 cache"""
        self.l1_cache.clear()
        self.l1_access_count.clear()
    
    def clear_l2(self) -> None:
        """Clear L2 cache"""
        if self.l2_available:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                print(f"L2 cache clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        l1_size = len(self.l1_cache)
        l1_max = self.l1_cache_size
        l1_usage = l1_size / l1_max
        
        l2_available = self.l2_available
        if l2_available:
            try:
                l2_size = self.redis_client.dbsize()
            except:
                l2_size = 0
        else:
            l2_size = 0
        
        return {
            'l1_cache': {
                'size': l1_size,
                'max_size': l1_max,
                'usage_ratio': l1_usage,
                'items': list(self.l1_cache.keys())
            },
            'l2_cache': {
                'available': l2_available,
                'size': l2_size
            }
        }

# Usage in PlayerRecognizer
class OptimizedPlayerRecognizer(PlayerRecognizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = MultiLevelCache(
            l1_cache_size=500,
            l2_cache_ttl=1800  # 30 minutes
        )
    
    @lru_cache(maxsize=100)
    def _cached_detection(self, image_hash: str, confidence_threshold: float):
        """Cached detection results"""
        # Actual detection logic here
        pass
    
    def detect_players(self, image: np.ndarray, **kwargs):
        # Create cache key
        image_hash = hash(image.tobytes())
        cache_key = f"detection_{image_hash}_{kwargs.get('confidence_threshold', 0.7)}"
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Run detection
        result = super().detect_players(image, **kwargs)
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
```

### Result Deduplication

```python
class ResultDeduplicator:
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.result_cache = []
        
    def deduplicate_results(self, results: List[Any]) -> List[Any]:
        """Remove duplicate or very similar results"""
        if not results:
            return results
        
        unique_results = []
        
        for result in results:
            is_duplicate = False
            
            for cached_result in self.result_cache:
                if self._results_similar(result, cached_result):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                self.result_cache.append(result)
                
                # Limit cache size
                if len(self.result_cache) > 1000:
                    self.result_cache.pop(0)
        
        return unique_results
    
    def _results_similar(self, result1: Any, result2: Any) -> bool:
        """Check if two results are similar enough to be considered duplicates"""
        if not hasattr(result1, 'bbox') or not hasattr(result2, 'bbox'):
            return False
        
        # Calculate IoU between bounding boxes
        bbox1 = np.array(result1.bbox)
        bbox2 = np.array(result2.bbox)
        
        if bbox1.shape != bbox2.shape:
            return False
        
        # Calculate average IoU
        avg_iou = 0
        for b1, b2 in zip(bbox1, bbox2):
            iou = self._calculate_iou(b1, b2)
            avg_iou += iou
        
        avg_iou /= len(bbox1)
        
        return avg_iou > self.similarity_threshold
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
```

## Monitoring and Profiling

### Real-time Performance Monitor

```python
import time
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PerformanceSnapshot:
    timestamp: float
    fps: float
    latency_ms: float
    gpu_utilization: float
    gpu_memory_gb: float
    cpu_utilization: float
    system_memory_gb: float

class RealTimeMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.snapshots: List[PerformanceSnapshot] = []
        self.start_time = time.time()
        
    def record_snapshot(self, 
                       current_fps: float,
                       current_latency_ms: float,
                       gpu_utilization: float = 0.0,
                       gpu_memory_gb: float = 0.0,
                       cpu_utilization: float = 0.0,
                       system_memory_gb: float = 0.0):
        """Record performance snapshot"""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            fps=current_fps,
            latency_ms=current_latency_ms,
            gpu_utilization=gpu_utilization,
            gpu_memory_gb=gpu_memory_gb,
            cpu_utilization=cpu_utilization,
            system_memory_gb=system_memory_gb
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots
        if len(self.snapshots) > self.window_size:
            self.snapshots.pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary over time window"""
        if not self.snapshots:
            return {}
        
        recent_snapshots = self.snapshots[-min(50, len(self.snapshots)):]  # Last 50 snapshots
        
        # Calculate statistics
        fps_values = [s.fps for s in recent_snapshots]
        latency_values = [s.latency_ms for s in recent_snapshots]
        
        return {
            'fps': {
                'current': recent_snapshots[-1].fps if recent_snapshots else 0,
                'average': np.mean(fps_values),
                'min': np.min(fps_values),
                'max': np.max(fps_values),
                'std': np.std(fps_values)
            },
            'latency_ms': {
                'current': recent_snapshots[-1].latency_ms if recent_snapshots else 0,
                'average': np.mean(latency_values),
                'min': np.min(latency_values),
                'max': np.max(latency_values),
                'p95': np.percentile(latency_values, 95),
                'p99': np.percentile(latency_values, 99)
            },
            'resources': {
                'gpu_utilization': np.mean([s.gpu_utilization for s in recent_snapshots]),
                'gpu_memory_gb': np.mean([s.gpu_memory_gb for s in recent_snapshots]),
                'cpu_utilization': np.mean([s.cpu_utilization for s in recent_snapshots]),
                'system_memory_gb': np.mean([s.system_memory_gb for s in recent_snapshots])
            },
            'uptime_seconds': time.time() - self.start_time
        }
    
    def detect_performance_anomalies(self) -> List[str]:
        """Detect performance anomalies"""
        if len(self.snapshots) < 10:
            return []
        
        anomalies = []
        recent = self.snapshots[-10:]
        
        # Check for low FPS
        avg_fps = np.mean([s.fps for s in recent])
        if avg_fps < 10:  # Less than 10 FPS
            anomalies.append(f"Low FPS detected: {avg_fps:.1f}")
        
        # Check for high latency
        avg_latency = np.mean([s.latency_ms for s in recent])
        if avg_latency > 100:  # More than 100ms
            anomalies.append(f"High latency detected: {avg_latency:.1f}ms")
        
        # Check for high GPU memory usage
        avg_gpu_memory = np.mean([s.gpu_memory_gb for s in recent])
        if avg_gpu_memory > 20:  # More than 20GB
            anomalies.append(f"High GPU memory usage: {avg_gpu_memory:.1f}GB")
        
        # Check for consistent low GPU utilization (might indicate CPU bottleneck)
        avg_gpu_util = np.mean([s.gpu_utilization for s in recent])
        if avg_gpu_util < 0.5 and avg_fps < 30:  # Low GPU utilization with low FPS
            anomalies.append("Possible CPU bottleneck: Low GPU utilization with low FPS")
        
        return anomalies

# Usage in the main application
class PerformanceAwarePlayerRecognizer(PlayerRecognizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = RealTimeMonitor()
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def analyze_scene(self, *args, **kwargs):
        """Override to add performance monitoring"""
        start_time = time.time()
        
        # Run original analysis
        result = super().analyze_scene(*args, **kwargs)
        
        # Record performance metrics
        elapsed_time = time.time() - start_time
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update FPS every second
            fps = self.frame_count / (current_time - self.last_fps_time)
            
            # Get resource usage
            gpu_stats = self._get_gpu_stats()
            cpu_stats = psutil.cpu_percent(interval=0.1)
            memory_stats = psutil.virtual_memory()
            
            # Record snapshot
            self.monitor.record_snapshot(
                current_fps=fps,
                current_latency_ms=elapsed_time * 1000,
                gpu_utilization=gpu_stats.get('utilization', 0),
                gpu_memory_gb=gpu_stats.get('memory_used_gb', 0),
                cpu_utilization=cpu_stats,
                system_memory_gb=memory_stats.used / (1024**3)
            )
            
            # Check for anomalies
            anomalies = self.monitor.detect_performance_anomalies()
            if anomalies:
                print("Performance anomalies detected:")
                for anomaly in anomalies:
                    print(f"  - {anomaly}")
            
            self.frame_count = 0
            self.last_fps_time = current_time
        
        return result
    
    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get current GPU statistics"""
        if not torch.cuda.is_available():
            return {}
        
        try:
            gpu = GPUtil.getGPUs()[0]
            return {
                'utilization': gpu.load * 100,
                'memory_used_gb': gpu.memoryUsed / 1024,
                'memory_total_gb': gpu.memoryTotal / 1024,
                'temperature': gpu.temperature
            }
        except:
            return {}
```

### Performance Profiling Integration

```python
class IntegratedProfiler:
    def __init__(self, model_name: str = "soccer_recognition"):
        self.model_name = model_name
        self.profile_data = {}
        self.active_sessions = {}
        
    def start_profiling_session(self, session_id: str):
        """Start a profiling session"""
        self.active_sessions[session_id] = {
            'start_time': time.time(),
            'operation_times': {},
            'resource_usage': []
        }
    
    def end_profiling_session(self, session_id: str) -> Dict[str, Any]:
        """End profiling session and return results"""
        if session_id not in self.active_sessions:
            return {}
        
        session_data = self.active_sessions[session_id]
        session_data['end_time'] = time.time()
        session_data['total_duration'] = session_data['end_time'] - session_data['start_time']
        
        # Calculate statistics
        results = self._calculate_session_statistics(session_data)
        
        # Store in profile data
        self.profile_data[session_id] = results
        
        # Clean up
        del self.active_sessions[session_id]
        
        return results
    
    def profile_operation(self, session_id: str, operation_name: str):
        """Decorator for profiling operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if session_id not in self.active_sessions:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_gpu_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_gpu_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                    
                    operation_time = end_time - start_time
                    memory_delta = end_gpu_memory - start_gpu_memory
                    
                    # Record operation metrics
                    if operation_name not in self.active_sessions[session_id]['operation_times']:
                        self.active_sessions[session_id]['operation_times'][operation_name] = {
                            'times': [],
                            'memory_usage': [],
                            'success_count': 0,
                            'error_count': 0
                        }
                    
                    op_data = self.active_sessions[session_id]['operation_times'][operation_name]
                    op_data['times'].append(operation_time)
                    op_data['memory_usage'].append(memory_delta)
                    
                    if success:
                        op_data['success_count'] += 1
                    else:
                        op_data['error_count'] += 1
                
                return result
            return wrapper
        return decorator
    
    def _calculate_session_statistics(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for a profiling session"""
        results = {
            'session_duration': session_data['total_duration'],
            'total_operations': 0,
            'operations': {}
        }
        
        for op_name, op_data in session_data['operation_times'].items():
            times = op_data['times']
            memory_usage = op_data['memory_usage']
            
            results['operations'][op_name] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'p95_time': np.percentile(times, 95),
                'total_time': np.sum(times),
                'avg_memory_gb': np.mean(memory_usage),
                'max_memory_gb': np.max(memory_usage),
                'success_rate': op_data['success_count'] / (op_data['success_count'] + op_data['error_count']),
                'throughput_per_second': len(times) / session_data['total_duration']
            }
            
            results['total_operations'] += len(times)
        
        return results
    
    def export_profile_report(self, session_id: str, output_path: str):
        """Export profiling report to file"""
        if session_id not in self.profile_data:
            print(f"Session {session_id} not found")
            return
        
        data = self.profile_data[session_id]
        
        # Create HTML report
        html_content = self._generate_html_report(session_id, data)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Profile report exported to {output_path}")
    
    def _generate_html_report(self, session_id: str, data: Dict[str, Any]) -> str:
        """Generate HTML profiling report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Profile - {session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .operation {{ border-left: 4px solid #2196F3; padding-left: 15px; margin: 15px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Profile Report</h1>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Duration:</strong> {data['session_duration']:.2f} seconds</p>
                <p><strong>Total Operations:</strong> {data['total_operations']}</p>
            </div>
            
            <h2>Operation Statistics</h2>
        """
        
        for op_name, op_stats in data['operations'].items():
            html += f"""
            <div class="operation">
                <h3>{op_name}</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Count</td><td>{op_stats['count']}</td></tr>
                    <tr><td>Average Time</td><td>{op_stats['avg_time']*1000:.2f} ms</td></tr>
                    <tr><td>Min Time</td><td>{op_stats['min_time']*1000:.2f} ms</td></tr>
                    <tr><td>Max Time</td><td>{op_stats['max_time']*1000:.2f} ms</td></tr>
                    <tr><td>P95 Time</td><td>{op_stats['p95_time']*1000:.2f} ms</td></tr>
                    <tr><td>Total Time</td><td>{op_stats['total_time']:.2f} s</td></tr>
                    <tr><td>Avg Memory</td><td>{op_stats['avg_memory_gb']:.2f} GB</td></tr>
                    <tr><td>Max Memory</td><td>{op_stats['max_memory_gb']:.2f} GB</td></tr>
                    <tr><td>Success Rate</td><td>{op_stats['success_rate']*100:.1f}%</td></tr>
                    <tr><td>Throughput</td><td>{op_stats['throughput_per_second']:.1f} ops/sec</td></tr>
                </table>
            </div>
            """
        
        html += "</body></html>"
        return html
```

This comprehensive performance optimization guide covers all aspects of optimizing the Soccer Player Recognition System. It includes strategies for hardware optimization, model optimization, inference optimization, memory management, parallelization, caching, and monitoring. The techniques described can significantly improve system performance across different deployment scenarios and hardware configurations.