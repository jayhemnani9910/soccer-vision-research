# Soccer Player Recognition System - Demo Applications

Comprehensive demonstration suite for the Soccer Player Recognition System featuring multiple advanced models and real-time processing capabilities.

## 🎯 Overview

This demo suite showcases the complete soccer player recognition pipeline including:
- **Object Detection** (RF-DETR) - Real-time football object detection
- **Segmentation** (SAM2) - Advanced player segmentation  
- **Identification** (SigLIP) - Multimodal player identification
- **Classification** (ResNet) - Player feature extraction and classification

## 📋 Demo Applications

### 1. Complete System Demo (`demos/complete_system_demo.py`)
**Full system integration demo showcasing the entire pipeline**

**Features:**
- End-to-end processing pipeline
- Object detection → Segmentation → Player identification
- Synthetic soccer field generation
- Comprehensive performance metrics
- Results visualization and reporting

**Usage:**
```bash
python demos/complete_system_demo.py
```

**Output:** Complete system demonstration with pipeline results saved to `outputs/complete_system_demo/`

---

### 2. Single Model Demo (`demos/single_model_demo.py`)
**Individual model testing and analysis**

**Features:**
- Test each model independently
- RF-DETR object detection testing
- SAM2 segmentation testing  
- SigLIP identification testing
- ResNet classification testing
- Interactive demo mode
- Detailed performance analysis

**Usage:**
```bash
python demos/single_model_demo.py
```

**Interactive Mode:** Choose individual models for focused testing

---

### 3. Real-Time Demo (`demos/real_time_demo.py`)
**Real-time processing capabilities demonstration**

**Features:**
- Live video processing simulation
- Multi-threaded processing
- Performance optimization
- Memory management
- Error handling and recovery
- Multiple processing configurations:
  - High-speed (60 FPS)
  - Standard (30 FPS) 
  - Low-power (15 FPS)

**Usage:**
```bash
python demos/real_time_demo.py
```

**Capabilities:**
- Real-time frame processing
- Threading optimization
- Resource monitoring
- Performance comparison across configurations

---

### 4. Benchmark Demo (`demos/benchmark_demo.py`)
**Comprehensive performance benchmarking suite**

**Features:**
- Model performance comparison
- Load testing (1-50 concurrent users)
- Stress testing (extended duration)
- Resource profiling (CPU, memory, GPU)
- Automated report generation
- Scalability analysis

**Test Types:**
- **Model Benchmarking:** Individual model performance testing
- **Load Testing:** Multi-user concurrent access testing
- **Stress Testing:** Long-duration sustained load testing
- **Resource Monitoring:** Real-time system resource tracking

**Usage:**
```bash
python demos/benchmark_demo.py
```

**Output:** Comprehensive performance reports in `outputs/benchmark_demo/`

---

### 5. GUI Interface (`UI/demo_interface.py`)
**Complete graphical user interface for easy interaction**

**Features:**
- Interactive model testing
- Real-time performance monitoring
- File browser and image viewer
- Configuration management
- Multi-tab interface:
  - Dashboard (main view)
  - Model Testing
  - Real-Time Processing
  - Benchmark Suite
  - Settings

**Usage:**
```bash
python UI/demo_interface.py
```

**Requirements:** tkinter (usually included with Python)

---

## 🚀 Quick Start

### Option 1: Use the Demo Launcher (Recommended)
```bash
python run_demos.py
```

**Interactive Menu Options:**
- Select individual demos (1-4)
- Run all demos (0)
- Launch GUI directly (`--gui`)
- Check dependencies (`--check-deps`)
- List available demos (`--list`)

### Option 2: Run Individual Demos
```bash
# Complete system demo
python demos/complete_system_demo.py

# Single model testing
python demos/single_model_demo.py

# Real-time processing
python demos/real_time_demo.py

# Performance benchmarking
python demos/benchmark_demo.py

# GUI interface
python UI/demo_interface.py
```

### Option 3: Command Line Options
```bash
# Check dependencies
python run_demos.py --check-deps

# List available demos
python run_demos.py --list

# Run specific demo
python run_demos.py --demo single_model_demo

# Launch GUI directly
python run_demos.py --gui
```

## 📊 Demo Outputs

All demos generate detailed results in the `outputs/` directory:

```
outputs/
├── complete_system_demo/
│   ├── demo_summary.json
│   └── pipeline_run_*.json
├── single_model_demo/
│   ├── demo_summary.json
│   └── {model}_results.json
├── real_time_demo/
│   └── demo_summary.json
└── benchmark_demo/
    ├── benchmark_summary.json
    └── *_results.json
```

### Output Features:
- **JSON Reports:** Structured data with all metrics
- **Performance Statistics:** Processing times, throughput, accuracy
- **Resource Usage:** CPU, memory, GPU utilization
- **Visualization Data:** Charts and graphs data
- **Configuration Details:** Test parameters and settings

## 🔧 Configuration

### Model Configurations
Each demo supports different model configurations:

**RF-DETR (Detection):**
- Input sizes: 640x640, 800x600, 1024x768
- Batch sizes: 1, 2, 4, 8
- Classes: player, ball, referee, goalkeeper

**SAM2 (Segmentation):**
- Input sizes: 512x512, 1024x1024, 1536x1024
- Batch sizes: 1, 2, 4
- Features: instance segmentation, mask generation

**SigLIP (Identification):**
- Input sizes: 224x224, 384x384, 512x512
- Batch sizes: 1, 4, 8, 16, 32
- Features: player ID, team classification

**ResNet (Classification):**
- Input sizes: 224x224, 256x256, 384x384
- Batch sizes: 1, 8, 16, 32, 64
- Features: feature extraction, classification

### Performance Settings
- **Real-time FPS:** 15, 30, 60 FPS
- **Threading:** Configurable worker threads
- **Memory Management:** Automatic cleanup and optimization
- **Error Handling:** Graceful degradation and recovery

## 📈 Performance Metrics

Each demo tracks comprehensive performance metrics:

**Processing Metrics:**
- Processing time (min, max, avg, std)
- Throughput (samples/requests per second)
- Frame rate (FPS)
- Memory usage (MB, percentage)
- CPU utilization (percentage)
- GPU utilization (if available)

**Quality Metrics:**
- Detection accuracy
- Segmentation coverage
- Identification confidence
- Classification accuracy (Top-1, Top-5)
- Success rate (percentage)

**Scalability Metrics:**
- Concurrent user handling
- Load test results
- Stress test endurance
- Resource scaling patterns

## 🛠️ Dependencies

**Required Packages:**
```
numpy>=1.20.0
opencv-python>=4.5.0
matplotlib>=3.3.0
Pillow>=8.0.0
psutil>=5.8.0
tkinter (usually included with Python)
```

**Optional (for enhanced features):**
```
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

**Installation:**
```bash
pip install numpy opencv-python matplotlib Pillow psutil
```

## 🎮 Interactive Features

### GUI Interface Tabs:

1. **Dashboard:** Overview and quick actions
2. **Model Testing:** Individual model controls
3. **Real-Time Processing:** Live stream simulation
4. **Benchmark Suite:** Comprehensive testing
5. **Settings:** Configuration management

### Interactive Controls:
- **File Browser:** Load images and videos
- **Model Selection:** Choose specific models to test
- **Parameter Tuning:** Adjust input sizes, batch sizes
- **Real-time Monitoring:** Live performance graphs
- **Result Visualization:** Charts and statistics

## 📝 Example Usage Scenarios

### Scenario 1: Model Development
```bash
# Test individual model performance
python run_demos.py --demo single_model_demo

# Focus on specific model
# Select "resnet" in interactive mode
# Test with different input sizes and batch sizes
```

### Scenario 2: System Integration Testing
```bash
# Run complete pipeline
python demos/complete_system_demo.py

# Check integration results
# Review generated synthetic soccer images
# Analyze pipeline performance metrics
```

### Scenario 3: Performance Optimization
```bash
# Benchmark all models
python demos/benchmark_demo.py

# Review performance report
# Identify bottlenecks
# Optimize configurations
```

### Scenario 4: Real-time Deployment Testing
```bash
# Test real-time capabilities
python demos/real_time_demo.py

# Compare different FPS settings
# Monitor resource usage
# Validate timing constraints
```

## 🔍 Troubleshooting

### Common Issues:

1. **Missing Dependencies:**
   ```bash
   python run_demos.py --check-deps
   pip install missing_packages
   ```

2. **GUI Not Starting:**
   ```bash
   # Install tkinter (Linux)
   sudo apt-get install python3-tk
   
   # Use command-line demos instead
   python demos/single_model_demo.py
   ```

3. **Performance Issues:**
   ```bash
   # Reduce batch sizes
   # Lower target FPS
   # Use CPU-only mode if GPU issues
   ```

4. **Memory Issues:**
   ```bash
   # Reduce input image sizes
   # Limit concurrent operations
   # Enable memory optimization
   ```

### Debug Mode:
All demos support debug logging by setting environment variable:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python demos/single_model_demo.py  # Will show detailed logs
```

## 📚 Advanced Usage

### Custom Configurations:
Each demo can be modified to test custom scenarios:
- Different image sizes
- Varying batch sizes
- Custom model parameters
- Specialized test data

### Integration with CI/CD:
Demos can be integrated into automated testing:
```bash
# Automated benchmark
python run_demos.py --demo benchmark_demo

# Integration testing
python run_demos.py --demo complete_system_demo

# Performance regression testing
python run_demos.py --demo benchmark_demo
```

## 🤝 Contributing

To extend the demo suite:

1. **Add New Models:** Follow existing model patterns
2. **Enhance Visualizations:** Add to visualization modules
3. **Improve Performance:** Optimize processing pipelines
4. **Add Test Cases:** Expand testing coverage

## 📄 License

This demo suite is part of the Soccer Player Recognition System.
See main project license for details.

## 🎉 Success Indicators

**All demos working correctly when:**
- ✅ All dependencies satisfied
- ✅ No import errors
- ✅ Performance metrics generated
- ✅ Output files created
- ✅ GUI launches successfully (if applicable)

**Ready for deployment when:**
- ✅ All benchmark tests pass
- ✅ Real-time processing meets requirements
- ✅ System integration validated
- ✅ Performance targets achieved

---

**Built with ❤️ by the Soccer Player Recognition Team**

*For technical support or questions, please refer to the main project documentation.*