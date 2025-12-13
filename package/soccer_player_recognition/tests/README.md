# Soccer Player Recognition Testing Framework

This directory contains a comprehensive testing framework for the Soccer Player Recognition System.

## Overview

The testing framework is organized into several layers:

### Test Categories

1. **Unit Tests** - Test individual components in isolation
   - `test_all_models.py` - Tests for all ML models (RF-DETR, SAM2, SigLIP, ResNet, MediaPipe)
   - `utils_tests.py` - Tests for utility modules (config, image processing, logging, etc.)

2. **Integration Tests** - Test interactions between components
   - `integration_tests.py` - End-to-end pipeline and component integration

3. **Performance Tests** - Benchmark and regression testing
   - `performance_tests.py` - Load testing, throughput, memory usage, etc.

4. **Cross-Platform Tests** - Ensure compatibility across platforms
   - Automated through CI for Windows, macOS, and Linux

### Testing Framework Structure

```
tests/
├── test_all_models.py      # Model functionality tests
├── performance_tests.py    # Performance benchmarking
├── integration_tests.py    # System integration tests
├── utils_tests.py         # Utility module tests
└── __init__.py            # Test package initialization
```

## Quick Start

### Running All Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run specific test categories
python -m pytest tests/test_all_models.py          # Model tests
python -m pytest tests/performance_tests.py        # Performance tests
python -m pytest tests/integration_tests.py        # Integration tests
python -m pytest tests/utils_tests.py             # Utility tests
```

### Individual Test Scripts

```bash
# Run model tests directly
python tests/test_all_models.py

# Run performance tests directly
python tests/performance_tests.py

# Run integration tests directly
python tests/integration_tests.py

# Run utility tests directly
python tests/utils_tests.py
```

## Test Categories

### Model Tests (`test_all_models.py`)

Tests all models in the system:
- **RF-DETR**: Object detection functionality
- **SAM2**: Image segmentation capabilities
- **ResNet**: Image classification
- **SigLIP**: Vision-language classification
- **MediaPipe**: Pose estimation

#### Key Test Classes:
- `TestModelInstances` - Individual model instance testing
- `TestModelManager` - Model management functionality
- `TestModelTypes` - Model type-specific testing
- `TestModelConfiguration` - Configuration handling

### Performance Tests (`performance_tests.py`)

Comprehensive performance benchmarking:
- Model loading/unloading speed
- Inference throughput (FPS measurements)
- Memory usage tracking
- Batch processing efficiency
- Device performance comparison (CPU vs GPU)
- Scalability testing
- Regression detection

#### Key Test Classes:
- `TestModelLoadingPerformance` - Loading time benchmarks
- `TestInferencePerformance` - Inference speed tests
- `TestBatchProcessingPerformance` - Batch efficiency
- `TestDevicePerformance` - Hardware comparisons
- `TestScalabilityPerformance` - System scaling
- `TestPerformanceRegression` - Regression detection

### Integration Tests (`integration_tests.py`)

End-to-end system testing:
- Complete pipeline workflows
- Data flow between components
- Configuration integration
- Error handling and recovery
- Multi-model coordination

#### Key Test Classes:
- `TestPipelineIntegration` - Complete workflow testing
- `TestDataFlowIntegration` - Data flow validation
- `TestConfigurationIntegration` - Config system testing
- `TestErrorHandlingIntegration` - Error recovery
- `TestPerformanceIntegration` - Integrated performance

### Utility Tests (`utils_tests.py`)

Testing for all utility modules:
- Configuration loader and validation
- Image processing utilities
- Video processing capabilities
- Drawing and visualization
- Performance monitoring
- Logging utilities
- Model-specific utilities

#### Key Test Classes:
- `TestConfigLoader` - Configuration management
- `TestPerformanceMonitor` - Performance tracking
- `TestImageUtils` - Image processing
- `TestVideoUtils` - Video handling
- `TestDrawUtils` - Visualization
- `TestLogger` - Logging functionality
- `TestModelSpecificUtils` - Model-specific utilities

## Continuous Integration

The framework includes comprehensive CI/CD setup in `CI/github-actions.yml`:

### CI Workflow

1. **Dependency Checks** - Verify all requirements can be installed
2. **Unit Tests** - Run model and utility tests
3. **Integration Tests** - Test component interactions
4. **Performance Tests** - Benchmark on both CPU and GPU
5. **Cross-Platform Tests** - Windows, macOS, Linux compatibility
6. **Code Quality** - Linting, formatting, security checks
7. **Documentation Tests** - Verify documentation builds

### CI Features

- **Parallel Execution**: Tests run in parallel for faster feedback
- **Cross-Platform**: Automated testing on Windows, macOS, Linux
- **Multiple Python Versions**: Tests with Python 3.9, 3.10, 3.11
- **GPU Testing**: Optional GPU-accelerated performance tests
- **Coverage Reporting**: Automated code coverage collection
- **Security Scanning**: Bandit and Safety dependency checks
- **Performance Regression**: Automated performance monitoring

### CI Artifacts

The CI pipeline generates several artifacts:
- Test results and coverage reports
- Performance benchmark results
- Security scan reports
- Documentation builds
- Log files and debugging information

## Configuration

### Test Configuration

Tests can be configured through:

1. **Environment Variables**:
   ```bash
   export TEST_DEVICE=cpu
   export TEST_BATCH_SIZE=8
   export TEST_TIMEOUT=300
   ```

2. **pytest Configuration** (`pytest.ini`):
   ```ini
   [tool:pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   addopts = -v --tb=short --strict-markers
   ```

3. **Test-Specific Configuration**:
   - Model-specific test configs in test files
   - Performance test parameters
   - Integration test scenarios

### Environment Setup

1. **Development Dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Test Data**:
   - Tests create temporary test data automatically
   - No manual setup required
   - Automatic cleanup after tests

3. **Optional Dependencies**:
   ```bash
   # For GPU tests
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For video processing
   sudo apt-get install ffmpeg  # Ubuntu/Debian
   ```

## Test Data Management

### Temporary Test Data

- Tests create temporary files in system temp directories
- Automatic cleanup after test execution
- No permanent test data storage required

### Test Fixtures

- **Mock Models**: Generated programmatically for testing
- **Test Images**: Synthetic test images with known properties
- **Test Videos**: Generated video sequences for testing
- **Configuration Files**: Sample YAML configs for testing

## Performance Benchmarks

### Baseline Metrics

The framework establishes baseline performance metrics:
- Model loading times
- Inference speeds (FPS)
- Memory usage patterns
- Batch processing efficiency

### Regression Detection

Automated performance regression detection:
- Compare against stored baselines
- Alert on significant performance degradation
- Historical performance tracking

### Performance Targets

Expected performance benchmarks:
- Model loading: < 5 seconds (CPU), < 2 seconds (GPU)
- Single inference: < 100ms (CPU), < 50ms (GPU)
- Memory usage: < 2GB for typical workloads
- Batch processing: Linear scalability up to batch size 32

## Error Handling

### Test Error Categories

1. **Expected Errors**: Known limitations or dependencies
2. **Environment Errors**: Missing dependencies or hardware
3. **Assertion Failures**: Actual test failures
4. **System Errors**: Unexpected system-level issues

### Error Reporting

- Detailed error messages with context
- Stack traces for debugging
- Performance metrics on failures
- Artifact collection for post-mortem analysis

## Debugging Tests

### Debug Mode

```bash
# Run with detailed output
python -m pytest -v -s --tb=long

# Run specific test with debugging
python -m pytest tests/test_all_models.py::TestModelInstances::test_model_loading -v -s

# Run with pdb on failure
python -m pytest --pdb
```

### Logging

Tests include detailed logging:
- Performance measurements
- Error contexts
- System information
- Test progress tracking

### Artifacts

Test artifacts are preserved for debugging:
- Temporary test files
- Performance measurement data
- Error screenshots or outputs
- System configuration information

## Best Practices

### Writing Tests

1. **Test Naming**: Use descriptive test method names
2. **Isolation**: Each test should be independent
3. **Assertions**: Use specific, meaningful assertions
4. **Setup/Teardown**: Proper resource management
5. **Documentation**: Document complex test scenarios

### Performance Testing

1. **Warmup Runs**: Include warmup for accurate measurements
2. **Multiple Runs**: Average results over multiple runs
3. **Resource Monitoring**: Track CPU, memory, GPU usage
4. **Baseline Comparison**: Compare against known baselines

### Integration Testing

1. **End-to-End Scenarios**: Test complete workflows
2. **Error Paths**: Test error handling and recovery
3. **Configuration Testing**: Test different config scenarios
4. **Resource Constraints**: Test under various resource conditions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH is set correctly
2. **Missing Dependencies**: Install all required packages
3. **GPU Issues**: Check CUDA installation for GPU tests
4. **Permission Errors**: Ensure write access to temp directories
5. **Memory Issues**: Reduce batch sizes for memory-constrained systems

### Getting Help

- Check CI logs for detailed error information
- Review test artifacts for debugging data
- Enable debug logging for additional context
- Use interactive debugging for complex issues

## Contributing

### Adding New Tests

1. Follow existing test patterns
2. Include both positive and negative test cases
3. Add appropriate documentation
4. Consider performance implications
5. Update this README if adding new test categories

### Test Maintenance

- Regularly update baseline performance metrics
- Review and update test scenarios
- Monitor test execution times
- Keep dependencies up to date

## License

This testing framework is part of the Soccer Player Recognition System project.