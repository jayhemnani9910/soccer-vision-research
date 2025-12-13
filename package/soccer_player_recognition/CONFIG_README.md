# Advanced Configuration Management System

## Overview

This advanced configuration management system provides a comprehensive solution for managing configurations across the Soccer Player Recognition project. The system includes centralized configuration management, model-specific schemas, system-wide settings, and robust validation capabilities.

## 🎯 Key Features

### 1. **Centralized Configuration Management**
- Hot-reloading of configuration files
- Environment variable substitution
- Configuration caching with invalidation
- Thread-safe operations
- Change notifications and callbacks

### 2. **Model-Specific Configuration Schemas**
- Dedicated schemas for RF-DETR, SAM2, SigLIP, and ResNet models
- Type validation and constraint checking
- Automatic default value generation
- Cross-model compatibility validation

### 3. **System-Wide Configuration**
- Automatic hardware detection (CPU, GPU, memory)
- Performance optimization based on system capabilities
- Device and memory management
- Runtime behavior configuration

### 4. **Comprehensive Validation**
- Schema validation for all configuration types
- Cross-configuration compatibility checking
- Dependency validation (file existence, format)
- Performance impact assessment
- Security validation

### 5. **Configuration Templates and Migration**
- Template generation for new configurations
- Version migration support
- Auto-fix common configuration issues
- Export capabilities with metadata

## 📁 System Architecture

```
config/
├── config_manager.py          # Core configuration manager
├── model_configs.py           # Model-specific configurations
├── system_config.py           # System-wide settings
├── model_config.yaml          # Existing model configuration
├── settings.py                # Original settings file
└── utils/
    └── config_validator.py    # Validation and compatibility checker
```

## 🚀 Quick Start

### Running the Demo

```bash
cd soccer_player_recognition
python config_demo.py
```

This will create and validate various configuration files, demonstrating all system features.

### Basic Usage

```python
# Load and manage configurations
from config.config_manager import ConfigManager

with ConfigManager("config") as manager:
    # Load configuration
    config = manager.load_config("model_config.yaml")
    
    # Get specific values
    batch_size = manager.get_config("models.rf_detr.batch_size")
    
    # Update configuration
    manager.set_config("models.rf_detr.batch_size", 4, save=True)
    
    # Create template
    manager.create_config_template("new_model.yaml", "model")
```

## 📋 Configuration Files

### 1. Model Configuration (`model_config.yaml`)

Contains configuration for all models in the system:

```yaml
# RF-DETR Detection Model
rf_detr:
  model_name: "rf-detr"
  model_path: "models/pretrained/rf_detr/rf_detr_epoch_20.pth"
  input_size: [640, 640]
  confidence_threshold: 0.5
  nms_threshold: 0.4
  batch_size: 1
  device: "cuda"
  class_names:
    - "player"
    - "ball"
    - "referee"
    - "goalkeeper"

# SAM2 Segmentation Model
sam2:
  model_name: "sam2"
  model_path: "models/pretrained/sam2/sam2_hiera_large.pt"
  input_size: [1024, 1024]
  confidence_threshold: 0.7
  batch_size: 1
  device: "cuda"
  use_multimask: true

# Ensemble Configuration
ensemble:
  enabled: true
  voting_strategy: "soft"
  weights:
    rf_detr: 0.3
    sam2: 0.2
    siglip: 0.3
    resnet: 0.2
```

### 2. System Configuration

Auto-generated system configuration with hardware detection:

```yaml
version: "1.0.0"
system_info:
  cpu_count: 32
  memory_total_gb: 60.54
  cuda_available: false
device:
  device: "cpu"
  max_memory_fraction: 0.8
  mixed_precision: false
performance:
  num_workers: 8
  enable_caching: true
  cache_size_mb: 512
```

### 3. Model-Specific Configurations

Individual configuration files for each model type:

- `detection_config.yaml` - RF-DETR specific settings
- `segmentation_config.yaml` - SAM2 specific settings
- `classification_config.yaml` - ResNet/SigLIP specific settings

## 🔧 Core Components

### ConfigManager

The central configuration manager providing:

- **Hot Reloading**: Automatically detects file changes
- **Caching**: Efficient configuration caching
- **Validation**: Schema and constraint validation
- **Environment Variables**: `${VAR_NAME}` substitution
- **Change Notifications**: Callback system for config changes

```python
from config.config_manager import ConfigManager

manager = ConfigManager("config", enable_hot_reload=True)

# Register change listener
def on_config_change(change):
    print(f"Config changed: {change.key}")

manager.register_change_listener(on_config_change)
```

### ModelConfigManager

Manages model-specific configurations with schemas:

```python
from config.model_configs import ModelConfigManager, ModelType

manager = ModelConfigManager("config")

# Load model configuration
detection_config = manager.load_model_config(ModelType.DETECTION)

# Validate configuration
is_valid = manager.validate_model_config(detection_config)

# Create template
manager.create_model_config_template(ModelType.DETECTION, "new_detection.yaml")
```

### SystemConfigManager

Handles system-wide configuration with auto-detection:

```python
from config.system_config import SystemConfigManager

manager = SystemConfigManager("config")

# Load system configuration
config = manager.load_system_config()

# Optimize for detected hardware
optimized = manager.optimize_config_for_system(config)

# Get recommended batch size
batch_size = manager.get_recommended_batch_size(500)  # MB
```

### ConfigValidator

Provides comprehensive validation:

```python
from utils.config_validator import ConfigValidator

validator = ConfigValidator(strict_mode=True)

# Validate single file
report = validator.validate_config_file("config/model_config.yaml")

# Validate all files in directory
reports = validator.validate_all_configs("config")

# Check cross-config compatibility
compatibility = validator.validate_cross_config_compatibility(configs)

# Generate validation report
report_text = validator.generate_config_report("config", "validation_report.md")
```

## 📊 Validation Features

### Schema Validation

- Required field checking
- Type validation
- Range constraint validation
- Custom constraint validation

### Compatibility Checking

- Device compatibility (CPU/GPU)
- Memory usage compatibility
- Batch size compatibility
- Input/output format compatibility

### Dependency Validation

- Model file existence
- Configuration file validity
- Dataset directory validation
- Weights file verification

### Performance Validation

- Memory usage estimation
- Batch size recommendations
- Worker configuration validation
- Mixed precision compatibility

## 🎨 Configuration Templates

The system provides templates for:

### Model Template
```yaml
model_name: "{{MODEL_NAME}}"
type: "{{MODEL_TYPE}}"
model_path: "models/pretrained/{{MODEL_NAME}}/model.pth"
input_size: [224, 224]
batch_size: 1
confidence_threshold: 0.5
pretrained: true
```

### System Template
```yaml
version: "1.0.0"
device:
  device: "auto"
  max_memory_fraction: 0.8
  mixed_precision: true
performance:
  num_workers: 4
  enable_caching: true
```

### Pipeline Template
```yaml
pipeline_name: "{{PIPELINE_NAME}}"
steps:
  - name: "detection"
    model: "rf_detr"
    enabled: true
  - name: "segmentation"
    model: "sam2"
    enabled: true
  - name: "classification"
    model: "resnet"
    enabled: true
```

## 📈 Generated Files

The demo generates:

```
config/
├── complete_config.yaml          # Main configuration
├── complete_config.json          # JSON version
├── detection_config.yaml         # RF-DETR specific config
├── segmentation_config.yaml      # SAM2 specific config
├── classification_config.yaml    # ResNet specific config
├── system_config.yaml            # System-wide config
├── model_registry.yaml           # Model registry
├── model_template.yaml           # Model template
├── system_template.yaml          # System template
├── pipeline_template.yaml        # Pipeline template
├── templates_usage_guide.md      # Template usage guide
├── validation_report.json        # Detailed validation report
└── validation_report.md          # Human-readable report
```

## 🛠 Best Practices

### 1. Configuration Organization

```python
# Good: Use dot notation for nested configs
batch_size = manager.get_config("models.rf_detr.batch_size")

# Good: Group related settings
detection_config = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "max_detections": 100
}
```

### 2. Validation

```python
# Always validate configurations
validator = ConfigValidator(strict_mode=True)
report = validator.validate_config_file("config/model_config.yaml")

if report.errors > 0:
    print("Configuration has errors!")
    for issue in report.issues:
        if issue.level == ValidationLevel.ERROR:
            print(f"ERROR: {issue.message}")
```

### 3. Error Handling

```python
try:
    config = manager.load_config("model_config.yaml")
except FileNotFoundError:
    print("Configuration file not found")
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
```

### 4. Template Usage

```python
# Use templates for new configurations
manager.create_config_template("new_model.yaml", "model")

# Customize templates with environment variables
# Use ${ENV_VAR} syntax for environment variable substitution
```

## 🔍 Troubleshooting

### Common Issues

1. **Configuration File Not Found**
   - Check file path and permissions
   - Ensure directory exists

2. **Validation Errors**
   - Review validation report
   - Check required fields and constraints
   - Use auto-fix for common issues

3. **Hot Reloading Not Working**
   - Check file monitoring permissions
   - Verify file timestamps are updating

4. **Memory Issues**
   - Adjust `max_memory_fraction` setting
   - Reduce batch sizes
   - Enable memory optimization

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check configuration status
manager = ConfigManager("config")
config = manager.get_all_configs()
print(f"Loaded {len(config)} configuration keys")
```

## 🚦 Integration Examples

### With Existing Code

```python
# Replace direct configuration loading
# OLD:
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# NEW:
from config.config_manager import ConfigManager
with ConfigManager("config") as manager:
    config = manager.load_config("config.yaml")
```

### With Model Training

```python
# Load model configuration
from config.model_configs import get_detection_config
detection_config = get_detection_config("config")

# Use in training
model = RFDETRModel(
    model_path=detection_config.model_path,
    batch_size=detection_config.batch_size,
    device=detection_config.device.value
)
```

### With System Monitoring

```python
# Monitor configuration changes
def on_config_change(change):
    if change.key.startswith("models.rf_detr"):
        print("RF-DETR configuration updated")
        # Trigger model reload or parameter update

manager.register_change_listener(on_config_change)
```

## 📚 API Reference

### ConfigManager Methods

- `load_config(file_path, scope)` - Load configuration file
- `save_config(config, file_path, format)` - Save configuration
- `get_config(key, default, file_path)` - Get configuration value
- `set_config(key, value, file_path, save)` - Set configuration value
- `register_change_listener(callback)` - Register change listener
- `export_config(file_path, format, include_metadata)` - Export configuration

### ModelConfigManager Methods

- `load_model_config(model_type)` - Load model configuration
- `save_model_config(config)` - Save model configuration
- `validate_model_config(config)` - Validate configuration
- `create_model_config_template(model_type, file_path)` - Create template

### SystemConfigManager Methods

- `load_system_config()` - Load system configuration
- `optimize_config_for_system(config)` - Optimize for hardware
- `get_recommended_batch_size(model_size_mb)` - Get batch size recommendation
- `benchmark_system_performance()` - Run performance benchmark

### ConfigValidator Methods

- `validate_config_file(file_path)` - Validate single file
- `validate_all_configs(config_dir)` - Validate all files
- `validate_cross_config_compatibility(configs)` - Check compatibility
- `generate_config_report(config_dir, output_file)` - Generate report
- `fix_common_issues(config)` - Auto-fix issues

## 🎓 Advanced Features

### Custom Validators

```python
def custom_constraint(value):
    # Your custom validation logic
    return value > 0

# Register with validator
validator.register_constraint("custom_field", custom_constraint)
```

### Environment Variable Substitution

```yaml
model_path: "${MODEL_PATH}/model.pth"  # Uses SOCCER_MODEL_PATH env var
api_key: "${API_KEY}"                  # Uses SOCCER_API_KEY env var
```

### Configuration Migration

```python
# Migrate configuration to new version
old_config = {"version": "1.0.0", "model_path": "old/path"}
new_config = manager.migrate_config(old_config, "1.1.0")
```

### Performance Profiling

```python
# Profile configuration loading
import time
start_time = time.time()
config = manager.load_config("config.yaml")
load_time = time.time() - start_time
print(f"Configuration loaded in {load_time:.3f}s")
```

## 🏁 Summary

This configuration management system provides:

- ✅ **Comprehensive validation** with detailed reporting
- ✅ **Flexible configuration** with multiple formats and schemas
- ✅ **Automatic optimization** based on system capabilities
- ✅ **Easy template generation** for new configurations
- ✅ **Hot-reloading** for development workflows
- ✅ **Cross-configuration compatibility** checking
- ✅ **Production-ready** with error handling and logging

The system is designed to be robust, scalable, and easy to use, providing a solid foundation for managing complex AI model configurations in production environments.