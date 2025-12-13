# Configuration Validation Report
Generated: 2025-11-04 12:32:12
Directory: config

## Summary
- **Total Files**: 11
- **Total Issues**: 10
- **Errors**: 10
- **Warnings**: 0
- **Info**: 0

## File Details
### classification_config.yaml
- **Status**: ✅ Valid
- **Issues**: 0

### complete_config.yaml
- **Status**: ✅ Valid
- **Issues**: 0

### detection_config.yaml
- **Status**: ✅ Valid
- **Issues**: 0

### model_config.yaml
- **Status**: ✅ Valid
- **Issues**: 0

### model_registry.yaml
- **Status**: ❌ Issues Found
- **Issues**: 9
- **Issues Found**:
  - resnet: Missing required field 'model_name'
  - resnet: Missing required field 'input_size'
  - resnet: Missing required field 'device'
  - rf_detr: Missing required field 'model_name'
  - rf_detr: Missing required field 'input_size'
  - rf_detr: Missing required field 'device'
  - sam2: Missing required field 'model_name'
  - sam2: Missing required field 'input_size'
  - sam2: Missing required field 'device'

### model_template.yaml
- **Status**: ✅ Valid
- **Issues**: 0

### pipeline_template.yaml
- **Status**: ❌ Issues Found
- **Issues**: 1
- **Issues Found**:
  - ensemble weights must be a dictionary

### segmentation_config.yaml
- **Status**: ✅ Valid
- **Issues**: 0

### system_config.yaml
- **Status**: ✅ Valid
- **Issues**: 0

### system_template.yaml
- **Status**: ✅ Valid
- **Issues**: 0

### complete_config.json
- **Status**: ✅ Valid
- **Issues**: 0

## Recommendations
- Please fix configuration errors before deployment.
