"""
Configuration Validator and Compatibility Checker
===============================================

This module provides comprehensive validation for all configuration files
in the soccer player recognition system. It ensures configuration compatibility,
validates schemas, checks dependencies, and provides detailed error reporting.

Features:
- Schema validation for all configuration types
- Cross-configuration compatibility checking
- Dependency validation (model files, datasets, etc.)
- Performance impact assessment
- Configuration migration support
- Detailed validation reports

Author: Advanced AI Assistant
Date: 2025-11-04
"""

import os
import json
import yaml
import logging
import re
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import importlib.util
import warnings


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class ValidationCategory(Enum):
    """Validation categories."""
    SCHEMA = "schema"
    COMPATIBILITY = "compatibility"
    DEPENDENCY = "dependency"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    level: ValidationLevel
    category: ValidationCategory
    message: str
    field_path: str
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "level": self.level.value,
            "category": self.category.value,
            "message": self.message,
            "field_path": self.field_path,
            "suggestion": self.suggestion,
            "details": self.details
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    config_file: str
    validation_time: str
    total_issues: int
    errors: int
    warnings: int
    infos: int
    issues: List[ValidationIssue]
    score: float  # 0-100 validation score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_file": self.config_file,
            "validation_time": self.validation_time,
            "total_issues": self.total_issues,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "issues": [issue.to_dict() for issue in self.issues],
            "score": self.score
        }
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
        self.total_issues += 1
        
        if issue.level == ValidationLevel.ERROR:
            self.errors += 1
        elif issue.level == ValidationLevel.WARNING:
            self.warnings += 1
        else:
            self.infos += 1


class ConfigValidator:
    """
    Comprehensive configuration validator for the soccer player recognition system.
    
    Validates:
    - Individual configuration files (model, system, etc.)
    - Cross-configuration compatibility
    - Dependencies and file existence
    - Performance implications
    - Security considerations
    """
    
    def __init__(self, strict_mode: bool = False):
        """Initialize the configuration validator."""
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)
        
        # Validation schemas
        self._init_schemas()
        
        # Known compatible combinations
        self._init_compatibility_matrix()
        
        # File checks
        self._init_file_checks()
    
    def _init_schemas(self):
        """Initialize validation schemas for different configuration types."""
        self.schemas = {
            "model_config": {
                "required_fields": ["models"],
                "nested_schemas": {
                    "models.*": {
                        "required_fields": ["model_name"],
                        "field_types": {
                            "input_size": (list, tuple),
                            "batch_size": int,
                            "confidence_threshold": (int, float),
                            "nms_threshold": (int, float),
                            "learning_rate": (int, float)
                        },
                        "constraints": {
                            "confidence_threshold": lambda x: 0 <= x <= 1,
                            "nms_threshold": lambda x: 0 <= x <= 1,
                            "batch_size": lambda x: x > 0,
                            "learning_rate": lambda x: x > 0
                        }
                    }
                }
            },
            "system_config": {
                "required_fields": ["device", "performance", "logging", "data"],
                "nested_schemas": {
                    "device.device": lambda x: x in ["cuda", "cpu", "mps", "auto"],
                    "performance.num_workers": lambda x: x >= 0,
                    "performance.cache_size_mb": lambda x: x > 0,
                    "logging.level": lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                }
            },
            "model_configs": {
                "detection_config": {
                    "required_fields": ["model_name", "input_size", "class_names"],
                    "field_types": {
                        "input_size": (list, tuple),
                        "confidence_threshold": (int, float),
                        "nms_threshold": (int, float),
                        "batch_size": int
                    },
                    "constraints": {
                        "confidence_threshold": lambda x: 0 <= x <= 1,
                        "nms_threshold": lambda x: 0 <= x <= 1,
                        "batch_size": lambda x: x > 0
                    }
                },
                "segmentation_config": {
                    "required_fields": ["model_name", "input_size"],
                    "field_types": {
                        "input_size": (list, tuple),
                        "confidence_threshold": (int, float),
                        "iou_threshold": (int, float)
                    },
                    "constraints": {
                        "confidence_threshold": lambda x: 0 <= x <= 1,
                        "iou_threshold": lambda x: 0 <= x <= 1
                    }
                },
                "classification_config": {
                    "required_fields": ["model_name", "num_classes"],
                    "field_types": {
                        "input_size": (list, tuple),
                        "num_classes": int,
                        "dropout": (int, float)
                    },
                    "constraints": {
                        "num_classes": lambda x: x > 0,
                        "dropout": lambda x: 0 <= x <= 1
                    }
                }
            }
        }
    
    def _init_compatibility_matrix(self):
        """Initialize compatibility matrix for model combinations."""
        self.compatibility_matrix = {
            # Device compatibility
            "device": {
                "cuda": ["detection", "segmentation", "classification"],
                "cpu": ["detection", "segmentation", "classification"],
                "mps": ["classification"],  # Apple Silicon
                "auto": ["detection", "segmentation", "classification"]
            },
            
            # Batch size compatibility
            "batch_size": {
                "rf_detr": {"min": 1, "max": 8},
                "sam2": {"min": 1, "max": 2},
                "siglip": {"min": 1, "max": 64},
                "resnet": {"min": 1, "max": 128}
            },
            
            # Input size compatibility
            "input_size": {
                "rf_detr": {"min": [320, 320], "max": [1024, 1024]},
                "sam2": {"min": [512, 512], "max": [2048, 2048]},
                "siglip": {"min": [224, 224], "max": [512, 512]},
                "resnet": {"min": [224, 224], "max": [512, 512]}
            }
        }
    
    def _init_file_checks(self):
        """Initialize file existence and dependency checks."""
        self.file_checks = {
            "model_path": self._check_model_file,
            "config_path": self._check_config_file,
            "dataset_path": self._check_dataset_file,
            "weights_path": self._check_weights_file
        }
    
    def validate_config_file(self, 
                           file_path: Union[str, Path],
                           config_type: str = "auto") -> ValidationReport:
        """
        Validate a configuration file.
        
        Args:
            file_path: Path to configuration file
            config_type: Type of configuration ("model_config", "system_config", "auto")
            
        Returns:
            Validation report
        """
        file_path = Path(file_path)
        validation_start = time.time()
        
        report = ValidationReport(
            config_file=str(file_path),
            validation_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_issues=0,
            errors=0,
            warnings=0,
            infos=0,
            issues=[],
            score=100.0
        )
        
        try:
            # Auto-detect config type if not specified
            if config_type == "auto":
                config_type = self._detect_config_type(file_path)
            
            # Load configuration
            config_data = self._load_config_file(file_path)
            
            # Basic validation
            self._validate_basic_structure(config_data, report, file_path)
            
            # Schema validation
            self._validate_schema(config_data, config_type, report)
            
            # File dependency validation
            self._validate_dependencies(config_data, report)
            
            # Performance validation
            self._validate_performance(config_data, report)
            
            # Security validation
            self._validate_security(config_data, report)
            
        except Exception as e:
            issue = ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SCHEMA,
                message=f"Failed to load or parse configuration file: {e}",
                field_path="file",
                suggestion="Check file format and syntax"
            )
            report.add_issue(issue)
        
        # Calculate validation score
        if report.total_issues > 0:
            error_penalty = report.errors * 10
            warning_penalty = report.warnings * 3
            info_penalty = report.infos * 1
            report.score = max(0, 100 - error_penalty - warning_penalty - info_penalty)
        
        self.logger.info(f"Validation complete for {file_path}: "
                        f"{report.errors} errors, {report.warnings} warnings, "
                        f"score: {report.score:.1f}/100")
        
        return report
    
    def validate_all_configs(self, config_dir: Union[str, Path]) -> List[ValidationReport]:
        """Validate all configuration files in a directory."""
        config_dir = Path(config_dir)
        reports = []
        
        # Find all configuration files
        config_files = []
        for pattern in ["*.yaml", "*.yml", "*.json"]:
            config_files.extend(config_dir.rglob(pattern))
        
        for config_file in config_files:
            try:
                report = self.validate_config_file(config_file)
                reports.append(report)
            except Exception as e:
                self.logger.error(f"Failed to validate {config_file}: {e}")
        
        return reports
    
    def validate_cross_config_compatibility(self, 
                                          configs: Dict[str, Dict[str, Any]]) -> ValidationReport:
        """
        Validate compatibility between different configuration files.
        
        Args:
            configs: Dictionary of configuration name -> configuration data
            
        Returns:
            Validation report for cross-config compatibility
        """
        report = ValidationReport(
            config_file="cross_config_compatibility",
            validation_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_issues=0,
            errors=0,
            warnings=0,
            infos=0,
            issues=[],
            score=100.0
        )
        
        # Check device compatibility
        self._check_device_compatibility(configs, report)
        
        # Check batch size compatibility
        self._check_batch_size_compatibility(configs, report)
        
        # Check memory compatibility
        self._check_memory_compatibility(configs, report)
        
        # Check input/output compatibility
        self._check_io_compatibility(configs, report)
        
        return report
    
    def _detect_config_type(self, file_path: Path) -> str:
        """Auto-detect configuration type from file content."""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    content = yaml.safe_load(f)
                else:
                    content = json.load(f)
            
            # Heuristics for configuration type detection
            if "models" in content:
                return "model_config"
            elif "device" in content and "performance" in content:
                return "system_config"
            elif any(model in str(content).lower() for model in ["rf_detr", "sam2", "siglip", "resnet"]):
                return "model_configs"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration file based on format."""
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _validate_basic_structure(self, config: Dict[str, Any], 
                                report: ValidationReport, file_path: Path):
        """Validate basic configuration structure."""
        if not isinstance(config, dict):
            issue = ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SCHEMA,
                message="Configuration must be a dictionary/object",
                field_path="root",
                suggestion="Ensure the root element is an object/dictionary"
            )
            report.add_issue(issue)
            return
        
        if not config:
            issue = ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SCHEMA,
                message="Configuration file is empty",
                field_path="root",
                suggestion="Add configuration settings or use a template"
            )
            report.add_issue(issue)
    
    def _validate_schema(self, config: Dict[str, Any], 
                        config_type: str, report: ValidationReport):
        """Validate configuration against schema."""
        schema = self.schemas.get(config_type)
        
        if not schema:
            issue = ValidationIssue(
                level=ValidationLevel.INFO,
                category=ValidationCategory.SCHEMA,
                message=f"No schema defined for configuration type: {config_type}",
                field_path="schema",
                suggestion="Define a schema or use a known configuration type"
            )
            report.add_issue(issue)
            return
        
        # Check required fields
        if "required_fields" in schema:
            missing_fields = []
            for field in schema["required_fields"]:
                if not self._has_field(config, field):
                    missing_fields.append(field)
            
            if missing_fields:
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SCHEMA,
                    message=f"Missing required fields: {missing_fields}",
                    field_path="required_fields",
                    suggestion="Add the missing required fields to the configuration"
                )
                report.add_issue(issue)
        
        # Check field types and constraints
        if "nested_schemas" in schema:
            for field_pattern, validator in schema["nested_schemas"].items():
                if isinstance(validator, dict):
                    self._validate_field_schema(config, field_pattern, validator, report)
                elif callable(validator):
                    self._validate_field_constraint(config, field_pattern, validator, report)
    
    def _validate_field_schema(self, config: Dict[str, Any], 
                             field_pattern: str, schema: Dict[str, Any], 
                             report: ValidationReport):
        """Validate a field against its schema."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated pattern matching
        fields = self._get_fields_by_pattern(config, field_pattern)
        
        for field_path, field_value in fields.items():
            # Check required fields for nested objects
            if "required_fields" in schema and isinstance(field_value, dict):
                missing = [f for f in schema["required_fields"] if f not in field_value]
                if missing:
                    issue = ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category=ValidationCategory.SCHEMA,
                        message=f"Missing required fields in {field_path}: {missing}",
                        field_path=field_path,
                        suggestion="Add missing required fields"
                    )
                    report.add_issue(issue)
            
            # Check field types
            if "field_types" in schema and isinstance(field_value, dict):
                for field_name, expected_type in schema["field_types"].items():
                    if field_name in field_value:
                        actual_value = field_value[field_name]
                        if not isinstance(actual_value, expected_type):
                            issue = ValidationIssue(
                                level=ValidationLevel.ERROR,
                                category=ValidationCategory.SCHEMA,
                                message=f"Field {field_path}.{field_name} has wrong type: "
                                       f"expected {expected_type.__name__}, got {type(actual_value).__name__}",
                                field_path=f"{field_path}.{field_name}",
                                suggestion=f"Convert field to {expected_type.__name__}"
                            )
                            report.add_issue(issue)
            
            # Check constraints
            if "constraints" in schema and isinstance(field_value, dict):
                for field_name, constraint in schema["constraints"].items():
                    if field_name in field_value:
                        try:
                            if not constraint(field_value[field_name]):
                                issue = ValidationIssue(
                                    level=ValidationLevel.WARNING,
                                    category=ValidationCategory.SCHEMA,
                                    message=f"Field {field_path}.{field_name} failed validation constraint",
                                    field_path=f"{field_path}.{field_name}",
                                    suggestion="Check the constraint definition and adjust the value"
                                )
                                report.add_issue(issue)
                        except Exception as e:
                            issue = ValidationIssue(
                                level=ValidationLevel.ERROR,
                                category=ValidationCategory.SCHEMA,
                                message=f"Error validating constraint for {field_path}.{field_name}: {e}",
                                field_path=f"{field_path}.{field_name}",
                                suggestion="Fix the constraint definition"
                            )
                            report.add_issue(issue)
    
    def _validate_field_constraint(self, config: Dict[str, Any], 
                                 field_pattern: str, constraint: callable, 
                                 report: ValidationReport):
        """Validate a field against a constraint function."""
        fields = self._get_fields_by_pattern(config, field_pattern)
        
        for field_path, field_value in fields.items():
            try:
                if not constraint(field_value):
                    issue = ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.SCHEMA,
                        message=f"Field {field_path} failed constraint validation",
                        field_path=field_path,
                        suggestion="Check the constraint and adjust the value"
                    )
                    report.add_issue(issue)
            except Exception as e:
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SCHEMA,
                    message=f"Error validating constraint for {field_path}: {e}",
                    field_path=field_path,
                    suggestion="Fix the constraint definition"
                )
                report.add_issue(issue)
    
    def _get_fields_by_pattern(self, config: Dict[str, Any], 
                             pattern: str) -> Dict[str, Any]:
        """Get fields matching a pattern (simplified implementation)."""
        # This is a simplified pattern matcher
        # In practice, you'd want more sophisticated pattern matching
        if pattern == "*":
            return {"root": config}
        elif pattern.startswith("models."):
            model_name = pattern.split(".", 1)[1]
            if model_name == "*" and "models" in config:
                return {f"models.{k}": v for k, v in config["models"].items()}
            elif "models" in config and model_name in config["models"]:
                return {pattern: config["models"][model_name]}
        
        return {}
    
    def _has_field(self, config: Dict[str, Any], field_path: str) -> bool:
        """Check if a field exists in the configuration."""
        keys = field_path.split(".")
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        
        return True
    
    def _validate_dependencies(self, config: Dict[str, Any], 
                             report: ValidationReport):
        """Validate file dependencies and external resources."""
        for field_name, checker in self.file_checks.items():
            self._check_field_dependencies(config, field_name, checker, report)
    
    def _check_field_dependencies(self, config: Dict[str, Any], 
                                field_name: str, checker: callable, 
                                report: ValidationReport):
        """Check dependencies for a specific field."""
        # Recursively find all instances of the field
        self._recursive_check(config, field_name, field_name, checker, report)
    
    def _recursive_check(self, obj: Any, field_path: str, 
                        original_field: str, checker: callable, 
                        report: ValidationReport):
        """Recursively check for field dependencies."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{field_path}.{key}" if field_path else key
                
                if key == original_field:
                    try:
                        result = checker(value)
                        if not result[0]:  # checker returns (exists, message)
                            issue = ValidationIssue(
                                level=ValidationLevel.WARNING,
                                category=ValidationCategory.DEPENDENCY,
                                message=result[1],
                                field_path=current_path,
                                suggestion="Verify the file path or download the required resource"
                            )
                            report.add_issue(issue)
                    except Exception as e:
                        issue = ValidationIssue(
                            level=ValidationLevel.ERROR,
                            category=ValidationCategory.DEPENDENCY,
                            message=f"Error checking dependency for {current_path}: {e}",
                            field_path=current_path,
                            suggestion="Fix the file path or dependency check"
                        )
                        report.add_issue(issue)
                
                self._recursive_check(value, current_path, original_field, checker, report)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{field_path}[{i}]"
                self._recursive_check(item, current_path, original_field, checker, report)
    
    def _check_model_file(self, path: str) -> Tuple[bool, str]:
        """Check if model file exists."""
        if not path:
            return False, "Model path is empty"
        
        model_path = Path(path)
        if not model_path.exists():
            return False, f"Model file not found: {path}"
        
        if model_path.stat().st_size == 0:
            return False, f"Model file is empty: {path}"
        
        return True, f"Model file exists: {path}"
    
    def _check_config_file(self, path: str) -> Tuple[bool, str]:
        """Check if config file exists."""
        if not path:
            return False, "Config path is empty"
        
        config_path = Path(path)
        if not config_path.exists():
            return False, f"Config file not found: {path}"
        
        return True, f"Config file exists: {path}"
    
    def _check_dataset_file(self, path: str) -> Tuple[bool, str]:
        """Check if dataset directory exists."""
        if not path:
            return False, "Dataset path is empty"
        
        dataset_path = Path(path)
        if not dataset_path.exists():
            return False, f"Dataset path not found: {path}"
        
        if not dataset_path.is_dir():
            return False, f"Dataset path is not a directory: {path}"
        
        # Check if directory has expected contents
        try:
            contents = list(dataset_path.iterdir())
            if not contents:
                return False, f"Dataset directory is empty: {path}"
        except PermissionError:
            return False, f"Permission denied accessing dataset: {path}"
        
        return True, f"Dataset directory exists: {path}"
    
    def _check_weights_file(self, path: str) -> Tuple[bool, str]:
        """Check if weights file exists."""
        if not path:
            return False, "Weights path is empty"
        
        weights_path = Path(path)
        if not weights_path.exists():
            return False, f"Weights file not found: {path}"
        
        if weights_path.stat().st_size < 1024:  # Less than 1KB is suspicious
            return False, f"Weights file seems too small: {path}"
        
        return True, f"Weights file exists: {path}"
    
    def _validate_performance(self, config: Dict[str, Any], 
                            report: ValidationReport):
        """Validate configuration for performance implications."""
        # Check memory usage
        self._check_memory_usage(config, report)
        
        # Check batch size vs memory
        self._check_batch_size_memory(config, report)
        
        # Check worker settings
        self._check_worker_settings(config, report)
        
        # Check mixed precision compatibility
        self._check_mixed_precision(config, report)
    
    def _check_memory_usage(self, config: Dict[str, Any], report: ValidationReport):
        """Check memory usage configuration."""
        # This is a simplified check
        # In practice, you'd want more sophisticated memory estimation
        
        max_memory_fraction = config.get("max_memory_fraction", 0.8)
        if max_memory_fraction > 0.9:
            issue = ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message="Very high memory usage may cause system instability",
                field_path="max_memory_fraction",
                suggestion="Consider reducing max_memory_fraction to 0.8 or lower"
            )
            report.add_issue(issue)
    
    def _check_batch_size_memory(self, config: Dict[str, Any], report: ValidationReport):
        """Check if batch size is reasonable for available memory."""
        batch_size = config.get("batch_size", 1)
        input_size = config.get("input_size", [640, 640])
        
        # Estimate memory usage per batch
        estimated_memory_per_batch = (input_size[0] * input_size[1] * 3 * 4) / (1024 * 1024)  # RGB float32
        
        if batch_size * estimated_memory_per_batch > 100:  # 100MB threshold
            issue = ValidationIssue(
                level=ValidationLevel.INFO,
                category=ValidationCategory.PERFORMANCE,
                message=f"Large batch size ({batch_size}) may require significant memory",
                field_path="batch_size",
                suggestion="Consider reducing batch size if experiencing memory issues"
            )
            report.add_issue(issue)
    
    def _check_worker_settings(self, config: Dict[str, Any], report: ValidationReport):
        """Check data loader worker settings."""
        num_workers = config.get("num_workers", 4)
        
        if num_workers == 0:
            issue = ValidationIssue(
                level=ValidationLevel.INFO,
                category=ValidationCategory.PERFORMANCE,
                message="Using 0 workers may slow down data loading",
                field_path="num_workers",
                suggestion="Consider using 2-8 workers for better data loading performance"
            )
            report.add_issue(issue)
    
    def _check_mixed_precision(self, config: Dict[str, Any], report: ValidationReport):
        """Check mixed precision configuration."""
        mixed_precision = config.get("mixed_precision", False)
        device = config.get("device", "cpu")
        
        if mixed_precision and device == "cpu":
            issue = ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message="Mixed precision enabled on CPU (no GPU detected)",
                field_path="mixed_precision",
                suggestion="Disable mixed precision for CPU inference"
            )
            report.add_issue(issue)
    
    def _validate_security(self, config: Dict[str, Any], report: ValidationReport):
        """Validate configuration for security issues."""
        # Check for suspicious file paths
        self._check_file_paths(config, report)
        
        # Check for sensitive information
        self._check_sensitive_info(config, report)
        
        # Check authentication settings
        self._check_auth_settings(config, report)
    
    def _check_file_paths(self, config: Dict[str, Any], report: ValidationReport):
        """Check file paths for security issues."""
        file_fields = ["model_path", "config_path", "weights_path"]
        
        for field in file_fields:
            if field in config:
                path = config[field]
                if isinstance(path, str):
                    # Check for directory traversal
                    if ".." in path:
                        issue = ValidationIssue(
                            level=ValidationLevel.WARNING,
                            category=ValidationCategory.SECURITY,
                            message=f"Path contains directory traversal: {path}",
                            field_path=field,
                            suggestion="Use absolute paths or sanitize relative paths"
                        )
                        report.add_issue(issue)
    
    def _check_sensitive_info(self, config: Dict[str, Any], report: ValidationReport):
        """Check for sensitive information in configuration."""
        config_str = json.dumps(config).lower()
        
        sensitive_patterns = ["password", "api_key", "secret", "token"]
        
        for pattern in sensitive_patterns:
            if pattern in config_str:
                issue = ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.SECURITY,
                    message=f"Potential sensitive information detected: {pattern}",
                    field_path="security_check",
                    suggestion="Move sensitive information to environment variables"
                )
                report.add_issue(issue)
                break
    
    def _check_auth_settings(self, config: Dict[str, Any], report: ValidationReport):
        """Check authentication settings."""
        security = config.get("security", {})
        
        if security.get("enable_auth", False) and not security.get("require_https", False):
            issue = ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SECURITY,
                message="Authentication enabled but HTTPS not required",
                field_path="security.require_https",
                suggestion="Enable HTTPS for secure authentication"
            )
            report.add_issue(issue)
    
    def _check_device_compatibility(self, configs: Dict[str, Dict[str, Any]], 
                                  report: ValidationReport):
        """Check device compatibility between configurations."""
        devices = {}
        
        for config_name, config in configs.items():
            device = config.get("device", {}).get("device", "unknown")
            devices[config_name] = device
        
        # Check for mixed CPU/GPU usage
        has_cuda = any(d == "cuda" for d in devices.values())
        has_cpu = any(d == "cpu" for d in devices.values())
        
        if has_cuda and has_cpu:
            issue = ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.COMPATIBILITY,
                message="Mixed CPU/GPU device usage detected",
                field_path="device",
                suggestion="Consider using consistent device types across all configurations"
            )
            report.add_issue(issue)
    
    def _check_batch_size_compatibility(self, configs: Dict[str, Dict[str, Any]], 
                                      report: ValidationReport):
        """Check batch size compatibility."""
        for config_name, config in configs.items():
            batch_size = config.get("batch_size", 1)
            model_name = config.get("model_name", "").lower()
            
            if model_name in self.compatibility_matrix.get("batch_size", {}):
                limits = self.compatibility_matrix["batch_size"][model_name]
                if batch_size < limits["min"] or batch_size > limits["max"]:
                    issue = ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.COMPATIBILITY,
                        message=f"Batch size {batch_size} outside recommended range "
                               f"({limits['min']}-{limits['max']}) for {model_name}",
                        field_path="batch_size",
                        suggestion=f"Adjust batch size to be within {limits['min']}-{limits['max']}"
                    )
                    report.add_issue(issue)
    
    def _check_memory_compatibility(self, configs: Dict[str, Dict[str, Any]], 
                                  report: ValidationReport):
        """Check memory compatibility between configurations."""
        max_memory_fractions = {}
        
        for config_name, config in configs.items():
            max_memory_fraction = config.get("max_memory_fraction", 0.8)
            max_memory_fractions[config_name] = max_memory_fraction
        
        total_memory_usage = sum(max_memory_fractions.values())
        
        if total_memory_usage > 1.0:
            issue = ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.COMPATIBILITY,
                message=f"Total memory usage {total_memory_usage:.2f} exceeds 100%",
                field_path="max_memory_fraction",
                suggestion="Reduce memory fractions to sum to 1.0 or less"
            )
            report.add_issue(issue)
    
    def _check_io_compatibility(self, configs: Dict[str, Dict[str, Any]], 
                              report: ValidationReport):
        """Check input/output compatibility."""
        input_sizes = {}
        
        for config_name, config in configs.items():
            input_size = config.get("input_size")
            if input_size:
                input_sizes[config_name] = tuple(input_size) if isinstance(input_size, list) else input_size
        
        # Check for incompatible input sizes if models need to share data
        if len(input_sizes) > 1:
            sizes = list(input_sizes.values())
            if not all(size == sizes[0] for size in sizes):
                issue = ValidationIssue(
                    level=ValidationLevel.INFO,
                    category=ValidationCategory.COMPATIBILITY,
                    message="Models have different input sizes",
                    field_path="input_size",
                    suggestion="Ensure compatible input sizes if models share data"
                )
                report.add_issue(issue)
    
    def generate_config_report(self, config_dir: Union[str, Path], 
                             output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive configuration validation report.
        
        Args:
            config_dir: Directory containing configuration files
            output_file: Optional file to save the report
            
        Returns:
            Formatted validation report
        """
        config_dir = Path(config_dir)
        
        # Validate all configurations
        reports = self.validate_all_configs(config_dir)
        
        # Generate report
        report_lines = []
        report_lines.append("# Configuration Validation Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Config Directory: {config_dir}")
        report_lines.append("")
        
        total_errors = sum(r.errors for r in reports)
        total_warnings = sum(r.warnings for r in reports)
        total_score = sum(r.score * r.total_issues for r in reports) / max(sum(r.total_issues for r in reports), 1)
        
        report_lines.append("## Summary")
        report_lines.append(f"- Total Files: {len(reports)}")
        report_lines.append(f"- Total Errors: {total_errors}")
        report_lines.append(f"- Total Warnings: {total_warnings}")
        report_lines.append(f"- Average Score: {total_score:.1f}/100")
        report_lines.append("")
        
        for report in reports:
            report_lines.append(f"## {report.config_file}")
            report_lines.append(f"Issues: {report.total_issues} (Errors: {report.errors}, Warnings: {report.warnings})")
            report_lines.append(f"Score: {report.score:.1f}/100")
            report_lines.append("")
            
            if report.issues:
                report_lines.append("### Issues:")
                for issue in report.issues:
                    report_lines.append(f"- **{issue.level.value}** [{issue.category.value}]: {issue.message}")
                    if issue.suggestion:
                        report_lines.append(f"  - Suggestion: {issue.suggestion}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Validation report saved to {output_path}")
        
        return report_text
    
    def fix_common_issues(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to automatically fix common configuration issues.
        
        Args:
            config: Configuration dictionary to fix
            
        Returns:
            Fixed configuration dictionary
        """
        fixed_config = config.copy()
        
        # Fix confidence thresholds
        for model_config in fixed_config.get("models", {}).values():
            if "confidence_threshold" in model_config:
                ct = model_config["confidence_threshold"]
                if ct > 1.0:
                    model_config["confidence_threshold"] = ct / 100.0
                elif ct < 0:
                    model_config["confidence_threshold"] = 0.1
        
        # Fix batch sizes
        for model_config in fixed_config.get("models", {}).values():
            if "batch_size" in model_config:
                bs = model_config["batch_size"]
                if bs <= 0:
                    model_config["batch_size"] = 1
        
        # Fix memory fractions
        if "device" in fixed_config:
            device_config = fixed_config["device"]
            if "max_memory_fraction" in device_config:
                mmf = device_config["max_memory_fraction"]
                if mmf > 1.0:
                    device_config["max_memory_fraction"] = 0.8
        
        return fixed_config


# Convenience functions
def validate_config_file(file_path: Union[str, Path]) -> ValidationReport:
    """Validate a single configuration file."""
    validator = ConfigValidator()
    return validator.validate_config_file(file_path)


def validate_config_directory(config_dir: Union[str, Path]) -> List[ValidationReport]:
    """Validate all configuration files in a directory."""
    validator = ConfigValidator()
    return validator.validate_all_configs(config_dir)


def generate_validation_report(config_dir: Union[str, Path], 
                              output_file: Optional[Union[str, Path]] = None) -> str:
    """Generate a comprehensive validation report."""
    validator = ConfigValidator()
    return validator.generate_config_report(config_dir, output_file)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Import required modules
    import time
    
    # Create validator
    validator = ConfigValidator()
    
    # Validate configuration directory
    config_dir = Path("config")
    reports = validator.validate_all_configs(config_dir)
    
    # Print results
    for report in reports:
        print(f"Validated {report.config_file}: {report.errors} errors, {report.warnings} warnings")
        if report.issues:
            for issue in report.issues:
                print(f"  - {issue.level.value}: {issue.message}")
    
    # Generate comprehensive report
    validator.generate_config_report(config_dir, "config/validation_report.md")
    
    print("\nValidation complete. Check config/validation_report.md for detailed report.")