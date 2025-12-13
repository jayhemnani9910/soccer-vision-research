"""
RF-DETR Detection Models Package

This package contains implementations of RF-DETR models for soccer player detection.
"""

from .rf_detr_model import (
    RFDETRModel,
    create_rf_detr_model,
    load_rf_detr_model,
    RFDETRBackbone,
    RFDETRTransformer
)

from .rf_detr_config import (
    RFDETRConfig,
    RFDETRSoccerConfigs,
    DEFAULT_CONFIG
)

__all__ = [
    'RFDETRModel',
    'create_rf_detr_model', 
    'load_rf_detr_model',
    'RFDETRBackbone',
    'RFDETRTransformer',
    'RFDETRConfig',
    'RFDETRSoccerConfigs',
    'DEFAULT_CONFIG'
]