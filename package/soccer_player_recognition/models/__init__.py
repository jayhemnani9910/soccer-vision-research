"""
Model Management Package for Soccer Player Recognition System

This package provides utilities for managing, loading, and monitoring ML models.
"""

from .model_registry import ModelRegistry, ModelType, registry
from .model_manager import ModelManager, ModelInstance, model_manager

__all__ = [
    'ModelRegistry',
    'ModelType', 
    'registry',
    'ModelManager',
    'ModelInstance',
    'model_manager'
]