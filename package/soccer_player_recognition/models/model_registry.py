"""
Model Registry for Soccer Player Recognition System

Provides a centralized registry for managing different types of models
including detection, classification, identification, and segmentation models.
"""

import os
import json
from typing import Dict, List, Optional, Type, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    IDENTIFICATION = "identification"
    SEGMENTATION = "segmentation"
    POSE = "pose"


class ModelRegistry:
    """
    Centralized registry for managing model information and configurations.
    
    Features:
    - Register and store model metadata
    - Track model versions and compatibility
    - Support for different model types
    - Configuration validation
    """
    
    def __init__(self, registry_path: str = "models/registry.json"):
        """Initialize the model registry.
        
        Args:
            registry_path: Path to store the registry file
        """
        self.registry_path = registry_path
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_types = set(model_type.value for model_type in ModelType)
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # Load existing registry
        self._load_registry()
    
    def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        model_path: str,
        config_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of the model
            model_path: Path to the model file
            config_path: Path to the model configuration
            metadata: Additional model metadata
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        if model_type.value not in self.model_types:
            logger.error(f"Invalid model type: {model_type.value}")
            return False
        
        # Validate paths exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False
        
        # Create model entry
        model_entry = {
            "model_id": model_id,
            "model_type": model_type.value,
            "model_path": model_path,
            "config_path": config_path,
            "metadata": metadata or {},
            "registered_at": self._get_timestamp(),
            "is_active": True
        }
        
        self.models[model_id] = model_entry
        logger.info(f"Model {model_id} registered successfully")
        
        # Save registry
        self._save_registry()
        return True
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Model information dictionary or None if not found
        """
        return self.models.get(model_id)
    
    def get_models_by_type(self, model_type: ModelType) -> List[Dict[str, Any]]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: Type of models to retrieve
            
        Returns:
            List of model information dictionaries
        """
        return [
            model_info for model_info in self.models.values()
            if model_info["model_type"] == model_type.value
        ]
    
    def activate_model(self, model_id: str) -> bool:
        """
        Activate a registered model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            True if activation successful, False otherwise
        """
        if model_id in self.models:
            self.models[model_id]["is_active"] = True
            self._save_registry()
            logger.info(f"Model {model_id} activated")
            return True
        return False
    
    def deactivate_model(self, model_id: str) -> bool:
        """
        Deactivate a registered model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            True if deactivation successful, False otherwise
        """
        if model_id in self.models:
            self.models[model_id]["is_active"] = False
            self._save_registry()
            logger.info(f"Model {model_id} deactivated")
            return True
        return False
    
    def get_active_models(self) -> List[Dict[str, Any]]:
        """
        Get all active models.
        
        Returns:
            List of active model information dictionaries
        """
        return [
            model_info for model_info in self.models.values()
            if model_info["is_active"]
        ]
    
    def list_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered models.
        
        Returns:
            Dictionary of all registered models
        """
        return self.models.copy()
    
    def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a registered model.
        
        Args:
            model_id: Unique identifier for the model
            metadata: New metadata to merge with existing metadata
            
        Returns:
            True if update successful, False otherwise
        """
        if model_id in self.models:
            self.models[model_id]["metadata"].update(metadata)
            self.models[model_id]["updated_at"] = self._get_timestamp()
            self._save_registry()
            logger.info(f"Metadata updated for model {model_id}")
            return True
        return False
    
    def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model from the registry.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            logger.info(f"Model {model_id} unregistered")
            return True
        return False
    
    def get_model_compatibility(self, model_id: str, requirement: str) -> bool:
        """
        Check if a model meets specific requirements.
        
        Args:
            model_id: Unique identifier for the model
            requirement: Requirement to check (e.g., 'gpu_required', 'min_memory_gb')
            
        Returns:
            True if requirement is met, False otherwise
        """
        if model_id not in self.models:
            return False
        
        metadata = self.models[model_id]["metadata"]
        return requirement in metadata and metadata[requirement] is True
    
    def _load_registry(self) -> None:
        """Load the registry from file."""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    self.models = json.load(f)
                logger.info(f"Registry loaded with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            self.models = {}
    
    def _save_registry(self) -> None:
        """Save the registry to file."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.models, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_registry(self) -> List[str]:
        """
        Validate the registry for any issues.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        for model_id, model_info in self.models.items():
            # Check required fields
            required_fields = ["model_id", "model_type", "model_path", "config_path"]
            for field in required_fields:
                if field not in model_info:
                    errors.append(f"Model {model_id}: Missing required field '{field}'")
            
            # Check if files exist
            if "model_path" in model_info:
                if not os.path.exists(model_info["model_path"]):
                    errors.append(f"Model {model_id}: Model file not found: {model_info['model_path']}")
            
            if "config_path" in model_info:
                if not os.path.exists(model_info["config_path"]):
                    errors.append(f"Model {model_id}: Config file not found: {model_info['config_path']}")
            
            # Check model type validity
            if "model_type" in model_info:
                if model_info["model_type"] not in self.model_types:
                    errors.append(f"Model {model_id}: Invalid model type: {model_info['model_type']}")
        
        return errors


# Global instance
registry = ModelRegistry()