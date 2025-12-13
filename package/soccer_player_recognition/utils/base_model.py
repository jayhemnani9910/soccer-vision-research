"""
Abstract base model for soccer player recognition components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models in the soccer player recognition pipeline."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            config: Model configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load model from path or initialize model."""
        pass
    
    @abstractmethod
    def save_model(self, save_path: str) -> None:
        """Save model to specified path."""
        pass
    
    @abstractmethod
    def preprocess(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess input data."""
        pass
    
    @abstractmethod
    def postprocess(self, raw_output: Any) -> Any:
        """Postprocess model output."""
        pass
    
    @abstractmethod
    def forward(self, input_data: torch.Tensor) -> Any:
        """Forward pass through the model."""
        pass
    
    def predict(self, input_data: Union[np.ndarray, torch.Tensor, str]) -> Any:
        """
        Make prediction on input data.
        
        Args:
            input_data: Input data (image, video, or file path)
            
        Returns:
            Model prediction
        """
        # Preprocess input
        processed_input = self.preprocess(input_data)
        
        # Forward pass
        with torch.no_grad():
            raw_output = self.forward(processed_input)
        
        # Postprocess output
        result = self.postprocess(raw_output)
        
        return result
    
    def to_device(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Move data to the model's device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.from_numpy(data).to(self.device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'config': self.config
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseModel':
        """Create model instance from configuration."""
        return cls(**config)