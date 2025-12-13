"""
ResNet Model for Player Recognition and Classification

This module implements a ResNet-based deep learning model for player classification
and identification, including pre-trained model loading, fine-tuning capabilities,
and output processing.

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from pathlib import Path
import json
from PIL import Image
import cv2

from soccer_player_recognition.utils.logger import get_logger

logger = get_logger(__name__)


class ResNetPlayerClassifier(nn.Module):
    """
    ResNet-based model for player classification and identification.
    
    This class provides:
    - Pre-trained model loading and initialization
    - Fine-tuning capabilities for custom datasets
    - Output processing for classification results
    - Feature extraction for further analysis
    """
    
    def __init__(self, 
                 num_players: int,
                 num_classes: int = 1000,
                 pretrained: bool = True,
                 freeze_features: bool = False,
                 dropout_rate: float = 0.5,
                 model_name: str = "resnet50"):
        """
        Initialize ResNet Player Classifier.
        
        Args:
            num_players: Number of unique players to classify
            num_classes: Number of output classes (default 1000 for ImageNet)
            pretrained: Whether to use pre-trained weights
            freeze_features: Whether to freeze feature extraction layers
            dropout_rate: Dropout rate for regularization
            model_name: ResNet architecture name ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        """
        super(ResNetPlayerClassifier, self).__init__()
        
        self.num_players = num_players
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_features = freeze_features
        self.dropout_rate = dropout_rate
        self.model_name = model_name
        
        # Load pre-trained ResNet model
        self.backbone = self._load_backbone()
        
        # Modify classifier head for player classification
        self.classifier = self._create_classifier()
        
        # Feature extraction layer
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Apply modifications
        self._apply_modifications()
        
        logger.info(f"Initialized {model_name} with {num_players} player classes")
    
    def _load_backbone(self) -> nn.Module:
        """Load pre-trained ResNet backbone."""
        try:
            if hasattr(models, self.model_name):
                backbone = getattr(models, self.model_name)(pretrained=self.pretrained)
            else:
                logger.warning(f"Unknown model {self.model_name}, defaulting to ResNet50")
                backbone = models.resnet50(pretrained=self.pretrained)
            
            # Get feature dimension
            if self.model_name in ['resnet18', 'resnet34']:
                self.feature_dim = 512
            else:  # resnet50, resnet101, resnet152
                self.feature_dim = 2048
                
            return backbone
        except Exception as e:
            logger.error(f"Error loading backbone: {e}")
            raise
    
    def _create_classifier(self) -> nn.Module:
        """Create custom classifier head."""
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 4, self.num_players)
        )
    
    def _apply_modifications(self):
        """Apply modifications based on configuration."""
        if self.freeze_features:
            self._freeze_features()
        
        # Replace final layer for player classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, self.num_players)
    
    def _freeze_features(self):
        """Freeze feature extraction layers for transfer learning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        logger.info("Feature extraction layers frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input images of shape (batch_size, 3, height, width)
            
        Returns:
            Classification logits of shape (batch_size, num_players)
        """
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input images of shape (batch_size, 3, height, width)
            
        Returns:
            Feature vectors of shape (batch_size, feature_dim)
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
        return features
    
    def get_layer_outputs(self, x: torch.Tensor, target_layer: str = 'layer4') -> Dict[str, torch.Tensor]:
        """
        Get outputs from intermediate layers for analysis.
        
        Args:
            x: Input images
            target_layer: Target layer name ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')
            
        Returns:
            Dictionary containing layer outputs
        """
        outputs = {}
        
        def hook_fn(module, input, output):
            outputs[target_layer] = output
        
        # Register hook on target layer
        if hasattr(self.backbone, target_layer):
            hook = getattr(self.backbone, target_layer).register_forward_hook(hook_fn)
            
            with torch.no_grad():
                _ = self.backbone(x)
            
            hook.remove()
        
        return outputs
    
    def predict(self, x: torch.Tensor, return_probabilities: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on input images.
        
        Args:
            x: Input images of shape (batch_size, 3, height, width)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Tuple of (predictions, probabilities) where:
            - predictions: Class indices of shape (batch_size,)
            - probabilities: Class probabilities of shape (batch_size, num_players)
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        if return_probabilities:
            return predictions, probabilities
        return predictions, None
    
    def save_model(self, path: str, metadata: Optional[Dict] = None):
        """
        Save model and metadata.
        
        Args:
            path: Path to save the model
            metadata: Additional metadata to save
        """
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            torch.save({
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'num_players': self.num_players,
                    'num_classes': self.num_classes,
                    'pretrained': self.pretrained,
                    'freeze_features': self.freeze_features,
                    'dropout_rate': self.dropout_rate,
                    'model_name': self.model_name,
                    'feature_dim': self.feature_dim
                },
                'metadata': metadata or {}
            }, path)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(cls, path: str, map_location: str = 'cpu') -> 'ResNetPlayerClassifier':
        """
        Load model from saved state.
        
        Args:
            path: Path to saved model
            map_location: Device to load model on
            
        Returns:
            Loaded ResNetPlayerClassifier instance
        """
        try:
            checkpoint = torch.load(path, map_location=map_location)
            config = checkpoint['model_config']
            
            model = cls(
                num_players=config['num_players'],
                num_classes=config['num_classes'],
                pretrained=False,  # Load from checkpoint
                freeze_features=config['freeze_features'],
                dropout_rate=config['dropout_rate'],
                model_name=config['model_name']
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Model loaded from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_players': self.num_players,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'pretrained': self.pretrained,
            'freeze_features': self.freeze_features,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class PlayerRecognitionModel:
    """
    High-level wrapper for player recognition tasks using ResNet.
    
    This class provides an easy-to-use interface for:
    - Model training and fine-tuning
    - Player identification and classification
    - Feature extraction and analysis
    - Model persistence and loading
    """
    
    def __init__(self, 
                 num_players: int,
                 model_name: str = "resnet50",
                 device: str = "auto",
                 pretrained: bool = True):
        """
        Initialize Player Recognition Model.
        
        Args:
            num_players: Number of unique players
            model_name: ResNet architecture name
            device: Device for computation ('auto', 'cpu', 'cuda')
            pretrained: Whether to use pre-trained weights
        """
        self.num_players = num_players
        self.model_name = model_name
        self.device = self._get_device(device)
        self.pretrained = pretrained
        
        # Initialize model
        self.model = ResNetPlayerClassifier(
            num_players=num_players,
            pretrained=pretrained,
            model_name=model_name
        ).to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Player information mapping
        self.player_info = {}
        self.class_to_player = {}
        
        logger.info(f"Initialized PlayerRecognitionModel with {num_players} players on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def setup_training(self, 
                      learning_rate: float = 0.001,
                      weight_decay: float = 1e-4,
                      optimizer_type: str = "adam",
                      scheduler_type: str = "step"):
        """
        Setup training configuration.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer ('adam', 'sgd')
            scheduler_type: Type of learning rate scheduler ('step', 'cosine', 'none')
        """
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                      lr=learning_rate, 
                                      weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), 
                                     lr=learning_rate, 
                                     momentum=0.9, 
                                     weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Setup scheduler
        if scheduler_type.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        logger.info(f"Training setup completed with {optimizer_type} optimizer")
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total,
            'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
        }
        
        logger.info(f"Training Epoch - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2f}%")
        return metrics
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
        
        logger.info(f"Validation - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2f}%")
        return metrics
    
    def identify_player(self, image: np.ndarray, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Identify player from image.
        
        Args:
            image: Input image as numpy array (BGR format)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing identification results
        """
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert BGR to RGB and normalize
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get player information
        player_id = self.class_to_player.get(predicted_class, f"Unknown_{predicted_class}")
        player_name = self.player_info.get(player_id, {}).get('name', f"Player_{player_id}")
        
        result = {
            'player_id': player_id,
            'player_name': player_name,
            'class_index': predicted_class
        }
        
        if return_confidence:
            result['confidence'] = confidence
            result['all_probabilities'] = probabilities[0].cpu().numpy().tolist()
        
        return result
    
    def extract_player_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from player image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Feature vector as numpy array
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        
        features = self.model.extract_features(input_tensor)
        return features.cpu().numpy().flatten()
    
    def save_model(self, path: str, player_info: Optional[Dict] = None):
        """
        Save complete model with metadata.
        
        Args:
            path: Path to save the model
            player_info: Player information mapping
        """
        metadata = {
            'model_info': self.model.get_model_info(),
            'player_info': player_info or self.player_info,
            'class_to_player': self.class_to_player
        }
        
        self.model.save_model(path, metadata)
    
    @classmethod
    def load_model(cls, path: str, device: str = "auto") -> 'PlayerRecognitionModel':
        """
        Load complete model with metadata.
        
        Args:
            path: Path to saved model
            device: Device to load model on
            
        Returns:
            Loaded PlayerRecognitionModel instance
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = cls(
            num_players=config['num_players'],
            model_name=config['model_name'],
            device=device,
            pretrained=False
        )
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.player_info = checkpoint['metadata'].get('player_info', {})
        model.class_to_player = checkpoint['metadata'].get('class_to_player', {})
        
        return model


# Example usage and testing functions
def create_sample_model() -> PlayerRecognitionModel:
    """Create a sample model for testing."""
    return PlayerRecognitionModel(num_players=10, model_name="resnet18", pretrained=True)


if __name__ == "__main__":
    # Example usage
    logger.info("Testing ResNet Player Recognition Model")
    
    # Create sample model
    model = create_sample_model()
    
    # Setup training
    model.setup_training(learning_rate=0.001)
    
    # Print model information
    info = model.model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    logger.info("ResNet Player Recognition Model test completed")