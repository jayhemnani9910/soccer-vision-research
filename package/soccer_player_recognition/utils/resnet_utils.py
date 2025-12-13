"""
ResNet Utility Functions for Image Preprocessing and Feature Extraction

This module provides utility functions for ResNet-based player recognition,
including image preprocessing, data augmentation, feature extraction,
and model evaluation utilities.

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any, Union
import logging
from pathlib import Path
import json
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

from .logger import get_logger

logger = get_logger(__name__)


class ResNetPreprocessor:
    """
    Image preprocessing utilities for ResNet models.
    
    This class provides methods for:
    - Standardizing input images for ResNet
    - Data augmentation for training
    - Image quality enhancement
    - Batch preprocessing
    """
    
    def __init__(self, 
                 input_size: int = 224,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 augmentation: bool = False):
        """
        Initialize ResNet Preprocessor.
        
        Args:
            input_size: Target image size for ResNet input
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            augmentation: Whether to enable data augmentation
        """
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        
        # Setup transforms
        self.train_transforms = self._get_train_transforms()
        self.val_transforms = self._get_validation_transforms()
        self.inference_transforms = self._get_inference_transforms()
        
        logger.info(f"ResNet Preprocessor initialized with input_size={input_size}")
    
    def _get_train_transforms(self) -> transforms.Compose:
        """Get training transforms with augmentation."""
        if not self.augmentation:
            return self._get_validation_transforms()
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size + 32, self.input_size + 32)),
            transforms.RandomCrop(self.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            transforms.RandomErasing(p=0.1)
        ])
    
    def _get_validation_transforms(self) -> transforms.Compose:
        """Get validation transforms without augmentation."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def _get_inference_transforms(self) -> transforms.Compose:
        """Get inference transforms for prediction."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def preprocess_image(self, 
                        image: np.ndarray, 
                        mode: str = 'inference') -> torch.Tensor:
        """
        Preprocess single image for ResNet.
        
        Args:
            image: Input image as numpy array (BGR format)
            mode: Processing mode ('train', 'val', 'inference')
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply appropriate transforms
            if mode == 'train':
                transform = self.train_transforms
            elif mode == 'val':
                transform = self.val_transforms
            else:  # inference
                transform = self.inference_transforms
            
            # Apply transforms
            tensor = transform(image)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return default tensor
            return torch.zeros((3, self.input_size, self.input_size))
    
    def preprocess_batch(self, 
                        images: List[np.ndarray], 
                        mode: str = 'inference') -> torch.Tensor:
        """
        Preprocess batch of images.
        
        Args:
            images: List of input images
            mode: Processing mode
            
        Returns:
            Batch of preprocessed image tensors
        """
        tensors = []
        for image in images:
            tensor = self.preprocess_image(image, mode)
            tensors.append(tensor)
        
        return torch.stack(tensors)
    
    def denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor for visualization.
        
        Args:
            tensor: Normalized image tensor
            
        Returns:
            Denormalized image as numpy array (0-255, RGB)
        """
        try:
            # Convert tensor to numpy
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # Remove batch dimension
            
            # Denormalize
            mean_tensor = torch.tensor(self.mean).view(3, 1, 1)
            std_tensor = torch.tensor(self.std).view(3, 1, 1)
            
            denorm_tensor = tensor * std_tensor + mean_tensor
            
            # Clamp to valid range
            denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
            
            # Convert to numpy and change to HWC format
            numpy_img = denorm_tensor.permute(1, 2, 0).numpy()
            
            # Convert to 0-255 and uint8
            numpy_img = (numpy_img * 255).astype(np.uint8)
            
            return numpy_img
            
        except Exception as e:
            logger.error(f"Error denormalizing image: {e}")
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better recognition.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Apply enhancements
            # 1. Increase contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(1.2)
            
            # 2. Increase sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # 3. Adjust brightness slightly
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            # Convert back to OpenCV format
            enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            return enhanced_cv
            
        except Exception as e:
            logger.warning(f"Error enhancing image quality: {e}")
            return image
    
    def detect_and_crop_player(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and crop player region from full frame.
        
        Args:
            image: Full frame image
            
        Returns:
            Cropped player image
        """
        try:
            # Apply player detection using simple heuristics
            # In a real implementation, this would use object detection models
            
            height, width = image.shape[:2]
            
            # Assume player is in center-lower portion of image
            # This is a simplified approach
            player_top = int(height * 0.1)   # Skip top 10% (background/sky)
            player_bottom = int(height * 0.9)  # Skip bottom 10% (grass/ground)
            player_left = int(width * 0.2)    # Skip left 20%
            player_right = int(width * 0.8)   # Skip right 20%
            
            cropped = image[player_top:player_bottom, player_left:player_right]
            
            return cropped
            
        except Exception as e:
            logger.warning(f"Error cropping player: {e}")
            return image


class ResNetFeatureExtractor:
    """
    Feature extraction utilities for ResNet models.
    
    This class provides methods for:
    - Extracting features from different layers
    - Computing feature statistics
    - Feature visualization
    - Feature clustering and analysis
    """
    
    def __init__(self, model: nn.Module, layer_name: str = 'layer4'):
        """
        Initialize Feature Extractor.
        
        Args:
            model: ResNet model
            layer_name: Name of layer to extract features from
        """
        self.model = model
        self.layer_name = layer_name
        self.feature_dim = self._get_feature_dimension()
        
        logger.info(f"Feature extractor initialized for layer '{layer_name}'")
    
    def _get_feature_dimension(self) -> int:
        """Get feature dimension for the specified layer."""
        try:
            if self.layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                return 2048  # ResNet feature dimension
            elif self.layer_name == 'avgpool':
                return 2048
            elif 'conv' in self.layer_name:
                return 512
            else:
                return 512
        except:
            return 512
    
    def extract_features(self, 
                        images: torch.Tensor, 
                        return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract features from images.
        
        Args:
            images: Batch of image tensors
            return_intermediate: Whether to return intermediate layer outputs
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Register hooks for intermediate layers
        hooks = self._register_hooks()
        
        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model(images)
            
            # Extract features from hooks
            for name, hook in hooks.items():
                if name in self.hook_outputs:
                    features[name] = self.hook_outputs[name]
            
        finally:
            # Remove hooks
            for hook in hooks.values():
                hook.remove()
        
        # Add final global average pooled features
        if 'layer4' in features:
            features['final'] = torch.mean(features['layer4'], dim=[2, 3])
        
        return features
    
    def _register_hooks(self) -> Dict[str, Any]:
        """Register forward hooks for feature extraction."""
        self.hook_outputs = {}
        hooks = {}
        
        def hook_fn(name):
            def forward_hook(module, input, output):
                self.hook_outputs[name] = output
            return forward_hook
        
        # Register hooks for common layers
        layer_mapping = {
            'conv1': self.model.conv1,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4,
            'avgpool': self.model.avgpool
        }
        
        for name, layer in layer_mapping.items():
            try:
                hooks[name] = layer.register_forward_hook(hook_fn(name))
            except:
                continue
        
        return hooks
    
    def compute_feature_statistics(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics of extracted features.
        
        Args:
            features: Feature tensor
            
        Returns:
            Dictionary containing feature statistics
        """
        stats = {
            'mean': float(torch.mean(features).item()),
            'std': float(torch.std(features).item()),
            'min': float(torch.min(features).item()),
            'max': float(torch.max(features).item()),
            'sparsity': float((features == 0).sum().item() / features.numel()),
            'l2_norm': float(torch.norm(features).item())
        }
        
        # Compute activation distribution
        hist, _ = torch.histogram(features, bins=20)
        stats['activation_spread'] = float(hist.std().item())
        
        return stats
    
    def visualize_features(self, 
                          features: torch.Tensor, 
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize extracted features.
        
        Args:
            features: Feature tensor
            save_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        try:
            if features.dim() == 2:
                # 2D features - show as heatmap
                feature_map = features.cpu().numpy()
                
                plt.figure(figsize=(10, 6))
                sns.heatmap(feature_map, cmap='viridis', cbar=True)
                plt.title('Feature Activation Heatmap')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Sample Index')
                
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    # Convert to numpy array
                    import io
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    buf.seek(0)
                    image = Image.open(buf)
                    return np.array(image)
            
            elif features.dim() == 4:
                # 4D feature maps - show first few channels
                feature_maps = features[0].cpu().numpy()  # First sample, all channels
                
                n_channels = min(16, feature_maps.shape[0])  # Show up to 16 channels
                
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                axes = axes.flatten()
                
                for i in range(n_channels):
                    axes[i].imshow(feature_maps[i], cmap='viridis')
                    axes[i].set_title(f'Channel {i}')
                    axes[i].axis('off')
                
                # Hide unused subplots
                for i in range(n_channels, 16):
                    axes[i].axis('off')
                
                plt.suptitle('Feature Maps Visualization')
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    # Convert to numpy array
                    import io
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    buf.seek(0)
                    image = Image.open(buf)
                    return np.array(image)
            
        except Exception as e:
            logger.error(f"Error visualizing features: {e}")
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def cluster_features(self, 
                        features: torch.Tensor, 
                        n_clusters: int = 5,
                        method: str = 'kmeans') -> Dict[str, Any]:
        """
        Cluster extracted features for analysis.
        
        Args:
            features: Feature tensor
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'dbscan')
            
        Returns:
            Dictionary containing clustering results
        """
        try:
            # Convert to numpy and flatten if needed
            features_np = features.cpu().numpy()
            if features_np.ndim > 2:
                features_np = features_np.reshape(features_np.shape[0], -1)
            
            if method.lower() == 'kmeans':
                # K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(features_np)
                centers = kmeans.cluster_centers_
                
                return {
                    'labels': labels,
                    'centers': centers,
                    'method': 'kmeans',
                    'n_clusters': n_clusters,
                    'inertia': float(kmeans.inertia_)
                }
            
            elif method.lower() == 'dbscan':
                from sklearn.cluster import DBSCAN
                
                # DBSCAN clustering
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                labels = dbscan.fit_predict(features_np)
                
                return {
                    'labels': labels,
                    'method': 'dbscan',
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'n_noise': int((labels == -1).sum())
                }
            
        except Exception as e:
            logger.error(f"Error clustering features: {e}")
            return {'error': str(e)}


class ResNetDatasetProcessor:
    """
    Dataset processing utilities for ResNet training and evaluation.
    
    This class provides methods for:
    - Loading and processing datasets
    - Creating data loaders
    - Dataset statistics analysis
    - Data validation
    """
    
    def __init__(self, preprocessor: ResNetPreprocessor):
        """
        Initialize Dataset Processor.
        
        Args:
            preprocessor: ResNet preprocessor instance
        """
        self.preprocessor = preprocessor
        self.dataset_stats = {}
        
        logger.info("ResNet Dataset Processor initialized")
    
    def analyze_dataset(self, dataset: Any) -> Dict[str, Any]:
        """
        Analyze dataset statistics.
        
        Args:
            dataset: PyTorch dataset or list of images
            
        Returns:
            Dataset statistics
        """
        stats = {
            'num_samples': 0,
            'image_sizes': [],
            'class_distribution': {},
            'color_statistics': {'mean': [], 'std': []}
        }
        
        try:
            if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
                # PyTorch dataset
                stats['num_samples'] = len(dataset)
                
                for i in range(min(100, len(dataset))):  # Sample first 100 images
                    if hasattr(dataset, '__getitem__'):
                        sample = dataset[i]
                        
                        # Handle different dataset formats
                        if isinstance(sample, tuple):
                            image = sample[0]
                            label = sample[1]
                            
                            # Update class distribution
                            if isinstance(label, torch.Tensor):
                                label = label.item()
                            stats['class_distribution'][label] = stats['class_distribution'].get(label, 0) + 1
                        
                        if isinstance(image, torch.Tensor):
                            # Convert tensor to numpy for analysis
                            if image.dim() == 3:
                                image_np = image.permute(1, 2, 0).numpy()
                            else:
                                continue
                        elif isinstance(image, np.ndarray):
                            image_np = image
                        else:
                            continue
                        
                        # Analyze image size
                        if image_np.ndim >= 2:
                            stats['image_sizes'].append(image_np.shape[:2])
                        
                        # Color statistics
                        if image_np.ndim == 3:
                            for c in range(min(3, image_np.shape[2])):
                                channel = image_np[:, :, c]
                                stats['color_statistics']['mean'].append(float(np.mean(channel)))
                                stats['color_statistics']['std'].append(float(np.std(channel)))
            
            # Compute summary statistics
            if stats['image_sizes']:
                sizes = np.array(stats['image_sizes'])
                stats['size_summary'] = {
                    'common_size': tuple(map(int, np.median(sizes, axis=0))),
                    'size_variety': len(set(map(tuple, stats['image_sizes'])))
                }
            
            if stats['color_statistics']['mean']:
                mean_vals = np.array(stats['color_statistics']['mean'])
                std_vals = np.array(stats['color_statistics']['std'])
                stats['color_summary'] = {
                    'overall_mean': float(np.mean(mean_vals)),
                    'overall_std': float(np.mean(std_vals))
                }
            
            self.dataset_stats = stats
            logger.info(f"Dataset analysis completed: {stats['num_samples']} samples")
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def validate_dataset(self, dataset: Any) -> Dict[str, Any]:
        """
        Validate dataset for training compatibility.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Validation report
        """
        report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        try:
            if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
                if len(dataset) == 0:
                    report['is_valid'] = False
                    report['errors'].append("Dataset is empty")
                    return report
                
                # Sample validation
                sample = dataset[0]
                
                if not isinstance(sample, tuple) or len(sample) < 2:
                    report['is_valid'] = False
                    report['errors'].append("Dataset samples should be (image, label) tuples")
                    return report
                
                image, label = sample
                
                # Check image format
                if isinstance(image, torch.Tensor):
                    if image.dim() not in [2, 3]:
                        report['errors'].append(f"Invalid image tensor dimensions: {image.dim()}")
                        report['is_valid'] = False
                    
                    if image.dim() == 3 and image.size(0) not in [1, 3]:
                        report['warnings'].append(f"Unexpected number of channels: {image.size(0)}")
                    
                elif isinstance(image, np.ndarray):
                    if image.ndim not in [2, 3]:
                        report['errors'].append(f"Invalid image array dimensions: {image.ndim}")
                        report['is_valid'] = False
                    
                    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
                        report['warnings'].append(f"Unexpected number of channels: {image.shape[2]}")
                
                # Check label format
                if not isinstance(label, (int, torch.Tensor)):
                    report['warnings'].append("Labels should be integers or tensors")
                
                # Check class balance
                if hasattr(dataset, '__len__') and len(dataset) > 10:
                    class_counts = {}
                    for i in range(min(1000, len(dataset))):
                        sample_label = dataset[i][1]
                        if isinstance(sample_label, torch.Tensor):
                            sample_label = sample_label.item()
                        class_counts[sample_label] = class_counts.get(sample_label, 0) + 1
                    
                    if len(class_counts) > 1:
                        counts = list(class_counts.values())
                        imbalance_ratio = max(counts) / min(counts)
                        if imbalance_ratio > 10:
                            report['warnings'].append(f"Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
                
                # Suggestions
                if hasattr(dataset, '__len__') and len(dataset) < 1000:
                    report['suggestions'].append("Dataset is small, consider data augmentation")
                
                report['warnings'] = list(set(report['warnings']))
                report['suggestions'] = list(set(report['suggestions']))
                
            else:
                report['is_valid'] = False
                report['errors'].append("Invalid dataset format")
        
        except Exception as e:
            report['is_valid'] = False
            report['errors'].append(f"Validation error: {str(e)}")
        
        logger.info(f"Dataset validation completed: {'Valid' if report['is_valid'] else 'Invalid'}")
        
        return report


class ResNetModelEvaluator:
    """
    Model evaluation utilities for ResNet models.
    
    This class provides methods for:
    - Computing model metrics
    - Generating evaluation reports
    - Model comparison
    - Performance visualization
    """
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        """
        Initialize Model Evaluator.
        
        Args:
            model: ResNet model to evaluate
            device: Device for computation
        """
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        
        logger.info(f"ResNet Model Evaluator initialized on device: {self.device}")
    
    def evaluate_model(self, 
                      data_loader: torch.utils.data.DataLoader,
                      return_predictions: bool = False) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            data_loader: Data loader for evaluation
            return_predictions: Whether to return predictions
            
        Returns:
            Evaluation metrics and results
        """
        self.model.eval()
        
        total_loss = 0.0
        predictions = []
        targets = []
        probabilities = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Collect predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                total_loss += loss.item()
                
                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        avg_loss = total_loss / len(data_loader)
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'num_samples': len(targets),
            'predictions': predictions if return_predictions else None,
            'targets': targets if return_predictions else None,
            'probabilities': probabilities if return_predictions else None
        }
        
        # Add detailed classification report
        if len(set(targets)) > 1:  # Only if multiple classes
            report = classification_report(targets, predictions, output_dict=True)
            metrics['classification_report'] = report
        
        logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, 
                             targets: List[int], 
                             predictions: List[int],
                             class_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Plot confusion matrix.
        
        Args:
            targets: True labels
            predictions: Predicted labels
            class_names: Class names for labels
            save_path: Path to save plot
            
        Returns:
            Confusion matrix image
        """
        try:
            cm = confusion_matrix(targets, predictions)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                return cm
            else:
                # Convert to numpy array
                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                
                buf.seek(0)
                image = Image.open(buf)
                return np.array(image)
        
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            return np.zeros((400, 400, 3), dtype=np.uint8)
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]], 
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
            
        Returns:
            Training history plot image
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            if 'train_loss' in history and 'val_loss' in history:
                axes[0].plot(history['train_loss'], label='Training Loss')
                axes[0].plot(history['val_loss'], label='Validation Loss')
                axes[0].set_title('Model Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
            
            # Accuracy plot
            if 'train_acc' in history and 'val_acc' in history:
                axes[1].plot(history['train_acc'], label='Training Accuracy')
                axes[1].plot(history['val_acc'], label='Validation Accuracy')
                axes[1].set_title('Model Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                # Convert to numpy array
                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                
                buf.seek(0)
                image = Image.open(buf)
                return np.array(image)
        
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            return np.zeros((300, 800, 3), dtype=np.uint8)


# Utility functions
def load_and_preprocess_image(image_path: str, 
                             preprocessor: ResNetPreprocessor,
                             enhance: bool = False) -> torch.Tensor:
    """
    Load and preprocess image from file.
    
    Args:
        image_path: Path to image file
        preprocessor: ResNet preprocessor instance
        enhance: Whether to enhance image quality
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Enhance if requested
        if enhance:
            image = preprocessor.enhance_image_quality(image)
        
        # Preprocess
        return preprocessor.preprocess_image(image, mode='inference')
        
    except Exception as e:
        logger.error(f"Error loading and preprocessing image {image_path}: {e}")
        return torch.zeros((3, 224, 224))


def save_image_tensor(tensor: torch.Tensor, 
                     path: str, 
                     preprocessor: ResNetPreprocessor):
    """
    Save image tensor to file.
    
    Args:
        tensor: Image tensor
        path: Output path
        preprocessor: ResNet preprocessor instance
    """
    try:
        # Denormalize
        image = preprocessor.denormalize_image(tensor)
        
        # Save
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info(f"Image saved to {path}")
        
    except Exception as e:
        logger.error(f"Error saving image tensor: {e}")


def compute_model_flops(model: nn.Module, input_size: Tuple[int, int] = (224, 224)) -> Dict[str, float]:
    """
    Compute model FLOPs and parameters.
    
    Args:
        model: PyTorch model
        input_size: Input image size
        
    Returns:
        Dictionary containing model statistics
    """
    try:
        from thop import profile
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        
        # Profile model
        flops, params = profile(model, inputs=(dummy_input,))
        
        stats = {
            'flops': flops,
            'parameters': params,
            'flops_gmac': flops / 1e9,
            'parameters_mb': params * 4 / 1e6,  # Assuming float32
            'input_size': input_size
        }
        
        logger.info(f"Model FLOPs: {stats['flops_gmac']:.2f} GMAC, Parameters: {stats['parameters_mb']:.2f} MB")
        
        return stats
        
    except ImportError:
        logger.warning("thop not installed, using basic parameter count")
        
        # Basic parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_mb': total_params * 4 / 1e6,
            'input_size': input_size
        }
    
    except Exception as e:
        logger.error(f"Error computing model FLOPs: {e}")
        return {'error': str(e)}


# Example usage and testing
def test_resnet_utils():
    """Test ResNet utilities."""
    logger.info("Testing ResNet Utilities")
    
    # Test preprocessor
    preprocessor = ResNetPreprocessor(input_size=224, augmentation=True)
    
    # Test feature extractor
    model = models.resnet18(pretrained=True)
    feature_extractor = ResNetFeatureExtractor(model)
    
    # Test dataset processor
    dataset_processor = ResNetDatasetProcessor(preprocessor)
    
    # Test model evaluator
    evaluator = ResNetModelEvaluator(model)
    
    logger.info("ResNet utilities test completed")
    
    return {
        'preprocessor': preprocessor,
        'feature_extractor': feature_extractor,
        'dataset_processor': dataset_processor,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    # Run test
    test_resnet_utils()