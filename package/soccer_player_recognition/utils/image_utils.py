"""
Image processing utilities for soccer player recognition.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
from pathlib import Path


class ImageProcessor:
    """Image processing utilities for soccer player recognition."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor.
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
        self.mean = (0.485, 0.456, 0.406)  # ImageNet mean
        self.std = (0.229, 0.224, 0.225)   # ImageNet std
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array (BGR format)
        """
        image_path = str(image_path)
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        return image
    
    def resize_image(self, image: np.ndarray, 
                    target_size: Optional[Tuple[int, int]] = None,
                    keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            keep_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        if keep_aspect_ratio:
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create new image with target size and pad if necessary
            new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            new_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return new_image
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize to [0, 1]
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image_normalized = (image_normalized - self.mean) / self.std
        
        return image_normalized
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image pixel values.
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image in [0, 255] range
        """
        # Reverse normalization
        image_denormalized = image * self.std + self.mean
        
        # Clip to [0, 1] range
        image_denormalized = np.clip(image_denormalized, 0, 1)
        
        # Convert to [0, 255] and RGB format
        image_denormalized = (image_denormalized * 255).astype(np.uint8)
        
        return image_denormalized
    
    def enhance_image(self, image: np.ndarray, 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     saturation: float = 1.0,
                     sharpness: float = 1.0) -> np.ndarray:
        """
        Enhance image quality using PIL.
        
        Args:
            image: Input image
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Apply enhancements
        if brightness != 1.0:
            pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
        
        if contrast != 1.0:
            pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
        
        if saturation != 1.0:
            pil_image = ImageEnhance.Color(pil_image).enhance(saturation)
        
        if sharpness != 1.0:
            pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)
        
        # Convert back to BGR
        enhanced_rgb = np.array(pil_image)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr
    
    def apply_noise_reduction(self, image: np.ndarray, 
                             method: str = 'gaussian') -> np.ndarray:
        """
        Apply noise reduction to image.
        
        Args:
            image: Input image
            method: Noise reduction method ('gaussian', 'median', 'bilateral')
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in image using Canny edge detection.
        
        Args:
            image: Input image
            
        Returns:
            Edge image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def crop_to_roi(self, image: np.ndarray, 
                   roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop image to region of interest.
        
        Args:
            image: Input image
            roi: Region of interest (x, y, w, h)
            
        Returns:
            Cropped image
        """
        x, y, w, h = roi
        return image[y:y+h, x:x+w]
    
    def rotate_image(self, image: np.ndarray, 
                    angle: float,
                    center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            center: Rotation center (x, y)
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return rotated
    
    def flip_image(self, image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
        """
        Flip image horizontally or vertically.
        
        Args:
            image: Input image
            direction: Flip direction ('horizontal', 'vertical', 'both')
            
        Returns:
            Flipped image
        """
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        elif direction == 'both':
            return cv2.flip(image, -1)
        else:
            raise ValueError(f"Invalid flip direction: {direction}")
    
    def apply_color_mask(self, image: np.ndarray, 
                        lower_bound: Tuple[int, int, int],
                        upper_bound: Tuple[int, int, int]) -> np.ndarray:
        """
        Apply color mask to image.
        
        Args:
            image: Input image
            lower_bound: Lower color bound (B, G, R)
            upper_bound: Upper color bound (B, G, R)
            
        Returns:
            Masked image
        """
        # Create mask
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        # Apply mask
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        return masked
    
    def get_image_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about the image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        stats = {
            'shape': image.shape,
            'dtype': image.dtype,
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'aspect_ratio': float(image.shape[1] / image.shape[0])
        }
        
        # Calculate histogram for each channel
        stats['histogram'] = {}
        for i, color in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            stats['histogram'][color] = hist.flatten().tolist()
        
        return stats


def create_torch_tensor(image: np.ndarray, 
                       target_size: Optional[Tuple[int, int]] = None,
                       normalize: bool = True,
                       to_tensor: bool = True) -> torch.Tensor:
    """
    Convert image to PyTorch tensor.
    
    Args:
        image: Input image (numpy array)
        target_size: Target size for resizing
        normalize: Whether to normalize the image
        to_tensor: Whether to convert to tensor
        
    Returns:
        PyTorch tensor
    """
    processor = ImageProcessor(target_size)
    
    # Resize image
    image = processor.resize_image(image)
    
    # Normalize if requested
    if normalize:
        image = processor.normalize_image(image)
    
    # Convert to tensor
    if to_tensor:
        # Convert from (H, W, C) to (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
    
    return image


def load_and_process_image(image_path: Union[str, Path],
                          target_size: Optional[Tuple[int, int]] = None,
                          enhance: bool = False,
                          denoise: bool = False) -> np.ndarray:
    """
    Load and process image with multiple enhancement options.
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        enhance: Whether to apply image enhancement
        denoise: Whether to apply noise reduction
        
    Returns:
        Processed image
    """
    processor = ImageProcessor(target_size)
    
    # Load image
    image = processor.load_image(image_path)
    
    # Apply enhancements
    if enhance:
        image = processor.enhance_image(image)
    
    if denoise:
        image = processor.apply_noise_reduction(image)
    
    return image


def batch_process_images(image_paths: List[Union[str, Path]],
                        target_size: Optional[Tuple[int, int]] = None,
                        **kwargs) -> torch.Tensor:
    """
    Process multiple images in batch.
    
    Args:
        image_paths: List of image paths
        target_size: Target size for resizing
        **kwargs: Additional processing arguments
        
    Returns:
        Batch of processed images as tensor
    """
    processed_images = []
    
    for image_path in image_paths:
        image = load_and_process_image(image_path, target_size, **kwargs)
        processed_images.append(image)
    
    # Stack images into batch
    if processed_images:
        batch = np.stack(processed_images)
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)  # (B, C, H, W)
        return batch_tensor
    
    return torch.empty(0)