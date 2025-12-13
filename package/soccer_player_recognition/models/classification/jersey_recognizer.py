"""
Jersey Number Recognition using OCR and Computer Vision

This module implements jersey number recognition for soccer players using 
OCR (Optical Character Recognition) and computer vision techniques.
It extracts jersey numbers from player images to assist in player identification.

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Tuple, Dict, Optional, Any, Union
import logging
from pathlib import Path
import re
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import json
from collections import Counter

from soccer_player_recognition.utils.logger import get_logger

logger = get_logger(__name__)


class JerseyNumberExtractor:
    """
    Jersey number extraction using computer vision and OCR techniques.
    
    This class provides methods to:
    - Detect jersey number regions in player images
    - Extract and recognize jersey numbers using OCR
    - Process and enhance jersey number images for better recognition
    - Validate and filter extracted numbers
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 min_number_size: int = 30,
                 max_number_size: int = 200,
                 confidence_threshold: float = 0.6):
        """
        Initialize Jersey Number Extractor.
        
        Args:
            config: Configuration dictionary for OCR and image processing
            min_number_size: Minimum size of number region in pixels
            max_number_size: Maximum size of number region in pixels
            confidence_threshold: Minimum confidence for number recognition
        """
        self.config = config or self._get_default_config()
        self.min_number_size = min_number_size
        self.max_number_size = max_number_size
        self.confidence_threshold = confidence_threshold
        
        # Configure Tesseract OCR
        self._configure_tesseract()
        
        # Color ranges for jersey number detection (in HSV)
        self.jersey_color_ranges = self._get_jersey_color_ranges()
        
        logger.info("Jersey Number Extractor initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for OCR and image processing."""
        return {
            'tesseract_config': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
            'blur_kernel_size': (3, 3),
            'morph_kernel_size': (3, 3),
            'canny_low_threshold': 50,
            'canny_high_threshold': 150,
            'dilate_iterations': 1,
            'erode_iterations': 1,
            'morph_operation': cv2.MORPH_CLOSE,
            'number_padding': 10
        }
    
    def _configure_tesseract(self):
        """Configure Tesseract OCR parameters."""
        try:
            # Test Tesseract installation
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.warning(f"Tesseract not properly configured: {e}")
            # Fallback configuration
            self.config['tesseract_config'] = '--psm 8 -c tessedit_char_whitelist=0123456789'
    
    def _get_jersey_color_ranges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get color ranges for jersey number detection in HSV.
        
        Returns:
            List of HSV color ranges for jersey colors
        """
        return [
            # White numbers
            ([0, 0, 200], [180, 30, 255]),
            # Black numbers
            ([0, 0, 0], [180, 255, 50]),
            # Yellow/Gold numbers
            ([15, 100, 100], [35, 255, 255]),
            # Red numbers
            ([0, 100, 100], [10, 255, 255]),
            # Blue numbers
            ([100, 50, 50], [130, 255, 255])
        ]
    
    def detect_jersey_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect jersey region in player image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Cropped jersey region or None if not detected
        """
        try:
            height, width = image.shape[:2]
            
            # Define jersey region (typically middle-lower portion of image)
            jersey_top = int(height * 0.3)  # Top 30% excluded (likely head/shoulders)
            jersey_bottom = int(height * 0.9)  # Bottom 10% excluded (likely shorts)
            
            jersey_region = image[jersey_top:jersey_bottom, :]
            
            if jersey_region.size == 0:
                logger.warning("Empty jersey region detected")
                return None
            
            return jersey_region
            
        except Exception as e:
            logger.error(f"Error detecting jersey region: {e}")
            return None
    
    def preprocess_jersey_region(self, jersey_region: np.ndarray) -> np.ndarray:
        """
        Preprocess jersey region for better number recognition.
        
        Args:
            jersey_region: Cropped jersey region image
            
        Returns:
            Preprocessed image for OCR
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, self.config['blur_kernel_size'], 0)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config['morph_kernel_size'])
            morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
            
            # Enhance contrast using PIL
            pil_image = Image.fromarray(morphed)
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(1.5)
            
            processed = np.array(enhanced)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing jersey region: {e}")
            return cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
    
    def extract_number_candidates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract candidate number regions using computer vision.
        
        Args:
            image: Preprocessed jersey region image
            
        Returns:
            List of candidate number regions with metadata
        """
        candidates = []
        
        try:
            # Apply edge detection
            edges = cv2.Canny(image, self.config['canny_low_threshold'], self.config['canny_high_threshold'])
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                area = w * h
                if area < self.min_number_size or area > self.max_number_size:
                    continue
                
                # Check aspect ratio (numbers are typically wider than tall)
                aspect_ratio = w / h
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    continue
                
                # Extract region
                x_start = max(0, x - self.config['number_padding'])
                y_start = max(0, y - self.config['number_padding'])
                x_end = min(image.shape[1], x + w + self.config['number_padding'])
                y_end = min(image.shape[0], y + h + self.config['number_padding'])
                
                region = image[y_start:y_end, x_start:x_end]
                
                if region.size == 0:
                    continue
                
                candidates.append({
                    'region': region,
                    'bbox': (x_start, y_start, x_end - x_start, y_end - y_start),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'confidence_score': self._calculate_confidence(region)
                })
            
            # Sort by confidence score
            candidates.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            logger.info(f"Found {len(candidates)} number candidates")
            
        except Exception as e:
            logger.error(f"Error extracting number candidates: {e}")
        
        return candidates
    
    def _calculate_confidence(self, region: np.ndarray) -> float:
        """
        Calculate confidence score for a number region.
        
        Args:
            region: Cropped number region
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Check if region has sufficient contrast
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Check for clear foreground/background separation
            non_zero_hist = hist[hist > 0]
            if len(non_zero_hist) < 3:
                return 0.0
            
            # Check for good contrast ratio
            max_hist = np.max(hist)
            non_zero_hist = non_zero_hist[non_zero_hist > max_hist * 0.1]
            
            if len(non_zero_hist) < 2:
                return 0.3
            
            # Simple contrast metric
            contrast = np.std(non_zero_hist) / np.mean(non_zero_hist)
            confidence = min(1.0, contrast / 2.0)
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.0
    
    def recognize_number(self, region: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize number from region using OCR.
        
        Args:
            region: Cropped number region
            
        Returns:
            Tuple of (recognized_number, confidence)
        """
        try:
            # Preprocess for OCR
            processed_region = self._preprocess_for_ocr(region)
            
            # Apply OCR
            results = pytesseract.image_to_string(
                processed_region, 
                config=self.config['tesseract_config']
            )
            
            # Clean and validate result
            number, confidence = self._clean_ocr_result(results)
            
            return number, confidence
            
        except Exception as e:
            logger.error(f"Error recognizing number: {e}")
            return None, 0.0
    
    def _preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """
        Preprocess region specifically for OCR.
        
        Args:
            region: Input region image
            
        Returns:
            Preprocessed image for OCR
        """
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        # Resize if too small
        min_size = 50
        h, w = gray.shape
        if h < min_size or w < min_size:
            scale = min_size / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _clean_ocr_result(self, result: str) -> Tuple[Optional[str], float]:
        """
        Clean and validate OCR result.
        
        Args:
            result: Raw OCR result
            
        Returns:
            Tuple of (cleaned_number, confidence)
        """
        # Remove whitespace and special characters
        cleaned = re.sub(r'[^\d]', '', result.strip())
        
        # Validate length (1-3 digits typically for soccer jersey numbers)
        if len(cleaned) == 0 or len(cleaned) > 3:
            return None, 0.0
        
        # Check if it's a valid jersey number (1-99 typically)
        try:
            number = int(cleaned)
            if 1 <= number <= 99:
                # Assign confidence based on length and validity
                if len(cleaned) == 2:
                    confidence = 0.8  # Most common jersey numbers
                elif len(cleaned) == 1:
                    confidence = 0.6
                else:  # 3 digits
                    confidence = 0.7
                return cleaned, confidence
        except ValueError:
            pass
        
        return None, 0.0
    
    def extract_jersey_number(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Complete jersey number extraction pipeline.
        
        Args:
            image: Input player image (BGR format)
            
        Returns:
            Dictionary containing extraction results
        """
        result = {
            'number': None,
            'confidence': 0.0,
            'candidates': [],
            'processing_info': {}
        }
        
        try:
            # Step 1: Detect jersey region
            jersey_region = self.detect_jersey_region(image)
            if jersey_region is None:
                result['processing_info']['error'] = 'Could not detect jersey region'
                return result
            
            # Step 2: Preprocess jersey region
            processed_region = self.preprocess_jersey_region(jersey_region)
            
            # Step 3: Extract number candidates
            candidates = self.extract_number_candidates(processed_region)
            result['candidates'] = [
                {
                    'bbox': c['bbox'],
                    'area': c['area'],
                    'aspect_ratio': c['aspect_ratio'],
                    'confidence_score': c['confidence_score']
                } for c in candidates
            ]
            
            if not candidates:
                result['processing_info']['error'] = 'No number candidates found'
                return result
            
            # Step 4: Recognize numbers from candidates
            best_recognition = None
            best_confidence = 0.0
            
            for candidate in candidates[:5]:  # Check top 5 candidates
                number, confidence = self.recognize_number(candidate['region'])
                
                if number is not None and confidence > best_confidence:
                    # Additional validation
                    if self._validate_number(number):
                        best_recognition = number
                        best_confidence = confidence
            
            if best_recognition is not None and best_confidence >= self.confidence_threshold:
                result['number'] = best_recognition
                result['confidence'] = best_confidence
                result['processing_info']['status'] = 'success'
            else:
                result['processing_info']['status'] = 'low_confidence'
                result['processing_info']['best_confidence'] = best_confidence
            
        except Exception as e:
            logger.error(f"Error in jersey number extraction: {e}")
            result['processing_info']['error'] = str(e)
        
        return result
    
    def _validate_number(self, number: str) -> bool:
        """
        Validate if number is a reasonable jersey number.
        
        Args:
            number: Recognized number string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            num = int(number)
            # Most jersey numbers are between 1-99, though 00 is sometimes used
            return 0 <= num <= 99
        except ValueError:
            return False
    
    def visualize_extraction(self, image: np.ndarray, result: Dict[str, Any], save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize jersey number extraction results.
        
        Args:
            image: Original input image
            result: Extraction result dictionary
            save_path: Path to save visualization image
            
        Returns:
            Visualization image
        """
        try:
            vis_image = image.copy()
            jersey_region = self.detect_jersey_region(image)
            
            if jersey_region is not None and result['candidates']:
                # Draw jersey region
                jersey_y_offset = int(image.shape[0] * 0.3)
                cv2.rectangle(vis_image, (0, jersey_y_offset), 
                            (image.shape[1], int(image.shape[0] * 0.9)), 
                            (0, 255, 0), 2)
            
            # Draw candidates
            for candidate in result['candidates']:
                bbox = candidate['bbox']
                # Adjust bbox for original image coordinates
                jersey_y_offset = int(image.shape[0] * 0.3)
                adjusted_bbox = (bbox[0], bbox[1] + jersey_y_offset, bbox[2], bbox[3])
                
                cv2.rectangle(vis_image, (adjusted_bbox[0], adjusted_bbox[1]), 
                            (adjusted_bbox[0] + adjusted_bbox[2], adjusted_bbox[1] + adjusted_bbox[3]), 
                            (255, 0, 0), 1)
            
            # Add text annotations
            if result['number']:
                text = f"Jersey: {result['number']} (conf: {result['confidence']:.2f})"
                cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
            else:
                cv2.putText(vis_image, "No jersey number detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if save_path:
                cv2.imwrite(save_path, vis_image)
                logger.info(f"Visualization saved to {save_path}")
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return image
    
    def batch_extract_numbers(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extract jersey numbers from multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.extract_jersey_number(image)
                result['image_index'] = i
                results.append(result)
                
                logger.info(f"Processed image {i+1}/{len(images)}: {result.get('number', 'No number')}")
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    'image_index': i,
                    'number': None,
                    'confidence': 0.0,
                    'processing_info': {'error': str(e)}
                })
        
        return results
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics from batch extraction results.
        
        Args:
            results: List of extraction results
            
        Returns:
            Dictionary containing statistics
        """
        successful_extractions = [r for r in results if r.get('number') is not None]
        total_confidence = sum(r.get('confidence', 0) for r in successful_extractions)
        
        # Number frequency analysis
        numbers = [r.get('number') for r in successful_extractions]
        number_frequency = Counter(numbers)
        
        return {
            'total_images': len(results),
            'successful_extractions': len(successful_extractions),
            'success_rate': len(successful_extractions) / len(results) if results else 0,
            'average_confidence': total_confidence / len(successful_extractions) if successful_extractions else 0,
            'number_frequency': dict(number_frequency),
            'most_common_number': number_frequency.most_common(1)[0] if number_frequency else None
        }


class JerseyRecognizer:
    """
    High-level jersey number recognition system.
    
    This class provides a simplified interface for jersey number recognition
    and integrates with other player recognition components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Jersey Recognizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.extractor = JerseyNumberExtractor(self.config)
        
        # Recognition cache for performance
        self.recognition_cache = {}
        self.cache_enabled = True
        
        logger.info("Jersey Recognizer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def recognize_jersey_number(self, image: np.ndarray, use_cache: bool = True) -> Dict[str, Any]:
        """
        Recognize jersey number from image.
        
        Args:
            image: Input image (BGR format)
            use_cache: Whether to use recognition cache
            
        Returns:
            Recognition result dictionary
        """
        # Check cache first
        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(image)
            if cache_key in self.recognition_cache:
                return self.recognition_cache[cache_key]
        
        # Perform extraction
        result = self.extractor.extract_jersey_number(image)
        
        # Add metadata
        result['recognizer_version'] = '1.0'
        result['cache_key'] = self._generate_cache_key(image) if use_cache else None
        
        # Cache result
        if use_cache and self.cache_enabled:
            self.recognition_cache[cache_key] = result
        
        return result
    
    def _generate_cache_key(self, image: np.ndarray) -> str:
        """Generate cache key for image."""
        # Simple hash-based key
        return str(hash(image.tobytes()))
    
    def process_video_frame(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Process video frame for jersey number recognition.
        
        Args:
            frame: Video frame image
            roi: Region of interest (x, y, w, h)
            
        Returns:
            Recognition result
        """
        try:
            # Apply ROI if provided
            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]
            
            return self.recognize_jersey_number(frame)
            
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return {
                'number': None,
                'confidence': 0.0,
                'processing_info': {'error': str(e)}
            }
    
    def clear_cache(self):
        """Clear recognition cache."""
        self.recognition_cache.clear()
        logger.info("Recognition cache cleared")
    
    def enable_cache(self, enable: bool = True):
        """Enable or disable recognition cache."""
        self.cache_enabled = enable
        logger.info(f"Recognition cache {'enabled' if enable else 'disabled'}")


# Example usage and testing functions
def test_jersey_recognition():
    """Test jersey number recognition with sample data."""
    logger.info("Testing Jersey Number Recognition")
    
    # Create sample recognizer
    recognizer = JerseyRecognizer()
    
    # Create sample image (placeholder)
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some fake jersey region and number for testing
    cv2.rectangle(sample_image, (200, 150), (400, 300), (255, 255, 255), -1)
    cv2.putText(sample_image, "10", (280, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Test recognition
    result = recognizer.recognize_jersey_number(sample_image)
    
    print("Recognition Result:")
    print(f"Number: {result.get('number')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Candidates found: {len(result.get('candidates', []))}")
    
    return result


if __name__ == "__main__":
    # Run test
    test_jersey_recognition()