"""
Configuration settings for Soccer Player Recognition project.
This file contains general settings and paths for the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DATA_DIR = DATA_DIR / "input"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DETECTION_OUTPUTS_DIR = OUTPUTS_DIR / "detection"
SEGMENTATION_OUTPUTS_DIR = OUTPUTS_DIR / "segmentation"
CLASSIFICATION_OUTPUTS_DIR = OUTPUTS_DIR / "classification"
IDENTIFICATION_OUTPUTS_DIR = OUTPUTS_DIR / "identification"

# Create directories if they don't exist
DIRECTORIES = [
    DATA_DIR,
    INPUT_DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PRETRAINED_MODELS_DIR,
    CUSTOM_MODELS_DIR,
    OUTPUTS_DIR,
    DETECTION_OUTPUTS_DIR,
    SEGMENTATION_OUTPUTS_DIR,
    CLASSIFICATION_OUTPUTS_DIR,
    IDENTIFICATION_OUTPUTS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# General settings
DEFAULT_DEVICE = "cuda"  # or "cpu"
DEFAULT_NUM_WORKERS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (224, 224)

# Confidence thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.5
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.7
TRACKING_CONFIDENCE_THRESHOLD = 0.3

# Image processing settings
MAX_IMAGE_SIZE = (1920, 1080)
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Cache settings
CACHE_ENABLED = True
CACHE_DIR = PROJECT_ROOT / "cache"

# Memory optimization
ENABLE_MEMORY_OPTIMIZATION = True
MAX_MEMORY_USAGE_PERCENT = 80