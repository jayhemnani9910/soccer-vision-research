#!/usr/bin/env python
"""Setup script for Soccer Player Recognition System."""

from setuptools import setup, find_packages
import os

# Read version from version.py
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'soccer_player_recognition', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '1.0.0'

# Read README for long description
def read_readme():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return 'Soccer Player Recognition System'

setup(
    name="soccer-player-recognition",
    version=read_version(),
    author="AI Development Team",
    author_email="ai@example.com",
    description="Advanced soccer player recognition and tracking system using computer vision and deep learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/soccer-player-recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "ultralytics>=8.0.0",
        "albumentations>=1.3.0",
        "timm>=0.9.0",
        "huggingface-hub>=0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch-audio-cuda",
            "torchvision-cuda",
        ],
        "tracking": [
            "face-recognition>=1.3.0",
            "deep-sort-realtime>=1.3.2",
        ],
        "visualization": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "soccer-recognition=soccer_player_recognition.cli:main",
            "soccer-detect=soccer_player_recognition.detection.run_detection",
            "soccer-identify=soccer_player_recognition.identification.run_identification",
        ],
    },
    include_package_data=True,
    package_data={
        "soccer_player_recognition": [
            "config/*.yaml",
            "config/*.json",
            "data/*",
            "docs/*",
            "models/*.pt",
            "models/*.pth",
            "models/*.onnx",
        ],
    },
    keywords=[
        "computer vision",
        "machine learning",
        "deep learning",
        "soccer",
        "football",
        "player recognition",
        "object detection",
        "player tracking",
        "sports analytics",
        "yolo",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/soccer-player-recognition/issues",
        "Source": "https://github.com/example/soccer-player-recognition",
        "Documentation": "https://soccer-player-recognition.readthedocs.io/",
    },
)