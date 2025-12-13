"""
UI Package for Soccer Player Recognition System

This package contains the graphical user interface components
for the soccer player recognition system demo applications.

Components:
- demo_interface: Main GUI application for testing and demonstrating the system
- visualization_tools: Additional visualization utilities
- interaction_handlers: User interaction management

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

__version__ = "1.0"
__author__ = "Soccer Player Recognition Team"

from .demo_interface import SoccerRecognitionGUI

__all__ = ["SoccerRecognitionGUI"]