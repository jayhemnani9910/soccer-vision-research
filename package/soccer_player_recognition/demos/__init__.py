"""
Demos Package for Soccer Player Recognition System

This package contains comprehensive demo applications for testing
and demonstrating the soccer player recognition system capabilities.

Available Demos:
- complete_system_demo.py: Full system demonstration with all models
- single_model_demo.py: Individual model testing and analysis
- real_time_demo.py: Real-time processing demonstrations
- benchmark_demo.py: Performance benchmarking and stress testing

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

__version__ = "1.0"
__author__ = "Soccer Player Recognition Team"

# Import main demo classes for easy access
try:
    from .complete_system_demo import CompleteSystemDemo
    from .single_model_demo import SingleModelDemo
    from .real_time_demo import RealTimeDemo
    from .benchmark_demo import BenchmarkDemo
except ImportError:
    # Handle cases where modules might not be available
    CompleteSystemDemo = None
    SingleModelDemo = None
    RealTimeDemo = None
    BenchmarkDemo = None

__all__ = [
    "CompleteSystemDemo",
    "SingleModelDemo", 
    "RealTimeDemo",
    "BenchmarkDemo"
]