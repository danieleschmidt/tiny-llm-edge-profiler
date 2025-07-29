"""
Tiny LLM Edge Profiler

A comprehensive profiling toolkit for running quantized LLMs on microcontrollers
and edge devices. Measure real-world performance on ARM Cortex-M, RISC-V, ESP32.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "dev@terragon.dev"

# Core profiling components
from .profiler import EdgeProfiler
from .models import QuantizedModel
from .results import ProfileResults
from .platforms import PlatformManager

# Analysis and optimization tools
from .analyzer import MetricsAnalyzer, ComparativeAnalyzer
from .optimizer import (
    EnergyOptimizer,
    MemoryOptimizer,
    PlatformOptimizer,
)

# Power profiling
from .power import PowerProfiler

# Benchmarking
from .benchmarks import StandardBenchmarks

__all__ = [
    # Core API
    "EdgeProfiler",
    "QuantizedModel", 
    "ProfileResults",
    "PlatformManager",
    
    # Analysis
    "MetricsAnalyzer",
    "ComparativeAnalyzer",
    
    # Optimization
    "EnergyOptimizer",
    "MemoryOptimizer", 
    "PlatformOptimizer",
    
    # Power profiling
    "PowerProfiler",
    
    # Benchmarking
    "StandardBenchmarks",
]