"""
Autonomous Model Architecture Search (AutoNAS)
=============================================

Advanced autonomous neural architecture search system that automatically
discovers optimal model architectures for edge deployment with minimal
human intervention.

Generation 5+ Features:
- Evolutionary neural architecture search with multi-objective optimization
- Hardware-aware architecture design for edge constraints
- Progressive architecture refinement with performance feedback
- Automated quantization-aware architecture optimization
- Self-supervised architecture evaluation and ranking
- Continuous architecture adaptation based on deployment feedback
"""

import asyncio
import json
import logging
import math
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import hashlib

from .profiler import EdgeProfiler, ProfileResults
from .models import QuantizedModel
from .self_evolving_optimization import SelfEvolvingOptimizer, OptimizationGenome, EvolutionStrategy
from .advanced_monitoring import MonitoringSystem


class LayerType(Enum):
    """Neural network layer types."""
    
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    DEPTHWISE_CONV = "depthwise_conv"
    SEPARABLE_CONV = "separable_conv"
    LINEAR = "linear"
    ATTENTION = "attention"
    MULTIHEAD_ATTENTION = "multihead_attention"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER_BLOCK = "transformer_block"
    RESIDUAL_BLOCK = "residual_block"
    BOTTLENECK = "bottleneck"
    SKIP_CONNECTION = "skip_connection"
    POOLING = "pooling"
    ADAPTIVE_POOLING = "adaptive_pooling"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    DROPOUT = "dropout"
    ACTIVATION = "activation"


class ActivationType(Enum):
    """Activation function types."""
    
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    HARD_SIGMOID = "hard_sigmoid"
    HARD_SWISH = "hard_swish"


class HardwareTarget(Enum):
    """Target hardware platforms for optimization."""
    
    ESP32 = "esp32"
    STM32 = "stm32"
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    CORAL_TPU = "coral_tpu"
    ARDUINO = "arduino"
    RISC_V = "risc_v"
    GENERIC_MCU = "generic_mcu"


@dataclass
class LayerSpec:
    """Specification for a neural network layer."""
    
    layer_type: LayerType
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    kernel_size: Optional[Union[int, Tuple[int, ...]]] = None
    stride: Optional[Union[int, Tuple[int, ...]]] = None
    padding: Optional[Union[str, int, Tuple[int, ...]]] = None
    activation: Optional[ActivationType] = None
    dropout_rate: Optional[float] = None
    use_bias: bool = True
    groups: int = 1
    dilation: Optional[Union[int, Tuple[int, ...]]] = None
    
    # Attention-specific parameters
    num_heads: Optional[int] = None
    embed_dim: Optional[int] = None
    
    # Quantization parameters
    quantization_bits: int = 8
    quantize_weights: bool = True
    quantize_activations: bool = True
    
    def __post_init__(self):
        # Auto-infer parameters for specific layer types
        if self.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
            if self.kernel_size is None:
                self.kernel_size = 3
            if self.stride is None:
                self.stride = 1
            if self.padding is None:
                self.padding = "same"
    
    def estimate_parameters(self) -> int:
        """Estimate number of parameters for this layer."""
        
        if self.layer_type == LayerType.LINEAR:
            if self.input_dim and self.output_dim:
                params = self.input_dim * self.output_dim
                if self.use_bias:
                    params += self.output_dim
                return params
                
        elif self.layer_type == LayerType.CONV2D:
            if self.input_dim and self.output_dim and self.kernel_size:
                if isinstance(self.kernel_size, int):
                    kernel_params = self.kernel_size * self.kernel_size
                else:
                    kernel_params = self.kernel_size[0] * self.kernel_size[1]
                
                params = kernel_params * self.input_dim * self.output_dim
                if self.use_bias:
                    params += self.output_dim
                return params
                
        elif self.layer_type == LayerType.ATTENTION:
            if self.embed_dim:
                # Query, Key, Value projections + output projection
                params = 4 * self.embed_dim * self.embed_dim
                if self.use_bias:
                    params += 4 * self.embed_dim
                return params
                
        elif self.layer_type == LayerType.LSTM:
            if self.input_dim and self.output_dim:
                # LSTM has 4 gates, each with input and hidden weights
                params = 4 * (self.input_dim * self.output_dim + self.output_dim * self.output_dim)
                if self.use_bias:
                    params += 4 * self.output_dim
                return params
        
        return 0  # Default for unknown layer types
    
    def estimate_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for this layer given input shape."""
        
        if self.layer_type == LayerType.LINEAR:
            if len(input_shape) >= 1 and self.output_dim:
                batch_size = input_shape[0] if len(input_shape) > 1 else 1
                return batch_size * self.input_dim * self.output_dim * 2  # Multiply-add
                
        elif self.layer_type == LayerType.CONV2D:
            if len(input_shape) >= 3 and self.output_dim and self.kernel_size:
                batch_size, height, width = input_shape[0], input_shape[-2], input_shape[-1]
                if isinstance(self.kernel_size, int):
                    kernel_ops = self.kernel_size * self.kernel_size
                else:
                    kernel_ops = self.kernel_size[0] * self.kernel_size[1]
                
                output_elements = height * width  # Simplified
                return batch_size * output_elements * kernel_ops * self.input_dim * self.output_dim
                
        elif self.layer_type == LayerType.ATTENTION:
            if len(input_shape) >= 2 and self.embed_dim:
                batch_size, seq_len = input_shape[0], input_shape[1]
                # Simplified attention FLOPs
                return batch_size * seq_len * seq_len * self.embed_dim * 4
        
        return 0


@dataclass  
class ModelArchitecture:
    """Complete model architecture specification."""
    
    architecture_id: str
    layers: List[LayerSpec]
    input_shape: Tuple[int, ...]
    output_dim: int
    generation: int = 0
    
    # Hardware constraints
    target_hardware: HardwareTarget = HardwareTarget.GENERIC_MCU
    max_parameters: int = 1000000
    max_memory_mb: float = 4.0
    max_latency_ms: float = 100.0
    max_power_mw: float = 50.0
    
    # Quantization strategy
    global_quantization_bits: int = 8
    mixed_precision: bool = False
    quantization_scheme: str = "uniform"
    
    # Performance metrics
    accuracy_scores: List[float] = field(default_factory=list)
    latency_measurements: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    energy_consumption: List[float] = field(default_factory=list)
    
    # Evolution tracking
    parent_architectures: List[str] = field(default_factory=list)
    mutation_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.architecture_id:
            # Generate unique ID based on architecture
            arch_str = self._architecture_string()
            self.architecture_id = hashlib.md5(arch_str.encode()).hexdigest()[:12]
    
    def _architecture_string(self) -> str:
        """Create string representation of architecture for hashing."""
        
        layer_strings = []
        for layer in self.layers:
            layer_str = f"{layer.layer_type.value}_{layer.input_dim}_{layer.output_dim}"
            if layer.kernel_size:
                layer_str += f"_k{layer.kernel_size}"
            if layer.activation:
                layer_str += f"_{layer.activation.value}"
            layer_strings.append(layer_str)
        
        return "_".join(layer_strings)
    
    def estimate_total_parameters(self) -> int:
        """Estimate total number of parameters."""
        
        total = 0
        current_dim = None
        
        # Infer dimensions through the network
        if self.input_shape:
            current_dim = self.input_shape[-1] if len(self.input_shape) > 1 else self.input_shape[0]
        
        for layer in self.layers:
            if layer.layer_type in [LayerType.LINEAR, LayerType.CONV2D, LayerType.ATTENTION]:
                if layer.input_dim is None and current_dim is not None:
                    layer.input_dim = current_dim
                
                total += layer.estimate_parameters()
                
                if layer.output_dim:
                    current_dim = layer.output_dim
        
        return total
    
    def estimate_total_flops(self) -> int:
        """Estimate total FLOPs for forward pass."""
        
        total = 0
        current_shape = self.input_shape
        
        for layer in self.layers:
            if layer.layer_type not in [LayerType.DROPOUT, LayerType.BATCH_NORM, LayerType.LAYER_NORM]:
                total += layer.estimate_flops(current_shape)
                
                # Update shape for next layer (simplified)
                if layer.output_dim and len(current_shape) > 0:
                    if layer.layer_type == LayerType.LINEAR:
                        current_shape = (current_shape[0], layer.output_dim)
                    elif layer.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
                        current_shape = (current_shape[0], current_shape[1], current_shape[2], layer.output_dim)
        
        return total
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        
        # Parameter memory
        param_memory = self.estimate_total_parameters() * 4 / (1024 * 1024)  # 4 bytes per float32
        
        # Activation memory (simplified)
        activation_memory = 0
        current_shape = self.input_shape
        
        for layer in self.layers:
            if layer.output_dim and len(current_shape) > 0:
                # Estimate activation size
                if layer.layer_type == LayerType.LINEAR:
                    elements = current_shape[0] * layer.output_dim
                elif layer.layer_type in [LayerType.CONV2D]:
                    elements = current_shape[0] * current_shape[1] * current_shape[2] * layer.output_dim
                else:
                    elements = np.prod(current_shape) * (layer.output_dim or 1)
                
                activation_memory += elements * 4 / (1024 * 1024)  # 4 bytes per float32
                
                # Update shape
                if layer.layer_type == LayerType.LINEAR:
                    current_shape = (current_shape[0], layer.output_dim)
        
        return param_memory + activation_memory
    
    def is_hardware_compatible(self) -> bool:
        """Check if architecture is compatible with target hardware."""
        
        params = self.estimate_total_parameters()
        memory = self.estimate_memory_usage()
        
        if params > self.max_parameters:
            return False
        if memory > self.max_memory_mb:
            return False
        
        # Additional hardware-specific checks
        if self.target_hardware == HardwareTarget.ESP32:
            if memory > 1.0:  # ESP32 has limited RAM
                return False
            if any(layer.layer_type == LayerType.TRANSFORMER_BLOCK for layer in self.layers):
                return False  # Too complex for ESP32
                
        elif self.target_hardware == HardwareTarget.ARDUINO:
            if memory > 0.1:  # Very limited memory
                return False
            if params > 10000:  # Very limited parameters
                return False
        
        return True
    
    def calculate_fitness(self) -> float:
        """Calculate overall fitness score for this architecture."""
        
        if not self.accuracy_scores:
            return 0.0
        
        # Multi-objective fitness
        accuracy = np.mean(self.accuracy_scores)
        
        # Hardware efficiency scores
        param_efficiency = min(1.0, self.max_parameters / max(self.estimate_total_parameters(), 1))
        memory_efficiency = min(1.0, self.max_memory_mb / max(self.estimate_memory_usage(), 0.001))
        
        # Latency and energy efficiency
        latency_score = 1.0
        energy_score = 1.0
        
        if self.latency_measurements:
            avg_latency = np.mean(self.latency_measurements)
            latency_score = min(1.0, self.max_latency_ms / max(avg_latency, 1.0))
        
        if self.energy_consumption:
            avg_energy = np.mean(self.energy_consumption)
            energy_score = min(1.0, self.max_power_mw / max(avg_energy, 0.1))
        
        # Hardware compatibility bonus
        compatibility_bonus = 1.1 if self.is_hardware_compatible() else 0.8
        
        # Weighted combination
        fitness = (
            0.4 * accuracy +
            0.2 * param_efficiency +
            0.2 * memory_efficiency +
            0.1 * latency_score +
            0.1 * energy_score
        ) * compatibility_bonus
        
        return min(1.0, max(0.0, fitness))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary."""
        
        return {
            'architecture_id': self.architecture_id,
            'layers': [asdict(layer) for layer in self.layers],
            'input_shape': list(self.input_shape),
            'output_dim': self.output_dim,
            'generation': self.generation,
            'target_hardware': self.target_hardware.value,
            'max_parameters': self.max_parameters,
            'max_memory_mb': self.max_memory_mb,
            'max_latency_ms': self.max_latency_ms,
            'max_power_mw': self.max_power_mw,
            'global_quantization_bits': self.global_quantization_bits,
            'mixed_precision': self.mixed_precision,
            'quantization_scheme': self.quantization_scheme,
            'accuracy_scores': self.accuracy_scores,
            'latency_measurements': self.latency_measurements,
            'memory_usage': self.memory_usage,
            'energy_consumption': self.energy_consumption,
            'parent_architectures': self.parent_architectures,
            'estimated_parameters': self.estimate_total_parameters(),
            'estimated_memory_mb': self.estimate_memory_usage(),
            'fitness_score': self.calculate_fitness()
        }


class ArchitectureMutator:
    """Handles architecture mutations for evolutionary search."""
    
    def __init__(self, mutation_rate: float = 0.3):
        self.mutation_rate = mutation_rate
        self.mutation_types = [
            self._mutate_layer_dimensions,
            self._mutate_activation_functions,
            self._mutate_layer_types,
            self._mutate_quantization,
            self._add_layer,
            self._remove_layer,
            self._mutate_layer_parameters,
            self._mutate_skip_connections
        ]
    
    def mutate(self, architecture: ModelArchitecture) -> ModelArchitecture:
        """Apply mutations to create a new architecture."""
        
        # Deep copy the architecture
        new_layers = []
        for layer in architecture.layers:
            new_layer = LayerSpec(
                layer_type=layer.layer_type,
                input_dim=layer.input_dim,
                output_dim=layer.output_dim,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                activation=layer.activation,
                dropout_rate=layer.dropout_rate,
                use_bias=layer.use_bias,
                groups=layer.groups,
                dilation=layer.dilation,
                num_heads=layer.num_heads,
                embed_dim=layer.embed_dim,
                quantization_bits=layer.quantization_bits,
                quantize_weights=layer.quantize_weights,
                quantize_activations=layer.quantize_activations
            )
            new_layers.append(new_layer)
        
        mutated_arch = ModelArchitecture(
            architecture_id="",  # Will be auto-generated
            layers=new_layers,
            input_shape=architecture.input_shape,
            output_dim=architecture.output_dim,
            generation=architecture.generation + 1,
            target_hardware=architecture.target_hardware,
            max_parameters=architecture.max_parameters,
            max_memory_mb=architecture.max_memory_mb,
            max_latency_ms=architecture.max_latency_ms,
            max_power_mw=architecture.max_power_mw,
            global_quantization_bits=architecture.global_quantization_bits,
            mixed_precision=architecture.mixed_precision,
            quantization_scheme=architecture.quantization_scheme
        )
        
        mutated_arch.parent_architectures = [architecture.architecture_id]
        
        # Apply random mutations
        mutations_applied = []
        
        for mutation_func in self.mutation_types:
            if np.random.random() < self.mutation_rate:
                try:
                    mutation_name = mutation_func.__name__
                    mutation_func(mutated_arch)
                    mutations_applied.append(mutation_name)
                except Exception as e:
                    # Skip failed mutations
                    pass
        
        mutated_arch.mutation_history = [{
            'generation': mutated_arch.generation,
            'mutations_applied': mutations_applied,
            'parent_id': architecture.architecture_id
        }]
        
        return mutated_arch
    
    def _mutate_layer_dimensions(self, architecture: ModelArchitecture) -> None:
        """Mutate layer dimensions."""
        
        if not architecture.layers:
            return
        
        layer_idx = np.random.randint(0, len(architecture.layers))
        layer = architecture.layers[layer_idx]
        
        if layer.layer_type in [LayerType.LINEAR, LayerType.CONV2D, LayerType.ATTENTION]:
            # Mutate output dimension
            if layer.output_dim:
                mutation_factor = np.random.uniform(0.5, 2.0)
                new_dim = max(1, int(layer.output_dim * mutation_factor))
                
                # Hardware-aware dimension constraints
                if architecture.target_hardware in [HardwareTarget.ESP32, HardwareTarget.ARDUINO]:
                    new_dim = min(new_dim, 128)  # Limit for MCUs
                
                layer.output_dim = new_dim
    
    def _mutate_activation_functions(self, architecture: ModelArchitecture) -> None:
        """Mutate activation functions."""
        
        if not architecture.layers:
            return
        
        layer_idx = np.random.randint(0, len(architecture.layers))
        layer = architecture.layers[layer_idx]
        
        # Choose hardware-appropriate activations
        if architecture.target_hardware in [HardwareTarget.ESP32, HardwareTarget.ARDUINO]:
            # Prefer simple activations for MCUs
            activations = [ActivationType.RELU, ActivationType.HARD_SIGMOID, ActivationType.HARD_SWISH]
        else:
            activations = list(ActivationType)
        
        layer.activation = np.random.choice(activations)
    
    def _mutate_layer_types(self, architecture: ModelArchitecture) -> None:
        """Mutate layer types while preserving compatibility."""
        
        if not architecture.layers:
            return
        
        layer_idx = np.random.randint(0, len(architecture.layers))
        layer = architecture.layers[layer_idx]
        
        # Define compatible layer transitions
        compatible_types = {
            LayerType.LINEAR: [LayerType.LINEAR, LayerType.CONV1D],
            LayerType.CONV2D: [LayerType.CONV2D, LayerType.DEPTHWISE_CONV, LayerType.SEPARABLE_CONV],
            LayerType.CONV1D: [LayerType.CONV1D, LayerType.LINEAR],
            LayerType.ATTENTION: [LayerType.ATTENTION, LayerType.LINEAR],
            LayerType.LSTM: [LayerType.LSTM, LayerType.GRU, LayerType.RNN]
        }
        
        if layer.layer_type in compatible_types:
            possible_types = compatible_types[layer.layer_type]
            
            # Filter based on hardware constraints
            if architecture.target_hardware in [HardwareTarget.ESP32, HardwareTarget.ARDUINO]:
                # Remove complex layers for MCUs
                possible_types = [t for t in possible_types 
                                if t not in [LayerType.ATTENTION, LayerType.TRANSFORMER_BLOCK]]
            
            if possible_types:
                layer.layer_type = np.random.choice(possible_types)
    
    def _mutate_quantization(self, architecture: ModelArchitecture) -> None:
        """Mutate quantization parameters."""
        
        if not architecture.layers:
            return
        
        # Mutate global quantization
        if np.random.random() < 0.5:
            quantization_options = [4, 6, 8, 16]
            architecture.global_quantization_bits = np.random.choice(quantization_options)
        
        # Mutate layer-specific quantization
        layer_idx = np.random.randint(0, len(architecture.layers))
        layer = architecture.layers[layer_idx]
        
        quantization_options = [2, 4, 6, 8, 16]
        layer.quantization_bits = np.random.choice(quantization_options)
        
        layer.quantize_weights = np.random.choice([True, False])
        layer.quantize_activations = np.random.choice([True, False])
    
    def _add_layer(self, architecture: ModelArchitecture) -> None:
        """Add a new layer to the architecture."""
        
        if len(architecture.layers) >= 20:  # Maximum layers
            return
        
        # Choose insertion position
        insert_pos = np.random.randint(0, len(architecture.layers) + 1)
        
        # Determine input/output dimensions based on context
        input_dim = None
        output_dim = None
        
        if insert_pos > 0:
            prev_layer = architecture.layers[insert_pos - 1]
            input_dim = prev_layer.output_dim
        elif architecture.input_shape:
            input_dim = architecture.input_shape[-1]
        
        if insert_pos < len(architecture.layers):
            next_layer = architecture.layers[insert_pos]
            output_dim = next_layer.input_dim or input_dim
        else:
            output_dim = architecture.output_dim
        
        # Choose layer type based on hardware constraints
        if architecture.target_hardware in [HardwareTarget.ESP32, HardwareTarget.ARDUINO]:
            layer_types = [LayerType.LINEAR, LayerType.CONV1D, LayerType.RELU, LayerType.DROPOUT]
        else:
            layer_types = [LayerType.LINEAR, LayerType.CONV2D, LayerType.CONV1D, 
                          LayerType.ATTENTION, LayerType.BATCH_NORM, LayerType.DROPOUT]
        
        layer_type = np.random.choice(layer_types)
        
        # Create new layer
        new_layer = LayerSpec(
            layer_type=layer_type,
            input_dim=input_dim,
            output_dim=output_dim or input_dim,
            activation=np.random.choice(list(ActivationType)),
            dropout_rate=np.random.uniform(0.0, 0.3) if layer_type == LayerType.DROPOUT else None
        )
        
        # Insert the layer
        architecture.layers.insert(insert_pos, new_layer)
    
    def _remove_layer(self, architecture: ModelArchitecture) -> None:
        """Remove a layer from the architecture."""
        
        if len(architecture.layers) <= 2:  # Keep minimum layers
            return
        
        # Don't remove critical layers (first and last)
        removable_indices = list(range(1, len(architecture.layers) - 1))
        
        if removable_indices:
            remove_idx = np.random.choice(removable_indices)
            architecture.layers.pop(remove_idx)
    
    def _mutate_layer_parameters(self, architecture: ModelArchitecture) -> None:
        """Mutate layer-specific parameters."""
        
        if not architecture.layers:
            return
        
        layer_idx = np.random.randint(0, len(architecture.layers))
        layer = architecture.layers[layer_idx]
        
        # Mutate layer-specific parameters
        if layer.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
            if layer.kernel_size and isinstance(layer.kernel_size, int):
                layer.kernel_size = np.random.choice([1, 3, 5, 7])
            
            if np.random.random() < 0.3:
                layer.stride = np.random.choice([1, 2])
        
        elif layer.layer_type == LayerType.ATTENTION:
            if layer.num_heads:
                layer.num_heads = np.random.choice([1, 2, 4, 8])
        
        elif layer.layer_type == LayerType.DROPOUT:
            layer.dropout_rate = np.random.uniform(0.0, 0.5)
        
        # Mutate bias usage
        layer.use_bias = np.random.choice([True, False])
    
    def _mutate_skip_connections(self, architecture: ModelArchitecture) -> None:
        """Add or modify skip connections (simplified)."""
        
        if len(architecture.layers) < 3:
            return
        
        # Add residual block pattern
        if np.random.random() < 0.5:
            # Find suitable position for residual block
            for i in range(1, len(architecture.layers) - 1):
                layer = architecture.layers[i]
                if layer.layer_type == LayerType.LINEAR:
                    # Convert to residual block
                    layer.layer_type = LayerType.RESIDUAL_BLOCK
                    break


class HardwareAwareEvaluator:
    """Evaluates architectures for hardware compatibility and performance."""
    
    def __init__(self, target_hardware: HardwareTarget):
        self.target_hardware = target_hardware
        self.hardware_specs = self._get_hardware_specs()
        self.performance_predictor = self._initialize_performance_predictor()
        
    def _get_hardware_specs(self) -> Dict[str, Any]:
        """Get hardware specifications for target platform."""
        
        specs = {
            HardwareTarget.ESP32: {
                'max_ram_mb': 0.5,
                'max_flash_mb': 4.0,
                'cpu_freq_mhz': 240,
                'has_fpu': True,
                'has_simd': False,
                'max_power_mw': 100,
                'preferred_ops': ['add', 'mul', 'relu', 'conv2d_small']
            },
            HardwareTarget.STM32: {
                'max_ram_mb': 1.0,
                'max_flash_mb': 2.0,
                'cpu_freq_mhz': 180,
                'has_fpu': True,
                'has_simd': True,
                'max_power_mw': 50,
                'preferred_ops': ['add', 'mul', 'relu', 'conv1d']
            },
            HardwareTarget.RASPBERRY_PI: {
                'max_ram_mb': 1024,
                'max_flash_mb': 32000,
                'cpu_freq_mhz': 1500,
                'has_fpu': True,
                'has_simd': True,
                'max_power_mw': 3000,
                'preferred_ops': ['add', 'mul', 'conv2d', 'attention_small']
            },
            HardwareTarget.JETSON_NANO: {
                'max_ram_mb': 4096,
                'max_flash_mb': 64000,
                'cpu_freq_mhz': 1430,
                'has_fpu': True,
                'has_simd': True,
                'has_gpu': True,
                'max_power_mw': 10000,
                'preferred_ops': ['conv2d', 'attention', 'transformer']
            },
            HardwareTarget.ARDUINO: {
                'max_ram_mb': 0.032,  # 32KB
                'max_flash_mb': 0.256,  # 256KB
                'cpu_freq_mhz': 16,
                'has_fpu': False,
                'has_simd': False,
                'max_power_mw': 20,
                'preferred_ops': ['add', 'relu']
            }
        }
        
        return specs.get(self.target_hardware, specs[HardwareTarget.GENERIC_MCU])
    
    def _initialize_performance_predictor(self) -> RandomForestRegressor:
        """Initialize ML model for performance prediction."""
        
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
    
    def evaluate_architecture(self, architecture: ModelArchitecture) -> Dict[str, float]:
        """Evaluate architecture for target hardware."""
        
        evaluation = {
            'hardware_compatibility': self._evaluate_compatibility(architecture),
            'performance_score': self._estimate_performance(architecture),
            'efficiency_score': self._evaluate_efficiency(architecture),
            'complexity_score': self._evaluate_complexity(architecture),
            'quantization_score': self._evaluate_quantization(architecture)
        }
        
        # Overall score
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        evaluation['overall_score'] = sum(w * s for w, s in zip(weights, evaluation.values()))
        
        return evaluation
    
    def _evaluate_compatibility(self, architecture: ModelArchitecture) -> float:
        """Evaluate hardware compatibility."""
        
        score = 1.0
        
        # Memory constraints
        estimated_memory = architecture.estimate_memory_usage()
        max_memory = self.hardware_specs['max_ram_mb']
        
        if estimated_memory > max_memory:
            score *= 0.1  # Heavy penalty for memory overflow
        else:
            score *= (1.0 - estimated_memory / max_memory * 0.3)  # Penalty for high usage
        
        # Parameter constraints
        param_count = architecture.estimate_total_parameters()
        
        # Hardware-specific parameter limits
        if self.target_hardware == HardwareTarget.ARDUINO:
            max_params = 1000
        elif self.target_hardware == HardwareTarget.ESP32:
            max_params = 100000
        else:
            max_params = 1000000
        
        if param_count > max_params:
            score *= 0.2
        
        # Layer type compatibility
        for layer in architecture.layers:
            if layer.layer_type not in self._get_supported_layers():
                score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _get_supported_layers(self) -> List[LayerType]:
        """Get supported layer types for target hardware."""
        
        if self.target_hardware == HardwareTarget.ARDUINO:
            return [LayerType.LINEAR, LayerType.ACTIVATION, LayerType.DROPOUT]
        elif self.target_hardware == HardwareTarget.ESP32:
            return [LayerType.LINEAR, LayerType.CONV1D, LayerType.CONV2D, LayerType.ACTIVATION, 
                   LayerType.DROPOUT, LayerType.BATCH_NORM, LayerType.POOLING]
        else:
            return list(LayerType)  # All types supported
    
    def _estimate_performance(self, architecture: ModelArchitecture) -> float:
        """Estimate performance score."""
        
        # FLOPS-based estimation
        flops = architecture.estimate_total_flops()
        cpu_freq = self.hardware_specs['cpu_freq_mhz'] * 1000000  # Convert to Hz
        
        # Rough estimate: 1 FLOP = 10 CPU cycles (varies by operation)
        estimated_latency_ms = (flops * 10) / cpu_freq * 1000
        
        # Normalize latency score
        max_acceptable_latency = 500  # ms
        latency_score = max(0.0, 1.0 - estimated_latency_ms / max_acceptable_latency)
        
        return latency_score
    
    def _evaluate_efficiency(self, architecture: ModelArchitecture) -> float:
        """Evaluate computational efficiency."""
        
        params = architecture.estimate_total_parameters()
        memory = architecture.estimate_memory_usage()
        
        # Efficiency = performance per resource unit
        if params == 0 or memory == 0:
            return 0.0
        
        # Simplified efficiency metric
        efficiency = 1.0 / (1.0 + np.log10(max(params, 1)) + np.log10(max(memory, 0.001)))
        
        return max(0.0, min(1.0, efficiency))
    
    def _evaluate_complexity(self, architecture: ModelArchitecture) -> float:
        """Evaluate architecture complexity."""
        
        complexity_factors = []
        
        # Layer diversity
        layer_types = set(layer.layer_type for layer in architecture.layers)
        type_diversity = len(layer_types) / len(LayerType)
        complexity_factors.append(type_diversity)
        
        # Depth
        depth_score = min(1.0, len(architecture.layers) / 20)  # Normalize by max depth
        complexity_factors.append(depth_score)
        
        # Skip connections and advanced features
        advanced_layers = sum(1 for layer in architecture.layers 
                            if layer.layer_type in [LayerType.ATTENTION, LayerType.TRANSFORMER_BLOCK, 
                                                   LayerType.RESIDUAL_BLOCK])
        advanced_score = min(1.0, advanced_layers / max(len(architecture.layers), 1))
        complexity_factors.append(advanced_score)
        
        # Return inverse complexity for hardware-constrained targets
        complexity = np.mean(complexity_factors)
        
        if self.target_hardware in [HardwareTarget.ARDUINO, HardwareTarget.ESP32]:
            return 1.0 - complexity  # Prefer simpler architectures
        else:
            return complexity  # Complex architectures are okay
    
    def _evaluate_quantization(self, architecture: ModelArchitecture) -> float:
        """Evaluate quantization effectiveness."""
        
        score = 0.0
        
        # Global quantization bonus
        if architecture.global_quantization_bits <= 8:
            score += 0.5
        if architecture.global_quantization_bits <= 4:
            score += 0.3
        
        # Layer-specific quantization
        quantized_layers = sum(1 for layer in architecture.layers 
                             if layer.quantize_weights or layer.quantize_activations)
        
        if quantized_layers > 0:
            quantization_ratio = quantized_layers / len(architecture.layers)
            score += 0.2 * quantization_ratio
        
        # Mixed precision bonus
        if architecture.mixed_precision:
            score += 0.1
        
        return min(1.0, score)


class AutonomousNAS:
    """Autonomous Neural Architecture Search system."""
    
    def __init__(
        self,
        target_hardware: HardwareTarget = HardwareTarget.GENERIC_MCU,
        population_size: int = 20,
        max_generations: int = 50,
        mutation_rate: float = 0.3
    ):
        self.target_hardware = target_hardware
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        
        # Components
        self.mutator = ArchitectureMutator(mutation_rate)
        self.evaluator = HardwareAwareEvaluator(target_hardware)
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Evolution state
        self.population: List[ModelArchitecture] = []
        self.generation = 0
        self.evolution_history: List[Dict] = []
        self.best_architecture: Optional[ModelArchitecture] = None
        
        # Performance tracking
        self.evaluated_architectures: Dict[str, Dict] = {}
        self.training_data: List[Tuple] = []  # For performance prediction
        
        self.logger = logging.getLogger(__name__)
        self.monitoring = MonitoringSystem()
    
    def initialize_population(
        self,
        input_shape: Tuple[int, ...],
        output_dim: int,
        seed_architectures: Optional[List[ModelArchitecture]] = None
    ) -> None:
        """Initialize the architecture population."""
        
        self.population = []
        
        if seed_architectures:
            # Use provided seed architectures
            self.population.extend(seed_architectures[:self.population_size])
        
        # Fill remaining slots with random architectures
        while len(self.population) < self.population_size:
            arch = self._generate_random_architecture(input_shape, output_dim)
            self.population.append(arch)
        
        self.logger.info(f"Initialized population with {len(self.population)} architectures")
    
    def _generate_random_architecture(
        self,
        input_shape: Tuple[int, ...],
        output_dim: int
    ) -> ModelArchitecture:
        """Generate a random architecture within hardware constraints."""
        
        # Hardware-aware layer selection
        if self.target_hardware == HardwareTarget.ARDUINO:
            available_layers = [LayerType.LINEAR, LayerType.ACTIVATION]
            max_layers = 3
            max_dim = 32
        elif self.target_hardware == HardwareTarget.ESP32:
            available_layers = [LayerType.LINEAR, LayerType.CONV1D, LayerType.ACTIVATION, LayerType.DROPOUT]
            max_layers = 8
            max_dim = 128
        else:
            available_layers = [LayerType.LINEAR, LayerType.CONV2D, LayerType.CONV1D, 
                              LayerType.ATTENTION, LayerType.ACTIVATION, LayerType.DROPOUT]
            max_layers = 15
            max_dim = 512
        
        # Generate layers
        layers = []
        current_dim = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
        num_layers = np.random.randint(2, max_layers + 1)
        
        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                layer_type = LayerType.LINEAR
                layer_output_dim = output_dim
            else:
                layer_type = np.random.choice(available_layers)
                layer_output_dim = np.random.randint(8, min(max_dim, current_dim * 2))
            
            layer = LayerSpec(
                layer_type=layer_type,
                input_dim=current_dim,
                output_dim=layer_output_dim,
                activation=np.random.choice(list(ActivationType)) if layer_type != LayerType.ACTIVATION else None,
                dropout_rate=np.random.uniform(0.0, 0.3) if layer_type == LayerType.DROPOUT else None,
                quantization_bits=np.random.choice([4, 8, 16]),
                quantize_weights=np.random.choice([True, False]),
                quantize_activations=np.random.choice([True, False])
            )
            
            layers.append(layer)
            current_dim = layer_output_dim
        
        # Create architecture
        architecture = ModelArchitecture(
            architecture_id="",  # Auto-generated
            layers=layers,
            input_shape=input_shape,
            output_dim=output_dim,
            target_hardware=self.target_hardware,
            global_quantization_bits=np.random.choice([4, 8, 16]),
            mixed_precision=np.random.choice([True, False])
        )
        
        return architecture
    
    async def search_architectures(
        self,
        input_shape: Tuple[int, ...],
        output_dim: int,
        evaluation_budget: int = 1000,
        seed_architectures: Optional[List[ModelArchitecture]] = None
    ) -> Dict[str, Any]:
        """Main architecture search loop."""
        
        # Initialize population
        self.initialize_population(input_shape, output_dim, seed_architectures)
        
        start_time = time.time()
        evaluations_used = 0
        
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Evaluate population
            fitness_scores = []
            
            for arch in self.population:
                if evaluations_used >= evaluation_budget:
                    break
                
                # Check if already evaluated
                if arch.architecture_id in self.evaluated_architectures:
                    cached_eval = self.evaluated_architectures[arch.architecture_id]
                    fitness = cached_eval['overall_score']
                else:
                    # Evaluate architecture
                    evaluation = await self._evaluate_architecture(arch)
                    fitness = evaluation['overall_score']
                    
                    # Cache evaluation
                    self.evaluated_architectures[arch.architecture_id] = evaluation
                    evaluations_used += 1
                
                # Update architecture performance
                arch.accuracy_scores.append(fitness)
                fitness_scores.append(fitness)
            
            if not fitness_scores:
                break
            
            # Track best architecture
            best_idx = np.argmax(fitness_scores)
            current_best = self.population[best_idx]
            
            if (self.best_architecture is None or 
                current_best.calculate_fitness() > self.best_architecture.calculate_fitness()):
                self.best_architecture = current_best
            
            # Log progress
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.logger.info(f"Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Record evolution statistics
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'population_size': len(self.population),
                'evaluations_used': evaluations_used,
                'best_architecture_id': current_best.architecture_id
            })
            
            # Check for early stopping
            if self._should_stop_early():
                self.logger.info(f"Early stopping at generation {generation}")
                break
            
            # Evolve population
            await self._evolve_population(fitness_scores)
            
            # Update performance predictor
            if evaluations_used % 50 == 0:  # Update every 50 evaluations
                await self._update_performance_predictor()
        
        search_time = time.time() - start_time
        
        # Prepare results
        result = {
            'best_architecture': self.best_architecture.to_dict() if self.best_architecture else None,
            'search_statistics': {
                'generations': self.generation + 1,
                'evaluations_used': evaluations_used,
                'search_time_seconds': search_time,
                'final_population_size': len(self.population),
                'unique_architectures_found': len(self.evaluated_architectures),
                'convergence_achieved': self._check_convergence()
            },
            'evolution_history': self.evolution_history,
            'hardware_target': self.target_hardware.value,
            'top_architectures': await self._get_top_architectures(10)
        }
        
        return result
    
    async def _evaluate_architecture(self, architecture: ModelArchitecture) -> Dict[str, float]:
        """Evaluate a single architecture."""
        
        # Hardware-aware evaluation
        hardware_eval = self.evaluator.evaluate_architecture(architecture)
        
        # Simulated performance evaluation (in real implementation, would train/test)
        simulated_accuracy = self._simulate_accuracy(architecture)
        simulated_latency = self._simulate_latency(architecture)
        simulated_energy = self._simulate_energy(architecture)
        
        # Update architecture performance metrics
        architecture.accuracy_scores.append(simulated_accuracy)
        architecture.latency_measurements.append(simulated_latency)
        architecture.energy_consumption.append(simulated_energy)
        
        # Combine evaluations
        combined_eval = hardware_eval.copy()
        combined_eval['accuracy'] = simulated_accuracy
        combined_eval['latency_ms'] = simulated_latency
        combined_eval['energy_mj'] = simulated_energy
        
        # Recalculate overall score with real performance
        performance_weight = 0.4
        hardware_weight = 0.6
        
        combined_eval['overall_score'] = (
            performance_weight * simulated_accuracy +
            hardware_weight * hardware_eval['overall_score']
        )
        
        # Store for training performance predictor
        self.training_data.append((architecture, combined_eval))
        
        return combined_eval
    
    def _simulate_accuracy(self, architecture: ModelArchitecture) -> float:
        """Simulate accuracy based on architecture characteristics."""
        
        # Heuristic accuracy estimation
        base_accuracy = 0.7
        
        # Parameter count bonus (up to a point)
        param_count = architecture.estimate_total_parameters()
        param_bonus = min(0.15, np.log10(max(param_count, 1)) / 20)
        
        # Depth bonus
        depth_bonus = min(0.1, len(architecture.layers) / 50)
        
        # Architecture quality bonuses
        has_attention = any(layer.layer_type in [LayerType.ATTENTION, LayerType.MULTIHEAD_ATTENTION] 
                          for layer in architecture.layers)
        has_residual = any(layer.layer_type == LayerType.RESIDUAL_BLOCK 
                         for layer in architecture.layers)
        has_normalization = any(layer.layer_type in [LayerType.BATCH_NORM, LayerType.LAYER_NORM]
                              for layer in architecture.layers)
        
        quality_bonus = (
            (0.05 if has_attention else 0.0) +
            (0.03 if has_residual else 0.0) +
            (0.02 if has_normalization else 0.0)
        )
        
        # Hardware penalty for overly complex models
        if not architecture.is_hardware_compatible():
            hardware_penalty = 0.2
        else:
            hardware_penalty = 0.0
        
        # Quantization impact
        quant_impact = -0.02 * (8 - architecture.global_quantization_bits) / 8  # Lower bits = lower accuracy
        
        accuracy = base_accuracy + param_bonus + depth_bonus + quality_bonus - hardware_penalty + quant_impact
        
        # Add noise
        accuracy += np.random.normal(0, 0.02)
        
        return max(0.0, min(1.0, accuracy))
    
    def _simulate_latency(self, architecture: ModelArchitecture) -> float:
        """Simulate latency based on architecture characteristics."""
        
        # FLOPS-based latency estimation
        flops = architecture.estimate_total_flops()
        
        # Hardware-specific performance
        hardware_specs = self.evaluator.hardware_specs
        cpu_ops_per_second = hardware_specs['cpu_freq_mhz'] * 1000000  # Base ops per second
        
        # Operation efficiency based on hardware capabilities
        if hardware_specs.get('has_simd', False):
            cpu_ops_per_second *= 4  # SIMD speedup
        if hardware_specs.get('has_fpu', False):
            cpu_ops_per_second *= 2  # FPU speedup
        
        # Layer-specific performance factors
        performance_factor = 1.0
        for layer in architecture.layers:
            if layer.layer_type == LayerType.ATTENTION:
                performance_factor *= 2.0  # Attention is expensive
            elif layer.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
                performance_factor *= 1.5  # Convolution overhead
        
        # Quantization speedup
        quant_speedup = 16 / architecture.global_quantization_bits
        
        estimated_latency_ms = (flops * performance_factor) / (cpu_ops_per_second * quant_speedup) * 1000
        
        # Add noise and minimum latency
        estimated_latency_ms += np.random.uniform(5, 15)
        
        return max(1.0, estimated_latency_ms)
    
    def _simulate_energy(self, architecture: ModelArchitecture) -> float:
        """Simulate energy consumption."""
        
        # Base energy consumption
        base_energy = 5.0  # mJ
        
        # Parameter-based energy
        param_energy = architecture.estimate_total_parameters() * 0.001  # 1µJ per parameter
        
        # FLOPS-based energy
        flops_energy = architecture.estimate_total_flops() * 0.0001  # 0.1µJ per FLOP
        
        # Memory access energy
        memory_energy = architecture.estimate_memory_usage() * 2.0  # 2mJ per MB
        
        # Hardware-specific efficiency
        max_power = self.evaluator.hardware_specs.get('max_power_mw', 100)
        efficiency_factor = min(1.0, 50 / max_power)  # More efficient on lower power devices
        
        total_energy = (base_energy + param_energy + flops_energy + memory_energy) * efficiency_factor
        
        # Add noise
        total_energy += np.random.uniform(0, total_energy * 0.1)
        
        return max(0.1, total_energy)
    
    async def _evolve_population(self, fitness_scores: List[float]) -> None:
        """Evolve the population for next generation."""
        
        if not fitness_scores:
            return
        
        # Selection: Tournament selection
        elite_size = max(2, self.population_size // 5)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite_architectures = [self.population[i] for i in elite_indices]
        
        # Generate new population
        new_population = elite_architectures.copy()  # Keep elite
        
        while len(new_population) < self.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover and mutation
            if np.random.random() < 0.7:  # Crossover probability
                child = self._crossover_architectures(parent1, parent2)
            else:
                child = parent1
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self.mutator.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
    
    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> ModelArchitecture:
        """Tournament selection for parent selection."""
        
        tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]
    
    def _crossover_architectures(self, parent1: ModelArchitecture, parent2: ModelArchitecture) -> ModelArchitecture:
        """Create child architecture from two parents."""
        
        # Simple layer-wise crossover
        child_layers = []
        min_layers = min(len(parent1.layers), len(parent2.layers))
        
        for i in range(min_layers):
            if np.random.random() < 0.5:
                child_layers.append(parent1.layers[i])
            else:
                child_layers.append(parent2.layers[i])
        
        # Add remaining layers from the longer parent
        if len(parent1.layers) > min_layers:
            child_layers.extend(parent1.layers[min_layers:])
        elif len(parent2.layers) > min_layers:
            child_layers.extend(parent2.layers[min_layers:])
        
        # Create child architecture
        child = ModelArchitecture(
            architecture_id="",  # Auto-generated
            layers=child_layers,
            input_shape=parent1.input_shape,
            output_dim=parent1.output_dim,
            generation=max(parent1.generation, parent2.generation) + 1,
            target_hardware=parent1.target_hardware,
            global_quantization_bits=np.random.choice([parent1.global_quantization_bits, parent2.global_quantization_bits])
        )
        
        child.parent_architectures = [parent1.architecture_id, parent2.architecture_id]
        
        return child
    
    async def _update_performance_predictor(self) -> None:
        """Update performance prediction model."""
        
        if len(self.training_data) < 20:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for architecture, evaluation in self.training_data[-100:]:  # Use last 100 examples
                features = self._extract_architecture_features(architecture)
                X.append(features)
                y.append(evaluation['overall_score'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Train predictor
            if len(X) > 10:
                self.performance_predictor.fit(X, y)
                
                # Log training score
                score = self.performance_predictor.score(X, y)
                self.logger.info(f"Performance predictor R² score: {score:.3f}")
                
        except Exception as e:
            self.logger.warning(f"Failed to update performance predictor: {e}")
    
    def _extract_architecture_features(self, architecture: ModelArchitecture) -> List[float]:
        """Extract features from architecture for ML predictor."""
        
        features = []
        
        # Basic architecture properties
        features.extend([
            len(architecture.layers),
            architecture.estimate_total_parameters(),
            architecture.estimate_memory_usage(),
            architecture.estimate_total_flops(),
            architecture.global_quantization_bits,
            int(architecture.mixed_precision)
        ])
        
        # Layer type distribution
        layer_type_counts = defaultdict(int)
        for layer in architecture.layers:
            layer_type_counts[layer.layer_type] += 1
        
        # Convert to feature vector (fixed size)
        for layer_type in LayerType:
            features.append(layer_type_counts[layer_type])
        
        # Activation function distribution
        activation_counts = defaultdict(int)
        for layer in architecture.layers:
            if layer.activation:
                activation_counts[layer.activation] += 1
        
        for activation_type in ActivationType:
            features.append(activation_counts[activation_type])
        
        # Hardware target encoding
        hardware_encoding = [0] * len(HardwareTarget)
        hardware_index = list(HardwareTarget).index(architecture.target_hardware)
        hardware_encoding[hardware_index] = 1
        features.extend(hardware_encoding)
        
        return features
    
    def _should_stop_early(self) -> bool:
        """Check if search should stop early."""
        
        if len(self.evolution_history) < 10:
            return False
        
        # Check for stagnation
        recent_best = [h['best_fitness'] for h in self.evolution_history[-10:]]
        improvement = recent_best[-1] - recent_best[0]
        
        return improvement < 0.01  # Stop if improvement < 1% over 10 generations
    
    def _check_convergence(self) -> bool:
        """Check if population has converged."""
        
        if len(self.evolution_history) < 5:
            return False
        
        recent_fitness = [h['best_fitness'] for h in self.evolution_history[-5:]]
        return np.std(recent_fitness) < 0.005  # Low variance indicates convergence
    
    async def _get_top_architectures(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N architectures from evaluation cache."""
        
        if not self.evaluated_architectures:
            return []
        
        # Sort by overall score
        sorted_archs = sorted(
            self.evaluated_architectures.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        top_architectures = []
        
        for arch_id, evaluation in sorted_archs[:n]:
            # Find architecture object
            arch_obj = None
            for arch in self.population:
                if arch.architecture_id == arch_id:
                    arch_obj = arch
                    break
            
            if arch_obj:
                arch_data = arch_obj.to_dict()
                arch_data['evaluation'] = evaluation
                top_architectures.append(arch_data)
        
        return top_architectures
    
    def save_search_state(self, filepath: Path) -> None:
        """Save NAS search state."""
        
        state = {
            'population': [arch.to_dict() for arch in self.population],
            'generation': self.generation,
            'evolution_history': self.evolution_history,
            'evaluated_architectures': self.evaluated_architectures,
            'best_architecture': self.best_architecture.to_dict() if self.best_architecture else None,
            'target_hardware': self.target_hardware.value,
            'search_params': {
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'mutation_rate': self.mutation_rate
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_search_state(self, filepath: Path) -> None:
        """Load NAS search state."""
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore state
        self.generation = state.get('generation', 0)
        self.evolution_history = state.get('evolution_history', [])
        self.evaluated_architectures = state.get('evaluated_architectures', {})
        
        # Restore target hardware
        if 'target_hardware' in state:
            self.target_hardware = HardwareTarget(state['target_hardware'])
            self.evaluator = HardwareAwareEvaluator(self.target_hardware)
        
        # Restore search parameters
        if 'search_params' in state:
            params = state['search_params']
            self.population_size = params.get('population_size', self.population_size)
            self.max_generations = params.get('max_generations', self.max_generations)
            self.mutation_rate = params.get('mutation_rate', self.mutation_rate)


# Factory functions and utilities
async def create_autonomous_nas(
    target_hardware: HardwareTarget = HardwareTarget.GENERIC_MCU,
    population_size: int = 20,
    max_generations: int = 30
) -> AutonomousNAS:
    """Factory function to create autonomous NAS system."""
    
    nas = AutonomousNAS(
        target_hardware=target_hardware,
        population_size=population_size,
        max_generations=max_generations
    )
    
    return nas


async def run_architecture_search_experiment(
    input_shape: Tuple[int, ...],
    output_dim: int,
    target_hardware: HardwareTarget = HardwareTarget.ESP32,
    evaluation_budget: int = 300,
    population_size: int = 15
) -> Dict[str, Any]:
    """Run architecture search experiment."""
    
    try:
        # Create NAS system
        nas = await create_autonomous_nas(
            target_hardware=target_hardware,
            population_size=population_size,
            max_generations=20
        )
        
        # Run search
        results = await nas.search_architectures(
            input_shape=input_shape,
            output_dim=output_dim,
            evaluation_budget=evaluation_budget
        )
        
        return {
            'search_results': results,
            'success': True
        }
        
    except Exception as e:
        return {
            'error': f"Architecture search failed: {e}",
            'success': False
        }


# Example usage and demonstration
if __name__ == "__main__":
    async def main():
        # Define search problem
        input_shape = (1, 28, 28)  # MNIST-like input
        output_dim = 10  # 10 classes
        target_hardware = HardwareTarget.ESP32
        
        print("Starting Autonomous Neural Architecture Search...")
        print("=" * 60)
        print(f"Input Shape: {input_shape}")
        print(f"Output Dimension: {output_dim}")
        print(f"Target Hardware: {target_hardware.value}")
        
        # Run architecture search
        results = await run_architecture_search_experiment(
            input_shape=input_shape,
            output_dim=output_dim,
            target_hardware=target_hardware,
            evaluation_budget=200,
            population_size=12
        )
        
        if results['success']:
            search_results = results['search_results']
            
            print("\nSearch Results:")
            print("-" * 40)
            stats = search_results['search_statistics']
            print(f"Generations: {stats['generations']}")
            print(f"Evaluations Used: {stats['evaluations_used']}")
            print(f"Search Time: {stats['search_time_seconds']:.1f}s")
            print(f"Unique Architectures: {stats['unique_architectures_found']}")
            print(f"Convergence: {'Yes' if stats['convergence_achieved'] else 'No'}")
            
            if search_results['best_architecture']:
                print("\nBest Architecture Found:")
                print("-" * 40)
                best_arch = search_results['best_architecture']
                print(f"Architecture ID: {best_arch['architecture_id']}")
                print(f"Fitness Score: {best_arch['fitness_score']:.4f}")
                print(f"Parameters: {best_arch['estimated_parameters']:,}")
                print(f"Memory Usage: {best_arch['estimated_memory_mb']:.2f} MB")
                print(f"Generation: {best_arch['generation']}")
                print(f"Layers: {len(best_arch['layers'])}")
                
                print("\nLayer Architecture:")
                for i, layer in enumerate(best_arch['layers']):
                    print(f"  {i+1}. {layer['layer_type']}: {layer.get('input_dim', '?')} -> {layer.get('output_dim', '?')}")
                    if layer.get('activation'):
                        print(f"     Activation: {layer['activation']}")
            
            print("\nTop 5 Architectures:")
            print("-" * 40)
            top_archs = search_results.get('top_architectures', [])[:5]
            for i, arch in enumerate(top_archs):
                print(f"{i+1}. ID: {arch['architecture_id'][:8]}... "
                      f"Fitness: {arch['fitness_score']:.3f}, "
                      f"Params: {arch['estimated_parameters']:,}, "
                      f"Layers: {len(arch['layers'])}")
        else:
            print(f"Search failed: {results['error']}")
    
    # Run demonstration
    asyncio.run(main())