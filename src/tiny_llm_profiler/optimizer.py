"""
Optimization tools for improving LLM performance on edge devices.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from .models import QuantizedModel
from .results import ProfileResults


@dataclass
class OptimalConfiguration:
    """Optimal operating configuration for a platform."""
    cpu_freq_mhz: int
    voltage_v: float
    memory_config: Dict[str, Any]
    energy_reduction: float
    performance_impact: float


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    original_performance: float
    optimized_performance: float
    improvement_factor: float
    optimizations_applied: List[str]


class EnergyOptimizer:
    """Optimize energy consumption for edge deployment."""
    
    def __init__(self):
        self.platform_configs = {
            "esp32": {
                "min_freq_mhz": 80,
                "max_freq_mhz": 240,
                "voltage_range": (2.7, 3.6),
                "power_curves": self._esp32_power_curve
            },
            "stm32f4": {
                "min_freq_mhz": 16,
                "max_freq_mhz": 168,
                "voltage_range": (1.8, 3.6),
                "power_curves": self._stm32_power_curve
            },
            "stm32f7": {
                "min_freq_mhz": 16,
                "max_freq_mhz": 216,
                "voltage_range": (1.8, 3.6),
                "power_curves": self._stm32_power_curve
            },
            "rp2040": {
                "min_freq_mhz": 10,
                "max_freq_mhz": 133,
                "voltage_range": (1.8, 3.3),
                "power_curves": self._rp2040_power_curve
            }
        }
    
    def find_optimal_configuration(
        self,
        model: QuantizedModel,
        platform: str,
        constraints: Dict[str, float],
        optimize_for: str = "energy_per_token"
    ) -> OptimalConfiguration:
        """
        Find optimal operating configuration for energy efficiency.
        
        Args:
            model: QuantizedModel to optimize for
            platform: Target platform
            constraints: Performance constraints (min_tokens_per_second, max_latency_ms)
            optimize_for: Optimization target (energy_per_token, total_energy, power_consumption)
            
        Returns:
            OptimalConfiguration with recommended settings
        """
        if platform not in self.platform_configs:
            raise ValueError(f"Unsupported platform: {platform}")
        
        config = self.platform_configs[platform]
        
        # Search space for optimization
        freq_range = np.linspace(config["min_freq_mhz"], config["max_freq_mhz"], 20)
        voltage_range = np.linspace(*config["voltage_range"], 10)
        
        best_config = None
        best_score = float('inf')
        
        for freq in freq_range:
            for voltage in voltage_range:
                # Estimate performance at this configuration
                performance_scaling = freq / config["max_freq_mhz"]
                # Estimate performance based on frequency scaling
                estimated_tps = 10.0 * performance_scaling  # Simplified estimation
                estimated_latency = 100.0 / performance_scaling  # Simplified estimation
                
                # Check constraints
                meets_constraints = True
                if "min_tokens_per_second" in constraints:
                    if estimated_tps < constraints["min_tokens_per_second"]:
                        meets_constraints = False
                        
                if "max_latency_ms" in constraints:
                    if estimated_latency > constraints["max_latency_ms"]:
                        meets_constraints = False
                
                if not meets_constraints:
                    continue
                
                # Calculate energy metrics
                power_mw = config["power_curves"](freq, voltage)
                energy_per_token = power_mw / (estimated_tps * 1000)  # mJ per token
                
                # Score based on optimization target
                if optimize_for == "energy_per_token":
                    score = energy_per_token
                elif optimize_for == "total_energy":
                    score = power_mw
                else:  # power_consumption
                    score = power_mw
                
                if score < best_score:
                    best_score = score
                    baseline_energy = config["power_curves"](config["max_freq_mhz"], config["voltage_range"][1])
                    energy_reduction = 1.0 - (power_mw / baseline_energy)
                    
                    best_config = OptimalConfiguration(
                        cpu_freq_mhz=int(freq),
                        voltage_v=voltage,
                        memory_config={"use_cache_optimization": True},
                        energy_reduction=energy_reduction,
                        performance_impact=1.0 - performance_scaling
                    )
        
        return best_config or OptimalConfiguration(
            cpu_freq_mhz=config["max_freq_mhz"],
            voltage_v=config["voltage_range"][1],
            memory_config={},
            energy_reduction=0.0,
            performance_impact=0.0
        )
                
                # Check constraints
                if "min_tokens_per_second" in constraints:
                    if estimated_tps < constraints["min_tokens_per_second"]:
                        continue
                
                if "max_latency_ms" in constraints:
                    estimated_latency = 1000.0 / estimated_tps if estimated_tps > 0 else float('inf')
                    if estimated_latency > constraints["max_latency_ms"]:
                        continue
                
                # Calculate power consumption
                power_mw = config["power_curves"](freq, voltage)
                energy_per_token = power_mw / estimated_tps if estimated_tps > 0 else float('inf')
                
                # Score based on optimization target
                if optimize_for == "energy_per_token":
                    score = energy_per_token
                elif optimize_for == "total_energy":
                    score = power_mw
                elif optimize_for == "power_consumption":
                    score = power_mw
                else:
                    score = energy_per_token
                
                if score < best_score:
                    best_score = score
                    best_config = {
                        "freq": freq,
                        "voltage": voltage,
                        "power": power_mw,
                        "energy_per_token": energy_per_token
                    }
        
        if best_config is None:
            # Fallback to maximum performance
            best_config = {
                "freq": config["max_freq_mhz"],
                "voltage": config["voltage_range"][1],
                "power": config["power_curves"](config["max_freq_mhz"], config["voltage_range"][1]),
                "energy_per_token": 0
            }
        
        # Calculate energy reduction compared to max frequency
        baseline_power = config["power_curves"](config["max_freq_mhz"], config["voltage_range"][1])
        energy_reduction = (baseline_power - best_config["power"]) / baseline_power
        
        return OptimalConfiguration(
            cpu_freq_mhz=int(best_config["freq"]),
            voltage_v=best_config["voltage"],
            memory_config={"enable_power_saving": True},
            energy_reduction=energy_reduction,
            performance_impact=best_config["freq"] / config["max_freq_mhz"]
        )
    
    def _esp32_power_curve(self, freq_mhz: float, voltage_v: float) -> float:
        """ESP32 power consumption model."""
        # Simplified power model: P = C * V^2 * f + P_static
        capacitance = 1e-11  # Effective switching capacitance
        static_power = 10  # Static power in mW
        
        dynamic_power = capacitance * (voltage_v ** 2) * (freq_mhz * 1e6) * 1000  # Convert to mW
        return dynamic_power + static_power
    
    def _stm32_power_curve(self, freq_mhz: float, voltage_v: float) -> float:
        """STM32 power consumption model."""
        # ARM Cortex-M power model
        capacitance = 8e-12
        static_power = 5
        
        dynamic_power = capacitance * (voltage_v ** 2) * (freq_mhz * 1e6) * 1000
        return dynamic_power + static_power
    
    def _rp2040_power_curve(self, freq_mhz: float, voltage_v: float) -> float:
        """RP2040 power consumption model."""
        capacitance = 6e-12
        static_power = 8
        
        dynamic_power = capacitance * (voltage_v ** 2) * (freq_mhz * 1e6) * 1000
        return dynamic_power + static_power


class MemoryOptimizer:
    """Optimize memory usage for constrained devices."""
    
    def __init__(self):
        self.optimization_techniques = [
            "kv_cache_quantization",
            "activation_checkpointing", 
            "flash_attention_lite",
            "weight_sharing",
            "dynamic_batching",
            "memory_pooling"
        ]
    
    def analyze_memory_usage(self, model: QuantizedModel) -> 'MemoryMap':
        """Analyze memory layout and usage patterns."""
        return MemoryMap(
            model_weights_kb=model.size_mb * 1024,
            activation_memory_kb=self._estimate_activation_memory(model),
            kv_cache_kb=self._estimate_kv_cache_memory(model),
            scratch_memory_kb=50,  # Estimated scratch space
            total_memory_kb=0  # Will be calculated
        )
    
    def optimize_layout(self, model: QuantizedModel, constraints: Dict[str, float]) -> OptimizationResult:
        """
        Optimize memory layout for given constraints.
        
        Args:
            model: QuantizedModel to optimize
            constraints: Memory constraints (ram_limit_kb, flash_limit_kb)
            
        Returns:
            OptimizationResult with memory optimizations
        """
        original_memory = self._estimate_total_memory(model)
        optimizations = []
        current_memory = original_memory
        
        # Apply optimization techniques
        if "ram_limit_kb" in constraints:
            ram_limit = constraints["ram_limit_kb"]
            
            if current_memory > ram_limit:
                # Try KV cache quantization
                if "kv_cache_quantization" in self.optimization_techniques:
                    kv_savings = self._estimate_kv_cache_memory(model) * 0.5  # 50% reduction
                    current_memory -= kv_savings
                    optimizations.append("kv_cache_quantization")
                
                # Try activation checkpointing
                if current_memory > ram_limit and "activation_checkpointing" in self.optimization_techniques:
                    activation_savings = self._estimate_activation_memory(model) * 0.6  # 60% reduction
                    current_memory -= activation_savings
                    optimizations.append("activation_checkpointing")
                
                # Try flash attention lite
                if current_memory > ram_limit and "flash_attention_lite" in self.optimization_techniques:
                    attention_savings = self._estimate_activation_memory(model) * 0.3  # 30% reduction
                    current_memory -= attention_savings
                    optimizations.append("flash_attention_lite")
        
        improvement_factor = original_memory / current_memory if current_memory > 0 else 1.0
        
        return OptimizationResult(
            original_performance=original_memory,
            optimized_performance=current_memory,
            improvement_factor=improvement_factor,
            optimizations_applied=optimizations
        )
    
    def _estimate_activation_memory(self, model: QuantizedModel) -> float:
        """Estimate activation memory requirements."""
        # Simplified estimation based on model parameters
        return max(model.size_mb * 1024 * 0.1, 20)  # At least 20KB for small models
    
    def _estimate_kv_cache_memory(self, model: QuantizedModel) -> float:
        """Estimate KV cache memory requirements."""
        # Estimate based on context length and hidden size
        context_length = getattr(model, 'context_length', 512)
        hidden_size = getattr(model, 'hidden_size', 768)
        return (context_length * hidden_size * 2 * 2) / 1024  # 2 for K+V, 2 bytes per value
    
    def _estimate_total_memory(self, model: QuantizedModel) -> float:
        """Estimate total memory requirements."""
        weights = model.size_mb * 1024
        activations = self._estimate_activation_memory(model)
        kv_cache = self._estimate_kv_cache_memory(model)
        scratch = 50
        
        return weights + activations + kv_cache + scratch


@dataclass
class MemoryMap:
    """Memory usage breakdown."""
    model_weights_kb: float
    activation_memory_kb: float
    kv_cache_kb: float
    scratch_memory_kb: float
    total_memory_kb: float
    
    def __post_init__(self):
        self.total_memory_kb = (
            self.model_weights_kb + 
            self.activation_memory_kb + 
            self.kv_cache_kb + 
            self.scratch_memory_kb
        )
    
    def visualize(self, output_path: str):
        """Create visualization of memory layout."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Warning: Plotly not available, skipping visualization")
            return
        
        labels = ["Model Weights", "Activations", "KV Cache", "Scratch"]
        values = [self.model_weights_kb, self.activation_memory_kb, 
                 self.kv_cache_kb, self.scratch_memory_kb]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            title="Memory Usage Breakdown"
        )])
        
        fig.write_html(output_path)


class PlatformOptimizer:
    """Platform-specific optimizations."""
    
    def __init__(self, platform: str):
        self.platform = platform
        self.optimizations = {
            "esp32": self._esp32_optimizations,
            "stm32f4": self._arm_cortex_m_optimizations,
            "stm32f7": self._arm_cortex_m7_optimizations,
            "rp2040": self._arm_cortex_m0_optimizations,
            "cortex-m7": self._arm_cortex_m7_optimizations
        }
    
    def optimize(self, model: QuantizedModel, **kwargs) -> QuantizedModel:
        """Apply platform-specific optimizations."""
        if self.platform not in self.optimizations:
            return model  # No optimizations available
        
        optimizer_func = self.optimizations[self.platform]
        return optimizer_func(model, **kwargs)
    
    def _esp32_optimizations(self, model: QuantizedModel, **kwargs) -> QuantizedModel:
        """ESP32-specific optimizations."""
        # Create optimized copy
        optimized = QuantizedModel(
            name=f"{model.name}_esp32_opt",
            size_mb=model.size_mb,
            quantization=model.quantization
        )
        
        # ESP32 optimizations
        if kwargs.get("use_psram", False):
            optimized.config["use_psram"] = True
            optimized.config["psram_cache_strategy"] = "sequential"
        
        if kwargs.get("use_dual_core", True):
            optimized.config["enable_dual_core"] = True
            optimized.config["core_affinity"] = "inference_core_1"
        
        if kwargs.get("optimize_flash_access", True):
            optimized.config["flash_mode"] = "qio"
            optimized.config["flash_freq"] = "80m"
        
        return optimized
    
    def _arm_cortex_m_optimizations(self, model: QuantizedModel, **kwargs) -> QuantizedModel:
        """ARM Cortex-M4 optimizations."""
        optimized = QuantizedModel(
            name=f"{model.name}_cortex_m4_opt",
            size_mb=model.size_mb,
            quantization=model.quantization
        )
        
        # ARM DSP optimizations
        if kwargs.get("use_dsp_instructions", True):
            optimized.config["enable_dsp"] = True
            optimized.config["simd_width"] = 4
        
        if kwargs.get("use_fpu", True):
            optimized.config["enable_fpu"] = True
            optimized.config["fp_precision"] = "single"
        
        optimized.config["loop_unrolling"] = kwargs.get("loop_unrolling", 2)
        
        return optimized
    
    def _arm_cortex_m7_optimizations(self, model: QuantizedModel, **kwargs) -> QuantizedModel:
        """ARM Cortex-M7 optimizations."""
        optimized = QuantizedModel(
            name=f"{model.name}_cortex_m7_opt",
            size_mb=model.size_mb,
            quantization=model.quantization
        )
        
        # M7 has better cache and FPU
        optimized.config["enable_dsp"] = kwargs.get("use_dsp_instructions", True)
        optimized.config["enable_fpu"] = kwargs.get("use_fpu", True)
        optimized.config["enable_cache"] = True
        optimized.config["cache_policy"] = "write_through"
        optimized.config["loop_unrolling"] = kwargs.get("loop_unrolling", 4)
        optimized.config["simd_width"] = 8  # Better SIMD than M4
        
        return optimized
    
    def _arm_cortex_m0_optimizations(self, model: QuantizedModel, **kwargs) -> QuantizedModel:
        """ARM Cortex-M0+ (RP2040) optimizations."""
        optimized = QuantizedModel(
            name=f"{model.name}_rp2040_opt", 
            size_mb=model.size_mb,
            quantization=model.quantization
        )
        
        # M0+ is simpler, focus on basic optimizations
        optimized.config["enable_dual_core"] = True
        optimized.config["optimize_for_size"] = True
        optimized.config["use_pio"] = kwargs.get("use_pio", False)  # Programmable I/O
        
        return optimized