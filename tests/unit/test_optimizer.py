"""
Unit tests for the optimizer module.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from pathlib import Path

from tiny_llm_profiler.optimizer import (
    EnergyOptimizer,
    MemoryOptimizer,
    PlatformOptimizer,
    OptimalConfiguration,
    OptimizationResult,
    MemoryMap
)
from tiny_llm_profiler.models import QuantizedModel


class TestOptimalConfiguration:
    """Test the OptimalConfiguration dataclass."""
    
    def test_creation(self):
        """Test configuration creation."""
        config = OptimalConfiguration(
            cpu_freq_mhz=120,
            voltage_v=3.3,
            memory_config={"enable_power_saving": True},
            energy_reduction=0.25,
            performance_impact=0.8
        )
        
        assert config.cpu_freq_mhz == 120
        assert config.voltage_v == 3.3
        assert config.memory_config["enable_power_saving"] is True
        assert config.energy_reduction == 0.25
        assert config.performance_impact == 0.8


class TestOptimizationResult:
    """Test the OptimizationResult dataclass."""
    
    def test_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            original_performance=100.0,
            optimized_performance=80.0,
            improvement_factor=1.25,
            optimizations_applied=["kv_cache_quantization", "activation_checkpointing"]
        )
        
        assert result.original_performance == 100.0
        assert result.optimized_performance == 80.0
        assert result.improvement_factor == 1.25
        assert len(result.optimizations_applied) == 2


class TestMemoryMap:
    """Test the MemoryMap dataclass."""
    
    def test_creation_and_post_init(self):
        """Test memory map creation and automatic total calculation."""
        memory_map = MemoryMap(
            model_weights_kb=1000.0,
            activation_memory_kb=200.0,
            kv_cache_kb=300.0,
            scratch_memory_kb=50.0,
            total_memory_kb=0.0  # Should be calculated
        )
        
        # Post-init should calculate total
        assert memory_map.total_memory_kb == 1550.0
    
    @patch('plotly.graph_objects.Figure')
    def test_visualize(self, mock_figure):
        """Test memory map visualization."""
        memory_map = MemoryMap(
            model_weights_kb=1000.0,
            activation_memory_kb=200.0,
            kv_cache_kb=300.0,
            scratch_memory_kb=50.0,
            total_memory_kb=0.0
        )
        
        mock_fig_instance = Mock()
        mock_figure.return_value = mock_fig_instance
        
        memory_map.visualize("test_output.html")
        
        mock_figure.assert_called_once()
        mock_fig_instance.write_html.assert_called_once_with("test_output.html")


class TestEnergyOptimizer:
    """Test the EnergyOptimizer class."""
    
    def test_initialization(self):
        """Test energy optimizer initialization."""
        optimizer = EnergyOptimizer()
        
        assert "esp32" in optimizer.platform_configs
        assert "stm32f4" in optimizer.platform_configs
        assert "rp2040" in optimizer.platform_configs
        
        # Check ESP32 config structure
        esp32_config = optimizer.platform_configs["esp32"]
        assert "min_freq_mhz" in esp32_config
        assert "max_freq_mhz" in esp32_config
        assert "voltage_range" in esp32_config
        assert "power_curves" in esp32_config
    
    def test_find_optimal_configuration_esp32(self):
        """Test finding optimal configuration for ESP32."""
        optimizer = EnergyOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.estimated_tokens_per_second = 10.0
        
        constraints = {
            "min_tokens_per_second": 5.0,
            "max_latency_ms": 500.0
        }
        
        config = optimizer.find_optimal_configuration(
            model=model,
            platform="esp32",
            constraints=constraints,
            optimize_for="energy_per_token"
        )
        
        assert isinstance(config, OptimalConfiguration)
        assert 80 <= config.cpu_freq_mhz <= 240  # ESP32 frequency range
        assert 2.7 <= config.voltage_v <= 3.6    # ESP32 voltage range
        assert isinstance(config.energy_reduction, float)
        assert isinstance(config.performance_impact, float)
    
    def test_find_optimal_configuration_unsupported_platform(self):
        """Test finding optimal configuration for unsupported platform."""
        optimizer = EnergyOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.estimated_tokens_per_second = 10.0
        
        with pytest.raises(ValueError):
            optimizer.find_optimal_configuration(
                model=model,
                platform="unsupported_platform",
                constraints={}
            )
    
    def test_find_optimal_configuration_strict_constraints(self):
        """Test finding optimal configuration with strict constraints."""
        optimizer = EnergyOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.estimated_tokens_per_second = 15.0
        
        # Very strict constraints that might not be met
        constraints = {
            "min_tokens_per_second": 50.0,  # Higher than model capability
            "max_latency_ms": 10.0          # Very low latency
        }
        
        config = optimizer.find_optimal_configuration(
            model=model,
            platform="esp32",
            constraints=constraints
        )
        
        # Should fallback to max performance configuration
        assert config.cpu_freq_mhz == 240  # Max ESP32 frequency
        assert config.voltage_v == 3.6     # Max ESP32 voltage
    
    def test_esp32_power_curve(self):
        """Test ESP32 power curve calculation."""
        optimizer = EnergyOptimizer()
        
        # Test different frequency and voltage combinations
        power_low = optimizer._esp32_power_curve(80, 2.7)
        power_high = optimizer._esp32_power_curve(240, 3.6)
        
        assert power_high > power_low  # Higher freq/voltage should use more power
        assert power_low > 0           # Should be positive
        assert power_high > 0
    
    def test_stm32_power_curve(self):
        """Test STM32 power curve calculation."""
        optimizer = EnergyOptimizer()
        
        power_low = optimizer._stm32_power_curve(16, 1.8)
        power_high = optimizer._stm32_power_curve(168, 3.6)
        
        assert power_high > power_low
        assert power_low > 0
        assert power_high > 0
    
    def test_rp2040_power_curve(self):
        """Test RP2040 power curve calculation."""
        optimizer = EnergyOptimizer()
        
        power_low = optimizer._rp2040_power_curve(10, 1.8)
        power_high = optimizer._rp2040_power_curve(133, 3.3)
        
        assert power_high > power_low
        assert power_low > 0
        assert power_high > 0
    
    def test_optimization_targets(self):
        """Test different optimization targets."""
        optimizer = EnergyOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.estimated_tokens_per_second = 10.0
        
        constraints = {"min_tokens_per_second": 5.0}
        
        # Test different optimization targets
        targets = ["energy_per_token", "total_energy", "power_consumption"]
        
        configs = {}
        for target in targets:
            configs[target] = optimizer.find_optimal_configuration(
                model=model,
                platform="esp32",
                constraints=constraints,
                optimize_for=target
            )
        
        # All should return valid configurations
        for target, config in configs.items():
            assert isinstance(config, OptimalConfiguration)
            assert config.cpu_freq_mhz > 0
            assert config.voltage_v > 0


class TestMemoryOptimizer:
    """Test the MemoryOptimizer class."""
    
    def test_initialization(self):
        """Test memory optimizer initialization."""
        optimizer = MemoryOptimizer()
        
        expected_techniques = [
            "kv_cache_quantization",
            "activation_checkpointing", 
            "flash_attention_lite",
            "weight_sharing",
            "dynamic_batching",
            "memory_pooling"
        ]
        
        for technique in expected_techniques:
            assert technique in optimizer.optimization_techniques
    
    def test_analyze_memory_usage(self):
        """Test memory usage analysis."""
        optimizer = MemoryOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.size_mb = 2.0
        
        memory_map = optimizer.analyze_memory_usage(model)
        
        assert isinstance(memory_map, MemoryMap)
        assert memory_map.model_weights_kb == 2048.0  # 2MB * 1024
        assert memory_map.activation_memory_kb > 0
        assert memory_map.kv_cache_kb > 0
        assert memory_map.scratch_memory_kb == 50.0
        assert memory_map.total_memory_kb > 0
    
    def test_estimate_activation_memory(self):
        """Test activation memory estimation."""
        optimizer = MemoryOptimizer()
        
        # Small model
        small_model = Mock(spec=QuantizedModel)
        small_model.size_mb = 0.5
        activation_memory = optimizer._estimate_activation_memory(small_model)
        assert activation_memory >= 20  # Minimum 20KB
        
        # Large model
        large_model = Mock(spec=QuantizedModel)
        large_model.size_mb = 10.0
        activation_memory_large = optimizer._estimate_activation_memory(large_model)
        assert activation_memory_large > activation_memory
    
    def test_estimate_kv_cache_memory(self):
        """Test KV cache memory estimation."""
        optimizer = MemoryOptimizer()
        
        # Model with default attributes
        model = Mock(spec=QuantizedModel)
        kv_memory_default = optimizer._estimate_kv_cache_memory(model)
        assert kv_memory_default > 0
        
        # Model with custom attributes
        model.context_length = 1024
        model.hidden_size = 1024
        kv_memory_custom = optimizer._estimate_kv_cache_memory(model)
        assert kv_memory_custom > kv_memory_default
    
    def test_estimate_total_memory(self):
        """Test total memory estimation."""
        optimizer = MemoryOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.size_mb = 2.0
        
        total_memory = optimizer._estimate_total_memory(model)
        
        # Should include all components
        weights = model.size_mb * 1024
        activations = optimizer._estimate_activation_memory(model)
        kv_cache = optimizer._estimate_kv_cache_memory(model)
        scratch = 50
        
        expected_total = weights + activations + kv_cache + scratch
        assert abs(total_memory - expected_total) < 1.0  # Allow small floating point differences
    
    def test_optimize_layout_no_constraints(self):
        """Test memory layout optimization with no constraints."""
        optimizer = MemoryOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.size_mb = 2.0
        
        result = optimizer.optimize_layout(model, {})
        
        assert isinstance(result, OptimizationResult)
        assert result.original_performance > 0
        assert result.optimized_performance > 0
        assert result.improvement_factor >= 1.0
        assert isinstance(result.optimizations_applied, list)
    
    def test_optimize_layout_with_ram_constraints(self):
        """Test memory layout optimization with RAM constraints."""
        optimizer = MemoryOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.size_mb = 4.0  # Larger model to trigger optimizations
        
        # Set tight RAM constraint
        constraints = {"ram_limit_kb": 3000}  # 3MB limit
        
        result = optimizer.optimize_layout(model, constraints)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.optimizations_applied) > 0
        assert result.improvement_factor > 1.0
        
        # Should apply memory optimizations
        possible_optimizations = [
            "kv_cache_quantization",
            "activation_checkpointing",
            "flash_attention_lite"
        ]
        
        applied_optimizations = set(result.optimizations_applied)
        expected_optimizations = set(possible_optimizations)
        
        assert len(applied_optimizations.intersection(expected_optimizations)) > 0
    
    def test_optimize_layout_very_tight_constraints(self):
        """Test optimization with very tight constraints."""
        optimizer = MemoryOptimizer()
        
        model = Mock(spec=QuantizedModel)
        model.size_mb = 8.0  # Large model
        
        # Very tight constraints
        constraints = {"ram_limit_kb": 2000}  # 2MB limit for 8MB model
        
        result = optimizer.optimize_layout(model, constraints)
        
        # Should apply all available optimizations
        assert len(result.optimizations_applied) >= 2
        assert "kv_cache_quantization" in result.optimizations_applied
        assert "activation_checkpointing" in result.optimizations_applied


class TestPlatformOptimizer:
    """Test the PlatformOptimizer class."""
    
    def test_initialization(self):
        """Test platform optimizer initialization."""
        optimizer = PlatformOptimizer("esp32")
        
        assert optimizer.platform == "esp32"
        assert "esp32" in optimizer.optimizations
        assert "stm32f4" in optimizer.optimizations
        assert "rp2040" in optimizer.optimizations
    
    def test_optimize_supported_platform(self):
        """Test optimization for supported platform."""
        optimizer = PlatformOptimizer("esp32")
        
        model = QuantizedModel(
            name="test_model",
            size_mb=2.0,
            quantization="4bit"
        )
        
        optimized_model = optimizer.optimize(model, use_psram=True, use_dual_core=True)
        
        assert optimized_model.name == "test_model_esp32_opt"
        assert optimized_model.size_mb == model.size_mb
        assert optimized_model.quantization == model.quantization
        assert optimized_model.config.get("use_psram") is True
        assert optimized_model.config.get("enable_dual_core") is True
    
    def test_optimize_unsupported_platform(self):
        """Test optimization for unsupported platform."""
        optimizer = PlatformOptimizer("unknown_platform")
        
        model = QuantizedModel(
            name="test_model",
            size_mb=2.0,
            quantization="4bit"
        )
        
        # Should return original model unchanged
        optimized_model = optimizer.optimize(model)
        
        assert optimized_model is model  # Same object
    
    def test_esp32_optimizations(self):
        """Test ESP32-specific optimizations."""
        optimizer = PlatformOptimizer("esp32")
        
        model = QuantizedModel("test_model", size_mb=2.0, quantization="4bit")
        
        optimized_model = optimizer.optimize(
            model,
            use_psram=True,
            use_dual_core=True,
            optimize_flash_access=True
        )
        
        config = optimized_model.config
        assert config.get("use_psram") is True
        assert config.get("psram_cache_strategy") == "sequential"
        assert config.get("enable_dual_core") is True
        assert config.get("core_affinity") == "inference_core_1"
        assert config.get("flash_mode") == "qio"
        assert config.get("flash_freq") == "80m"
    
    def test_arm_cortex_m4_optimizations(self):
        """Test ARM Cortex-M4 optimizations."""
        optimizer = PlatformOptimizer("stm32f4")
        
        model = QuantizedModel("test_model", size_mb=1.5, quantization="4bit")
        
        optimized_model = optimizer.optimize(
            model,
            use_dsp_instructions=True,
            use_fpu=True,
            loop_unrolling=4
        )
        
        config = optimized_model.config
        assert config.get("enable_dsp") is True
        assert config.get("simd_width") == 4
        assert config.get("enable_fpu") is True
        assert config.get("fp_precision") == "single"
        assert config.get("loop_unrolling") == 4
    
    def test_arm_cortex_m7_optimizations(self):
        """Test ARM Cortex-M7 optimizations."""
        optimizer = PlatformOptimizer("stm32f7")
        
        model = QuantizedModel("test_model", size_mb=2.0, quantization="4bit")
        
        optimized_model = optimizer.optimize(
            model,
            use_dsp_instructions=True,
            use_fpu=True,
            loop_unrolling=8
        )
        
        config = optimized_model.config
        assert config.get("enable_dsp") is True
        assert config.get("enable_fpu") is True
        assert config.get("enable_cache") is True
        assert config.get("cache_policy") == "write_through"
        assert config.get("loop_unrolling") == 8
        assert config.get("simd_width") == 8  # Better SIMD than M4
    
    def test_arm_cortex_m0_optimizations(self):
        """Test ARM Cortex-M0+ (RP2040) optimizations."""
        optimizer = PlatformOptimizer("rp2040")
        
        model = QuantizedModel("test_model", size_mb=1.0, quantization="4bit")
        
        optimized_model = optimizer.optimize(
            model,
            use_pio=True
        )
        
        config = optimized_model.config
        assert config.get("enable_dual_core") is True
        assert config.get("optimize_for_size") is True
        assert config.get("use_pio") is True
    
    def test_optimization_with_default_kwargs(self):
        """Test optimization with default keyword arguments."""
        optimizer = PlatformOptimizer("esp32")
        
        model = QuantizedModel("test_model", size_mb=2.0, quantization="4bit")
        
        # Don't pass any kwargs, should use defaults
        optimized_model = optimizer.optimize(model)
        
        config = optimized_model.config
        assert config.get("use_psram") is False  # Default
        assert config.get("enable_dual_core") is True  # Default
        assert config.get("optimize_flash_access") is True  # Default
    
    def test_optimization_preserves_model_properties(self):
        """Test that optimization preserves essential model properties."""
        optimizer = PlatformOptimizer("esp32")
        
        original_model = QuantizedModel(
            name="original_model",
            size_mb=3.5,
            quantization="2bit"
        )
        original_model.config["custom_setting"] = "preserved"
        
        optimized_model = optimizer.optimize(original_model)
        
        # Essential properties should be preserved
        assert optimized_model.size_mb == original_model.size_mb
        assert optimized_model.quantization == original_model.quantization
        
        # Original custom settings should be preserved
        assert optimized_model.config.get("custom_setting") == "preserved"
        
        # New optimizations should be added
        assert len(optimized_model.config) > len(original_model.config)


if __name__ == "__main__":
    pytest.main([__file__])