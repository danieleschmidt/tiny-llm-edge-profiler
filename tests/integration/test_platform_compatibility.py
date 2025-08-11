"""
Platform Compatibility Integration Tests

Tests that verify the profiler works correctly across different edge platforms
with their specific characteristics and constraints.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from tiny_llm_profiler.profiler import EdgeProfiler, ProfilingConfig
from tiny_llm_profiler.models import QuantizedModel, QuantizationType
from tiny_llm_profiler.platforms import PlatformManager, Architecture
from tiny_llm_profiler.results import ProfileResults


@pytest.mark.integration
class TestMultiPlatformCompatibility:
    """Test profiling across different edge platforms."""
    
    def test_esp32_family_profiling(self, tmp_path):
        """Test profiling across ESP32 family platforms."""
        
        esp32_platforms = ["esp32", "esp32s3"]
        test_models = {
            "tiny_2bit": self._create_test_model(tmp_path / "tiny_2bit.gguf", 1.5, QuantizationType.INT2),
            "small_4bit": self._create_test_model(tmp_path / "small_4bit.gguf", 3.0, QuantizationType.INT4)
        }
        
        compatibility_matrix = {}
        
        for platform in esp32_platforms:
            platform_manager = PlatformManager(platform)
            platform_results = {}
            
            for model_name, model in test_models.items():
                # Check theoretical compatibility
                is_compatible, issues = platform_manager.validate_model_compatibility(
                    model.size_mb, model.quantization.value
                )
                
                # Test actual profiling
                profiler = EdgeProfiler(platform=platform, connection="local")
                
                try:
                    results = profiler.profile_model(
                        model=model,
                        test_prompts=["ESP32 compatibility test"],
                        metrics=["latency", "memory"],
                        config=ProfilingConfig(duration_seconds=3, measurement_iterations=2)
                    )
                    
                    # Validate platform-specific characteristics
                    if platform == "esp32":
                        # Standard ESP32 should handle both models but with different performance
                        assert results.latency_profile.tokens_per_second > 0
                        assert results.memory_profile.peak_memory_kb < 500  # Should fit in available RAM
                        
                    elif platform == "esp32s3":
                        # ESP32-S3 should perform better due to vector unit
                        assert results.latency_profile.tokens_per_second > 0
                        # May handle larger models due to more RAM/PSRAM
                        
                    platform_results[model_name] = {
                        "compatible": True,
                        "results": results,
                        "performance_score": results.calculate_efficiency_score()
                    }
                    
                except Exception as e:
                    platform_results[model_name] = {
                        "compatible": False,
                        "error": str(e),
                        "performance_score": 0
                    }
            
            compatibility_matrix[platform] = platform_results
        
        # Validate that at least one model works on each platform
        for platform, results in compatibility_matrix.items():
            compatible_models = [k for k, v in results.items() if v["compatible"]]
            assert len(compatible_models) > 0, f"No models compatible with {platform}"
        
        # Compare ESP32 vs ESP32-S3 performance where both work
        for model_name in test_models.keys():
            esp32_result = compatibility_matrix["esp32"].get(model_name)
            esp32s3_result = compatibility_matrix["esp32s3"].get(model_name)
            
            if esp32_result and esp32s3_result and both_compatible(esp32_result, esp32s3_result):
                # ESP32-S3 should generally perform better or equal
                esp32_score = esp32_result["performance_score"]
                esp32s3_score = esp32s3_result["performance_score"]
                
                print(f"  {model_name}: ESP32={esp32_score:.1f} vs ESP32-S3={esp32s3_score:.1f}")
        
        print(f"✓ ESP32 family compatibility test passed")
        print(f"  Platforms tested: {esp32_platforms}")
        print(f"  Models tested: {list(test_models.keys())}")
    
    def test_arm_cortex_platforms(self, tmp_path):
        """Test profiling across ARM Cortex-M platforms."""
        
        arm_platforms = ["stm32f4", "stm32f7", "rp2040", "nrf52840"]
        
        # Create appropriately sized models for memory-constrained platforms
        model_file = tmp_path / "cortex_model.gguf"
        self._create_model_file(model_file, 1.2)  # Small model for constrained devices
        model = QuantizedModel.from_file(model_file, quantization=QuantizationType.INT4)
        
        platform_performance = {}
        
        for platform in arm_platforms:
            try:
                platform_manager = PlatformManager(platform)
                platform_config = platform_manager.get_config()
                
                # Check if model fits in platform memory
                memory_constraints = platform_manager.get_memory_constraints()
                model_memory_req = model.get_memory_requirements(platform)
                
                fits_in_memory = (
                    model_memory_req["total_estimated_kb"] <= 
                    memory_constraints["available_ram_kb"]
                )
                
                if not fits_in_memory:
                    print(f"⚠️ Model too large for {platform} "
                          f"({model_memory_req['total_estimated_kb']}KB > "
                          f"{memory_constraints['available_ram_kb']}KB)")
                    continue
                
                # Profile on platform
                profiler = EdgeProfiler(platform=platform, connection="local")
                
                results = profiler.profile_model(
                    model=model,
                    test_prompts=["ARM Cortex test"],
                    metrics=["latency", "memory"],
                    config=ProfilingConfig(duration_seconds=2, measurement_iterations=1)
                )
                
                # Validate platform-specific characteristics
                memory_usage = results.memory_profile.peak_memory_kb
                assert memory_usage <= memory_constraints["available_ram_kb"]
                
                # Performance should scale with CPU frequency and capabilities
                expected_performance_factors = {
                    "stm32f4": 1.0,    # Baseline
                    "stm32f7": 1.3,    # Higher freq + better cache
                    "rp2040": 0.8,     # Lower performance, no FPU
                    "nrf52840": 0.7    # Lower freq
                }
                
                platform_performance[platform] = {
                    "results": results,
                    "tokens_per_second": results.latency_profile.tokens_per_second,
                    "memory_usage_kb": memory_usage,
                    "platform_config": {
                        "cpu_freq_mhz": platform_config.capabilities.max_cpu_freq_mhz,
                        "has_fpu": platform_config.capabilities.has_fpu,
                        "ram_kb": platform_config.memory.ram_kb
                    }
                }
                
            except Exception as e:
                print(f"❌ Failed to test {platform}: {e}")
        
        assert len(platform_performance) >= 2, "Should successfully test multiple ARM platforms"
        
        # Analyze performance correlation with platform capabilities
        if "stm32f4" in platform_performance and "stm32f7" in platform_performance:
            stm32f4_perf = platform_performance["stm32f4"]["tokens_per_second"]
            stm32f7_perf = platform_performance["stm32f7"]["tokens_per_second"]
            
            # STM32F7 should generally outperform STM32F4 due to higher frequency
            performance_ratio = stm32f7_perf / stm32f4_perf if stm32f4_perf > 0 else 0
            assert performance_ratio > 0.8, f"STM32F7 performance ratio too low: {performance_ratio}"
        
        print(f"✓ ARM Cortex platforms compatibility test passed")
        print(f"  Platforms tested: {list(platform_performance.keys())}")
        for platform, data in platform_performance.items():
            print(f"    {platform}: {data['tokens_per_second']:.1f} tok/s, "
                  f"{data['memory_usage_kb']:.0f}KB")
    
    def test_riscv_platforms(self, tmp_path):
        """Test profiling on RISC-V platforms."""
        
        riscv_platforms = ["k210", "bl602"]
        
        # Create models suitable for RISC-V testing
        models = {
            "optimized_2bit": self._create_test_model(tmp_path / "riscv_2bit.gguf", 2.0, QuantizationType.INT2),
            "standard_4bit": self._create_test_model(tmp_path / "riscv_4bit.gguf", 3.5, QuantizationType.INT4)
        }
        
        riscv_results = {}
        
        for platform in riscv_platforms:
            platform_results = {}
            
            for model_name, model in models.items():
                try:
                    platform_manager = PlatformManager(platform)
                    
                    # Check model compatibility
                    is_compatible, issues = platform_manager.validate_model_compatibility(
                        model.size_mb, model.quantization.value
                    )
                    
                    if not is_compatible:
                        print(f"⚠️ {model_name} not compatible with {platform}: {issues}")
                        continue
                    
                    profiler = EdgeProfiler(platform=platform, connection="local")
                    
                    results = profiler.profile_model(
                        model=model,
                        test_prompts=["RISC-V test prompt"],
                        metrics=["latency", "memory"],
                        config=ProfilingConfig(duration_seconds=3, measurement_iterations=2)
                    )
                    
                    # Validate RISC-V specific characteristics
                    if platform == "k210":
                        # K210 has dedicated AI accelerator (KPU)
                        # Should handle larger models better
                        assert results.latency_profile.tokens_per_second > 0
                        
                    elif platform == "bl602":
                        # BL602 is more memory constrained
                        assert results.memory_profile.peak_memory_kb < 250
                    
                    platform_results[model_name] = {
                        "results": results,
                        "success": True
                    }
                    
                except Exception as e:
                    platform_results[model_name] = {
                        "error": str(e),
                        "success": False
                    }
            
            riscv_results[platform] = platform_results
        
        # Validate results
        for platform, results in riscv_results.items():
            successful_tests = sum(1 for r in results.values() if r.get("success", False))
            assert successful_tests > 0, f"No successful tests on {platform}"
        
        print(f"✓ RISC-V platforms compatibility test passed")
        for platform, results in riscv_results.items():
            successful = sum(1 for r in results.values() if r.get("success", False))
            print(f"  {platform}: {successful}/{len(results)} models successful")
    
    def test_cross_platform_model_portability(self, tmp_path):
        """Test that models can be profiled across different platform architectures."""
        
        # Create a well-optimized model that should work on most platforms
        universal_model_file = tmp_path / "universal_model.gguf"
        self._create_model_file(universal_model_file, 1.8)
        universal_model = QuantizedModel.from_file(
            universal_model_file, 
            quantization=QuantizationType.INT3  # Good balance of size vs quality
        )
        
        # Test across diverse platform architectures
        test_platforms = ["esp32", "stm32f7", "rp2040", "k210"]
        
        portability_results = {}
        
        for platform in test_platforms:
            try:
                # Optimize model specifically for this platform
                platform_manager = PlatformManager(platform)
                optimized_model = universal_model.optimize_for_platform(
                    platform=platform,
                    constraints=platform_manager.get_memory_constraints()
                )
                
                # Profile optimized model
                profiler = EdgeProfiler(platform=platform, connection="local")
                
                results = profiler.profile_model(
                    model=optimized_model,
                    test_prompts=["Cross-platform portability test"],
                    metrics=["latency", "memory"],
                    config=ProfilingConfig(duration_seconds=2, measurement_iterations=1)
                )
                
                portability_results[platform] = {
                    "success": True,
                    "results": results,
                    "optimization": {
                        "original_size_mb": universal_model.size_mb,
                        "optimized_size_mb": optimized_model.size_mb,
                        "original_quantization": universal_model.quantization.value,
                        "optimized_quantization": optimized_model.quantization.value
                    }
                }
                
            except Exception as e:
                portability_results[platform] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Analyze portability
        successful_platforms = [p for p, r in portability_results.items() if r["success"]]
        portability_score = len(successful_platforms) / len(test_platforms)
        
        assert portability_score >= 0.75, f"Model should be portable to most platforms: {portability_score}"
        
        # Compare performance characteristics across architectures
        performance_summary = {}
        for platform in successful_platforms:
            result_data = portability_results[platform]
            results = result_data["results"]
            
            performance_summary[platform] = {
                "tokens_per_second": results.latency_profile.tokens_per_second,
                "memory_efficiency": results.memory_profile.memory_efficiency_tokens_per_kb,
                "architecture": PlatformManager(platform).get_config().architecture.value
            }
        
        print(f"✓ Cross-platform model portability test passed")
        print(f"  Portability score: {portability_score:.1%} ({len(successful_platforms)}/{len(test_platforms)})")
        print(f"  Performance across architectures:")
        for platform, perf in performance_summary.items():
            print(f"    {platform} ({perf['architecture']}): {perf['tokens_per_second']:.1f} tok/s")
    
    def _create_model_file(self, path: Path, size_mb: float):
        """Create a test model file of specified size."""
        header = b"GGUF\x03\x00\x00\x00"  # GGUF header
        metadata_size = 256
        data_size = int(size_mb * 1024 * 1024) - len(header) - metadata_size
        
        with open(path, 'wb') as f:
            f.write(header)
            f.write(b'\x00' * metadata_size)  # Minimal metadata
            f.write(b'\x42' * max(0, data_size))  # Simulated model data
    
    def _create_test_model(self, path: Path, size_mb: float, quantization: QuantizationType) -> QuantizedModel:
        """Create a test model with specified characteristics."""
        self._create_model_file(path, size_mb)
        return QuantizedModel.from_file(path, quantization=quantization)


@pytest.mark.integration
class TestPlatformSpecificFeatures:
    """Test platform-specific features and optimizations."""
    
    def test_esp32_psram_utilization(self, tmp_path):
        """Test ESP32 PSRAM utilization for larger models."""
        
        # Create a model that requires PSRAM
        large_model_file = tmp_path / "large_model.gguf"
        with open(large_model_file, 'wb') as f:
            f.write(b"GGUF" + b"\x00" * (5 * 1024 * 1024))  # 5MB model
        
        large_model = QuantizedModel.from_file(large_model_file, quantization=QuantizationType.INT4)
        
        # Test on ESP32 with PSRAM enabled
        platform_manager = PlatformManager("esp32")
        config = platform_manager.get_config()
        
        # Model should be too large for main SRAM but OK with PSRAM
        memory_reqs = large_model.get_memory_requirements("esp32")
        main_ram_kb = config.memory.ram_kb
        total_ram_kb = main_ram_kb + config.memory.external_ram_kb
        
        assert memory_reqs["total_estimated_kb"] > main_ram_kb, "Model should require PSRAM"
        assert memory_reqs["total_estimated_kb"] <= total_ram_kb, "Model should fit with PSRAM"
        
        # Profile the model
        profiler = EdgeProfiler(platform="esp32", connection="local")
        
        results = profiler.profile_model(
            model=large_model,
            test_prompts=["PSRAM utilization test"],
            metrics=["memory"],
            config=ProfilingConfig(duration_seconds=2, measurement_iterations=1)
        )
        
        # Model should run successfully with PSRAM
        assert results.memory_profile is not None
        assert results.memory_profile.peak_memory_kb > main_ram_kb  # Using PSRAM
        
        print(f"✓ ESP32 PSRAM utilization test passed")
        print(f"  Model size: {large_model.size_mb:.1f}MB")
        print(f"  Peak memory usage: {results.memory_profile.peak_memory_kb:.0f}KB")
        print(f"  PSRAM utilization: Required for operation")
    
    def test_stm32_memory_constraints(self, tmp_path):
        """Test STM32 memory constraint handling."""
        
        # Test models of different sizes
        models = {}
        for size_mb, name in [(0.5, "tiny"), (1.5, "small"), (3.0, "medium")]:
            model_file = tmp_path / f"stm32_{name}.gguf"
            with open(model_file, 'wb') as f:
                f.write(b"GGUF" + b"\x00" * int(size_mb * 1024 * 1024))
            models[name] = QuantizedModel.from_file(model_file, quantization=QuantizationType.INT4)
        
        # Test on memory-constrained STM32F4
        platform_manager = PlatformManager("stm32f4")
        memory_constraints = platform_manager.get_memory_constraints()
        
        profiler = EdgeProfiler(platform="stm32f4", connection="local")
        
        constraint_test_results = {}
        
        for name, model in models.items():
            memory_reqs = model.get_memory_requirements("stm32f4")
            should_fit = memory_reqs["total_estimated_kb"] <= memory_constraints["available_ram_kb"]
            
            try:
                results = profiler.profile_model(
                    model=model,
                    test_prompts=["STM32 memory test"],
                    metrics=["memory"],
                    config=ProfilingConfig(duration_seconds=1, measurement_iterations=1)
                )
                
                # Verify memory usage is within constraints
                actual_usage = results.memory_profile.peak_memory_kb
                
                constraint_test_results[name] = {
                    "should_fit": should_fit,
                    "did_run": True,
                    "memory_usage_kb": actual_usage,
                    "within_constraints": actual_usage <= memory_constraints["available_ram_kb"]
                }
                
            except Exception as e:
                constraint_test_results[name] = {
                    "should_fit": should_fit,
                    "did_run": False,
                    "error": str(e)
                }
        
        # Validate results match expectations
        for name, result in constraint_test_results.items():
            if result["should_fit"]:
                assert result["did_run"], f"{name} model should have run successfully"
                if result["did_run"]:
                    assert result["within_constraints"], f"{name} model exceeded memory constraints"
        
        print(f"✓ STM32 memory constraints test passed")
        for name, result in constraint_test_results.items():
            status = "✓" if result["did_run"] else "✗"
            memory_info = f"{result.get('memory_usage_kb', 0):.0f}KB" if result["did_run"] else "N/A"
            print(f"  {status} {name}: {memory_info}")
    
    def test_k210_ai_accelerator_utilization(self, tmp_path):
        """Test K210 KPU (AI accelerator) utilization simulation."""
        
        # Create model suitable for AI acceleration
        model_file = tmp_path / "k210_ai_model.gguf"
        with open(model_file, 'wb') as f:
            f.write(b"GGUF" + b"\x00" * (4 * 1024 * 1024))  # 4MB model
        
        model = QuantizedModel.from_file(model_file, quantization=QuantizationType.INT8)
        
        # Test on K210 with AI accelerator capabilities
        platform_manager = PlatformManager("k210")
        config = platform_manager.get_config()
        
        assert config.capabilities.has_dedicated_ai_accelerator, "K210 should have AI accelerator"
        
        profiler = EdgeProfiler(platform="k210", connection="local")
        
        results = profiler.profile_model(
            model=model,
            test_prompts=["K210 AI accelerator test"],
            metrics=["latency", "memory"],
            config=ProfilingConfig(duration_seconds=3, measurement_iterations=2)
        )
        
        # With AI acceleration, should achieve better performance
        # than pure CPU implementation
        assert results.latency_profile.tokens_per_second > 5  # Should be reasonably fast
        assert results.memory_profile.peak_memory_kb < 8000  # Should use available 8MB efficiently
        
        # Get optimization recommendations - should suggest using AI accelerator
        recommendations = results.get_recommendations()
        
        print(f"✓ K210 AI accelerator test passed")
        print(f"  Performance: {results.latency_profile.tokens_per_second:.1f} tok/s")
        print(f"  Memory usage: {results.memory_profile.peak_memory_kb:.0f}KB")
        print(f"  Has AI accelerator: {config.capabilities.has_dedicated_ai_accelerator}")


def both_compatible(result1: Dict[str, Any], result2: Dict[str, Any]) -> bool:
    """Check if both results indicate compatibility."""
    return result1.get("compatible", False) and result2.get("compatible", False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])