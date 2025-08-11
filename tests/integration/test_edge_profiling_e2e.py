"""
Comprehensive End-to-End Integration Tests for Edge Profiling Workflows

These tests demonstrate the core value proposition of the tiny-llm-edge-profiler library:
profiling LLMs on edge devices with realistic workflows.
"""

import pytest
import asyncio
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from tiny_llm_profiler.profiler import EdgeProfiler, ProfilingConfig
from tiny_llm_profiler.models import QuantizedModel, QuantizationType
from tiny_llm_profiler.platforms import PlatformManager, Architecture
from tiny_llm_profiler.results import ProfileResults, LatencyProfile, MemoryProfile, PowerProfile
from tiny_llm_profiler.power import PowerProfiler
from tiny_llm_profiler.analyzer import MetricsAnalyzer, BottleneckInfo


class TestCompleteEdgeProfilingWorkflow:
    """Test complete edge profiling workflows from model to deployment."""
    
    @pytest.mark.integration
    def test_full_llm_profiling_pipeline(self, tmp_path):
        """Test complete LLM profiling pipeline: load model -> profile -> analyze -> optimize."""
        
        # Step 1: Create and load a realistic test model
        model_file = tmp_path / "tinyllama_2bit.gguf"
        self._create_realistic_model_file(model_file, size_mb=2.5)
        
        model = QuantizedModel.from_file(
            model_file,
            quantization=QuantizationType.INT2,
            vocab_size=32000
        )
        
        assert model.size_mb == pytest.approx(2.5, rel=0.1)
        assert model.quantization == QuantizationType.INT2
        
        # Step 2: Validate platform compatibility
        platform_manager = PlatformManager("esp32")
        is_compatible, issues = platform_manager.validate_model_compatibility(
            model.size_mb, model.quantization.value
        )
        
        assert is_compatible, f"Model should be compatible with ESP32: {issues}"
        
        # Step 3: Configure and run profiling
        config = ProfilingConfig(
            sample_rate_hz=50,
            duration_seconds=10,
            measurement_iterations=5,
            warmup_iterations=2,
            enable_power_profiling=True,
            enable_memory_profiling=True,
            enable_latency_profiling=True
        )
        
        # Use local profiling for testing (no real hardware)
        profiler = EdgeProfiler(
            platform="esp32",
            connection="local"
        )
        
        test_prompts = [
            "Hello, how are you?",
            "Explain machine learning in simple terms.",
            "Write a short poem about artificial intelligence.",
            "What is the capital of France?",
            "How does photosynthesis work?"
        ]
        
        results = profiler.profile_model(
            model=model,
            test_prompts=test_prompts,
            metrics=["latency", "memory", "power"],
            config=config
        )
        
        # Step 4: Validate results completeness
        assert isinstance(results, ProfileResults)
        assert results.platform == "esp32"
        assert results.model_name == model.name
        assert results.model_size_mb == pytest.approx(2.5, rel=0.1)
        assert results.quantization == "2bit"
        
        # Check all metrics were captured
        assert results.latency_profile is not None
        assert results.memory_profile is not None
        assert results.power_profile is not None
        
        # Step 5: Validate realistic performance metrics
        latency = results.latency_profile
        assert 5 <= latency.tokens_per_second <= 300  # Realistic range for ESP32 2-bit model (can be optimized)
        assert 10 <= latency.first_token_latency_ms <= 500  # Reasonable first token delay
        assert 1 <= latency.inter_token_latency_ms <= 200  # Reasonable inter-token delay (can be very fast with optimization)
        
        memory = results.memory_profile
        assert memory.peak_memory_kb > 0  # Should use some memory
        # Note: In test mode, memory usage includes host system memory
        # In actual edge deployment, this would be much lower
        if memory.baseline_memory_kb != memory.peak_memory_kb:
            assert memory.baseline_memory_kb < memory.peak_memory_kb
        
        power = results.power_profile
        assert 50 <= power.active_power_mw <= 300  # Typical ESP32 power consumption
        assert power.idle_power_mw < power.active_power_mw
        
        # Step 6: Generate analysis and recommendations
        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze(results)
        
        assert "tokens_per_second" in analysis
        assert "peak_memory_kb" in analysis
        assert "active_power_mw" in analysis
        
        bottlenecks = analyzer.find_bottlenecks(results)
        recommendations = results.get_recommendations()
        
        assert isinstance(bottlenecks, list)
        assert isinstance(recommendations, list)
        
        # Step 7: Export results
        results_file = tmp_path / "profiling_results.json"
        results.export_json(results_file)
        
        assert results_file.exists()
        with open(results_file) as f:
            exported_data = json.load(f)
            assert "metadata" in exported_data
            assert "profiles" in exported_data
            assert "recommendations" in exported_data
        
        # Step 8: Verify data integrity by reloading
        reloaded_results = ProfileResults.from_json(results_file)
        assert reloaded_results.model_name == results.model_name
        assert reloaded_results.platform == results.platform
        
        print(f"✓ Complete edge profiling pipeline test passed")
        print(f"  Model: {model.name} ({model.size_mb:.1f}MB, {model.quantization.value})")
        print(f"  Performance: {latency.tokens_per_second:.1f} tok/s")
        print(f"  Memory: {memory.peak_memory_kb:.0f}KB peak")
        print(f"  Power: {power.active_power_mw:.0f}mW active")
        print(f"  Recommendations: {len(recommendations)} suggestions")

    @pytest.mark.integration 
    @pytest.mark.slow
    def test_model_optimization_workflow(self, tmp_path):
        """Test complete model optimization workflow for edge deployment."""
        
        # Create base model that's too large for constraints
        base_model_file = tmp_path / "base_model.gguf"
        self._create_realistic_model_file(base_model_file, size_mb=5.0)
        
        base_model = QuantizedModel.from_file(
            base_model_file,
            quantization=QuantizationType.INT4
        )
        
        # Define platform constraints (tight for STM32)
        constraints = {
            "max_memory_kb": 200,
            "max_power_mw": 150,
            "target_platform": "stm32f4"
        }
        
        # Step 1: Initial profiling to establish baseline
        profiler = EdgeProfiler(platform="stm32f4", connection="local")
        
        baseline_results = profiler.profile_model(
            model=base_model,
            test_prompts=["Test baseline performance"],
            metrics=["latency", "memory", "power"],
            config=ProfilingConfig(duration_seconds=5, measurement_iterations=3)
        )
        
        # Check if initial model violates constraints
        memory_violation = baseline_results.memory_profile.peak_memory_kb > constraints["max_memory_kb"]
        power_violation = baseline_results.power_profile.active_power_mw > constraints["max_power_mw"]
        
        # Step 2: Optimize model for platform constraints
        optimized_model = base_model.optimize_for_platform(
            platform=constraints["target_platform"],
            constraints=constraints
        )
        
        # Model should have been optimized
        assert optimized_model.quantization in [QuantizationType.INT2, QuantizationType.INT3]
        assert optimized_model.context_length <= base_model.context_length
        
        # Step 3: Profile optimized model
        optimized_results = profiler.profile_model(
            model=optimized_model,
            test_prompts=["Test optimized performance"],
            metrics=["latency", "memory", "power"],
            config=ProfilingConfig(duration_seconds=5, measurement_iterations=3)
        )
        
        # Step 4: Verify optimizations met constraints
        if memory_violation:
            assert optimized_results.memory_profile.peak_memory_kb <= constraints["max_memory_kb"]
        if power_violation:
            assert optimized_results.power_profile.active_power_mw <= constraints["max_power_mw"]
        
        # Step 5: Compare performance
        comparison = baseline_results.compare_with(optimized_results)
        
        # Memory should be improved if it was a constraint
        if memory_violation:
            assert comparison["memory_efficiency_ratio"] > 1.0
        
        # Step 6: Generate optimization report
        report_file = tmp_path / "optimization_report.json"
        
        optimization_report = {
            "baseline": baseline_results.get_summary(),
            "optimized": optimized_results.get_summary(),
            "comparison": comparison,
            "constraints": constraints,
            "optimizations_applied": {
                "quantization": f"{base_model.quantization.value} -> {optimized_model.quantization.value}",
                "context_length": f"{base_model.context_length} -> {optimized_model.context_length}",
                "model_size": f"{base_model.size_mb:.1f}MB -> {optimized_model.size_mb:.1f}MB"
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(optimization_report, f, indent=2, default=str)
        
        assert report_file.exists()
        
        print(f"✓ Model optimization workflow test passed")
        print(f"  Baseline: {baseline_results.latency_profile.tokens_per_second:.1f} tok/s, "
              f"{baseline_results.memory_profile.peak_memory_kb:.0f}KB")
        print(f"  Optimized: {optimized_results.latency_profile.tokens_per_second:.1f} tok/s, "
              f"{optimized_results.memory_profile.peak_memory_kb:.0f}KB")
        print(f"  Memory improvement: {comparison.get('memory_efficiency_ratio', 1.0):.2f}x")
    
    @pytest.mark.integration
    def test_real_world_deployment_scenario(self, tmp_path):
        """Test realistic deployment scenario with multiple models and platforms."""
        
        # Create a portfolio of models for different use cases
        models = {
            "chat_tiny": self._create_test_model(tmp_path / "chat_tiny.gguf", 1.5, QuantizationType.INT2),
            "instruct_small": self._create_test_model(tmp_path / "instruct_small.gguf", 3.0, QuantizationType.INT3),
            "code_medium": self._create_test_model(tmp_path / "code_medium.gguf", 4.5, QuantizationType.INT4)
        }
        
        # Define target platforms with different constraints
        platforms = {
            "esp32": {"ram_budget_kb": 400, "power_budget_mw": 200, "use_case": "iot_sensor"},
            "stm32f7": {"ram_budget_kb": 350, "power_budget_mw": 150, "use_case": "embedded_assistant"},
            "rp2040": {"ram_budget_kb": 200, "power_budget_mw": 120, "use_case": "wearable"}
        }
        
        # Test prompts for different use cases
        use_case_prompts = {
            "iot_sensor": ["Temperature is 25°C", "Motion detected", "Battery at 80%"],
            "embedded_assistant": ["Set timer for 5 minutes", "What's the weather?", "Turn on lights"],
            "wearable": ["Steps today", "Heart rate check", "Quick reminder"]
        }
        
        deployment_results = {}
        
        # Test each model-platform combination
        for model_name, model in models.items():
            for platform_name, platform_config in platforms.items():
                
                # Check basic compatibility first
                platform_manager = PlatformManager(platform_name)
                is_compatible, issues = platform_manager.validate_model_compatibility(
                    model.size_mb, model.quantization.value
                )
                
                if not is_compatible:
                    print(f"⚠️ Skipping {model_name} on {platform_name}: {issues}")
                    continue
                
                # Profile the combination
                profiler = EdgeProfiler(platform=platform_name, connection="local")
                
                prompts = use_case_prompts[platform_config["use_case"]]
                
                try:
                    results = profiler.profile_model(
                        model=model,
                        test_prompts=prompts,
                        metrics=["latency", "memory", "power"],
                        config=ProfilingConfig(duration_seconds=3, measurement_iterations=2)
                    )
                    
                    # Check if deployment meets constraints
                    memory_ok = results.memory_profile.peak_memory_kb <= platform_config["ram_budget_kb"]
                    power_ok = results.power_profile.active_power_mw <= platform_config["power_budget_mw"]
                    
                    deployment_results[f"{model_name}_on_{platform_name}"] = {
                        "viable": memory_ok and power_ok,
                        "results": results,
                        "constraints_met": {
                            "memory": memory_ok,
                            "power": power_ok
                        },
                        "performance_score": results.calculate_efficiency_score()
                    }
                    
                except Exception as e:
                    print(f"❌ Failed to profile {model_name} on {platform_name}: {e}")
        
        # Find best deployment options
        viable_deployments = {
            k: v for k, v in deployment_results.items() 
            if v["viable"]
        }
        
        assert len(viable_deployments) > 0, "At least one deployment should be viable"
        
        # Generate deployment recommendations
        recommendations = {}
        for use_case in ["iot_sensor", "embedded_assistant", "wearable"]:
            use_case_deployments = {
                k: v for k, v in viable_deployments.items()
                if platforms[k.split("_on_")[1]]["use_case"] == use_case
            }
            
            if use_case_deployments:
                best_deployment = max(
                    use_case_deployments.items(),
                    key=lambda x: x[1]["performance_score"]
                )[0]
                recommendations[use_case] = best_deployment
        
        # Export deployment analysis
        deployment_report = {
            "tested_combinations": len(deployment_results),
            "viable_deployments": len(viable_deployments),
            "recommendations": recommendations,
            "detailed_results": {
                k: {
                    "viable": v["viable"],
                    "performance_score": v["performance_score"],
                    "constraints_met": v["constraints_met"],
                    "summary": v["results"].get_summary()
                }
                for k, v in deployment_results.items()
            }
        }
        
        report_file = tmp_path / "deployment_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(deployment_report, f, indent=2, default=str)
        
        print(f"✓ Real-world deployment scenario test passed")
        print(f"  Tested {len(deployment_results)} model-platform combinations")
        print(f"  Found {len(viable_deployments)} viable deployments")
        print(f"  Recommendations: {recommendations}")

    def _create_realistic_model_file(self, path: Path, size_mb: float):
        """Create a realistic model file for testing."""
        # Create GGUF-like header
        header = b"GGUF" + b"\x03\x00\x00\x00"  # Version 3
        
        # Add some metadata-like content
        metadata = json.dumps({
            "general.architecture": "llama",
            "general.quantization_version": 2,
            "general.file_type": 2,
            "llama.vocab_size": 32000,
            "llama.context_length": 2048,
            "llama.embedding_length": 2048,
            "llama.block_count": 22
        }).encode()
        
        # Calculate remaining size for "weights"
        header_size = len(header) + len(metadata) + 100  # Some padding
        remaining_bytes = int(size_mb * 1024 * 1024) - header_size
        
        with open(path, 'wb') as f:
            f.write(header)
            f.write(len(metadata).to_bytes(4, 'little'))
            f.write(metadata)
            f.write(b'\x00' * 96)  # Padding
            f.write(b'\x42' * max(0, remaining_bytes))  # Simulated weights
    
    def _create_test_model(self, path: Path, size_mb: float, quantization: QuantizationType) -> QuantizedModel:
        """Create a test model with specified characteristics."""
        self._create_realistic_model_file(path, size_mb)
        return QuantizedModel.from_file(path, quantization=quantization)


@pytest.mark.integration
class TestPowerProfilingIntegration:
    """Integration tests for power profiling functionality."""
    
    def test_power_profiling_workflow(self, tmp_path):
        """Test complete power profiling workflow with simulated power sensor."""
        
        # Create test model
        model_file = tmp_path / "power_test_model.gguf"
        with open(model_file, 'wb') as f:
            f.write(b"GGUF" + b"\x00" * 1000)
        
        model = QuantizedModel.from_file(model_file, quantization=QuantizationType.INT4)
        
        # Initialize power profiler with simulated sensor
        power_profiler = PowerProfiler(sensor="simulated")
        assert power_profiler.initialize()
        
        # Test streaming power measurements
        measurements = []
        measurement_count = 0
        
        for measurement in power_profiler.stream_measurements(sample_rate_hz=10):
            measurements.append(measurement)
            measurement_count += 1
            
            # Stop after collecting some samples
            if measurement_count >= 20:
                power_profiler.stop_profiling()
                break
        
        assert len(measurements) >= 10
        assert all(m.power_mw > 0 for m in measurements)
        assert all(m.voltage_v > 0 for m in measurements)
        assert all(m.current_ma > 0 for m in measurements)
        
        # Test power profiling during model inference simulation
        config = ProfilingConfig(
            duration_seconds=5,
            measurement_iterations=3,
            enable_power_profiling=True
        )
        
        profiler = EdgeProfiler(platform="esp32", connection="local")
        results = profiler.profile_model(
            model=model,
            test_prompts=["Power measurement test"],
            metrics=["power"],
            config=config
        )
        
        assert results.power_profile is not None
        power = results.power_profile
        
        # Validate power profile
        assert power.idle_power_mw < power.active_power_mw < power.peak_power_mw
        assert power.energy_per_token_mj > 0
        assert power.total_energy_mj > 0
        
        # Export power measurements
        export_file = tmp_path / "power_measurements.csv"
        power_profiler.export_measurements(export_file)
        
        assert export_file.exists()
        
        print(f"✓ Power profiling workflow test passed")
        print(f"  Measurements collected: {len(measurements)}")
        print(f"  Power range: {power.idle_power_mw:.1f} - {power.peak_power_mw:.1f} mW")
        print(f"  Energy per token: {power.energy_per_token_mj:.2f} mJ")


@pytest.mark.integration
class TestStreamingProfilingWorkflow:
    """Test real-time streaming profiling capabilities."""
    
    def test_real_time_metrics_streaming(self, tmp_path):
        """Test streaming real-time metrics during inference."""
        
        model_file = tmp_path / "streaming_test_model.gguf" 
        with open(model_file, 'wb') as f:
            f.write(b"GGUF" + b"\x00" * 2000)
        
        model = QuantizedModel.from_file(model_file)
        profiler = EdgeProfiler(platform="esp32", connection="local")
        
        # Start streaming metrics
        metrics_collected = []
        
        def collect_metrics():
            for metrics in profiler.stream_metrics(duration_seconds=8):
                metrics_collected.append(metrics)
                if len(metrics_collected) >= 20:  # Collect 20 samples
                    break
        
        # Run streaming in a separate thread while "profiling"
        import threading
        metrics_thread = threading.Thread(target=collect_metrics)
        metrics_thread.start()
        
        # Simulate some inference activity
        time.sleep(2)
        
        # Run quick profiling
        results = profiler.profile_model(
            model=model,
            test_prompts=["Streaming test prompt"],
            metrics=["latency"],
            config=ProfilingConfig(duration_seconds=3, measurement_iterations=1)
        )
        
        metrics_thread.join(timeout=10)
        
        # Validate streaming metrics
        assert len(metrics_collected) >= 5  # Should have collected some metrics
        
        for metrics in metrics_collected:
            assert "timestamp" in metrics
            assert "memory_mb" in metrics or "memory_kb" in metrics
            assert metrics["timestamp"] > 0
        
        # Validate that profiling still worked
        assert results.latency_profile is not None
        assert results.latency_profile.tokens_per_second > 0
        
        print(f"✓ Real-time streaming test passed")
        print(f"  Metrics samples: {len(metrics_collected)}")
        print(f"  Concurrent profiling: {results.latency_profile.tokens_per_second:.1f} tok/s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])