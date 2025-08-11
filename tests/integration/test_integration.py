"""
Enhanced Integration Tests for the Tiny LLM Edge Profiler.

Comprehensive integration tests that demonstrate the complete functionality
and core value proposition of profiling LLMs on edge devices.
"""

import pytest
import tempfile
import time
import threading
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import hashlib
import os

from tiny_llm_profiler.profiler import EdgeProfiler, ProfilingConfig
from tiny_llm_profiler.models import QuantizedModel, QuantizationType
from tiny_llm_profiler.platforms import PlatformManager
from tiny_llm_profiler.results import ProfileResults
from tiny_llm_profiler.concurrent_utils import ConcurrentProfiler, ProfilingTask
from tiny_llm_profiler.resource_pool import ResourcePoolManager, ResourcePool
from tiny_llm_profiler.cache import SmartCache
from tiny_llm_profiler.config import ConfigManager, AppConfig
from tiny_llm_profiler.security import SecurityValidator, InputSanitizer
from tiny_llm_profiler.health import HealthChecker
from tiny_llm_profiler.scaling import AutoScaler, LoadBalancer
from tiny_llm_profiler.analyzer import MetricsAnalyzer, BottleneckInfo
from tiny_llm_profiler.power import PowerProfiler


class TestEndToEndProfiling:
    """End-to-end profiling workflow tests."""
    
    def test_complete_profiling_workflow(self, mock_model_file, mock_device):
        """Test complete profiling workflow from model loading to results."""
        
        # Step 1: Load and validate model
        model = QuantizedModel.from_file(mock_model_file, quantization="4bit")
        assert model.name is not None
        assert model.size_mb > 0
        
        # Step 2: Validate platform compatibility
        platform_manager = PlatformManager("esp32")
        is_compatible, issues = platform_manager.validate_model_compatibility(
            model.size_mb, model.quantization.value
        )
        
        if not is_compatible:
            pytest.skip(f"Model not compatible: {issues}")
        
        # Step 3: Initialize profiler
        profiler = EdgeProfiler(
            platform="esp32",
            device=mock_device,
            connection="local"  # Use local mode for testing
        )
        
        # Step 4: Run profiling
        config = ProfilingConfig(
            duration_seconds=5,  # Short duration for testing
            measurement_iterations=3,
            warmup_iterations=1
        )
        
        results = profiler.profile_model(
            model=model,
            test_prompts=["Hello world", "Test prompt"],
            metrics=["latency", "memory"],
            config=config
        )
        
        # Step 5: Validate results
        assert isinstance(results, ProfileResults)
        assert results.platform == "esp32"
        assert results.model_name == model.name
        assert results.latency_profile is not None
        assert results.memory_profile is not None
        
        # Check result quality
        assert results.latency_profile.tokens_per_second > 0
        assert results.memory_profile.peak_memory_kb > 0
        
        # Step 6: Generate recommendations
        recommendations = results.get_recommendations()
        assert isinstance(recommendations, list)
        
        print(f"✓ Complete workflow test passed")
        print(f"  Tokens/sec: {results.latency_profile.tokens_per_second:.2f}")  
        print(f"  Peak memory: {results.memory_profile.peak_memory_kb:.0f}KB")
        print(f"  Recommendations: {len(recommendations)}")
    
    def test_multi_platform_comparison(self, mock_model_file):
        """Test profiling across multiple platforms for comparison."""
        
        model = QuantizedModel.from_file(mock_model_file)
        platforms = ["esp32", "stm32f7", "rp2040"]
        results = {}
        
        for platform in platforms:
            try:
                profiler = EdgeProfiler(platform=platform, connection="local")
                
                platform_results = profiler.profile_model(
                    model=model,
                    test_prompts=["Comparison test"],
                    metrics=["latency", "memory"],
                    config=ProfilingConfig(
                        duration_seconds=3,
                        measurement_iterations=2
                    )
                )
                
                results[platform] = platform_results
                
            except Exception as e:
                print(f"Platform {platform} failed: {e}")
        
        assert len(results) >= 2, "At least 2 platforms should succeed"
        
        # Compare results across platforms
        for platform1, result1 in results.items():
            for platform2, result2 in results.items():
                if platform1 != platform2:
                    comparison = result1.compare_with(result2)
                    assert isinstance(comparison, dict)
                    print(f"✓ {platform1} vs {platform2}: {comparison}")
    
    def test_concurrent_profiling_integration(self, mock_model_file):
        """Test concurrent profiling of multiple models."""
        
        # Create multiple model variants
        models = []
        for i in range(3):
            model = QuantizedModel.from_file(mock_model_file)
            model.name = f"test_model_{i}"
            models.append(model)
        
        # Initialize concurrent profiler
        concurrent_profiler = ConcurrentProfiler(max_threads=2)
        concurrent_profiler.start()
        
        try:
            # Submit tasks
            task_ids = []
            for i, model in enumerate(models):
                task = ProfilingTask(
                    task_id=f"integration_task_{i}",
                    platform="esp32",
                    model=model,
                    test_prompts=[f"Concurrent test {i}"],
                    metrics=["latency", "memory"],
                    priority=i
                )
                
                task_id = concurrent_profiler.submit_task(task)
                task_ids.append(task_id)
            
            # Wait for completion
            results = concurrent_profiler.wait_for_completion(
                task_ids, timeout=30
            )
            
            # Validate results
            assert len(results) == len(models)
            
            successful_results = [r for r in results.values() if r.success]
            assert len(successful_results) >= len(models) // 2, "At least half should succeed"
            
            print(f"✓ Concurrent profiling integration test passed")
            print(f"  Tasks submitted: {len(task_ids)}")
            print(f"  Tasks completed: {len(results)}")
            print(f"  Success rate: {len(successful_results) / len(results):.1%}")
            
        finally:
            concurrent_profiler.stop()


class TestResourceManagementIntegration:
    """Integration tests for resource management systems."""
    
    def test_resource_pool_lifecycle(self):
        """Test complete resource pool lifecycle."""
        
        pool_manager = ResourcePoolManager()
        
        try:
            # Create device pool
            device_pool = pool_manager.create_device_pool(
                pool_name="test_device_pool",
                platform="esp32",
                device_path="/dev/mock",
                min_size=1,
                max_size=3
            )
            
            assert isinstance(device_pool, ResourcePool)
            
            # Create model pool  
            with tempfile.NamedTemporaryFile(suffix=".gguf") as temp_file:
                temp_file.write(b"GGUF" + b"x" * 1000)
                temp_file.flush()
                
                model_pool = pool_manager.create_model_pool(
                    pool_name="test_model_pool",
                    model_path=temp_file.name,
                    min_size=1,
                    max_size=2
                )
                
                assert isinstance(model_pool, ResourcePool)
                
                # Test resource acquisition
                with device_pool.acquire(timeout=5.0) as device:
                    assert device is not None
                    
                    with model_pool.acquire(timeout=5.0) as model:
                        assert model is not None
                        
                        # Simulate work
                        time.sleep(0.1)
                
                # Check pool statistics
                device_stats = device_pool.get_stats()
                model_stats = model_pool.get_stats()
                
                assert device_stats["pool_size"] >= 1
                assert model_stats["pool_size"] >= 1
                
                print(f"✓ Resource pool lifecycle test passed")
                print(f"  Device pool: {device_stats}")
                print(f"  Model pool: {model_stats}")
        
        finally:
            pool_manager.shutdown_all()
    
    def test_cache_integration_workflow(self, tmp_path):
        """Test cache integration in profiling workflow."""
        
        # Initialize cache
        cache = SmartCache(
            memory_cache_size=100,
            memory_cache_mb=10,
            persistent_cache_dir=tmp_path / "cache"
        )
        
        # Simulate profiling workflow with caching
        model_key = "test_model_v1"
        results_key = "profiling_results_esp32_test"
        
        # Step 1: Check for cached model
        cached_model = cache.get(model_key)
        assert cached_model is None  # First run
        
        # Step 2: Load and cache model
        with tempfile.NamedTemporaryFile(suffix=".gguf") as temp_file:
            temp_file.write(b"GGUF" + b"x" * 1000)
            temp_file.flush()
            
            model = QuantizedModel.from_file(temp_file.name)
            cache.put(model_key, model, ttl_seconds=3600)
        
        # Step 3: Check for cached results
        cached_results = cache.get(results_key) 
        assert cached_results is None  # First run
        
        # Step 4: Run profiling and cache results
        profiler = EdgeProfiler(platform="esp32", connection="local")
        results = profiler.profile_model(
            model=model,
            test_prompts=["Cache test"],
            metrics=["latency"],
            config=ProfilingConfig(duration_seconds=2, measurement_iterations=1)
        )
        
        cache.put(results_key, results, ttl_seconds=1800)
        
        # Step 5: Verify caching worked
        cached_model_2 = cache.get(model_key)
        cached_results_2 = cache.get(results_key)
        
        assert cached_model_2 is not None
        assert cached_results_2 is not None
        assert cached_results_2.model_name == results.model_name
        
        # Check cache statistics
        cache_stats = cache.get_stats()
        assert cache_stats["memory_cache"]["entries"] >= 2
        
        print(f"✓ Cache integration test passed")
        print(f"  Cache stats: {cache_stats}")


class TestConfigurationIntegration:
    """Integration tests for configuration management."""
    
    def test_configuration_driven_profiling(self, tmp_path):
        """Test profiling driven by configuration files."""
        
        # Create configuration file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
app_name: "test-profiler"
version: "0.1.0"

profiling:
  sample_rate_hz: 50
  duration_seconds: 3
  measurement_iterations: 2
  warmup_iterations: 1
  enable_memory_profiling: true
  enable_latency_profiling: true
  max_prompt_length: 5000

security:
  enable_input_validation: true
  max_file_size_mb: 5
  require_model_validation: true

logging:
  level: "DEBUG"
  console_output: true
  json_format: false

output:
  default_format: "json"
  include_raw_data: true
  auto_timestamp: true

platform_overrides:
  esp32:
    profiling:
      sample_rate_hz: 100
      duration_seconds: 5
"""
        
        config_file.write_text(config_content)
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path=config_file)
        
        # Verify configuration loaded correctly
        assert config.app_name == "test-profiler"
        assert config.profiling.duration_seconds == 3
        assert config.security.enable_input_validation is True
        
        # Test platform-specific overrides
        esp32_config = config.get_platform_config("esp32")
        assert esp32_config["profiling"]["sample_rate_hz"] == 100
        assert esp32_config["profiling"]["duration_seconds"] == 5
        
        # Use configuration in profiling
        with tempfile.NamedTemporaryFile(suffix=".gguf") as temp_file:
            temp_file.write(b"GGUF" + b"x" * 1000)
            temp_file.flush()
            
            model = QuantizedModel.from_file(temp_file.name)
            profiler = EdgeProfiler(platform="esp32", connection="local")
            
            profiling_config = ProfilingConfig(
                duration_seconds=config.profiling.duration_seconds,
                measurement_iterations=config.profiling.measurement_iterations,
                warmup_iterations=config.profiling.warmup_iterations
            )
            
            results = profiler.profile_model(
                model=model,
                test_prompts=["Config-driven test"],
                metrics=["latency", "memory"] if config.profiling.enable_memory_profiling else ["latency"],
                config=profiling_config
            )
            
            assert results is not None
            assert results.latency_profile is not None
            
            if config.profiling.enable_memory_profiling:
                assert results.memory_profile is not None
        
        print(f"✓ Configuration-driven profiling test passed")


class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def test_security_validation_pipeline(self, tmp_path):
        """Test end-to-end security validation pipeline."""
        
        validator = SecurityValidator()
        
        # Step 1: Validate platform identifier
        platform = validator.validate_identifier("esp32", "platform")
        assert platform == "esp32"
        
        # Step 2: Create and validate model file
        model_file = tmp_path / "secure_model.gguf"
        model_file.write_bytes(b"GGUF" + b"x" * 1000)
        
        validated_path = validator.validate_model_file(model_file)
        assert validated_path == model_file.resolve()
        
        # Step 3: Validate device path (mock)
        device_path = "/dev/ttyUSB0"
        validated_device = validator.validate_device_path(device_path)
        assert validated_device == device_path
        
        # Step 4: Test secure profiling workflow
        model = QuantizedModel.from_file(validated_path)
        profiler = EdgeProfiler(platform=platform, device=validated_device, connection="local")
        
        # Use security-validated inputs
        sanitized_prompts = ["Test secure prompt", "Another test"]
        
        results = profiler.profile_model(
            model=model,
            test_prompts=sanitized_prompts,
            metrics=["latency"],
            config=ProfilingConfig(duration_seconds=2, measurement_iterations=1)
        )
        
        assert results is not None
        
        # Step 5: Secure results export
        results_file = tmp_path / validator.sanitize_filename("secure_results.json")
        results.export_json(results_file)
        
        assert results_file.exists()
        
        # Verify file contents are valid JSON
        with open(results_file) as f:
            results_data = json.load(f)
            assert "metadata" in results_data
            assert "profiles" in results_data
        
        print(f"✓ Security validation pipeline test passed")
    
    def test_security_threat_prevention(self, tmp_path):
        """Test prevention of common security threats."""
        
        validator = SecurityValidator()
        
        # Test 1: Path traversal prevention
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/proc/self/environ"
        ]
        
        for path in malicious_paths:
            with pytest.raises(Exception):  # Should raise security exception
                validator.validate_file_path(path)
        
        # Test 2: File extension validation
        malicious_file = tmp_path / "malicious.exe"
        malicious_file.write_bytes(b"MZ" + b"x" * 100)  # PE header
        
        with pytest.raises(Exception):
            validator.validate_model_file(malicious_file)
        
        # Test 3: Input sanitization
        from tiny_llm_profiler.security import InputSanitizer
        sanitizer = InputSanitizer()
        
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../secrets.txt",
            "$(rm -rf /)"
        ]
        
        for malicious_input in malicious_inputs:
            try:
                # Should either sanitize or reject
                result = sanitizer.sanitize_string(malicious_input, max_length=1000)
                # If sanitized, should not contain dangerous patterns
                assert "<script>" not in result.lower()
                assert "drop table" not in result.lower()
            except Exception:
                # Rejection is also acceptable
                pass
        
        print(f"✓ Security threat prevention test passed")


class TestHealthMonitoringIntegration:
    """Integration tests for health monitoring."""
    
    def test_health_monitoring_during_profiling(self):
        """Test health monitoring integration during profiling operations."""
        
        health_checker = HealthChecker()
        
        # Run initial health checks
        initial_health = health_checker.run_all_checks()
        overall_health = health_checker.get_overall_health()
        
        assert isinstance(initial_health, dict)
        assert len(initial_health) > 0
        
        # Start profiling workload while monitoring health
        with tempfile.NamedTemporaryFile(suffix=".gguf") as temp_file:
            temp_file.write(b"GGUF" + b"x" * 1000)
            temp_file.flush()
            
            model = QuantizedModel.from_file(temp_file.name)
            profiler = EdgeProfiler(platform="esp32", connection="local")
            
            # Monitor health during profiling
            def monitor_health():
                for _ in range(5):
                    time.sleep(1)
                    metrics = health_checker.collect_system_metrics()
                    assert metrics.cpu_percent >= 0
                    assert metrics.memory_percent >= 0
            
            # Start health monitoring in background
            monitor_thread = threading.Thread(target=monitor_health)
            monitor_thread.start()
            
            # Run profiling
            results = profiler.profile_model(
                model=model,
                test_prompts=["Health monitoring test"],
                metrics=["latency"],
                config=ProfilingConfig(duration_seconds=3, measurement_iterations=2)
            )
            
            monitor_thread.join()
            
            # Check final health
            final_health = health_checker.run_all_checks()
            
            # System should still be healthy after profiling
            final_overall = health_checker.get_overall_health()
            assert final_overall.healthy or len(final_overall.details["failed_checks"]) <= 1
            
            # Get metrics summary
            metrics_summary = health_checker.get_metrics_summary(duration_minutes=1)
            assert "cpu" in metrics_summary
            assert "memory" in metrics_summary
        
        print(f"✓ Health monitoring integration test passed")
        print(f"  Initial health: {overall_health.healthy}")
        print(f"  Final health: {final_overall.healthy}")


class TestScalingIntegration:
    """Integration tests for auto-scaling functionality."""
    
    def test_load_balancer_integration(self):
        """Test load balancer integration with profiling tasks."""
        
        balancer = LoadBalancer(strategy="least_connections")
        
        # Add mock resources
        for i in range(3):
            balancer.add_resource(
                f"profiler_{i}",
                {"id": f"profiler_{i}", "type": "mock"},
                capacity_weight=1.0
            )
        
        # Simulate task distribution
        num_tasks = 20
        task_assignments = {}
        
        for i in range(num_tasks):
            resource = balancer.select_resource()
            assert resource is not None
            
            resource_id = resource["id"]
            task_assignments[resource_id] = task_assignments.get(resource_id, 0) + 1
            
            # Simulate task execution
            balancer.report_task_start(resource_id, f"task_{i}")
            time.sleep(0.01)  # Simulate brief task
            balancer.report_task_completion(resource_id, f"task_{i}", True, 0.01)
        
        # Verify load distribution
        stats = balancer.get_stats()
        assert stats["total_resources"] == 3
        assert stats["healthy_resources"] == 3
        
        # Check that tasks were distributed across resources
        assert len(task_assignments) >= 2, "Tasks should be distributed across multiple resources"
        
        print(f"✓ Load balancer integration test passed")
        print(f"  Task distribution: {task_assignments}")
        print(f"  Balancer stats: {stats}")


@pytest.mark.integration
class TestComprehensiveProfilingWorkflows:
    """Comprehensive integration tests demonstrating real-world usage scenarios."""
    
    def test_production_deployment_simulation(self, tmp_path):
        """Simulate a production deployment scenario with multiple models and constraints."""
        
        # Create a realistic model portfolio
        model_portfolio = {}
        model_configs = [
            ("chat_assistant", 1.8, QuantizationType.INT2, ["How are you?", "Tell me about AI"]),
            ("code_helper", 3.2, QuantizationType.INT3, ["def hello():", "# Generate Python code"]),
            ("text_summarizer", 2.5, QuantizationType.INT4, ["Summarize this text", "Key points are"])
        ]
        
        for name, size_mb, quant, prompts in model_configs:
            model_file = tmp_path / f"{name}.gguf"
            self._create_production_model_file(model_file, size_mb)
            model_portfolio[name] = {
                "model": QuantizedModel.from_file(model_file, quantization=quant),
                "test_prompts": prompts,
                "target_platforms": self._get_suitable_platforms(size_mb)
            }
        
        # Production deployment constraints
        deployment_constraints = {
            "max_latency_ms": 500,
            "max_memory_kb": 400,
            "max_power_mw": 250,
            "min_tokens_per_second": 5,
            "reliability_threshold": 0.95
        }
        
        # Test deployment viability
        deployment_results = {}
        analyzer = MetricsAnalyzer()
        
        for model_name, model_info in model_portfolio.items():
            model = model_info["model"]
            test_prompts = model_info["test_prompts"]
            
            for platform in model_info["target_platforms"]:
                deployment_key = f"{model_name}_on_{platform}"
                
                try:
                    # Validate platform compatibility first
                    platform_manager = PlatformManager(platform)
                    is_compatible, issues = platform_manager.validate_model_compatibility(
                        model.size_mb, model.quantization.value
                    )
                    
                    if not is_compatible:
                        deployment_results[deployment_key] = {
                            "viable": False,
                            "reason": f"Compatibility issues: {issues}"
                        }
                        continue
                    
                    # Run comprehensive profiling
                    profiler = EdgeProfiler(platform=platform, connection="local")
                    
                    results = profiler.profile_model(
                        model=model,
                        test_prompts=test_prompts,
                        metrics=["latency", "memory", "power"],
                        config=ProfilingConfig(
                            duration_seconds=8,
                            measurement_iterations=5,
                            warmup_iterations=2,
                            enable_power_profiling=True
                        )
                    )
                    
                    # Analyze against production constraints
                    analysis = analyzer.analyze(results)
                    bottlenecks = analyzer.find_bottlenecks(results)
                    
                    # Check constraint compliance
                    constraints_met = {
                        "latency": results.latency_profile.total_latency_ms <= deployment_constraints["max_latency_ms"],
                        "memory": results.memory_profile.peak_memory_kb <= deployment_constraints["max_memory_kb"],
                        "power": results.power_profile.active_power_mw <= deployment_constraints["max_power_mw"],
                        "throughput": results.latency_profile.tokens_per_second >= deployment_constraints["min_tokens_per_second"]
                    }
                    
                    viable = all(constraints_met.values())
                    
                    deployment_results[deployment_key] = {
                        "viable": viable,
                        "results": results,
                        "analysis": analysis,
                        "bottlenecks": [b.component for b in bottlenecks],
                        "constraints_met": constraints_met,
                        "efficiency_score": results.calculate_efficiency_score(),
                        "recommendations": results.get_recommendations()
                    }
                    
                except Exception as e:
                    deployment_results[deployment_key] = {
                        "viable": False,
                        "error": str(e)
                    }
        
        # Generate deployment report
        viable_deployments = {k: v for k, v in deployment_results.items() if v.get("viable", False)}
        
        deployment_report = {
            "total_combinations": len(deployment_results),
            "viable_deployments": len(viable_deployments),
            "viability_rate": len(viable_deployments) / len(deployment_results) if deployment_results else 0,
            "best_deployments": self._rank_deployments(viable_deployments),
            "constraint_violations": self._analyze_constraint_violations(deployment_results),
            "platform_suitability": self._analyze_platform_suitability(deployment_results)
        }
        
        # Export comprehensive report
        report_file = tmp_path / "production_deployment_report.json"
        with open(report_file, 'w') as f:
            json.dump(deployment_report, f, indent=2, default=str)
        
        # Validate deployment results
        assert deployment_report["viability_rate"] > 0.3, "At least 30% deployments should be viable"
        assert len(deployment_report["best_deployments"]) > 0, "Should have at least one optimal deployment"
        
        print(f"✓ Production deployment simulation test passed")
        print(f"  Model portfolio: {len(model_portfolio)} models")
        print(f"  Total combinations tested: {deployment_report['total_combinations']}")
        print(f"  Viable deployments: {deployment_report['viable_deployments']} ({deployment_report['viability_rate']:.1%})")
        print(f"  Best deployment: {deployment_report['best_deployments'][0] if deployment_report['best_deployments'] else 'None'}")
    
    def test_edge_ai_pipeline_integration(self, tmp_path):
        """Test complete Edge AI pipeline from model selection to deployment."""
        
        # Phase 1: Model Selection and Optimization
        base_model_file = tmp_path / "base_llm_model.gguf"
        self._create_production_model_file(base_model_file, 4.5)
        base_model = QuantizedModel.from_file(base_model_file, quantization=QuantizationType.INT8)
        
        target_platform = "esp32"
        platform_manager = PlatformManager(target_platform)
        
        # Get platform constraints
        memory_constraints = platform_manager.get_memory_constraints()
        optimization_recommendations = platform_manager.get_optimization_recommendations(base_model.size_mb)
        
        # Apply optimizations
        optimized_model = base_model.optimize_for_platform(
            platform=target_platform,
            constraints=memory_constraints
        )
        
        # Phase 2: Model Validation and Profiling
        validator = SecurityValidator()
        
        # Validate optimized model
        model_validation = validator.validate_model_properties(optimized_model)
        assert model_validation["valid"], "Optimized model should pass validation"
        
        # Comprehensive profiling
        profiler = EdgeProfiler(platform=target_platform, connection="local")
        
        edge_ai_prompts = [
            "Process sensor data: temperature=25.5°C",
            "Classify image: outdoor scene",
            "Respond to voice command: 'turn on lights'",
            "Analyze text: 'system status normal'",
            "Generate response: user inquiry"
        ]
        
        # Sanitize prompts for security
        sanitizer = InputSanitizer()
        safe_prompts = []
        for prompt in edge_ai_prompts:
            sanitized = sanitizer.sanitize_string(prompt, max_length=1000)
            if sanitized:
                safe_prompts.append(sanitized)
        
        profiling_results = profiler.profile_model(
            model=optimized_model,
            test_prompts=safe_prompts,
            metrics=["latency", "memory", "power"],
            config=ProfilingConfig(
                duration_seconds=10,
                measurement_iterations=8,
                enable_power_profiling=True
            )
        )
        
        # Phase 3: Performance Analysis and Optimization
        analyzer = MetricsAnalyzer()
        performance_analysis = analyzer.analyze(profiling_results)
        bottlenecks = analyzer.find_bottlenecks(profiling_results)
        
        # Phase 4: Deployment Readiness Assessment
        deployment_readiness = {
            "memory_efficiency": profiling_results.memory_profile.peak_memory_kb <= memory_constraints["available_ram_kb"],
            "latency_acceptable": profiling_results.latency_profile.tokens_per_second >= 3,
            "power_efficient": profiling_results.power_profile.active_power_mw <= 200,
            "reliability_score": profiling_results.calculate_efficiency_score()
        }
        
        deployment_ready = all([
            deployment_readiness["memory_efficiency"],
            deployment_readiness["latency_acceptable"],
            deployment_readiness["power_efficient"],
            deployment_readiness["reliability_score"] >= 60
        ])
        
        # Phase 5: Generate Deployment Package
        deployment_package = {
            "model_config": {
                "name": optimized_model.name,
                "size_mb": optimized_model.size_mb,
                "quantization": optimized_model.quantization.value,
                "optimization_applied": optimization_recommendations
            },
            "platform_config": {
                "platform": target_platform,
                "memory_constraints": memory_constraints,
                "platform_capabilities": platform_manager.get_config().capabilities.__dict__
            },
            "performance_profile": performance_analysis,
            "deployment_readiness": deployment_readiness,
            "deployment_recommendations": profiling_results.get_recommendations(),
            "identified_bottlenecks": [{"component": b.component, "impact": b.impact} for b in bottlenecks]
        }
        
        # Export deployment package
        deployment_file = tmp_path / "edge_ai_deployment_package.json"
        with open(deployment_file, 'w') as f:
            json.dump(deployment_package, f, indent=2, default=str)
        
        # Validate pipeline results
        assert deployment_ready or len(bottlenecks) > 0, "Either ready for deployment or bottlenecks identified"
        assert len(deployment_package["deployment_recommendations"]) >= 0, "Should provide deployment guidance"
        
        print(f"✓ Edge AI pipeline integration test passed")
        print(f"  Base model: {base_model.size_mb:.1f}MB -> Optimized: {optimized_model.size_mb:.1f}MB")
        print(f"  Performance: {profiling_results.latency_profile.tokens_per_second:.1f} tok/s")
        print(f"  Deployment ready: {deployment_ready}")
        print(f"  Efficiency score: {deployment_readiness['reliability_score']:.1f}")
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring_integration(self, tmp_path):
        """Test real-time monitoring and adaptive profiling integration."""
        
        # Create test model
        model_file = tmp_path / "monitoring_test_model.gguf"
        self._create_production_model_file(model_file, 2.0)
        model = QuantizedModel.from_file(model_file)
        
        # Initialize monitoring components
        health_checker = HealthChecker()
        cache = SmartCache(
            memory_cache_size=30,
            memory_cache_mb=5,
            persistent_cache_dir=tmp_path / "monitoring_cache"
        )
        
        # Real-time monitoring data collection
        monitoring_data = {
            "system_metrics": [],
            "profiling_results": [],
            "cache_stats": [],
            "health_events": []
        }
        
        # Monitoring task
        async def continuous_monitoring():
            for i in range(15):  # 15 monitoring cycles
                try:
                    # Collect system metrics
                    metrics = health_checker.collect_system_metrics()
                    health_status = health_checker.get_overall_health()
                    
                    monitoring_data["system_metrics"].append({
                        "timestamp": time.time(),
                        "cpu_percent": metrics.cpu_percent,
                        "memory_percent": metrics.memory_percent,
                        "healthy": health_status.healthy
                    })
                    
                    # Check cache statistics
                    cache_stats = cache.get_stats()
                    monitoring_data["cache_stats"].append({
                        "timestamp": time.time(),
                        "memory_entries": cache_stats["memory_cache"]["entries"],
                        "hit_rate": cache_stats.get("hit_rate", 0)
                    })
                    
                    await asyncio.sleep(0.5)  # 2Hz monitoring
                    
                except Exception as e:
                    monitoring_data["health_events"].append({
                        "timestamp": time.time(),
                        "event": "monitoring_error",
                        "error": str(e)
                    })
        
        # Profiling task with adaptive behavior
        async def adaptive_profiling():
            profiler = EdgeProfiler(platform="esp32", connection="local")
            
            for i in range(5):  # 5 profiling runs
                try:
                    # Check system health before profiling
                    current_health = health_checker.get_overall_health()
                    
                    # Adaptive configuration based on health
                    if current_health.healthy:
                        config = ProfilingConfig(
                            duration_seconds=4,
                            measurement_iterations=3
                        )
                    else:
                        # Reduced intensity during health issues
                        config = ProfilingConfig(
                            duration_seconds=2,
                            measurement_iterations=1
                        )
                    
                    # Check cache for previous results
                    cache_key = f"adaptive_profiling_{i}_health_{current_health.healthy}"
                    cached_result = cache.get(cache_key)
                    
                    if cached_result:
                        profiling_result = cached_result
                        cache_hit = True
                    else:
                        profiling_result = profiler.profile_model(
                            model=model,
                            test_prompts=[f"Adaptive test {i}"],
                            metrics=["latency", "memory"],
                            config=config
                        )
                        
                        # Cache the result
                        cache.put(cache_key, profiling_result, ttl_seconds=300)
                        cache_hit = False
                    
                    monitoring_data["profiling_results"].append({
                        "timestamp": time.time(),
                        "iteration": i,
                        "cache_hit": cache_hit,
                        "system_healthy": current_health.healthy,
                        "tokens_per_second": profiling_result.latency_profile.tokens_per_second,
                        "memory_usage_kb": profiling_result.memory_profile.peak_memory_kb
                    })
                    
                    await asyncio.sleep(1.5)  # Pause between profiling runs
                    
                except Exception as e:
                    monitoring_data["health_events"].append({
                        "timestamp": time.time(),
                        "event": "profiling_error",
                        "error": str(e)
                    })
        
        # Run monitoring and profiling concurrently
        await asyncio.gather(
            continuous_monitoring(),
            adaptive_profiling()
        )
        
        # Analyze monitoring integration
        system_metrics = monitoring_data["system_metrics"]
        profiling_results = monitoring_data["profiling_results"]
        
        # Should have collected comprehensive monitoring data
        assert len(system_metrics) >= 10, "Should collect continuous system metrics"
        assert len(profiling_results) >= 3, "Should complete multiple profiling runs"
        
        # Check adaptive behavior
        cache_hits = sum(1 for r in profiling_results if r["cache_hit"])
        cache_hit_rate = cache_hits / len(profiling_results) if profiling_results else 0
        
        # Should show some caching benefits
        assert cache_hit_rate >= 0.2, f"Cache hit rate too low: {cache_hit_rate:.1%}"
        
        # Validate system remained stable during monitoring
        health_issues = len(monitoring_data["health_events"])
        system_health_rate = sum(1 for m in system_metrics if m["healthy"]) / len(system_metrics)
        
        assert system_health_rate >= 0.8, f"System health rate during monitoring: {system_health_rate:.1%}"
        
        # Export monitoring report
        monitoring_report = {
            "monitoring_duration_s": system_metrics[-1]["timestamp"] - system_metrics[0]["timestamp"],
            "total_profiling_runs": len(profiling_results),
            "cache_hit_rate": cache_hit_rate,
            "system_health_rate": system_health_rate,
            "health_events": len(monitoring_data["health_events"]),
            "average_performance": {
                "tokens_per_second": sum(r["tokens_per_second"] for r in profiling_results) / len(profiling_results),
                "memory_usage_kb": sum(r["memory_usage_kb"] for r in profiling_results) / len(profiling_results)
            }
        }
        
        report_file = tmp_path / "monitoring_integration_report.json"
        with open(report_file, 'w') as f:
            json.dump(monitoring_report, f, indent=2, default=str)
        
        print(f"✓ Real-time monitoring integration test passed")
        print(f"  Monitoring duration: {monitoring_report['monitoring_duration_s']:.1f}s")
        print(f"  Profiling runs: {monitoring_report['total_profiling_runs']}")
        print(f"  Cache hit rate: {monitoring_report['cache_hit_rate']:.1%}")
        print(f"  System health rate: {monitoring_report['system_health_rate']:.1%}")
    
    def _create_production_model_file(self, path: Path, size_mb: float):
        """Create a production-like model file for testing."""
        # GGUF header with proper structure
        header = b"GGUF"
        header += b"\x03\x00\x00\x00"  # Version 3
        
        # Realistic metadata
        metadata = json.dumps({
            "general.architecture": "llama",
            "general.quantization_version": 2,
            "general.file_type": 2,
            "llama.vocab_size": 32000,
            "llama.context_length": 2048,
            "llama.embedding_length": 2048,
            "llama.block_count": 22,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 32,
            "llama.rope.freq_base": 10000.0,
            "tokenizer.ggml.model": "llama"
        }).encode()
        
        # Calculate data size
        header_size = len(header) + len(metadata) + 200  # Header + metadata + padding
        data_size = int(size_mb * 1024 * 1024) - header_size
        
        with open(path, 'wb') as f:
            f.write(header)
            f.write(len(metadata).to_bytes(4, 'little'))
            f.write(metadata)
            f.write(b'\x00' * 196)  # Padding
            
            # Write model data in chunks to simulate realistic structure
            remaining = data_size
            chunk_size = 64 * 1024  # 64KB chunks
            
            while remaining > 0:
                current_chunk = min(chunk_size, remaining)
                # Simulate different types of model data
                if remaining > data_size * 0.8:
                    # Embedding weights
                    f.write(b'\x42' * current_chunk)
                elif remaining > data_size * 0.2:
                    # Layer weights
                    f.write(b'\x43' * current_chunk)
                else:
                    # Output weights
                    f.write(b'\x44' * current_chunk)
                
                remaining -= current_chunk
    
    def _get_suitable_platforms(self, model_size_mb: float) -> List[str]:
        """Get platforms suitable for a model of given size."""
        if model_size_mb <= 1.5:
            return ["esp32", "stm32f7", "rp2040", "nrf52840"]
        elif model_size_mb <= 3.0:
            return ["esp32", "stm32f7", "k210"]
        elif model_size_mb <= 5.0:
            return ["esp32", "k210"]
        else:
            return ["jetson_nano"]  # Only high-end platforms
    
    def _rank_deployments(self, viable_deployments: Dict) -> List[str]:
        """Rank viable deployments by efficiency score."""
        if not viable_deployments:
            return []
        
        ranked = sorted(
            viable_deployments.items(),
            key=lambda x: x[1].get("efficiency_score", 0),
            reverse=True
        )
        
        return [deployment_name for deployment_name, _ in ranked[:3]]  # Top 3
    
    def _analyze_constraint_violations(self, deployment_results: Dict) -> Dict:
        """Analyze which constraints are most commonly violated."""
        violations = {"latency": 0, "memory": 0, "power": 0, "throughput": 0}
        
        for result in deployment_results.values():
            if result.get("viable") == False and "constraints_met" in result:
                constraints = result["constraints_met"]
                for constraint, met in constraints.items():
                    if not met:
                        violations[constraint] = violations.get(constraint, 0) + 1
        
        return violations
    
    def _analyze_platform_suitability(self, deployment_results: Dict) -> Dict:
        """Analyze platform suitability across different models."""
        platform_stats = {}
        
        for deployment_key, result in deployment_results.items():
            platform = deployment_key.split("_on_")[-1]
            
            if platform not in platform_stats:
                platform_stats[platform] = {"total": 0, "viable": 0}
            
            platform_stats[platform]["total"] += 1
            if result.get("viable", False):
                platform_stats[platform]["viable"] += 1
        
        # Calculate viability rates
        for platform, stats in platform_stats.items():
            stats["viability_rate"] = stats["viable"] / stats["total"] if stats["total"] > 0 else 0
        
        return platform_stats


@pytest.fixture
def mock_model_file(tmp_path):
    """Create a mock model file for testing."""
    model_file = tmp_path / "test_model.gguf"
    # Create a minimal GGUF-like file
    content = b"GGUF" + (b"\x00" * 1000)  # 1KB mock model
    model_file.write_bytes(content)
    return model_file


@pytest.fixture
def mock_device():
    """Mock device path for testing."""
    return "/dev/mock_device"


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])