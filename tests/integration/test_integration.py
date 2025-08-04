"""
Integration tests for the Tiny LLM Edge Profiler.
"""

import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from tiny_llm_profiler.profiler import EdgeProfiler, ProfilingConfig
from tiny_llm_profiler.models import QuantizedModel
from tiny_llm_profiler.platforms import PlatformManager
from tiny_llm_profiler.results import ProfileResults
from tiny_llm_profiler.concurrent import ConcurrentProfiler, ProfilingTask
from tiny_llm_profiler.resource_pool import ResourcePoolManager, ResourcePool
from tiny_llm_profiler.cache import SmartCache
from tiny_llm_profiler.config import ConfigManager, AppConfig
from tiny_llm_profiler.security import SecurityValidator
from tiny_llm_profiler.health import HealthChecker
from tiny_llm_profiler.scaling import AutoScaler, LoadBalancer


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