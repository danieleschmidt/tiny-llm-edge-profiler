"""
Comprehensive test suite for core functionality.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_llm_profiler.core_lite import (
    SimpleProfiler, SimplePlatformManager, SimplifiedProfile,
    run_basic_benchmark, analyze_benchmark_results
)
from tiny_llm_profiler.health_monitor import HealthMonitor, get_health_monitor
from tiny_llm_profiler.performance_cache import PerformanceCache, OptimizedProfiler
from tiny_llm_profiler.scalable_profiler import ConcurrentProfiler, ProfileTask


class TestSimplePlatformManager(unittest.TestCase):
    """Test SimplePlatformManager functionality."""
    
    def test_get_supported_platforms(self):
        """Test platform enumeration."""
        platforms = SimplePlatformManager.get_supported_platforms()
        
        self.assertIsInstance(platforms, list)
        self.assertGreater(len(platforms), 0)
        self.assertIn("esp32", platforms)
        self.assertIn("stm32f4", platforms)
    
    def test_get_platform_info(self):
        """Test platform information retrieval."""
        info = SimplePlatformManager.get_platform_info("esp32")
        
        self.assertIsNotNone(info)
        self.assertIn("name", info)
        self.assertIn("ram_kb", info)
        self.assertIn("flash_kb", info)
        self.assertEqual(info["name"], "ESP32")
        self.assertGreater(info["ram_kb"], 0)
    
    def test_get_platform_info_invalid(self):
        """Test platform info for invalid platform."""
        info = SimplePlatformManager.get_platform_info("invalid_platform")
        self.assertIsNone(info)


class TestSimpleProfiler(unittest.TestCase):
    """Test SimpleProfiler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = SimpleProfiler("esp32")
        self.test_prompts = ["Hello world", "Generate code", "Explain AI"]
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        self.assertEqual(self.profiler.platform, "esp32")
        self.assertEqual(self.profiler.connection, "local")
        self.assertIsNotNone(self.profiler.platform_info)
    
    def test_profiler_initialization_invalid_platform(self):
        """Test profiler initialization with invalid platform."""
        with self.assertRaises(ValueError):
            SimpleProfiler("invalid_platform")
    
    def test_simulate_profiling(self):
        """Test basic profiling simulation."""
        result = self.profiler.simulate_profiling(self.test_prompts)
        
        self.assertIsInstance(result, SimplifiedProfile)
        self.assertTrue(result.success)
        self.assertGreater(result.tokens_per_second, 0)
        self.assertGreater(result.latency_ms, 0)
        self.assertGreater(result.memory_kb, 0)
        self.assertGreater(result.power_mw, 0)
        self.assertEqual(result.error_message, "")
    
    def test_simulate_profiling_empty_prompts(self):
        """Test profiling with empty prompts."""
        result = self.profiler.simulate_profiling([])
        
        self.assertIsInstance(result, SimplifiedProfile)
        self.assertTrue(result.success)
    
    def test_simulate_profiling_different_platforms(self):
        """Test profiling across different platforms."""
        platforms = ["esp32", "stm32f4", "stm32f7", "rp2040"]
        results = {}
        
        for platform in platforms:
            profiler = SimpleProfiler(platform)
            result = profiler.simulate_profiling(self.test_prompts)
            results[platform] = result
            
            self.assertTrue(result.success)
            self.assertGreater(result.tokens_per_second, 0)
        
        # ESP32 should generally be fastest
        self.assertGreaterEqual(
            results["esp32"].tokens_per_second,
            results["rp2040"].tokens_per_second
        )


class TestBenchmarkFunctions(unittest.TestCase):
    """Test benchmark utility functions."""
    
    def test_run_basic_benchmark(self):
        """Test basic benchmark execution."""
        results = run_basic_benchmark()
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check each platform result
        for platform, result in results.items():
            self.assertIn("success", result)
            if result["success"]:
                self.assertIn("tokens_per_second", result)
                self.assertIn("memory_kb", result)
                self.assertGreater(result["tokens_per_second"], 0)
    
    def test_analyze_benchmark_results(self):
        """Test benchmark results analysis."""
        # Create mock results
        mock_results = {
            "esp32": {
                "success": True,
                "tokens_per_second": 10.0,
                "memory_kb": 150.0,
                "power_mw": 200.0
            },
            "stm32f4": {
                "success": True,
                "tokens_per_second": 6.0,
                "memory_kb": 60.0,
                "power_mw": 120.0
            }
        }
        
        analysis = analyze_benchmark_results(mock_results)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("platforms_tested", analysis)
        self.assertIn("best_performer", analysis)
        self.assertIn("performance_stats", analysis)
        
        self.assertEqual(analysis["platforms_tested"], 2)
        self.assertEqual(analysis["best_performer"], "esp32")
        
        stats = analysis["performance_stats"]
        self.assertIn("avg_tokens_per_second", stats)
        self.assertEqual(stats["max_tokens_per_second"], 10.0)
    
    def test_analyze_benchmark_results_no_success(self):
        """Test analysis with no successful results."""
        mock_results = {
            "esp32": {"success": False, "error": "Test error"}
        }
        
        analysis = analyze_benchmark_results(mock_results)
        self.assertIn("error", analysis)


class TestHealthMonitor(unittest.TestCase):
    """Test health monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = HealthMonitor(check_interval=1)
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        self.assertFalse(self.monitor.is_monitoring)
        self.assertGreater(len(self.monitor.health_checks), 0)
    
    def test_get_current_health(self):
        """Test getting current health status."""
        health = self.monitor.get_current_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        # Should be unknown initially
        self.assertIn(health["status"], ["unknown", "healthy", "warning", "critical"])
    
    def test_add_custom_check(self):
        """Test adding custom health checks."""
        initial_count = len(self.monitor.health_checks)
        
        def test_check():
            return True
        
        self.monitor.add_custom_check(
            name="test_check",
            check_function=test_check,
            critical=False
        )
        
        self.assertEqual(len(self.monitor.health_checks), initial_count + 1)
        
        # Find the added check
        test_check_found = any(
            check.name == "test_check" for check in self.monitor.health_checks
        )
        self.assertTrue(test_check_found)


class TestPerformanceCache(unittest.TestCase):
    """Test performance caching functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = PerformanceCache(max_size=10, ttl_seconds=60)
        self.test_prompts = ["Hello", "Test"]
        self.test_config = {"param": "value"}
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        self.assertEqual(self.cache.max_size, 10)
        self.assertEqual(self.cache.ttl_seconds, 60)
        self.assertEqual(len(self.cache.cache), 0)
    
    def test_cache_key_creation(self):
        """Test cache key creation."""
        key = self.cache.create_key(
            platform="esp32",
            model_size_mb=2.5,
            quantization="4bit",
            prompts=self.test_prompts,
            config=self.test_config
        )
        
        self.assertEqual(key.platform, "esp32")
        self.assertEqual(key.model_size_mb, 2.5)
        self.assertEqual(key.quantization, "4bit")
        self.assertIsNotNone(key.prompts_hash)
        self.assertIsNotNone(key.config_hash)
    
    def test_cache_put_get(self):
        """Test cache put and get operations."""
        key = self.cache.create_key(
            platform="esp32",
            model_size_mb=2.5,
            quantization="4bit",
            prompts=self.test_prompts
        )
        
        test_result = {"tokens_per_second": 10.0, "success": True}
        
        # Cache miss initially
        result = self.cache.get(key)
        self.assertIsNone(result)
        
        # Put result in cache
        self.cache.put(key, test_result)
        
        # Cache hit
        result = self.cache.get(key)
        self.assertIsNotNone(result)
        self.assertEqual(result["tokens_per_second"], 10.0)
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("size", stats)
        self.assertIn("max_size", stats)
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertEqual(stats["max_size"], 10)


class TestOptimizedProfiler(unittest.TestCase):
    """Test optimized profiler with caching."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = OptimizedProfiler(cache_size=10, cache_ttl=60)
        self.test_prompts = ["Hello", "Test"]
    
    def test_profile_with_cache(self):
        """Test profiling with cache."""
        # First call - cache miss
        result1 = self.profiler.profile_with_cache(
            platform="esp32",
            model_size_mb=2.5,
            quantization="4bit",
            prompts=self.test_prompts
        )
        
        self.assertIsInstance(result1, dict)
        self.assertIn("success", result1)
        self.assertTrue(result1["success"])
        
        # Second call - cache hit (should be faster)
        result2 = self.profiler.profile_with_cache(
            platform="esp32",
            model_size_mb=2.5,
            quantization="4bit",
            prompts=self.test_prompts
        )
        
        self.assertIsInstance(result2, dict)
        
        # Remove metadata for comparison
        result1_clean = {k: v for k, v in result1.items() if k != "profiling_metadata"}
        result2_clean = {k: v for k, v in result2.items() if k != "profiling_metadata"}
        
        self.assertEqual(result1_clean, result2_clean)
    
    def test_get_performance_stats(self):
        """Test performance statistics."""
        # Generate some activity
        self.profiler.profile_with_cache(
            platform="esp32",
            model_size_mb=2.5,
            quantization="4bit",
            prompts=self.test_prompts
        )
        
        stats = self.profiler.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("cache_stats", stats)
        self.assertIn("profiler_stats", stats)


class TestConcurrentProfiler(unittest.TestCase):
    """Test concurrent profiler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = ConcurrentProfiler(max_workers=2)
        self.test_prompts = ["Hello", "Test"]
    
    def test_profiler_initialization(self):
        """Test concurrent profiler initialization."""
        self.assertEqual(self.profiler.max_workers, 2)
        self.assertFalse(self.profiler.is_running)
        self.assertEqual(len(self.profiler.workers), 0)
    
    def test_start_stop_profiler(self):
        """Test starting and stopping the profiler."""
        self.profiler.start()
        self.assertTrue(self.profiler.is_running)
        self.assertEqual(len(self.profiler.workers), 2)
        
        self.profiler.stop()
        self.assertFalse(self.profiler.is_running)
        self.assertEqual(len(self.profiler.workers), 0)
    
    def test_submit_task(self):
        """Test task submission."""
        self.profiler.start()
        
        try:
            task_id = self.profiler.submit_task(
                platform="esp32",
                test_prompts=self.test_prompts
            )
            
            self.assertIsInstance(task_id, str)
            self.assertIn("task_", task_id)
            
            # Wait for completion
            result = self.profiler.get_result(task_id, timeout=5.0)
            self.assertIsNotNone(result)
            self.assertEqual(result.task_id, task_id)
            self.assertEqual(result.platform, "esp32")
            
        finally:
            self.profiler.stop()
    
    def test_get_stats(self):
        """Test getting profiler statistics."""
        stats = self.profiler.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("runtime_seconds", stats)
        self.assertIn("tasks_submitted", stats)
        self.assertIn("active_workers", stats)


class TestProfileTask(unittest.TestCase):
    """Test ProfileTask functionality."""
    
    def test_task_creation(self):
        """Test task creation."""
        task = ProfileTask(
            task_id="test_task",
            platform="esp32",
            test_prompts=["Hello"],
            config={},
            priority=5
        )
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.platform, "esp32")
        self.assertEqual(task.priority, 5)
    
    def test_task_comparison(self):
        """Test task priority comparison."""
        task1 = ProfileTask("task1", "esp32", ["Hello"], {}, priority=1)
        task2 = ProfileTask("task2", "esp32", ["Hello"], {}, priority=5)
        
        # Lower priority number = higher priority
        self.assertTrue(task1 < task2)
        self.assertFalse(task2 < task1)


if __name__ == "__main__":
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSimplePlatformManager,
        TestSimpleProfiler,
        TestBenchmarkFunctions,
        TestHealthMonitor,
        TestPerformanceCache,
        TestOptimizedProfiler,
        TestConcurrentProfiler,
        TestProfileTask
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if not result.failures and not result.errors else 1
    sys.exit(exit_code)