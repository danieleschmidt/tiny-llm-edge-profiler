"""
Test suite for lightweight functionality without numpy dependencies.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "tiny_llm_profiler"))

# Direct imports to avoid numpy dependency
from core_lite import (
    SimpleProfiler, SimplePlatformManager, SimplifiedProfile,
    run_basic_benchmark, analyze_benchmark_results
)
from health_monitor import HealthMonitor
from performance_cache import PerformanceCache, OptimizedProfiler


class TestLiteCore(unittest.TestCase):
    """Test lightweight core functionality."""
    
    def test_platforms(self):
        """Test platform management."""
        platforms = SimplePlatformManager.get_supported_platforms()
        self.assertGreater(len(platforms), 0)
        self.assertIn("esp32", platforms)
        
        info = SimplePlatformManager.get_platform_info("esp32")
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "ESP32")
    
    def test_profiler(self):
        """Test basic profiling."""
        profiler = SimpleProfiler("esp32")
        result = profiler.simulate_profiling(["Hello", "World"])
        
        self.assertIsInstance(result, SimplifiedProfile)
        self.assertTrue(result.success)
        self.assertGreater(result.tokens_per_second, 0)
    
    def test_benchmark(self):
        """Test benchmark functionality."""
        results = run_basic_benchmark()
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        analysis = analyze_benchmark_results(results)
        self.assertIn("best_performer", analysis)
    
    def test_health_monitor(self):
        """Test health monitoring."""
        monitor = HealthMonitor(check_interval=1)
        health = monitor.get_current_health()
        
        self.assertIn("status", health)
    
    def test_performance_cache(self):
        """Test performance caching."""
        cache = PerformanceCache(max_size=10)
        key = cache.create_key(
            platform="esp32",
            model_size_mb=2.5,
            quantization="4bit",
            prompts=["test"]
        )
        
        # Cache miss
        result = cache.get(key)
        self.assertIsNone(result)
        
        # Cache put/get
        test_data = {"tokens_per_second": 10.0}
        cache.put(key, test_data)
        
        result = cache.get(key)
        self.assertIsNotNone(result)
        self.assertEqual(result["tokens_per_second"], 10.0)
    
    def test_optimized_profiler(self):
        """Test optimized profiler with caching."""
        profiler = OptimizedProfiler()
        
        # First call
        result1 = profiler.profile_with_cache(
            platform="esp32",
            model_size_mb=2.5,
            quantization="4bit",
            prompts=["test"]
        )
        
        # Second call (should hit cache)
        result2 = profiler.profile_with_cache(
            platform="esp32",
            model_size_mb=2.5,
            quantization="4bit",
            prompts=["test"]
        )
        
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)
        
        stats = profiler.get_performance_stats()
        self.assertIn("cache_stats", stats)


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)