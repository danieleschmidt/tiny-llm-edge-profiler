"""
Resource Management and Caching Integration Tests

Tests that verify resource pooling, caching mechanisms, and memory management
in realistic profiling scenarios.
"""

import pytest
import asyncio
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import pickle
import json

from tiny_llm_profiler.profiler import EdgeProfiler, ProfilingConfig
from tiny_llm_profiler.models import QuantizedModel, QuantizationType
from tiny_llm_profiler.results import ProfileResults
from tiny_llm_profiler.resource_pool import ResourcePoolManager, ResourcePool
from tiny_llm_profiler.cache import SmartCache
from tiny_llm_profiler.performance_cache import PerformanceCache


@pytest.mark.integration
class TestResourcePoolIntegration:
    """Test resource pool management in realistic scenarios."""
    
    def test_device_pool_lifecycle_management(self, tmp_path):
        """Test complete device pool lifecycle with realistic usage patterns."""
        
        pool_manager = ResourcePoolManager()
        
        try:
            # Create device pools for different platforms
            pools = {}
            
            for platform in ["esp32", "stm32f7", "rp2040"]:
                pool = pool_manager.create_device_pool(
                    pool_name=f"{platform}_device_pool",
                    platform=platform,
                    device_path=f"/dev/mock_{platform}",
                    min_size=2,
                    max_size=5
                )
                pools[platform] = pool
                
                assert isinstance(pool, ResourcePool)
                assert pool.platform == platform
                assert pool.min_size == 2
                assert pool.max_size == 5
            
            # Test concurrent device acquisition
            def acquire_and_use_device(platform: str, hold_time: float, results: list):
                """Acquire device, simulate usage, then release."""
                try:
                    pool = pools[platform]
                    with pool.acquire(timeout=10.0) as device:
                        assert device is not None
                        
                        # Simulate device usage
                        start_time = time.time()
                        time.sleep(hold_time)
                        end_time = time.time()
                        
                        results.append({
                            "platform": platform,
                            "hold_time": hold_time,
                            "actual_time": end_time - start_time,
                            "device_id": getattr(device, 'device_id', 'unknown'),
                            "success": True
                        })
                        
                except Exception as e:
                    results.append({
                        "platform": platform,
                        "error": str(e),
                        "success": False
                    })
            
            # Launch concurrent device usage
            threads = []
            results = []
            
            for platform in pools.keys():
                for i in range(8):  # More threads than max pool size
                    thread = threading.Thread(
                        target=acquire_and_use_device,
                        args=(platform, 0.5, results)
                    )
                    threads.append(thread)
                    thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join(timeout=30)
            
            # Analyze results
            successful_acquisitions = [r for r in results if r["success"]]
            failed_acquisitions = [r for r in results if not r["success"]]
            
            # All acquisitions should eventually succeed
            assert len(successful_acquisitions) >= len(threads) * 0.9, "Most acquisitions should succeed"
            
            # Check pool statistics
            for platform, pool in pools.items():
                stats = pool.get_stats()
                assert stats["pool_size"] >= pool.min_size
                assert stats["pool_size"] <= pool.max_size
                assert stats["total_acquisitions"] > 0
                
                print(f"  {platform} pool stats: {stats}")
        
        finally:
            pool_manager.shutdown_all()
        
        print(f"✓ Device pool lifecycle test passed")
        print(f"  Pools created: {len(pools)}")
        print(f"  Successful acquisitions: {len(successful_acquisitions)}")
    
    def test_model_pool_with_optimization_caching(self, tmp_path):
        """Test model pooling with platform-specific optimization caching."""
        
        pool_manager = ResourcePoolManager()
        
        try:
            # Create base models of different sizes
            base_models = {}
            for i, size_mb in enumerate([1.5, 2.5, 4.0]):
                model_file = tmp_path / f"base_model_{i}.gguf"
                self._create_model_file(model_file, size_mb)
                base_models[f"model_{i}"] = str(model_file)
            
            # Create model pools for different platforms
            model_pools = {}
            
            for platform in ["esp32", "stm32f4"]:
                for model_name, model_path in base_models.items():
                    pool_name = f"{model_name}_{platform}_pool"
                    
                    pool = pool_manager.create_model_pool(
                        pool_name=pool_name,
                        model_path=model_path,
                        platform=platform,
                        min_size=1,
                        max_size=3
                    )
                    
                    model_pools[pool_name] = pool
            
            # Test model acquisition and platform optimization
            optimization_results = {}
            
            for pool_name, pool in model_pools.items():
                parts = pool_name.split("_")
                model_name = f"{parts[0]}_{parts[1]}"
                platform = parts[2]
                
                # Acquire model from pool
                with pool.acquire(timeout=5.0) as model:
                    assert model is not None
                    assert isinstance(model, QuantizedModel)
                    
                    # Model should be optimized for the platform
                    memory_reqs = model.get_memory_requirements(platform)
                    
                    optimization_results[pool_name] = {
                        "model_size_mb": model.size_mb,
                        "quantization": model.quantization.value,
                        "memory_requirements": memory_reqs,
                        "platform": platform
                    }
            
            # Validate platform-specific optimizations
            for pool_name, result in optimization_results.items():
                platform = result["platform"]
                memory_reqs = result["memory_requirements"]
                
                if platform == "stm32f4":
                    # STM32F4 has limited memory, should be more aggressively optimized
                    assert memory_reqs["total_estimated_kb"] <= 150, f"STM32F4 model too large: {memory_reqs}"
                
                elif platform == "esp32":
                    # ESP32 has more memory, can handle larger models
                    assert memory_reqs["total_estimated_kb"] <= 400, f"ESP32 model too large: {memory_reqs}"
            
            # Test pool statistics
            for pool_name, pool in model_pools.items():
                stats = pool.get_stats()
                assert stats["pool_size"] >= 1
                assert stats["total_acquisitions"] >= 1
        
        finally:
            pool_manager.shutdown_all()
        
        print(f"✓ Model pool with optimization caching test passed")
        print(f"  Model pools created: {len(model_pools)}")
        print(f"  Optimization results: {len(optimization_results)}")
    
    def test_resource_pool_under_pressure(self, tmp_path):
        """Test resource pool behavior under high pressure scenarios."""
        
        pool_manager = ResourcePoolManager()
        
        try:
            # Create a small pool with high contention
            model_file = tmp_path / "pressure_test_model.gguf"
            self._create_model_file(model_file, 1.8)
            
            pool = pool_manager.create_model_pool(
                pool_name="pressure_test_pool",
                model_path=str(model_file),
                platform="esp32",
                min_size=1,
                max_size=2  # Very small pool
            )
            
            # Launch many concurrent acquisitions
            def acquire_under_pressure(thread_id: int, results: list):
                try:
                    acquire_start = time.time()
                    
                    with pool.acquire(timeout=5.0) as model:
                        acquire_time = time.time() - acquire_start
                        assert model is not None
                        
                        # Hold the resource briefly
                        time.sleep(0.1)
                        
                        results.append({
                            "thread_id": thread_id,
                            "acquire_time_s": acquire_time,
                            "success": True
                        })
                        
                except Exception as e:
                    results.append({
                        "thread_id": thread_id,
                        "error": str(e),
                        "success": False
                    })
            
            threads = []
            results = []
            num_threads = 20  # Much more than pool size
            
            start_time = time.time()
            
            for i in range(num_threads):
                thread = threading.Thread(
                    target=acquire_under_pressure,
                    args=(i, results)
                )
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=15)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Analyze pressure test results
            successful_acquisitions = [r for r in results if r["success"]]
            failed_acquisitions = [r for r in results if not r["success"]]
            
            # Most threads should succeed despite contention
            success_rate = len(successful_acquisitions) / len(results)
            assert success_rate >= 0.8, f"Success rate under pressure too low: {success_rate:.1%}"
            
            # Check acquisition times (some should wait due to contention)
            acquisition_times = [r["acquire_time_s"] for r in successful_acquisitions]
            avg_acquire_time = sum(acquisition_times) / len(acquisition_times)
            
            # Some acquisitions should have waited (indicating proper queuing)
            long_waits = [t for t in acquisition_times if t > 0.05]  # 50ms+ wait
            assert len(long_waits) > 0, "Should have some acquisition delays due to contention"
            
            # Get final pool stats
            final_stats = pool.get_stats()
            
            print(f"✓ Resource pool pressure test passed")
            print(f"  Threads: {num_threads}, Success rate: {success_rate:.1%}")
            print(f"  Average acquire time: {avg_acquire_time:.3f}s")
            print(f"  Long waits: {len(long_waits)}")
            print(f"  Final pool stats: {final_stats}")
        
        finally:
            pool_manager.shutdown_all()
    
    def _create_model_file(self, path: Path, size_mb: float):
        """Create a test model file of specified size."""
        header = b"GGUF\x03\x00\x00\x00"
        metadata = b'\x00' * 256
        data_size = int(size_mb * 1024 * 1024) - len(header) - len(metadata)
        
        with open(path, 'wb') as f:
            f.write(header)
            f.write(metadata)
            f.write(b'\x42' * max(0, data_size))


@pytest.mark.integration  
class TestCachingIntegration:
    """Test caching mechanisms in realistic profiling workflows."""
    
    def test_smart_cache_profiling_workflow(self, tmp_path):
        """Test smart caching integration in complete profiling workflows."""
        
        # Initialize cache with both memory and persistent storage
        cache = SmartCache(
            memory_cache_size=50,
            memory_cache_mb=10,
            persistent_cache_dir=tmp_path / "cache_test"
        )
        
        # Create test models
        models = {}
        for i, (size, quant) in enumerate([(1.5, QuantizationType.INT2), (2.5, QuantizationType.INT4)]):
            model_file = tmp_path / f"cache_test_model_{i}.gguf"
            self._create_model_file(model_file, size)
            models[f"model_{i}"] = QuantizedModel.from_file(model_file, quantization=quant)
        
        # Test caching workflow
        cache_hits = 0
        cache_misses = 0
        
        for iteration in range(3):  # Multiple iterations to test caching
            for model_name, model in models.items():
                for platform in ["esp32", "stm32f7"]:
                    
                    # Generate cache keys
                    model_cache_key = f"optimized_model_{model_name}_{platform}"
                    results_cache_key = f"profile_results_{model_name}_{platform}_quick"
                    
                    # Check for cached optimized model
                    cached_model = cache.get(model_cache_key)
                    
                    if cached_model is not None:
                        cache_hits += 1
                        optimized_model = cached_model
                    else:
                        cache_misses += 1
                        # Optimize model for platform
                        optimized_model = model.optimize_for_platform(
                            platform=platform,
                            constraints={"max_memory_kb": 300 if platform == "esp32" else 150}
                        )
                        
                        # Cache optimized model
                        cache.put(model_cache_key, optimized_model, ttl_seconds=3600)
                    
                    # Check for cached profiling results
                    cached_results = cache.get(results_cache_key)
                    
                    if cached_results is not None:
                        cache_hits += 1
                        results = cached_results
                    else:
                        cache_misses += 1
                        # Run profiling
                        profiler = EdgeProfiler(platform=platform, connection="local")
                        
                        results = profiler.profile_model(
                            model=optimized_model,
                            test_prompts=[f"Cache test {iteration}"],
                            metrics=["latency", "memory"],
                            config=ProfilingConfig(duration_seconds=2, measurement_iterations=1)
                        )
                        
                        # Cache results
                        cache.put(results_cache_key, results, ttl_seconds=1800)
                    
                    # Validate results
                    assert results is not None
                    assert results.latency_profile is not None
                    assert results.memory_profile is not None
        
        # Analyze caching effectiveness
        total_operations = cache_hits + cache_misses
        hit_rate = cache_hits / total_operations if total_operations > 0 else 0
        
        # First iteration should be all misses, subsequent should have hits
        assert hit_rate > 0.3, f"Cache hit rate too low: {hit_rate:.1%}"
        
        # Check cache statistics
        cache_stats = cache.get_stats()
        assert cache_stats["memory_cache"]["entries"] > 0
        assert cache_stats["persistent_cache"]["entries"] > 0
        
        print(f"✓ Smart cache profiling workflow test passed")
        print(f"  Cache operations: {total_operations} (hits: {cache_hits}, misses: {cache_misses})")
        print(f"  Hit rate: {hit_rate:.1%}")
        print(f"  Cache stats: {cache_stats}")
    
    def test_performance_cache_optimization_tracking(self, tmp_path):
        """Test performance cache for tracking optimization history."""
        
        perf_cache = PerformanceCache(cache_dir=tmp_path / "perf_cache")
        
        # Create test model
        model_file = tmp_path / "perf_cache_model.gguf"
        self._create_model_file(model_file, 2.0)
        model = QuantizedModel.from_file(model_file)
        
        # Test different optimization strategies
        optimization_strategies = [
            {"quantization": "2bit", "context_length": 512, "use_flash": True},
            {"quantization": "3bit", "context_length": 1024, "use_flash": False},
            {"quantization": "4bit", "context_length": 2048, "use_flash": False},
        ]
        
        platform = "esp32"
        
        # Run profiling with different optimizations and cache results
        for i, strategy in enumerate(optimization_strategies):
            profiler = EdgeProfiler(platform=platform, connection="local")
            
            # Apply optimization strategy (simulated)
            optimized_model = model.optimize_for_platform(
                platform=platform,
                constraints=strategy
            )
            
            results = profiler.profile_model(
                model=optimized_model,
                test_prompts=[f"Performance cache test {i}"],
                metrics=["latency", "memory", "power"],
                config=ProfilingConfig(duration_seconds=2, measurement_iterations=1)
            )
            
            # Cache performance results with strategy metadata
            cache_key = f"perf_{platform}_{i}"
            
            performance_record = {
                "strategy": strategy,
                "results": results,
                "model_config": {
                    "size_mb": optimized_model.size_mb,
                    "quantization": optimized_model.quantization.value,
                    "context_length": optimized_model.context_length
                },
                "timestamp": time.time()
            }
            
            perf_cache.store_performance_record(cache_key, performance_record)
        
        # Query cached performance data
        cached_records = perf_cache.get_performance_records(platform=platform)
        assert len(cached_records) == len(optimization_strategies)
        
        # Find best performing strategy
        best_record = perf_cache.get_best_performance(
            platform=platform,
            metric="tokens_per_second"
        )
        
        assert best_record is not None
        best_strategy = best_record["strategy"]
        best_results = best_record["results"]
        
        # Validate performance tracking
        assert best_results.latency_profile.tokens_per_second > 0
        
        # Test performance trend analysis
        trends = perf_cache.analyze_performance_trends(platform=platform)
        
        assert "optimization_impact" in trends
        assert "best_strategies" in trends
        
        print(f"✓ Performance cache optimization tracking test passed")
        print(f"  Cached records: {len(cached_records)}")
        print(f"  Best strategy: {best_strategy}")
        print(f"  Best performance: {best_results.latency_profile.tokens_per_second:.1f} tok/s")
    
    def test_cache_invalidation_and_cleanup(self, tmp_path):
        """Test cache invalidation and cleanup mechanisms."""
        
        cache = SmartCache(
            memory_cache_size=10,  # Small cache to force evictions
            memory_cache_mb=1,
            persistent_cache_dir=tmp_path / "cleanup_cache"
        )
        
        # Fill cache with test data
        test_data = {}
        
        for i in range(20):  # More than cache size
            key = f"test_key_{i}"
            value = {
                "data": f"test_data_{i}" * 100,  # Some sizeable data
                "timestamp": time.time(),
                "iteration": i
            }
            
            # Set different TTLs
            ttl = 1 if i % 5 == 0 else 3600  # Some short-lived entries
            
            cache.put(key, value, ttl_seconds=ttl)
            test_data[key] = value
        
        # Check initial cache state
        initial_stats = cache.get_stats()
        initial_memory_entries = initial_stats["memory_cache"]["entries"]
        initial_persistent_entries = initial_stats["persistent_cache"]["entries"]
        
        # Memory cache should be limited by size
        assert initial_memory_entries <= 10
        
        # Wait for some entries to expire
        time.sleep(1.5)
        
        # Trigger cleanup by accessing cache
        cache.get("test_key_0")  # Should trigger TTL cleanup
        
        # Force cleanup
        cache.cleanup_expired()
        
        # Check cache state after cleanup
        after_cleanup_stats = cache.get_stats()
        after_memory_entries = after_cleanup_stats["memory_cache"]["entries"]
        
        # Some entries should have been cleaned up
        assert after_memory_entries <= initial_memory_entries
        
        # Test cache invalidation patterns
        pattern_keys = [f"pattern_test_{i}" for i in range(5)]
        
        for key in pattern_keys:
            cache.put(key, {"pattern_data": key}, ttl_seconds=3600)
        
        # Invalidate by pattern
        cache.invalidate_pattern("pattern_test_*")
        
        # Check that pattern keys were invalidated
        remaining_pattern_keys = [
            key for key in pattern_keys 
            if cache.get(key) is not None
        ]
        
        assert len(remaining_pattern_keys) == 0, "Pattern invalidation should remove matching keys"
        
        # Test memory pressure cleanup
        large_data = {"large": "x" * (100 * 1024)}  # 100KB
        
        for i in range(15):  # Add large items to trigger memory pressure
            cache.put(f"large_item_{i}", large_data, ttl_seconds=3600)
        
        final_stats = cache.get_stats()
        final_memory_mb = final_stats["memory_cache"]["size_mb"]
        
        # Should stay within memory limits
        assert final_memory_mb <= 1.2, f"Cache exceeded memory limit: {final_memory_mb}MB"
        
        print(f"✓ Cache invalidation and cleanup test passed")
        print(f"  Initial entries: {initial_memory_entries}, After cleanup: {after_memory_entries}")
        print(f"  Final memory usage: {final_memory_mb:.2f}MB")
    
    def test_distributed_cache_coordination(self, tmp_path):
        """Test coordination between multiple cache instances (simulating distributed scenario)."""
        
        # Create multiple cache instances sharing persistent storage
        shared_cache_dir = tmp_path / "shared_cache"
        
        caches = []
        for i in range(3):
            cache = SmartCache(
                memory_cache_size=20,
                memory_cache_mb=5,
                persistent_cache_dir=shared_cache_dir,
                instance_id=f"cache_instance_{i}"
            )
            caches.append(cache)
        
        # Test data sharing between instances
        test_models = {}
        
        for i in range(5):
            model_file = tmp_path / f"distributed_model_{i}.gguf"
            self._create_model_file(model_file, 1.5)
            test_models[f"model_{i}"] = QuantizedModel.from_file(model_file)
        
        # Each cache instance stores different models
        for i, (model_name, model) in enumerate(test_models.items()):
            cache_instance = caches[i % len(caches)]
            
            # Store model in cache
            cache_key = f"shared_model_{model_name}"
            cache_instance.put(cache_key, model, ttl_seconds=3600)
        
        # Test that other instances can access the data
        shared_access_results = {}
        
        for model_name in test_models.keys():
            cache_key = f"shared_model_{model_name}"
            
            # Try to access from each cache instance
            for i, cache_instance in enumerate(caches):
                cached_model = cache_instance.get(cache_key)
                
                instance_key = f"instance_{i}_{model_name}"
                shared_access_results[instance_key] = cached_model is not None
        
        # Analyze sharing effectiveness
        successful_shares = sum(1 for success in shared_access_results.values() if success)
        total_attempts = len(shared_access_results)
        share_rate = successful_shares / total_attempts
        
        # Should have good sharing via persistent storage
        assert share_rate >= 0.6, f"Cache sharing rate too low: {share_rate:.1%}"
        
        # Test cache statistics from different instances
        all_stats = []
        for i, cache in enumerate(caches):
            stats = cache.get_stats()
            stats["instance_id"] = i
            all_stats.append(stats)
        
        print(f"✓ Distributed cache coordination test passed")
        print(f"  Cache instances: {len(caches)}")
        print(f"  Sharing success rate: {share_rate:.1%}")
        print(f"  Instance statistics:")
        for stats in all_stats:
            print(f"    Instance {stats['instance_id']}: "
                  f"Memory: {stats['memory_cache']['entries']}, "
                  f"Persistent: {stats['persistent_cache']['entries']}")
    
    def _create_model_file(self, path: Path, size_mb: float):
        """Create a test model file of specified size."""
        header = b"GGUF\x03\x00\x00\x00"
        metadata = b'\x00' * 256
        data_size = int(size_mb * 1024 * 1024) - len(header) - len(metadata)
        
        with open(path, 'wb') as f:
            f.write(header)
            f.write(metadata)
            f.write(b'\x42' * max(0, data_size))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])