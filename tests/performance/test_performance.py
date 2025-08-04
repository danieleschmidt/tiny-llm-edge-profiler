"""
Performance benchmarks and tests for the Tiny LLM Edge Profiler.
"""

import time
import statistics
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Callable
import pytest
import numpy as np
from pathlib import Path

from tiny_llm_profiler.cache import InMemoryCache, PersistentCache, SmartCache
from tiny_llm_profiler.concurrent import ConcurrentProfiler, ProfilingTask
from tiny_llm_profiler.resource_pool import ResourcePool, ResourceFactory
from tiny_llm_profiler.models import QuantizedModel
from tiny_llm_profiler.profiler import EdgeProfiler
from tiny_llm_profiler.scaling import LoadBalancer, AutoScaler


class MockResourceFactory(ResourceFactory):
    """Mock resource factory for testing."""
    
    def create_resource(self, **kwargs):
        return {"id": f"resource_{time.time()}", "created_at": time.time()}
    
    def validate_resource(self, resource):
        return True
    
    def cleanup_resource(self, resource):
        pass
    
    def reset_resource(self, resource):
        return True


class TestCachePerformance:
    """Performance tests for caching systems."""
    
    def test_memory_cache_throughput(self):
        """Test memory cache read/write throughput."""
        cache = InMemoryCache(max_size=10000, max_memory_mb=50)
        
        # Warm up
        for i in range(100):
            cache.put(f"key_{i}", f"value_{i}" * 100)
        
        # Test write performance
        start_time = time.time()
        write_count = 1000
        
        for i in range(write_count):
            cache.put(f"perf_key_{i}", f"perf_value_{i}" * 100)
        
        write_time = time.time() - start_time
        write_ops_per_sec = write_count / write_time
        
        # Test read performance
        start_time = time.time()
        read_count = 1000
        hits = 0
        
        for i in range(read_count):
            if cache.get(f"perf_key_{i % write_count}"):
                hits += 1
        
        read_time = time.time() - start_time
        read_ops_per_sec = read_count / read_time
        
        # Performance assertions
        assert write_ops_per_sec > 1000, f"Write performance too low: {write_ops_per_sec:.1f} ops/sec"
        assert read_ops_per_sec > 5000, f"Read performance too low: {read_ops_per_sec:.1f} ops/sec"
        assert hits / read_count > 0.9, f"Hit rate too low: {hits / read_count:.2%}"
        
        print(f"Memory Cache Performance:")
        print(f"  Write: {write_ops_per_sec:.1f} ops/sec")  
        print(f"  Read: {read_ops_per_sec:.1f} ops/sec")
        print(f"  Hit rate: {hits / read_count:.2%}")
    
    def test_persistent_cache_throughput(self, tmp_path):
        """Test persistent cache performance."""
        cache = PersistentCache(cache_dir=tmp_path / "perf_cache", max_size_gb=0.1)
        
        # Test write performance  
        start_time = time.time()
        write_count = 100  # Smaller count for disk operations
        
        for i in range(write_count):
            cache.put(f"persist_key_{i}", {"data": f"value_{i}" * 50, "index": i})
        
        write_time = time.time() - start_time
        write_ops_per_sec = write_count / write_time
        
        # Test read performance
        start_time = time.time()
        read_count = 100
        hits = 0
        
        for i in range(read_count):
            if cache.get(f"persist_key_{i}"):
                hits += 1
        
        read_time = time.time() - start_time
        read_ops_per_sec = read_count / read_time
        
        # Performance assertions (lower expectations for disk I/O)
        assert write_ops_per_sec > 50, f"Persistent write performance too low: {write_ops_per_sec:.1f} ops/sec"
        assert read_ops_per_sec > 100, f"Persistent read performance too low: {read_ops_per_sec:.1f} ops/sec"
        assert hits == read_count, f"All items should be found in persistent cache"
        
        print(f"Persistent Cache Performance:")
        print(f"  Write: {write_ops_per_sec:.1f} ops/sec")
        print(f"  Read: {read_ops_per_sec:.1f} ops/sec")
    
    def test_cache_memory_efficiency(self):
        """Test cache memory usage efficiency."""
        cache = InMemoryCache(max_size=1000, max_memory_mb=10)
        
        # Add items and track memory usage
        item_size = 1024  # 1KB per item
        num_items = 100
        
        for i in range(num_items):
            data = "x" * item_size
            cache.put(f"mem_key_{i}", data)
        
        stats = cache.get_stats()
        
        # Check memory efficiency
        expected_memory_mb = (num_items * item_size) / (1024 * 1024)
        actual_memory_mb = stats["memory_usage_mb"]
        
        # Allow some overhead but not too much
        efficiency = expected_memory_mb / actual_memory_mb if actual_memory_mb > 0 else 0
        
        assert efficiency > 0.5, f"Memory efficiency too low: {efficiency:.2%}"
        assert stats["entries"] == num_items
        
        print(f"Cache Memory Efficiency:")
        print(f"  Expected: {expected_memory_mb:.2f} MB")
        print(f"  Actual: {actual_memory_mb:.2f} MB")
        print(f"  Efficiency: {efficiency:.2%}")


class TestConcurrentPerformance:
    """Performance tests for concurrent processing."""
    
    def test_concurrent_profiler_throughput(self):
        """Test concurrent profiler task throughput."""
        profiler = ConcurrentProfiler(max_threads=4, max_processes=2)
        profiler.start()
        
        try:
            # Create mock tasks
            num_tasks = 20
            task_duration = 0.1  # Simulate 100ms tasks
            
            start_time = time.time()
            task_ids = []
            
            for i in range(num_tasks):
                task = ProfilingTask(
                    task_id=f"perf_task_{i}",
                    platform="mock_platform",
                    model=MockQuantizedModel(),
                    test_prompts=[f"prompt_{i}"],
                    priority=i % 3  # Vary priorities
                )
                
                task_id = profiler.submit_task(task)
                task_ids.append(task_id)
            
            # Wait for completion
            results = profiler.wait_for_completion(task_ids, timeout=30)
            
            total_time = time.time() - start_time
            throughput = len(results) / total_time
            
            # Performance assertions
            assert len(results) == num_tasks, f"Not all tasks completed: {len(results)}/{num_tasks}"
            assert throughput > 2, f"Throughput too low: {throughput:.1f} tasks/sec"
            
            # Check parallelization efficiency
            sequential_time = num_tasks * task_duration
            parallelization_factor = sequential_time / total_time
            
            assert parallelization_factor > 2, f"Poor parallelization: {parallelization_factor:.1f}x"
            
            print(f"Concurrent Profiler Performance:")
            print(f"  Tasks: {num_tasks}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} tasks/sec")
            print(f"  Parallelization: {parallelization_factor:.1f}x")
            
        finally:
            profiler.stop()
    
    def test_resource_pool_performance(self):
        """Test resource pool performance under load."""
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=2, max_size=10)
        
        # Test resource acquisition performance
        num_acquisitions = 100
        start_time = time.time()
        
        def acquire_and_release():
            with pool.acquire(timeout=5.0) as resource:
                # Simulate work
                time.sleep(0.01)
                return resource["id"]
        
        # Run concurrent acquisitions
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(acquire_and_release) for _ in range(num_acquisitions)]
            results = [f.result() for f in as_completed(futures)]
        
        total_time = time.time() - start_time
        throughput = len(results) / total_time
        
        # Performance assertions
        assert len(results) == num_acquisitions
        assert throughput > 20, f"Resource pool throughput too low: {throughput:.1f} acq/sec"
        
        # Check resource reuse
        stats = pool.get_stats()
        reuse_efficiency = stats["stats"]["acquired"] / stats["stats"]["created"]
        assert reuse_efficiency > 5, f"Poor resource reuse: {reuse_efficiency:.1f}"
        
        print(f"Resource Pool Performance:")
        print(f"  Acquisitions: {num_acquisitions}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} acq/sec")
        print(f"  Reuse efficiency: {reuse_efficiency:.1f}")
        
        pool.shutdown()
    
    def test_load_balancer_distribution_performance(self):
        """Test load balancer distribution efficiency."""
        balancer = LoadBalancer(strategy="least_connections")
        
        # Add resources with different capacities
        for i in range(5):
            balancer.add_resource(
                f"resource_{i}",
                {"id": f"resource_{i}", "capacity": (i + 1) * 100},
                capacity_weight=(i + 1)
            )
        
        # Simulate task distribution
        num_tasks = 1000
        start_time = time.time()
        
        task_assignments = {}
        
        for i in range(num_tasks):
            resource = balancer.select_resource()
            if resource:
                resource_id = resource["id"]
                task_assignments[resource_id] = task_assignments.get(resource_id, 0) + 1
                
                # Simulate task start/completion
                balancer.report_task_start(resource_id, f"task_{i}")
                balancer.report_task_completion(resource_id, f"task_{i}", True, 0.1)
        
        distribution_time = time.time() - start_time
        distribution_rate = num_tasks / distribution_time
        
        # Performance assertions
        assert distribution_rate > 10000, f"Distribution rate too low: {distribution_rate:.1f} tasks/sec"
        
        # Check distribution fairness
        assignment_counts = list(task_assignments.values())
        fairness_ratio = min(assignment_counts) / max(assignment_counts)
        assert fairness_ratio > 0.1, f"Poor load distribution fairness: {fairness_ratio:.2f}"
        
        print(f"Load Balancer Performance:")
        print(f"  Distribution rate: {distribution_rate:.1f} tasks/sec")
        print(f"  Task assignments: {dict(task_assignments)}")
        print(f"  Fairness ratio: {fairness_ratio:.2f}")


class TestScalingPerformance:
    """Performance tests for auto-scaling functionality."""
    
    def test_metrics_collection_overhead(self):
        """Test overhead of metrics collection."""
        from tiny_llm_profiler.scaling import AutoScaler
        from tiny_llm_profiler.resource_pool import ResourcePoolManager
        from tiny_llm_profiler.concurrent import ConcurrentProfiler
        
        # Create mock components
        pool_manager = ResourcePoolManager()
        concurrent_profiler = ConcurrentProfiler()
        
        scaler = AutoScaler(
            resource_pool_manager=pool_manager,
            concurrent_profiler=concurrent_profiler,
            check_interval_seconds=1
        )
        
        # Test metrics collection performance
        num_collections = 100
        start_time = time.time()
        
        for _ in range(num_collections):
            metrics = scaler._collect_metrics()
            assert metrics is not None
        
        collection_time = time.time() - start_time
        collection_rate = num_collections / collection_time
        
        # Performance assertions
        assert collection_rate > 50, f"Metrics collection too slow: {collection_rate:.1f} collections/sec"
        
        # Test memory usage stability
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Collect metrics for extended period
        for _ in range(1000):
            scaler._collect_metrics()
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        assert memory_growth < 10, f"Excessive memory growth during metrics collection: {memory_growth:.1f} MB"
        
        print(f"Metrics Collection Performance:")
        print(f"  Collection rate: {collection_rate:.1f} collections/sec")
        print(f"  Memory growth: {memory_growth:.1f} MB")


class TestMemoryPerformance:
    """Memory usage and leak tests."""
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in core components."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run operations that might leak memory
        for cycle in range(10):
            # Create and destroy cache
            cache = InMemoryCache(max_size=1000)
            for i in range(1000):
                cache.put(f"key_{i}", f"value_{i}" * 100)
            
            # Create and destroy profiler
            profiler = ConcurrentProfiler(max_threads=2)
            profiler.start()
            profiler.stop()
            
            # Create and destroy resource pool
            factory = MockResourceFactory()
            pool = ResourcePool(factory, min_size=1, max_size=5)
            with pool.acquire() as resource:
                pass
            pool.shutdown()
            
            # Force garbage collection
            gc.collect()
            
            # Check memory growth
            current_memory = process.memory_info().rss
            memory_growth = (current_memory - initial_memory) / (1024 * 1024)  # MB
            
            print(f"Cycle {cycle + 1}: Memory usage = {memory_growth:.1f} MB growth")
        
        # Final memory check
        final_memory = process.memory_info().rss
        total_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        # Allow some growth but not excessive
        assert total_growth < 50, f"Excessive memory growth detected: {total_growth:.1f} MB"
        
        print(f"Memory Leak Test:")
        print(f"  Initial memory: {initial_memory / (1024 * 1024):.1f} MB")
        print(f"  Final memory: {final_memory / (1024 * 1024):.1f} MB")
        print(f"  Total growth: {total_growth:.1f} MB")


class TestLatencyBenchmarks:
    """Latency benchmark tests."""
    
    def test_operation_latency_distribution(self):
        """Test latency distribution of key operations."""
        
        def measure_operation_latency(operation: Callable, num_samples: int = 100) -> Dict[str, float]:
            """Measure latency statistics for an operation."""
            latencies = []
            
            for _ in range(num_samples):
                start_time = time.perf_counter()
                operation()
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            return {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "max": max(latencies),
                "std": statistics.stdev(latencies)
            }
        
        # Test cache operations
        cache = InMemoryCache(max_size=1000)
        
        # Cache write latency
        write_stats = measure_operation_latency(
            lambda: cache.put(f"key_{time.time()}", "test_value" * 10)
        )
        
        # Cache read latency (with hits)
        for i in range(100):
            cache.put(f"read_key_{i}", f"read_value_{i}")
        
        read_stats = measure_operation_latency(
            lambda: cache.get(f"read_key_{np.random.randint(0, 100)}")
        )
        
        # Performance assertions
        assert write_stats["p95"] < 1.0, f"Cache write P95 latency too high: {write_stats['p95']:.2f}ms"
        assert read_stats["p95"] < 0.5, f"Cache read P95 latency too high: {read_stats['p95']:.2f}ms"
        
        print(f"Cache Operation Latencies:")
        print(f"  Write P95: {write_stats['p95']:.2f}ms")
        print(f"  Read P95: {read_stats['p95']:.2f}ms")
        
        # Test resource pool operations
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=5, max_size=10)
        
        acquire_stats = measure_operation_latency(
            lambda: pool.acquire(timeout=5.0).__enter__()
        )
        
        assert acquire_stats["p95"] < 10.0, f"Resource acquisition P95 latency too high: {acquire_stats['p95']:.2f}ms"
        
        print(f"Resource Pool Latencies:")
        print(f"  Acquire P95: {acquire_stats['p95']:.2f}ms")
        
        pool.shutdown()


class MockQuantizedModel:
    """Mock quantized model for testing."""
    
    def __init__(self):
        self.name = "mock_model"
        self.size_mb = 2.5
        self.quantization = "4bit"
    
    def validate(self):
        return True, []


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests."""
    
    def test_baseline_performance_metrics(self):
        """Test that performance meets baseline requirements."""
        
        # Define performance baselines
        baselines = {
            "cache_write_ops_per_sec": 1000,
            "cache_read_ops_per_sec": 5000,
            "concurrent_task_throughput": 2,
            "resource_pool_throughput": 20,
            "metrics_collection_rate": 50,
            "memory_growth_limit_mb": 50
        }
        
        results = {}
        
        # Run performance tests and collect results
        cache = InMemoryCache(max_size=1000)
        
        # Cache performance
        start_time = time.time()
        for i in range(1000):
            cache.put(f"baseline_key_{i}", f"value_{i}")
        write_time = time.time() - start_time
        results["cache_write_ops_per_sec"] = 1000 / write_time
        
        start_time = time.time()
        for i in range(5000):
            cache.get(f"baseline_key_{i % 1000}")
        read_time = time.time() - start_time
        results["cache_read_ops_per_sec"] = 5000 / read_time
        
        # Validate against baselines
        for metric, baseline in baselines.items():
            if metric in results:
                actual = results[metric]
                assert actual >= baseline, f"Performance regression in {metric}: {actual:.1f} < {baseline}"
                print(f"✓ {metric}: {actual:.1f} (baseline: {baseline})")
        
        print("\nPerformance Baseline Test: PASSED")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])