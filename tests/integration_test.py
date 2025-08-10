"""
Integration test and performance benchmark for the complete system.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "tiny_llm_profiler"))

from core_lite import run_basic_benchmark
from performance_cache import OptimizedProfiler
from scalable_profiler import ConcurrentProfiler
from health_monitor import get_health_monitor, start_global_monitoring


def test_basic_functionality():
    """Test basic functionality end-to-end."""
    print("🔧 Testing basic functionality...")
    
    results = run_basic_benchmark()
    
    # Validate results
    assert len(results) > 0, "No benchmark results"
    
    successful_results = [r for r in results.values() if r.get("success", False)]
    assert len(successful_results) > 0, "No successful results"
    
    print(f"✓ Basic benchmark completed with {len(successful_results)} successful results")
    return True


def test_caching_performance():
    """Test caching performance improvements."""
    print("🚀 Testing caching performance...")
    
    profiler = OptimizedProfiler(cache_size=100)
    
    # Test data
    platforms = ["esp32", "stm32f4", "stm32f7"]
    prompts = ["Hello world", "Generate code"]
    
    # First run (cache misses)
    start_time = time.time()
    for platform in platforms:
        profiler.profile_with_cache(
            platform=platform,
            model_size_mb=2.5,
            quantization="4bit",
            prompts=prompts
        )
    first_run_time = time.time() - start_time
    
    # Second run (cache hits)
    start_time = time.time()
    for platform in platforms:
        profiler.profile_with_cache(
            platform=platform,
            model_size_mb=2.5,
            quantization="4bit",
            prompts=prompts
        )
    second_run_time = time.time() - start_time
    
    # Calculate speedup
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    
    # Validate performance improvement
    assert speedup > 10, f"Caching speedup too low: {speedup:.1f}x"
    
    stats = profiler.get_performance_stats()
    hit_rate = stats["profiler_stats"]["cache_hit_rate_percent"]
    
    assert hit_rate >= 50, f"Cache hit rate too low: {hit_rate:.1f}%"
    
    print(f"✓ Caching achieved {speedup:.1f}x speedup with {hit_rate:.1f}% hit rate")
    return True


def test_concurrent_processing():
    """Test concurrent processing performance."""
    print("⚡ Testing concurrent processing...")
    
    profiler = ConcurrentProfiler(max_workers=4)
    profiler.start()
    
    try:
        # Submit multiple tasks
        task_ids = []
        platforms = ["esp32", "stm32f4", "stm32f7", "rp2040"]
        prompts = ["Hello", "Test", "Concurrent"]
        
        start_time = time.time()
        
        for i in range(12):  # 3 tasks per platform
            platform = platforms[i % len(platforms)]
            task_id = profiler.submit_task(
                platform=platform,
                test_prompts=prompts
            )
            task_ids.append(task_id)
        
        # Wait for completion
        completed = profiler.wait_for_completion(timeout=30)
        execution_time = time.time() - start_time
        
        assert completed, "Not all tasks completed within timeout"
        
        # Collect results
        results = profiler.get_all_results()
        successful = [r for r in results if r.result.success]
        
        assert len(successful) >= 10, f"Too many failures: {len(successful)}/12"
        
        # Check performance
        stats = profiler.get_stats()
        throughput = stats["throughput_tasks_per_second"]
        
        assert throughput > 10, f"Throughput too low: {throughput:.1f} tasks/sec"
        
        print(f"✓ Concurrent processing: {throughput:.1f} tasks/sec, {len(successful)}/12 successful")
        return True
        
    finally:
        profiler.stop()


def test_health_monitoring():
    """Test health monitoring system."""
    print("🏥 Testing health monitoring...")
    
    start_global_monitoring()
    monitor = get_health_monitor()
    
    # Wait for some metrics collection
    time.sleep(2)
    
    # Check health status
    health = monitor.get_current_health()
    
    assert "status" in health, "Health status missing"
    assert health["status"] in ["healthy", "warning", "critical", "unknown"]
    
    # Check metrics summary
    summary = monitor.get_metrics_summary(hours=1)
    
    # Should have at least some metrics
    if "error" not in summary:
        assert "samples_count" in summary, "No metrics samples"
    
    print(f"✓ Health monitoring active, status: {health['status']}")
    return True


def performance_stress_test():
    """Run a stress test to measure peak performance."""
    print("💪 Running performance stress test...")
    
    # Test concurrent profiler under load
    profiler = ConcurrentProfiler(max_workers=8)
    profiler.start()
    
    try:
        # Submit many tasks rapidly
        num_tasks = 50
        start_time = time.time()
        
        task_ids = []
        for i in range(num_tasks):
            task_id = profiler.submit_task(
                platform="esp32",
                test_prompts=[f"Task {i}", "Stress test"],
                priority=1 + (i % 3)  # Vary priorities
            )
            task_ids.append(task_id)
        
        # Wait for completion
        completed = profiler.wait_for_completion(timeout=60)
        total_time = time.time() - start_time
        
        if not completed:
            print("⚠️  Warning: Not all stress test tasks completed")
        
        # Measure performance
        results = profiler.get_all_results()
        successful = [r for r in results if r.result.success]
        
        stats = profiler.get_stats()
        peak_throughput = stats["throughput_tasks_per_second"]
        
        print(f"✓ Stress test: {len(successful)}/{num_tasks} tasks completed")
        print(f"  Peak throughput: {peak_throughput:.1f} tasks/sec")
        print(f"  Total time: {total_time:.1f}s")
        
        return len(successful) >= num_tasks * 0.8  # 80% success rate minimum
        
    finally:
        profiler.stop()


def main():
    """Run comprehensive integration testing."""
    print("🧪 Starting Comprehensive Integration Test")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Caching Performance", test_caching_performance),
        ("Concurrent Processing", test_concurrent_processing),
        ("Health Monitoring", test_health_monitoring),
        ("Performance Stress Test", performance_stress_test)
    ]
    
    passed = 0
    failed = 0
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            failed += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"📊 Integration Test Results:")
    print(f"  ✓ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏱️  Total time: {total_time:.1f}s")
    print(f"  📈 Success rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review issues before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())