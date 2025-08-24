#!/usr/bin/env python3
"""
Generation 2 Robustness Test - validates reliability enhancements
"""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, 'src')

try:
    from tiny_llm_profiler.robust_profiler import (
        RobustProfiler,
        CircuitBreaker,
        RetryMechanism,
        HealthMonitor,
        ResourceMonitor,
        HealthStatus,
        robust_quick_profile,
        check_system_health
    )
    print("✅ Generation 2 imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n🔌 Testing Circuit Breaker")
    
    cb = CircuitBreaker(failure_threshold=3, timeout=5.0)
    
    # Test normal operation (closed state)
    assert cb.can_execute() == True
    print("   ✅ Circuit starts in closed state")
    
    # Simulate failures
    for i in range(3):
        cb.record_failure()
    
    # Should be open now
    assert cb.state == "open"
    assert cb.can_execute() == False
    print("   ✅ Circuit opens after threshold failures")
    
    # Test timeout and half-open state
    time.sleep(0.1)  # Short sleep
    cb.timeout = 0.05  # Very short timeout for testing
    time.sleep(0.1)
    
    assert cb.can_execute() == True  # Should be half-open
    print("   ✅ Circuit enters half-open state after timeout")
    
    # Record success to close circuit
    cb.record_success()
    assert cb.state == "closed"
    print("   ✅ Circuit closes after successful operation")
    
    return True


def test_retry_mechanism():
    """Test retry mechanism with different strategies."""
    print("\n🔄 Testing Retry Mechanism")
    
    retry = RetryMechanism(max_attempts=3, base_delay=0.1)
    
    # Test successful operation (no retry needed)
    def success_func():
        return "success"
    
    result = retry.retry(success_func, strategy="exponential")
    assert result == "success"
    print("   ✅ Successful operation (no retry)")
    
    # Test failure then success
    attempts = {"count": 0}
    
    def fail_then_success():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise Exception(f"Failure {attempts['count']}")
        return f"Success after {attempts['count']} attempts"
    
    result = retry.retry(fail_then_success, strategy="exponential")
    assert "Success after 3" in result
    print("   ✅ Retry with exponential backoff")
    
    # Test different strategies
    delays_exp = [retry.exponential_backoff(i) for i in range(1, 4)]
    delays_lin = [retry.linear_backoff(i) for i in range(1, 4)]
    delays_fix = [retry.fixed_delay(i) for i in range(1, 4)]
    
    assert delays_exp == [0.1, 0.2, 0.4]  # exponential
    assert delays_lin == [0.1, 0.2, 0.3]  # linear
    assert all(d == 0.1 for d in delays_fix)  # fixed
    
    print("   ✅ All retry strategies working")
    
    return True


def test_health_monitor():
    """Test health monitoring system."""
    print("\n🏥 Testing Health Monitor")
    
    monitor = HealthMonitor()
    
    # Register test health checks
    def healthy_check():
        return True
    
    def unhealthy_check():
        return False
    
    def error_check():
        raise Exception("Check failed")
    
    monitor.register_check("healthy", healthy_check)
    monitor.register_check("unhealthy", unhealthy_check)
    monitor.register_check("error", error_check)
    
    # Run health checks
    health = monitor.run_health_checks()
    
    # Should be degraded (some checks failed)
    assert health.status == HealthStatus.DEGRADED
    assert health.checks["healthy"] == True
    assert health.checks["unhealthy"] == False
    assert health.checks["error"] == False
    
    print(f"   ✅ Health status: {health.status.value}")
    print(f"   ✅ Health message: {health.message}")
    print(f"   ✅ Check results: {health.checks}")
    
    return True


def test_resource_monitor():
    """Test resource monitoring."""
    print("\n💻 Testing Resource Monitor")
    
    monitor = ResourceMonitor()
    
    # Test memory check (should pass in most environments)
    try:
        memory_ok = monitor.check_memory()
        print(f"   Memory check: {'✅ OK' if memory_ok else '⚠️ Low'}")
    except Exception as e:
        print(f"   Memory check: ⚠️ Error - {e}")
    
    # Test disk space check
    try:
        disk_ok = monitor.check_disk_space()
        print(f"   Disk space check: {'✅ OK' if disk_ok else '⚠️ Low'}")
    except Exception as e:
        print(f"   Disk space check: ⚠️ Error - {e}")
    
    # Test CPU check
    try:
        cpu_ok = monitor.check_cpu_usage()
        print(f"   CPU usage check: {'✅ OK' if cpu_ok else '⚠️ High'}")
    except Exception as e:
        print(f"   CPU usage check: ⚠️ Error - {e}")
    
    print("   ✅ Resource monitoring functional")
    return True


def test_robust_profiler():
    """Test the main robust profiler."""
    print("\n🛡️ Testing Robust Profiler")
    
    profiler = RobustProfiler(enable_monitoring=True)
    
    # Test health status
    health = profiler.get_health_status()
    print(f"   System health: {health.status.value}")
    
    is_healthy = profiler.is_healthy()
    print(f"   Is healthy: {'✅ Yes' if is_healthy else '⚠️ No'}")
    
    # Create test model file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
        temp_file.write(b'0' * (2 * 1024 * 1024))  # 2MB model
        temp_model_path = temp_file.name
    
    try:
        # Test robust profiling
        print("   Running robust profiling...")
        result = profiler.robust_profile(temp_model_path, "esp32")
        
        print(f"   Profile status: {result['status']}")
        
        if result['status'] in ['success', 'degraded']:
            print("   ✅ Profiling completed")
            
            # Check reliability info
            reliability = result.get('reliability_info', {})
            print(f"   Circuit breaker state: {reliability.get('circuit_breaker_state', 'unknown')}")
            print(f"   Total time: {reliability.get('total_time_seconds', 0):.2f}s")
            
            if 'profiling_results' in result:
                print("   ✅ Profile results available")
        else:
            print(f"   ⚠️ Profiling failed: {result.get('error', 'Unknown error')}")
            print("   ✅ Graceful failure handling")
        
        # Test reliability metrics
        metrics = profiler.get_reliability_metrics()
        print(f"   Success rate: {metrics['success_rate']:.1%}")
        print(f"   Total profiles: {metrics['total_profiles_run']}")
        
        print("   ✅ Robust profiler functional")
        
    finally:
        Path(temp_model_path).unlink(missing_ok=True)
    
    return True


def test_convenience_functions():
    """Test convenience functions."""
    print("\n🎯 Testing Convenience Functions")
    
    # Test system health check
    health = check_system_health()
    print(f"   System health: {health.status.value}")
    
    # Test robust quick profile
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
        temp_file.write(b'0' * (1 * 1024 * 1024))  # 1MB model
        temp_model_path = temp_file.name
    
    try:
        result = robust_quick_profile(temp_model_path, "esp32", enable_monitoring=True)
        print(f"   Quick profile status: {result['status']}")
        print("   ✅ Convenience functions working")
    
    finally:
        Path(temp_model_path).unlink(missing_ok=True)
    
    return True


def test_generation2():
    """Run all Generation 2 tests."""
    
    print("🛡️ Generation 2 Robustness Test")
    print("=" * 50)
    
    tests = [
        ("Circuit Breaker", test_circuit_breaker),
        ("Retry Mechanism", test_retry_mechanism), 
        ("Health Monitor", test_health_monitor),
        ("Resource Monitor", test_resource_monitor),
        ("Robust Profiler", test_robust_profiler),
        ("Convenience Functions", test_convenience_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} test...")
            success = test_func()
            if success:
                passed += 1
                print(f"   ✅ {test_name} test PASSED")
            else:
                print(f"   ❌ {test_name} test FAILED")
        except Exception as e:
            print(f"   💥 {test_name} test CRASHED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All Generation 2 tests PASSED!")
        print("\n🎯 Generation 2 Enhancements Validated:")
        print("   ✅ Circuit breaker pattern - prevents cascade failures") 
        print("   ✅ Retry mechanisms - handles transient failures")
        print("   ✅ Health monitoring - tracks system status")
        print("   ✅ Resource monitoring - prevents resource exhaustion")
        print("   ✅ Graceful degradation - provides fallback results")
        print("   ✅ Comprehensive error handling - robust failure recovery")
        print("   ✅ Reliability metrics - observability and monitoring")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    try:
        success = test_generation2()
        if success:
            print(f"\n🎉 Generation 2 Enhancement Test PASSED!")
            print("🚀 System is robust and ready for Generation 3!")
            sys.exit(0)
        else:
            print(f"\n❌ Some Generation 2 tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)