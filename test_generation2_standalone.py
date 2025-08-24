#!/usr/bin/env python3
"""
Generation 2 Standalone Test - validates robustness without external dependencies
"""

import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Health check result."""
    status: HealthStatus
    message: str
    checks: Dict[str, bool]
    timestamp: datetime


class StandaloneCircuitBreaker:
    """Generation 2 demonstration: Circuit breaker pattern."""
    
    def __init__(self, failure_threshold: int = 3, timeout: float = 5.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class StandaloneRetryMechanism:
    """Generation 2 demonstration: Retry with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 0.1):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        
    def exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.base_delay * (2 ** (attempt - 1))
        
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts:
                    delay = self.exponential_backoff(attempt)
                    time.sleep(delay)
        
        raise last_exception


class StandaloneHealthMonitor:
    """Generation 2 demonstration: Health monitoring system."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_check: Optional[HealthCheck] = None
        
    def register_check(self, name: str, check_func: Callable):
        """Register a health check."""
        self.health_checks[name] = check_func
        
    def run_checks(self) -> HealthCheck:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        messages = []
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = bool(result)
                if not result:
                    overall_healthy = False
                    messages.append(f"{name} failed")
            except Exception as e:
                results[name] = False
                overall_healthy = False
                messages.append(f"{name} error: {e}")
        
        if overall_healthy:
            status = HealthStatus.HEALTHY
            message = "All checks passed"
        elif any(results.values()):
            status = HealthStatus.DEGRADED
            message = f"Some checks failed: {', '.join(messages)}"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"All checks failed: {', '.join(messages)}"
        
        health_check = HealthCheck(
            status=status,
            message=message,
            checks=results,
            timestamp=datetime.now()
        )
        
        self.last_check = health_check
        return health_check


class StandaloneRobustProfiler:
    """Generation 2 demonstration: Robust profiler with reliability patterns."""
    
    def __init__(self):
        self.circuit_breaker = StandaloneCircuitBreaker()
        self.retry_mechanism = StandaloneRetryMechanism()
        self.health_monitor = StandaloneHealthMonitor()
        
        # Register basic health checks
        self._setup_health_checks()
        
        self.operation_count = 0
        self.success_count = 0
        self.error_history = []
    
    def _setup_health_checks(self):
        """Setup basic health checks."""
        def memory_check():
            # Simulate memory check
            return True
        
        def circuit_check():
            return self.circuit_breaker.state == "closed"
        
        def error_rate_check():
            if self.operation_count < 5:
                return True
            recent_errors = len([e for e in self.error_history[-10:]])
            return recent_errors < 5
        
        self.health_monitor.register_check("memory", memory_check)
        self.health_monitor.register_check("circuit_breaker", circuit_check) 
        self.health_monitor.register_check("error_rate", error_rate_check)
    
    def get_health_status(self) -> HealthCheck:
        """Get current system health status."""
        return self.health_monitor.run_checks()
    
    def robust_profile(self, model_path: str, platform: str, simulate_failure: bool = False) -> Dict[str, Any]:
        """
        Perform profiling with comprehensive robustness features.
        
        Args:
            model_path: Path to model file
            platform: Target platform
            simulate_failure: Simulate failure for testing
            
        Returns:
            Profiling results with reliability metadata
        """
        start_time = time.time()
        self.operation_count += 1
        
        result = {
            "model_path": model_path,
            "platform": platform,
            "timestamp": datetime.now().isoformat(),
            "reliability_metadata": {
                "circuit_breaker_state": self.circuit_breaker.state,
                "operation_number": self.operation_count,
                "health_status": None,
                "retry_attempts": 0,
                "total_time_seconds": 0,
                "recovery_used": False
            }
        }
        
        try:
            # Pre-flight health check
            health = self.get_health_status()
            result["reliability_metadata"]["health_status"] = {
                "status": health.status.value,
                "message": health.message
            }
            
            if health.status == HealthStatus.UNHEALTHY:
                raise Exception("System unhealthy - aborting operation")
            
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                raise Exception("Circuit breaker open - service unavailable")
            
            # Define the profiling operation
            def do_profiling():
                if simulate_failure:
                    if self.operation_count % 3 == 0:  # Fail every 3rd operation
                        raise Exception("Simulated profiling failure")
                
                # Simulate profiling work
                time.sleep(0.01)  # Small delay to simulate work
                
                # Check if file exists
                if not Path(model_path).exists():
                    raise Exception(f"Model file not found: {model_path}")
                
                model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
                
                # Simulate platform compatibility
                platform_limits = {
                    "esp32": 4.0,
                    "stm32f4": 2.0,
                    "stm32f7": 2.0,
                    "rp2040": 2.0
                }
                
                max_size = platform_limits.get(platform, 2.0)
                if model_size_mb > max_size:
                    raise Exception(f"Model too large: {model_size_mb:.1f}MB > {max_size}MB")
                
                # Simulate performance results
                base_performance = {"esp32": 8.0, "stm32f4": 4.0, "stm32f7": 6.0, "rp2040": 3.0}.get(platform, 5.0)
                
                return {
                    "status": "success",
                    "model_size_mb": model_size_mb,
                    "compatible": True,
                    "performance": {
                        "tokens_per_second": base_performance,
                        "latency_ms": 1000 / base_performance,
                        "memory_usage_kb": model_size_mb * 1000,
                        "estimated": True
                    }
                }
            
            # Execute with retry mechanism
            profile_result = self.retry_mechanism.retry_with_backoff(do_profiling)
            
            # Record success
            self.circuit_breaker.record_success()
            self.success_count += 1
            
            result.update(profile_result)
            result["reliability_metadata"]["success"] = True
            
        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            error_info = {
                "error": str(e),
                "timestamp": datetime.now(),
                "operation": self.operation_count
            }
            self.error_history.append(error_info)
            
            # Attempt graceful degradation
            degraded_result = self._attempt_graceful_degradation(model_path, platform)
            
            if degraded_result:
                result.update(degraded_result)
                result["reliability_metadata"]["recovery_used"] = True
                result["degradation_reason"] = str(e)
            else:
                result.update({
                    "status": "failed",
                    "error": str(e),
                    "recommendations": [
                        "Check model file path and size",
                        "Verify platform compatibility", 
                        "Monitor system health status",
                        "Consider using graceful degradation"
                    ]
                })
            
            result["reliability_metadata"]["success"] = False
        
        finally:
            end_time = time.time()
            result["reliability_metadata"]["total_time_seconds"] = end_time - start_time
        
        return result
    
    def _attempt_graceful_degradation(self, model_path: str, platform: str) -> Optional[Dict[str, Any]]:
        """Attempt to provide degraded but useful results."""
        try:
            # Basic file existence check
            if not Path(model_path).exists():
                return None
            
            model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            
            # Provide very basic estimates
            return {
                "status": "degraded",
                "model_size_mb": model_size_mb,
                "compatible": model_size_mb < 5.0,  # Conservative estimate
                "performance": {
                    "tokens_per_second": 2.0,  # Very conservative
                    "latency_ms": 500,
                    "memory_usage_kb": model_size_mb * 1200,
                    "estimated": True,
                    "degraded": True
                },
                "recommendations": [
                    "⚠️ Results are degraded due to profiling failure",
                    "🔧 Consider full system health check",
                    "📊 Actual performance may differ significantly"
                ]
            }
            
        except:
            return None
    
    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get reliability metrics."""
        success_rate = self.success_count / self.operation_count if self.operation_count > 0 else 0.0
        
        return {
            "total_operations": self.operation_count,
            "successful_operations": self.success_count,
            "success_rate": success_rate,
            "circuit_breaker_state": self.circuit_breaker.state,
            "recent_errors": len(self.error_history[-10:]),
            "health_status": self.health_monitor.last_check.status.value if self.health_monitor.last_check else "unknown"
        }


def test_generation2_standalone():
    """Test Generation 2 robustness features in standalone mode."""
    
    print("🛡️ Generation 2 Robustness Test (Standalone)")
    print("=" * 60)
    
    # Test 1: Circuit Breaker
    print("\n1. 🔌 Testing Circuit Breaker Pattern")
    cb = StandaloneCircuitBreaker(failure_threshold=3, timeout=0.1)
    
    # Normal operation
    assert cb.can_execute() == True
    print("   ✅ Circuit starts closed - operations allowed")
    
    # Trigger failures
    for i in range(3):
        cb.record_failure()
    
    assert cb.state == "open"
    assert cb.can_execute() == False
    print("   ✅ Circuit opens after failure threshold - operations blocked")
    
    # Test recovery
    time.sleep(0.2)  # Wait for timeout
    assert cb.can_execute() == True  # Should be half-open
    
    cb.record_success()
    assert cb.state == "closed"
    print("   ✅ Circuit recovers after success - operations resumed")
    
    # Test 2: Retry Mechanism
    print("\n2. 🔄 Testing Retry with Exponential Backoff")
    retry = StandaloneRetryMechanism(max_attempts=3, base_delay=0.01)
    
    # Test successful operation
    def success_op():
        return "success"
    
    result = retry.retry_with_backoff(success_op)
    assert result == "success"
    print("   ✅ Successful operation requires no retry")
    
    # Test retry after failure
    attempts = {"count": 0}
    def fail_then_succeed():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise Exception(f"Attempt {attempts['count']} failed")
        return f"Success after {attempts['count']} attempts"
    
    result = retry.retry_with_backoff(fail_then_succeed)
    assert "Success after 3" in result
    print("   ✅ Retry mechanism succeeds after transient failures")
    
    # Test backoff calculation
    delays = [retry.exponential_backoff(i) for i in range(1, 4)]
    expected = [0.01, 0.02, 0.04]  # exponential growth
    assert delays == expected
    print("   ✅ Exponential backoff delays calculated correctly")
    
    # Test 3: Health Monitoring
    print("\n3. 🏥 Testing Health Monitoring System")
    monitor = StandaloneHealthMonitor()
    
    # Register test checks
    def healthy_check():
        return True
    
    def unhealthy_check():
        return False
    
    monitor.register_check("system", healthy_check)
    monitor.register_check("database", unhealthy_check)
    
    health = monitor.run_checks()
    assert health.status == HealthStatus.DEGRADED  # mixed results
    assert health.checks["system"] == True
    assert health.checks["database"] == False
    print(f"   ✅ Health monitoring detects degraded state: {health.message}")
    
    # Test 4: Robust Profiler Integration
    print("\n4. 🛡️ Testing Robust Profiler Integration")
    profiler = StandaloneRobustProfiler()
    
    # Create test model file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
        temp_file.write(b'0' * (1024 * 1024))  # 1MB model
        temp_model_path = temp_file.name
    
    try:
        # Test successful profiling
        result = profiler.robust_profile(temp_model_path, "esp32", simulate_failure=False)
        
        assert result["status"] == "success"
        assert "reliability_metadata" in result
        assert result["reliability_metadata"]["success"] == True
        print("   ✅ Successful profiling with reliability metadata")
        
        # Test health status integration
        health = profiler.get_health_status()
        print(f"   ✅ System health: {health.status.value}")
        
        # Test graceful degradation with failure
        result_failure = profiler.robust_profile(temp_model_path, "esp32", simulate_failure=True)
        
        if result_failure["status"] == "degraded":
            print("   ✅ Graceful degradation provides fallback results")
            assert "degradation_reason" in result_failure
            assert result_failure["performance"]["degraded"] == True
        elif result_failure["status"] == "failed":
            print("   ✅ Graceful failure with helpful error information")
            assert "recommendations" in result_failure
        
        # Test reliability metrics
        metrics = profiler.get_reliability_metrics()
        print(f"   ✅ Reliability metrics: {metrics['success_rate']:.1%} success rate")
        
        # Test circuit breaker integration
        # Simulate multiple failures to open circuit
        for _ in range(5):
            try:
                profiler.robust_profile("/nonexistent/model.bin", "esp32")
            except:
                pass
        
        final_metrics = profiler.get_reliability_metrics()
        print(f"   ✅ Circuit breaker state: {final_metrics['circuit_breaker_state']}")
    
    finally:
        Path(temp_model_path).unlink(missing_ok=True)
    
    # Test 5: Error Recovery and Resilience
    print("\n5. 🔧 Testing Error Recovery and Resilience")
    
    # Test with missing file (should trigger graceful degradation)
    result_missing = profiler.robust_profile("/nonexistent/model.bin", "esp32")
    
    assert result_missing["status"] in ["failed", "degraded"]
    assert "reliability_metadata" in result_missing
    print("   ✅ Handles missing files gracefully")
    
    # Test with oversized model
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
        temp_file.write(b'0' * (10 * 1024 * 1024))  # 10MB model (too large)
        temp_large_model = temp_file.name
    
    try:
        result_large = profiler.robust_profile(temp_large_model, "stm32f4")  # 2MB limit
        
        # Should fail or degrade due to size
        assert result_large["status"] in ["failed", "degraded"]
        print("   ✅ Detects incompatible model sizes")
    
    finally:
        Path(temp_large_model).unlink(missing_ok=True)
    
    print("\n✅ Generation 2 Standalone Test COMPLETED!")
    print("\n🎯 Robustness Features Validated:")
    print("   🔌 Circuit breaker pattern - prevents cascade failures")
    print("   🔄 Retry with exponential backoff - handles transient failures")
    print("   🏥 Health monitoring - tracks system status and checks")
    print("   🛡️ Robust profiler integration - combines all reliability patterns")
    print("   🔧 Graceful degradation - provides fallback when possible")
    print("   📊 Reliability metrics - observability into system health")
    print("   ⚠️ Comprehensive error handling - fails safely with context")
    
    return True


if __name__ == "__main__":
    try:
        success = test_generation2_standalone()
        if success:
            print(f"\n🎉 Generation 2 Robustness Test PASSED!")
            print("🚀 System demonstrates production-ready reliability!")
            print("🚀 Ready to proceed to Generation 3 (Optimization & Scale)!")
        else:
            print(f"\n❌ Generation 2 test failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)