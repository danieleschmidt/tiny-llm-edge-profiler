"""
Generation 2 Enhancement: Robust Profiler with Advanced Reliability
Enhanced profiler with comprehensive error handling, recovery, and resilience patterns.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import logging
import contextlib
from enum import Enum

from .core_lite import QuickStartProfiler, QuickProfileResult, MinimalPlatformManager
from .exceptions import (
    TinyLLMProfilerError,
    DeviceTimeoutError, 
    DeviceConnectionError,
    ProfilingError
)

# Configure logging for robustness
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass 
class ProfilingContext:
    """Context for robust profiling operations."""
    model_path: str
    platform: str
    attempt: int = 1
    max_attempts: int = 3
    timeout_seconds: float = 30.0
    retry_delay: float = 1.0
    graceful_degradation: bool = True
    health_checks_enabled: bool = True


class CircuitBreaker:
    """Generation 2: Simple but effective circuit breaker pattern."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
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
            if time.time() - self.last_failure_time > self.timeout:
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


class RetryMechanism:
    """Generation 2: Enhanced retry mechanism with multiple strategies."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        
    def exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.base_delay * (2 ** (attempt - 1))
    
    def linear_backoff(self, attempt: int) -> float:
        """Calculate linear backoff delay."""
        return self.base_delay * attempt
    
    def fixed_delay(self, attempt: int) -> float:
        """Fixed delay between retries."""
        return self.base_delay
    
    def retry(self, func: Callable, *args, strategy: str = "exponential", **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                logger.info(f"Operation succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt}/{self.max_attempts} failed: {e}")
                
                if attempt < self.max_attempts:
                    if strategy == "exponential":
                        delay = self.exponential_backoff(attempt)
                    elif strategy == "linear":
                        delay = self.linear_backoff(attempt)
                    else:
                        delay = self.fixed_delay(attempt)
                    
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
        
        # All attempts failed
        logger.error(f"All {self.max_attempts} attempts failed")
        raise last_exception


class HealthMonitor:
    """Generation 2: Health monitoring and status tracking."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_time = None
        self.last_health_status = HealthStatus.UNKNOWN
        
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> HealthCheck:
        """Run all health checks and return overall status."""
        check_results = {}
        metrics = {}
        overall_healthy = True
        messages = []
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_func()
                end_time = time.time()
                
                check_results[name] = bool(result)
                metrics[f"{name}_duration_ms"] = (end_time - start_time) * 1000
                
                if not result:
                    overall_healthy = False
                    messages.append(f"{name} check failed")
                    
            except Exception as e:
                check_results[name] = False
                overall_healthy = False
                messages.append(f"{name} check error: {e}")
                logger.error(f"Health check {name} failed: {e}")
        
        # Determine overall status
        if overall_healthy:
            status = HealthStatus.HEALTHY
            message = "All checks passed"
        elif any(check_results.values()):
            status = HealthStatus.DEGRADED
            message = f"Some checks failed: {', '.join(messages)}"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"All checks failed: {', '.join(messages)}"
        
        self.last_check_time = datetime.now()
        self.last_health_status = status
        
        return HealthCheck(
            status=status,
            message=message,
            checks=check_results,
            metrics=metrics
        )


class ResourceMonitor:
    """Generation 2: Monitor system resources and prevent exhaustion."""
    
    def __init__(self):
        self.memory_threshold_mb = 100  # Minimum free memory
        self.disk_threshold_mb = 500    # Minimum free disk
        self.cpu_threshold_percent = 90  # Maximum CPU usage
        
    def check_memory(self) -> bool:
        """Check if sufficient memory is available."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            free_memory_mb = memory.available / (1024 * 1024)
            return free_memory_mb > self.memory_threshold_mb
        except ImportError:
            # Fallback: assume memory is ok if we can't check
            logger.warning("psutil not available, skipping memory check")
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
    
    def check_disk_space(self) -> bool:
        """Check if sufficient disk space is available."""
        try:
            import shutil
            free_space = shutil.disk_usage('.').free
            free_space_mb = free_space / (1024 * 1024)
            return free_space_mb > self.disk_threshold_mb
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return False
    
    def check_cpu_usage(self) -> bool:
        """Check if CPU usage is within acceptable limits."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < self.cpu_threshold_percent
        except ImportError:
            # Fallback: assume CPU is ok if we can't check
            logger.warning("psutil not available, skipping CPU check")
            return True
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return False


class RobustProfiler:
    """
    Generation 2: Robust profiler with comprehensive reliability features.
    
    Features:
    - Circuit breaker pattern for fast failure
    - Retry mechanisms with multiple strategies
    - Health monitoring and status checks
    - Resource monitoring and limits
    - Graceful degradation
    - Comprehensive error handling
    - Recovery mechanisms
    """
    
    def __init__(self, enable_monitoring: bool = True):
        self.base_profiler = QuickStartProfiler()
        self.circuit_breaker = CircuitBreaker()
        self.retry_mechanism = RetryMechanism()
        self.health_monitor = HealthMonitor()
        self.resource_monitor = ResourceMonitor()
        
        self.enable_monitoring = enable_monitoring
        self.profiling_history: List[Dict[str, Any]] = []
        self.error_history: List[Dict[str, Any]] = []
        
        # Register default health checks
        if enable_monitoring:
            self._register_default_health_checks()
        
        logger.info("RobustProfiler initialized with comprehensive reliability features")
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        self.health_monitor.register_check("memory", self.resource_monitor.check_memory)
        self.health_monitor.register_check("disk_space", self.resource_monitor.check_disk_space)
        self.health_monitor.register_check("cpu_usage", self.resource_monitor.check_cpu_usage)
        self.health_monitor.register_check("circuit_breaker", lambda: self.circuit_breaker.state == "closed")
    
    def get_health_status(self) -> HealthCheck:
        """Get current health status."""
        return self.health_monitor.run_health_checks()
    
    def is_healthy(self) -> bool:
        """Quick health check - returns True if system is healthy."""
        health = self.get_health_status()
        return health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def robust_profile(self, 
                      model_path: str, 
                      platform: str,
                      context: Optional[ProfilingContext] = None) -> Dict[str, Any]:
        """
        Perform robust profiling with comprehensive error handling.
        
        Args:
            model_path: Path to model file
            platform: Target platform
            context: Optional profiling context for advanced configuration
            
        Returns:
            Dictionary with profiling results and reliability metadata
        """
        if context is None:
            context = ProfilingContext(
                model_path=model_path,
                platform=platform
            )
        
        start_time = time.time()
        result = {
            "model_path": model_path,
            "platform": platform,
            "timestamp": datetime.now().isoformat(),
            "reliability_info": {
                "circuit_breaker_state": self.circuit_breaker.state,
                "health_status": None,
                "attempts_made": 0,
                "total_time_seconds": 0,
                "errors_encountered": []
            }
        }
        
        try:
            # Pre-profiling health check
            if self.enable_monitoring:
                health = self.get_health_status()
                result["reliability_info"]["health_status"] = {
                    "status": health.status.value,
                    "message": health.message,
                    "checks": health.checks
                }
                
                if health.status == HealthStatus.UNHEALTHY:
                    raise ProfilingError("System health check failed - aborting profiling")
            
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                raise ProfilingError("Circuit breaker is open - service temporarily unavailable")
            
            # Execute profiling with retry mechanism
            def _do_profiling():
                return self.base_profiler.quick_profile(model_path, platform)
            
            profile_result = self.retry_mechanism.retry(
                _do_profiling,
                strategy="exponential"
            )
            
            # Record success
            self.circuit_breaker.record_success()
            
            # Convert QuickProfileResult to dict if needed
            if hasattr(profile_result, '__dict__'):
                profile_data = profile_result.__dict__
            else:
                profile_data = profile_result
            
            result.update({
                "status": "success",
                "profiling_results": profile_data,
                "recommendations": self._generate_robust_recommendations(profile_data, context)
            })
            
            # Record successful profiling
            self.profiling_history.append({
                "timestamp": datetime.now(),
                "model_path": model_path,
                "platform": platform,
                "success": True
            })
            
        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now()
            }
            
            result["reliability_info"]["errors_encountered"].append(error_info)
            self.error_history.append(error_info)
            
            # Attempt graceful degradation
            if context.graceful_degradation:
                degraded_result = self._attempt_graceful_degradation(model_path, platform, e)
                if degraded_result:
                    result.update({
                        "status": "degraded",
                        "profiling_results": degraded_result,
                        "degradation_reason": str(e),
                        "recommendations": ["Results may be less accurate due to error recovery"]
                    })
                else:
                    result.update({
                        "status": "failed",
                        "error": str(e),
                        "recommendations": self._generate_failure_recommendations(e)
                    })
            else:
                result.update({
                    "status": "failed", 
                    "error": str(e),
                    "recommendations": self._generate_failure_recommendations(e)
                })
        
        finally:
            end_time = time.time()
            result["reliability_info"]["total_time_seconds"] = end_time - start_time
            result["reliability_info"]["attempts_made"] = context.attempt
        
        return result
    
    def _attempt_graceful_degradation(self, model_path: str, platform: str, error: Exception) -> Optional[Dict[str, Any]]:
        """Attempt to provide degraded but useful results when full profiling fails."""
        try:
            logger.info(f"Attempting graceful degradation for {model_path} on {platform}")
            
            # Try basic compatibility check only
            compatibility = self.base_profiler.check_model_compatibility(model_path, platform)
            
            if compatibility.get("compatible", False):
                # Provide basic estimates based on model size and platform
                model_size_mb = compatibility.get("model_size_mb", 0)
                platform_info = MinimalPlatformManager().get_platform_info(platform)
                
                if platform_info:
                    # Very conservative estimates
                    estimated_tps = platform_info["max_freq_mhz"] / 200  # Very conservative
                    estimated_memory = min(platform_info["ram_kb"] * 0.5, model_size_mb * 1000)
                    
                    return {
                        "status": "estimated",
                        "tokens_per_second": estimated_tps,
                        "latency_ms": 1000 / estimated_tps if estimated_tps > 0 else 1000,
                        "memory_usage_kb": estimated_memory,
                        "power_consumption_mw": 75,  # Conservative estimate
                        "compatibility": compatibility,
                        "note": "Estimates only - actual profiling failed"
                    }
            
            return None
            
        except Exception as degradation_error:
            logger.error(f"Graceful degradation also failed: {degradation_error}")
            return None
    
    def _generate_robust_recommendations(self, profile_data: Dict[str, Any], context: ProfilingContext) -> List[str]:
        """Generate recommendations including reliability considerations."""
        recommendations = []
        
        # Add performance recommendations
        if "performance" in profile_data:
            perf = profile_data["performance"]
            tps = perf.get("tokens_per_second", 0)
            
            if tps > 10:
                recommendations.append("✅ Excellent performance - ready for production deployment")
            elif tps > 5:
                recommendations.append("🟡 Good performance - suitable for most use cases")
            else:
                recommendations.append("🔴 Performance needs optimization before production")
        
        # Add reliability recommendations
        circuit_state = self.circuit_breaker.state
        if circuit_state != "closed":
            recommendations.append(f"⚠️ Circuit breaker is {circuit_state} - monitor system health")
        
        error_count = len(self.error_history[-10:])  # Last 10 operations
        if error_count > 3:
            recommendations.append("🔧 High error rate detected - consider system maintenance")
        
        # Add resource recommendations
        if self.enable_monitoring:
            health = self.get_health_status()
            if health.status == HealthStatus.DEGRADED:
                recommendations.append("⚠️ System resources are constrained - monitor performance")
        
        return recommendations
    
    def _generate_failure_recommendations(self, error: Exception) -> List[str]:
        """Generate recommendations for handling failures."""
        recommendations = []
        
        error_type = type(error).__name__
        
        if "timeout" in error_type.lower():
            recommendations.extend([
                "⏱️ Operation timed out - try increasing timeout duration",
                "🔧 Check system performance and reduce load",
                "🌐 Verify network connectivity if using remote resources"
            ])
        elif "connection" in error_type.lower():
            recommendations.extend([
                "🔌 Connection failed - check device connectivity",
                "🔧 Verify device drivers and permissions",
                "🔄 Try reconnecting or restart the device"
            ])
        elif "memory" in error_type.lower():
            recommendations.extend([
                "💾 Memory issue detected - free up system memory",
                "📉 Consider using a smaller model",
                "🔧 Restart the profiling process"
            ])
        else:
            recommendations.extend([
                "🔧 General error - check system logs for details",
                "🔄 Retry the operation with different parameters",
                "🐛 Report this issue if problem persists"
            ])
        
        return recommendations
    
    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get reliability metrics and statistics."""
        recent_errors = self.error_history[-50:]  # Last 50 errors
        recent_profiles = self.profiling_history[-100:]  # Last 100 profiles
        
        success_rate = 0.0
        if recent_profiles:
            successful = sum(1 for p in recent_profiles if p.get("success", False))
            success_rate = successful / len(recent_profiles)
        
        return {
            "circuit_breaker_state": self.circuit_breaker.state,
            "success_rate": success_rate,
            "error_count_last_24h": len([e for e in recent_errors 
                                        if (datetime.now() - e["timestamp"]).days == 0]),
            "total_profiles_run": len(self.profiling_history),
            "total_errors": len(self.error_history),
            "health_status": self.health_monitor.last_health_status.value if self.health_monitor.last_health_status else "unknown",
            "last_health_check": self.health_monitor.last_check_time.isoformat() if self.health_monitor.last_check_time else None
        }


# Convenience functions for Generation 2

def robust_quick_profile(model_path: str, platform: str, enable_monitoring: bool = True) -> Dict[str, Any]:
    """
    Quick robust profiling function with built-in reliability.
    
    Args:
        model_path: Path to model file
        platform: Target platform
        enable_monitoring: Enable health monitoring
        
    Returns:
        Comprehensive profiling results with reliability metadata
    """
    profiler = RobustProfiler(enable_monitoring=enable_monitoring)
    return profiler.robust_profile(model_path, platform)


def check_system_health() -> HealthCheck:
    """Quick system health check."""
    profiler = RobustProfiler()
    return profiler.get_health_status()


# Export main classes and functions
__all__ = [
    "RobustProfiler",
    "HealthStatus",
    "HealthCheck", 
    "ProfilingContext",
    "CircuitBreaker",
    "RetryMechanism",
    "HealthMonitor",
    "ResourceMonitor",
    "robust_quick_profile",
    "check_system_health"
]