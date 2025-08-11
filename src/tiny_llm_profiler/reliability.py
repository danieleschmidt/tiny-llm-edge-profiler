"""
Reliability utilities and resilience patterns for the Tiny LLM Edge Profiler.

This module provides production-ready reliability patterns including:
- Retry mechanisms with exponential backoff
- Circuit breaker patterns
- Timeout handling
- Graceful degradation
- Health checking
- Resource management
"""

import time
import asyncio
import threading
import functools
from typing import (
    Any, Callable, Dict, List, Optional, Union, TypeVar, Generic,
    Awaitable, Type, Tuple
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import random
import contextlib
from pathlib import Path
import weakref

from .exceptions import (
    TinyLLMProfilerError, DeviceTimeoutError, DeviceConnectionError,
    ResourceError, ProfilingTimeoutError
)
from .logging_config import get_logger

logger = get_logger("reliability")

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class RetryStrategy(str, Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    backoff_multiplier: float = 2.0
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
    on_retry: Optional[Callable[[int, Exception], None]] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    success_threshold: int = 3
    monitor_window_seconds: float = 300.0  # 5 minutes
    failure_rate_threshold: float = 0.5  # 50% failure rate


@dataclass 
class TimeoutConfig:
    """Configuration for timeout handling."""
    default_timeout: float = 30.0
    connection_timeout: float = 10.0
    operation_timeout: float = 300.0
    heartbeat_interval: float = 5.0


class RetryableError(TinyLLMProfilerError):
    """Exception that indicates an operation should be retried."""
    
    def __init__(self, message: str, original_error: Exception, retry_after: Optional[float] = None):
        super().__init__(message, "RETRYABLE_ERROR", {"original_error": str(original_error)})
        self.original_error = original_error
        self.retry_after = retry_after


class CircuitOpenError(TinyLLMProfilerError):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, circuit_name: str, failure_count: int, last_failure_time: datetime):
        message = f"Circuit '{circuit_name}' is open due to {failure_count} failures"
        super().__init__(
            message, 
            "CIRCUIT_OPEN", 
            {
                "circuit_name": circuit_name,
                "failure_count": failure_count,
                "last_failure_time": last_failure_time.isoformat()
            }
        )


class RetryMechanism:
    """Implements various retry strategies with backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = get_logger("retry")
    
    def __call__(self, func: F) -> F:
        """Decorator for adding retry logic to functions."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_with_retry(func, args, kwargs)
        
        return wrapper
    
    def _execute_with_retry(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.logger.debug(f"Attempting {func.__name__}, attempt {attempt}/{self.config.max_attempts}")
                result = func(*args, **kwargs)
                
                if attempt > 1:
                    self.logger.info(f"{func.__name__} succeeded on attempt {attempt}")
                
                return result
                
            except self.config.exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts:
                    self.logger.error(f"{func.__name__} failed after {attempt} attempts: {e}")
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                self.logger.warning(
                    f"{func.__name__} failed on attempt {attempt}: {e}. Retrying in {delay:.2f}s"
                )
                
                # Call retry callback if provided
                if self.config.on_retry:
                    try:
                        self.config.on_retry(attempt, e)
                    except Exception as callback_error:
                        self.logger.error(f"Retry callback failed: {callback_error}")
                
                time.sleep(delay)
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(
                self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1)),
                self.config.max_delay
            )
            
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(
                self.config.base_delay * attempt,
                self.config.max_delay
            )
            
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            fib_value = self._fibonacci(attempt)
            delay = min(
                self.config.base_delay * fib_value,
                self.config.max_delay
            )
        else:
            delay = self.config.base_delay
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number for backoff."""
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_changed_time = datetime.now()
        self.failure_history: List[datetime] = []
        self.logger = get_logger(f"circuit_breaker.{name}")
        self._lock = threading.Lock()
    
    def __call__(self, func: F) -> F:
        """Decorator for protecting functions with circuit breaker."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_protected(func, args, kwargs)
        
        return wrapper
    
    def _execute_protected(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.logger.warning(f"Circuit {self.name} is OPEN, failing fast")
                    raise CircuitOpenError(
                        self.name, 
                        self.failure_count, 
                        self.last_failure_time or datetime.now()
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                    
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
                self.failure_history.clear()
    
    def _on_failure(self, exception: Exception):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            self.failure_history.append(self.last_failure_time)
            
            # Clean old failures outside monitoring window
            cutoff_time = self.last_failure_time - timedelta(seconds=self.config.monitor_window_seconds)
            self.failure_history = [
                ts for ts in self.failure_history 
                if ts > cutoff_time
            ]
            
            self.logger.warning(f"Circuit {self.name} failure #{self.failure_count}: {exception}")
            
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to_open()
                
            elif self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened."""
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate within monitoring window
        if len(self.failure_history) >= 2:  # Need at least 2 data points
            window_start = datetime.now() - timedelta(seconds=self.config.monitor_window_seconds)
            recent_failures = len(self.failure_history)
            
            # For simplicity, assume equal number of successes and failures
            # In production, you'd track successes too
            total_operations = recent_failures * 2  # Rough estimate
            failure_rate = recent_failures / total_operations if total_operations > 0 else 0
            
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from open state."""
        if not self.last_failure_time:
            return False
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_seconds
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        self.logger.info(f"Circuit {self.name} transitioning to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.failure_history.clear()
        self.state_changed_time = datetime.now()
    
    def _transition_to_open(self):
        """Transition to open state."""
        self.logger.warning(f"Circuit {self.name} transitioning to OPEN")
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.state_changed_time = datetime.now()
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        self.logger.info(f"Circuit {self.name} transitioning to HALF-OPEN")
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.state_changed_time = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "state_changed_time": self.state_changed_time.isoformat(),
                "failures_in_window": len(self.failure_history)
            }
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self.logger.info(f"Manually resetting circuit {self.name}")
            self._transition_to_closed()


class TimeoutManager:
    """Manages timeouts for operations."""
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.logger = get_logger("timeout")
    
    @contextlib.contextmanager
    def timeout(self, seconds: Optional[float] = None, operation_name: str = "operation"):
        """Context manager for timeout protection."""
        timeout_duration = seconds or self.config.default_timeout
        start_time = time.time()
        
        try:
            self.logger.debug(f"Starting {operation_name} with {timeout_duration}s timeout")
            yield
            
            duration = time.time() - start_time
            self.logger.debug(f"{operation_name} completed in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            
            if duration >= timeout_duration:
                self.logger.error(f"{operation_name} timed out after {duration:.2f}s")
                raise ProfilingTimeoutError(operation_name, timeout_duration)
            else:
                self.logger.error(f"{operation_name} failed after {duration:.2f}s: {e}")
                raise
    
    def with_timeout(self, timeout_seconds: Optional[float] = None):
        """Decorator for adding timeout to functions."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.timeout(timeout_seconds, func.__name__):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator


class ResourceManager:
    """Manages system resources and prevents leaks."""
    
    def __init__(self):
        self.active_resources: weakref.WeakSet = weakref.WeakSet()
        self.resource_registry: Dict[str, List[Any]] = {}
        self.logger = get_logger("resource_manager")
        self._lock = threading.Lock()
    
    def register_resource(self, resource: Any, category: str = "general", cleanup_func: Optional[Callable] = None):
        """Register a resource for tracking."""
        with self._lock:
            if category not in self.resource_registry:
                self.resource_registry[category] = []
            
            resource_info = {
                "resource": resource,
                "created_at": datetime.now(),
                "cleanup_func": cleanup_func
            }
            
            self.resource_registry[category].append(resource_info)
            
            # Also add to weak set for automatic cleanup
            try:
                self.active_resources.add(resource)
            except TypeError:
                # Resource is not weakly referenceable
                pass
            
            self.logger.debug(f"Registered {category} resource: {type(resource).__name__}")
    
    def cleanup_resources(self, category: Optional[str] = None):
        """Clean up resources in a category or all resources."""
        with self._lock:
            categories = [category] if category else list(self.resource_registry.keys())
            
            for cat in categories:
                if cat in self.resource_registry:
                    resources = self.resource_registry[cat]
                    self.logger.info(f"Cleaning up {len(resources)} {cat} resources")
                    
                    for resource_info in resources[:]:  # Copy to avoid modification during iteration
                        try:
                            if resource_info["cleanup_func"]:
                                resource_info["cleanup_func"]()
                            elif hasattr(resource_info["resource"], "close"):
                                resource_info["resource"].close()
                            elif hasattr(resource_info["resource"], "cleanup"):
                                resource_info["resource"].cleanup()
                            
                            self.resource_registry[cat].remove(resource_info)
                            
                        except Exception as e:
                            self.logger.error(f"Error cleaning up {cat} resource: {e}")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about managed resources."""
        with self._lock:
            stats = {
                "total_categories": len(self.resource_registry),
                "active_weak_refs": len(self.active_resources),
                "by_category": {}
            }
            
            for category, resources in self.resource_registry.items():
                stats["by_category"][category] = {
                    "count": len(resources),
                    "oldest": min((r["created_at"] for r in resources), default=None),
                    "newest": max((r["created_at"] for r in resources), default=None)
                }
                
                # Convert datetime to string for JSON serialization
                if stats["by_category"][category]["oldest"]:
                    stats["by_category"][category]["oldest"] = stats["by_category"][category]["oldest"].isoformat()
                if stats["by_category"][category]["newest"]:
                    stats["by_category"][category]["newest"] = stats["by_category"][category]["newest"].isoformat()
            
            return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all resources."""
        self.cleanup_resources()


class GracefulDegradation:
    """Implements graceful degradation patterns."""
    
    def __init__(self):
        self.fallback_functions: Dict[str, List[Callable]] = {}
        self.logger = get_logger("graceful_degradation")
    
    def register_fallback(self, operation_name: str, fallback_func: Callable, priority: int = 0):
        """Register a fallback function for an operation."""
        if operation_name not in self.fallback_functions:
            self.fallback_functions[operation_name] = []
        
        self.fallback_functions[operation_name].append({
            "func": fallback_func,
            "priority": priority
        })
        
        # Sort by priority (lower number = higher priority)
        self.fallback_functions[operation_name].sort(key=lambda x: x["priority"])
        
        self.logger.debug(f"Registered fallback for {operation_name} with priority {priority}")
    
    def with_fallback(self, operation_name: str, *fallback_funcs: Callable):
        """Decorator for adding fallback functions to operations."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_fallback(operation_name, func, args, kwargs, fallback_funcs)
            
            return wrapper
        return decorator
    
    def _execute_with_fallback(
        self, 
        operation_name: str, 
        primary_func: Callable, 
        args: tuple, 
        kwargs: dict,
        additional_fallbacks: Tuple[Callable, ...] = ()
    ) -> Any:
        """Execute function with fallback options."""
        # Try primary function first
        try:
            self.logger.debug(f"Executing primary function for {operation_name}")
            return primary_func(*args, **kwargs)
        except Exception as primary_error:
            self.logger.warning(f"Primary function failed for {operation_name}: {primary_error}")
            
            # Collect all fallback functions
            all_fallbacks = []
            
            # Add registered fallbacks
            if operation_name in self.fallback_functions:
                all_fallbacks.extend([fb["func"] for fb in self.fallback_functions[operation_name]])
            
            # Add additional fallbacks from decorator
            all_fallbacks.extend(additional_fallbacks)
            
            # Try fallbacks in order
            for i, fallback_func in enumerate(all_fallbacks):
                try:
                    self.logger.info(f"Attempting fallback #{i+1} for {operation_name}")
                    result = fallback_func(*args, **kwargs)
                    self.logger.info(f"Fallback #{i+1} succeeded for {operation_name}")
                    return result
                    
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback #{i+1} failed for {operation_name}: {fallback_error}")
                    continue
            
            # All fallbacks failed
            self.logger.error(f"All fallbacks failed for {operation_name}")
            raise primary_error  # Re-raise the original error


class ReliabilityManager:
    """Central manager for all reliability patterns."""
    
    def __init__(self):
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.timeout_manager = TimeoutManager(TimeoutConfig())
        self.resource_manager = ResourceManager()
        self.graceful_degradation = GracefulDegradation()
        self.logger = get_logger("reliability_manager")
    
    def create_retry_decorator(self, name: str, config: Optional[RetryConfig] = None) -> Callable:
        """Create a retry decorator with the specified configuration."""
        if name not in self.retry_configs:
            self.retry_configs[name] = config or RetryConfig()
        
        return RetryMechanism(self.retry_configs[name])
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create or get a circuit breaker."""
        if name not in self.circuit_breakers:
            breaker_config = config or CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(name, breaker_config)
        
        return self.circuit_breakers[name]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all reliability components."""
        circuit_statuses = {
            name: cb.get_status() 
            for name, cb in self.circuit_breakers.items()
        }
        
        resource_stats = self.resource_manager.get_resource_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": circuit_statuses,
            "resource_stats": resource_stats,
            "retry_configs": {
                name: {
                    "max_attempts": config.max_attempts,
                    "strategy": config.strategy.value,
                    "base_delay": config.base_delay
                }
                for name, config in self.retry_configs.items()
            }
        }
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers to closed state."""
        for cb in self.circuit_breakers.values():
            cb.reset()
        
        self.logger.info(f"Reset {len(self.circuit_breakers)} circuit breakers")
    
    def cleanup_all_resources(self):
        """Cleanup all managed resources."""
        self.resource_manager.cleanup_resources()
        self.logger.info("Cleaned up all managed resources")


# Global reliability manager instance
reliability_manager = ReliabilityManager()

# Convenience functions and decorators
def retry(name: str = "default", **config_kwargs) -> Callable:
    """Convenient retry decorator."""
    config = RetryConfig(**config_kwargs) if config_kwargs else None
    return reliability_manager.create_retry_decorator(name, config)


def circuit_breaker(name: str, **config_kwargs) -> Callable:
    """Convenient circuit breaker decorator."""
    config = CircuitBreakerConfig(**config_kwargs) if config_kwargs else None
    cb = reliability_manager.create_circuit_breaker(name, config)
    return cb


def with_timeout(seconds: float) -> Callable:
    """Convenient timeout decorator."""
    return reliability_manager.timeout_manager.with_timeout(seconds)


def with_fallback(*fallback_funcs: Callable) -> Callable:
    """Convenient fallback decorator."""
    def decorator(func: F) -> F:
        operation_name = func.__name__
        return reliability_manager.graceful_degradation.with_fallback(operation_name, *fallback_funcs)(func)
    return decorator


@contextlib.contextmanager
def managed_resource(resource: Any, category: str = "general", cleanup_func: Optional[Callable] = None):
    """Context manager for automatic resource management."""
    reliability_manager.resource_manager.register_resource(resource, category, cleanup_func)
    try:
        yield resource
    finally:
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as e:
                logger.error(f"Error in cleanup function: {e}")