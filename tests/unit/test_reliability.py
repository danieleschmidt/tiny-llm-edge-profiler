"""
Unit tests for reliability and robustness features.

Tests the retry mechanisms, circuit breakers, timeout handling,
and other reliability patterns implemented in Generation 2.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from tiny_llm_profiler.reliability import (
    RetryMechanism, RetryConfig, RetryStrategy, CircuitBreaker,
    CircuitBreakerConfig, CircuitState, TimeoutManager, TimeoutConfig,
    ResourceManager, GracefulDegradation, ReliabilityManager,
    RetryableError, CircuitOpenError
)
from tiny_llm_profiler.exceptions import (
    DeviceConnectionError, ProfilingTimeoutError
)


class TestRetryMechanism:
    """Test retry mechanisms and strategies."""
    
    def test_successful_operation_no_retry(self):
        """Test that successful operations don't trigger retries."""
        config = RetryConfig(max_attempts=3)
        retry_mechanism = RetryMechanism(config)
        
        mock_func = Mock(return_value="success")
        result = retry_mechanism._execute_with_retry(mock_func, (), {})
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_on_failure(self):
        """Test that failures trigger retries."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)  # Fast for testing
        retry_mechanism = RetryMechanism(config)
        
        # Mock function that fails twice then succeeds
        mock_func = Mock(side_effect=[Exception("fail1"), Exception("fail2"), "success"])
        
        result = retry_mechanism._execute_with_retry(mock_func, (), {})
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_max_attempts_exceeded(self):
        """Test that failures beyond max attempts raise the last exception."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        retry_mechanism = RetryMechanism(config)
        
        # Mock function that always fails
        test_exception = Exception("always fails")
        mock_func = Mock(side_effect=test_exception)
        
        with pytest.raises(Exception, match="always fails"):
            retry_mechanism._execute_with_retry(mock_func, (), {})
        
        assert mock_func.call_count == 2
    
    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False  # Disable jitter for predictable testing
        )
        retry_mechanism = RetryMechanism(config)
        
        # Test delay calculation for different attempts
        assert retry_mechanism._calculate_delay(1) == 1.0  # 1.0 * 2^0
        assert retry_mechanism._calculate_delay(2) == 2.0  # 1.0 * 2^1
        assert retry_mechanism._calculate_delay(3) == 4.0  # 1.0 * 2^2
    
    def test_linear_backoff_delay(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=1.0,
            jitter=False
        )
        retry_mechanism = RetryMechanism(config)
        
        assert retry_mechanism._calculate_delay(1) == 1.0  # 1.0 * 1
        assert retry_mechanism._calculate_delay(2) == 2.0  # 1.0 * 2
        assert retry_mechanism._calculate_delay(3) == 3.0  # 1.0 * 3
    
    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            max_delay=5.0,
            backoff_multiplier=2.0,
            jitter=False
        )
        retry_mechanism = RetryMechanism(config)
        
        # At attempt 10, exponential backoff would be 512s, but should be capped at 5s
        assert retry_mechanism._calculate_delay(10) == 5.0
    
    def test_retry_decorator(self):
        """Test the retry decorator functionality."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_decorator = RetryMechanism(config)
        
        call_count = 0
        
        @retry_decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("fail")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_callback(self):
        """Test that retry callbacks are called on failures."""
        callback_calls = []
        
        def retry_callback(attempt, exception):
            callback_calls.append((attempt, str(exception)))
        
        config = RetryConfig(max_attempts=3, base_delay=0.01, on_retry=retry_callback)
        retry_mechanism = RetryMechanism(config)
        
        mock_func = Mock(side_effect=[Exception("fail1"), Exception("fail2"), "success"])
        retry_mechanism._execute_with_retry(mock_func, (), {})
        
        assert len(callback_calls) == 2
        assert callback_calls[0] == (1, "fail1")
        assert callback_calls[1] == (2, "fail2")


class TestCircuitBreaker:
    """Test circuit breaker patterns."""
    
    def test_initial_state(self):
        """Test that circuit breaker starts in closed state."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker("test", config)
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_successful_operations(self):
        """Test that successful operations don't affect the circuit."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        mock_func = Mock(return_value="success")
        
        for _ in range(5):
            result = cb._execute_protected(mock_func, (), {})
            assert result == "success"
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_opens_on_failures(self):
        """Test that circuit opens after failure threshold is reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        mock_func = Mock(side_effect=Exception("fail"))
        
        # First 3 failures should execute and increment failure count
        for i in range(3):
            with pytest.raises(Exception, match="fail"):
                cb._execute_protected(mock_func, (), {})
        
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3
        
        # Subsequent calls should fail fast without executing the function
        mock_func.reset_mock()
        with pytest.raises(CircuitOpenError):
            cb._execute_protected(mock_func, (), {})
        
        assert mock_func.call_count == 0  # Function not called
    
    def test_circuit_half_open_transition(self):
        """Test transition to half-open state after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.1)
        cb = CircuitBreaker("test", config)
        
        # Trigger circuit to open
        mock_func = Mock(side_effect=Exception("fail"))
        for _ in range(2):
            with pytest.raises(Exception):
                cb._execute_protected(mock_func, (), {})
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout and try again
        time.sleep(0.15)
        
        # Next call should transition to half-open and execute
        mock_func.side_effect = None
        mock_func.return_value = "success"
        
        result = cb._execute_protected(mock_func, (), {})
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_circuit_closes_after_success_threshold(self):
        """Test that circuit closes after enough successes in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=3, timeout_seconds=0.1)
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        mock_func = Mock(side_effect=Exception("fail"))
        for _ in range(2):
            with pytest.raises(Exception):
                cb._execute_protected(mock_func, (), {})
        
        # Wait and transition to half-open
        time.sleep(0.15)
        mock_func.side_effect = None
        mock_func.return_value = "success"
        
        # First success transitions to half-open
        cb._execute_protected(mock_func, (), {})
        assert cb.state == CircuitState.HALF_OPEN
        
        # Additional successes should close the circuit
        for _ in range(2):  # Need 2 more for total of 3
            cb._execute_protected(mock_func, (), {})
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_decorator(self):
        """Test the circuit breaker decorator functionality."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)
        
        call_count = 0
        
        @cb
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("fail")
            return "success"
        
        # First 2 calls fail and open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                test_function()
        
        # Third call should fail fast with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            test_function()
        
        assert call_count == 2  # Function only called twice
    
    def test_get_status(self):
        """Test circuit breaker status reporting."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)
        
        status = cb.get_status()
        
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert "state_changed_time" in status


class TestTimeoutManager:
    """Test timeout handling."""
    
    def test_timeout_context_success(self):
        """Test that successful operations within timeout don't raise exceptions."""
        config = TimeoutConfig(default_timeout=1.0)
        timeout_manager = TimeoutManager(config)
        
        with timeout_manager.timeout(operation_name="test"):
            time.sleep(0.1)  # Short operation
        
        # Should complete without exception
    
    def test_timeout_decorator(self):
        """Test the timeout decorator."""
        config = TimeoutConfig()
        timeout_manager = TimeoutManager(config)
        
        @timeout_manager.with_timeout(0.1)
        def quick_function():
            time.sleep(0.05)
            return "success"
        
        result = quick_function()
        assert result == "success"


class TestResourceManager:
    """Test resource management."""
    
    def test_resource_registration(self):
        """Test resource registration and tracking."""
        manager = ResourceManager()
        
        mock_resource = Mock()
        manager.register_resource(mock_resource, "test_category")
        
        stats = manager.get_resource_stats()
        assert stats["by_category"]["test_category"]["count"] == 1
    
    def test_resource_cleanup(self):
        """Test resource cleanup."""
        manager = ResourceManager()
        
        mock_resource = Mock()
        cleanup_func = Mock()
        
        manager.register_resource(mock_resource, "test", cleanup_func)
        manager.cleanup_resources("test")
        
        cleanup_func.assert_called_once()
        
        # Resource should be removed from registry
        stats = manager.get_resource_stats()
        assert stats["by_category"]["test"]["count"] == 0
    
    def test_context_manager(self):
        """Test resource manager context manager."""
        cleanup_called = False
        
        def cleanup_func():
            nonlocal cleanup_called
            cleanup_called = True
        
        with ResourceManager() as manager:
            mock_resource = Mock()
            manager.register_resource(mock_resource, "test", cleanup_func)
        
        # Cleanup should be called on context exit
        assert cleanup_called


class TestGracefulDegradation:
    """Test graceful degradation patterns."""
    
    def test_fallback_execution(self):
        """Test that fallback functions are called when primary fails."""
        degradation = GracefulDegradation()
        
        def primary_func():
            raise Exception("primary failed")
        
        def fallback_func():
            return "fallback success"
        
        @degradation.with_fallback("test_op", fallback_func)
        def decorated_func():
            return primary_func()
        
        result = decorated_func()
        assert result == "fallback success"
    
    def test_multiple_fallbacks(self):
        """Test that multiple fallbacks are tried in order."""
        degradation = GracefulDegradation()
        
        def fallback1():
            raise Exception("fallback1 failed")
        
        def fallback2():
            return "fallback2 success"
        
        @degradation.with_fallback("test_op", fallback1, fallback2)
        def primary_func():
            raise Exception("primary failed")
        
        result = primary_func()
        assert result == "fallback2 success"
    
    def test_registered_fallbacks(self):
        """Test using registered fallback functions."""
        degradation = GracefulDegradation()
        
        def fallback_func():
            return "registered fallback"
        
        degradation.register_fallback("test_op", fallback_func)
        
        def primary_func():
            raise Exception("primary failed")
        
        result = degradation._execute_with_fallback("test_op", primary_func, (), {})
        assert result == "registered fallback"


class TestReliabilityManager:
    """Test the central reliability manager."""
    
    def test_create_retry_decorator(self):
        """Test creating retry decorators through the manager."""
        manager = ReliabilityManager()
        
        retry_config = RetryConfig(max_attempts=2)
        retry_decorator = manager.create_retry_decorator("test", retry_config)
        
        assert "test" in manager.retry_configs
        assert isinstance(retry_decorator, RetryMechanism)
    
    def test_create_circuit_breaker(self):
        """Test creating circuit breakers through the manager."""
        manager = ReliabilityManager()
        
        cb_config = CircuitBreakerConfig(failure_threshold=3)
        cb = manager.create_circuit_breaker("test", cb_config)
        
        assert "test" in manager.circuit_breakers
        assert isinstance(cb, CircuitBreaker)
    
    def test_health_status(self):
        """Test getting overall health status."""
        manager = ReliabilityManager()
        
        # Create some components
        manager.create_retry_decorator("test_retry")
        manager.create_circuit_breaker("test_cb")
        
        health_status = manager.get_health_status()
        
        assert "timestamp" in health_status
        assert "circuit_breakers" in health_status
        assert "resource_stats" in health_status
        assert "retry_configs" in health_status
        
        assert "test_cb" in health_status["circuit_breakers"]
        assert "test_retry" in health_status["retry_configs"]


@pytest.fixture
def sample_exceptions():
    """Fixture providing sample exceptions for testing."""
    return [
        DeviceConnectionError("/dev/ttyUSB0", "esp32", Exception("Connection failed")),
        ProfilingTimeoutError("inference", 30.0),
        RetryableError("Retryable operation failed", Exception("Original error")),
    ]


class TestReliabilityIntegration:
    """Integration tests for reliability patterns."""
    
    def test_retry_with_circuit_breaker(self):
        """Test retry mechanism combined with circuit breaker."""
        # This is a more realistic scenario where retry and circuit breaker work together
        
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        cb_config = CircuitBreakerConfig(failure_threshold=2)
        
        retry_mechanism = RetryMechanism(retry_config)
        cb = CircuitBreaker("test", cb_config)
        
        failure_count = 0
        
        def flaky_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 5:
                raise Exception("operation failed")
            return "success"
        
        # Apply both retry and circuit breaker
        @retry_mechanism
        @cb  
        def protected_operation():
            return flaky_operation()
        
        # First call: retry will attempt 3 times, all fail, circuit breaker sees 3 failures
        with pytest.raises(Exception):
            protected_operation()
        
        # Second call: should still execute (circuit not open yet)
        with pytest.raises(Exception):
            protected_operation()
        
        # Circuit should now be open, subsequent calls should fail fast
        assert cb.state == CircuitState.OPEN
    
    def test_timeout_with_cleanup(self):
        """Test timeout handling with resource cleanup."""
        config = TimeoutConfig(default_timeout=0.1)
        timeout_manager = TimeoutManager(config)
        
        cleanup_called = False
        
        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True
        
        with ResourceManager() as resource_manager:
            mock_resource = Mock()
            resource_manager.register_resource(mock_resource, "test", cleanup)
            
            try:
                with timeout_manager.timeout(0.05, "test_operation"):
                    time.sleep(0.2)  # This should timeout
            except ProfilingTimeoutError:
                pass
        
        # Cleanup should still be called even after timeout
        assert cleanup_called


if __name__ == "__main__":
    pytest.main([__file__])