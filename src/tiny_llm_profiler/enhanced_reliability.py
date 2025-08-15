"""
Enhanced Reliability System for Self-Healing Pipelines
Advanced circuit breakers, bulkheads, and chaos engineering
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
import json
import threading
from functools import wraps
import inspect

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failure state, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class ChaosExperimentType(Enum):
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    fallback_function: Optional[Callable] = None
    half_open_max_calls: int = 3


@dataclass
class CircuitBreakerStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


@dataclass
class BulkheadConfig:
    max_concurrent_calls: int = 10
    queue_size: int = 50
    timeout_seconds: int = 30


@dataclass
class ChaosExperiment:
    experiment_id: str
    experiment_type: ChaosExperimentType
    target_service: str
    parameters: Dict[str, Any]
    enabled: bool = True
    probability: float = 0.1  # 10% chance
    duration_seconds: int = 60


class EnhancedCircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._call_with_circuit_breaker_async(func, *args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        with self._lock:
            self.stats.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.stats.state_changes += 1
                    logger.info(f"Circuit breaker {func.__name__} moved to HALF_OPEN state")
                else:
                    if self.config.fallback_function:
                        logger.warning(f"Circuit breaker {func.__name__} OPEN, using fallback")
                        return self.config.fallback_function(*args, **kwargs)
                    else:
                        raise Exception(f"Circuit breaker {func.__name__} is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
                
            except self.config.expected_exception as e:
                self._record_failure()
                raise
    
    async def _call_with_circuit_breaker_async(self, func: Callable, *args, **kwargs):
        async with asyncio.Lock():
            self.stats.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.stats.state_changes += 1
                    logger.info(f"Circuit breaker {func.__name__} moved to HALF_OPEN state")
                else:
                    if self.config.fallback_function:
                        logger.warning(f"Circuit breaker {func.__name__} OPEN, using fallback")
                        if inspect.iscoroutinefunction(self.config.fallback_function):
                            return await self.config.fallback_function(*args, **kwargs)
                        else:
                            return self.config.fallback_function(*args, **kwargs)
                    else:
                        raise Exception(f"Circuit breaker {func.__name__} is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result
                
            except self.config.expected_exception as e:
                self._record_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.config.recovery_timeout
    
    def _record_success(self) -> None:
        self.stats.successful_calls += 1
        self.stats.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.stats.state_changes += 1
                logger.info(f"Circuit breaker moved to CLOSED state after successful recovery")
    
    def _record_failure(self) -> None:
        self.stats.failed_calls += 1
        self.stats.last_failure_time = datetime.now()
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.stats.failed_calls >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.stats.state_changes += 1
                logger.warning(f"Circuit breaker moved to OPEN state after {self.stats.failed_calls} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.stats.state_changes += 1
            logger.warning(f"Circuit breaker moved back to OPEN state during recovery attempt")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "success_rate": self.stats.successful_calls / max(1, self.stats.total_calls),
            "state_changes": self.stats.state_changes,
            "last_failure": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
        }
    
    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.stats = CircuitBreakerStats()
            self.last_failure_time = None
            self.half_open_calls = 0
            logger.info("Circuit breaker manually reset")


class BulkheadPattern:
    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        self.active_calls = 0
        self.total_calls = 0
        self.rejected_calls = 0
        self.timeout_calls = 0
        
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self._execute_with_bulkhead(func, *args, **kwargs)
        return wrapper
    
    async def _execute_with_bulkhead(self, func: Callable, *args, **kwargs):
        self.total_calls += 1
        
        try:
            # Try to acquire semaphore with timeout
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.timeout_seconds
            )
            
            self.active_calls += 1
            
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            finally:
                self.active_calls -= 1
                self.semaphore.release()
                
        except asyncio.TimeoutError:
            self.timeout_calls += 1
            raise Exception(f"Bulkhead timeout: {func.__name__} took longer than {self.config.timeout_seconds}s")
        except Exception as e:
            self.rejected_calls += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_calls": self.active_calls,
            "total_calls": self.total_calls,
            "rejected_calls": self.rejected_calls,
            "timeout_calls": self.timeout_calls,
            "success_rate": (self.total_calls - self.rejected_calls - self.timeout_calls) / max(1, self.total_calls),
            "queue_size": self.queue.qsize(),
            "max_concurrent": self.config.max_concurrent_calls
        }


class ChaosEngineer:
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.active_experiments: Set[str] = set()
        self.experiment_history: List[Dict[str, Any]] = []
        
    def register_experiment(self, experiment: ChaosExperiment) -> None:
        self.experiments[experiment.experiment_id] = experiment
        logger.info(f"Registered chaos experiment: {experiment.experiment_id}")
    
    def should_inject_chaos(self, service_name: str, experiment_type: ChaosExperimentType) -> bool:
        for exp_id, experiment in self.experiments.items():
            if (experiment.enabled and 
                experiment.target_service == service_name and
                experiment.experiment_type == experiment_type and
                random.random() < experiment.probability):
                return True
        return False
    
    async def inject_latency(self, service_name: str, base_latency: float = 0.1) -> None:
        if self.should_inject_chaos(service_name, ChaosExperimentType.LATENCY_INJECTION):
            # Find matching experiment
            experiment = next(
                (exp for exp in self.experiments.values()
                 if exp.target_service == service_name and 
                 exp.experiment_type == ChaosExperimentType.LATENCY_INJECTION),
                None
            )
            
            if experiment:
                additional_delay = experiment.parameters.get("delay_ms", 1000) / 1000.0
                total_delay = base_latency + additional_delay
                
                logger.warning(f"Chaos Engineering: Injecting {additional_delay:.3f}s latency to {service_name}")
                await asyncio.sleep(total_delay)
                
                self._record_experiment_execution(experiment.experiment_id, {
                    "delay_injected": additional_delay,
                    "total_delay": total_delay
                })
    
    def inject_error(self, service_name: str, error_message: str = "Chaos-induced error") -> None:
        if self.should_inject_chaos(service_name, ChaosExperimentType.ERROR_INJECTION):
            experiment = next(
                (exp for exp in self.experiments.values()
                 if exp.target_service == service_name and 
                 exp.experiment_type == ChaosExperimentType.ERROR_INJECTION),
                None
            )
            
            if experiment:
                error_type = experiment.parameters.get("error_type", "generic")
                logger.warning(f"Chaos Engineering: Injecting {error_type} error to {service_name}")
                
                self._record_experiment_execution(experiment.experiment_id, {
                    "error_type": error_type,
                    "error_message": error_message
                })
                
                if error_type == "timeout":
                    raise TimeoutError(f"Chaos-induced timeout in {service_name}")
                elif error_type == "connection":
                    raise ConnectionError(f"Chaos-induced connection error in {service_name}")
                else:
                    raise Exception(f"Chaos-induced error in {service_name}: {error_message}")
    
    def _record_experiment_execution(self, experiment_id: str, details: Dict[str, Any]) -> None:
        execution_record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.experiment_history.append(execution_record)
        
        # Keep only recent history
        if len(self.experiment_history) > 1000:
            self.experiment_history = self.experiment_history[-500:]
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        return {
            "total_experiments": len(self.experiments),
            "active_experiments": len(self.active_experiments),
            "executions_count": len(self.experiment_history),
            "experiments_by_type": {
                exp_type.value: len([
                    exp for exp in self.experiments.values()
                    if exp.experiment_type == exp_type
                ])
                for exp_type in ChaosExperimentType
            }
        }


class RetryWithBackoff:
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._retry_with_backoff(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._retry_with_backoff_async(func, *args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    def _retry_with_backoff(self, func: Callable, *args, **kwargs):
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    logger.error(f"Final retry attempt failed for {func.__name__}: {str(e)}")
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Retry attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s: {str(e)}")
                time.sleep(delay)
        
        raise last_exception
    
    async def _retry_with_backoff_async(self, func: Callable, *args, **kwargs):
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    logger.error(f"Final retry attempt failed for {func.__name__}: {str(e)}")
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Retry attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s: {str(e)}")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add ±25% jitter
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


class ReliabilityManager:
    def __init__(self):
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadPattern] = {}
        self.chaos_engineer = ChaosEngineer()
        self.retry_policies: Dict[str, RetryWithBackoff] = {}
        
        # Statistics
        self.total_protected_calls = 0
        self.total_circuit_breaker_trips = 0
        self.total_bulkhead_rejections = 0
        self.total_retry_attempts = 0
        
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> EnhancedCircuitBreaker:
        circuit_breaker = EnhancedCircuitBreaker(config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def create_bulkhead(self, name: str, config: BulkheadConfig) -> BulkheadPattern:
        bulkhead = BulkheadPattern(config)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def create_retry_policy(self, name: str, 
                          max_attempts: int = 3,
                          base_delay: float = 1.0) -> RetryWithBackoff:
        retry_policy = RetryWithBackoff(max_attempts=max_attempts, base_delay=base_delay)
        self.retry_policies[name] = retry_policy
        return retry_policy
    
    def protect_function(self, 
                        func: Callable,
                        circuit_breaker_name: Optional[str] = None,
                        bulkhead_name: Optional[str] = None,
                        retry_policy_name: Optional[str] = None,
                        chaos_service_name: Optional[str] = None) -> Callable:
        
        protected_func = func
        
        # Apply retry policy
        if retry_policy_name and retry_policy_name in self.retry_policies:
            protected_func = self.retry_policies[retry_policy_name](protected_func)
        
        # Apply circuit breaker
        if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
            protected_func = self.circuit_breakers[circuit_breaker_name](protected_func)
        
        # Apply bulkhead
        if bulkhead_name and bulkhead_name in self.bulkheads:
            protected_func = self.bulkheads[bulkhead_name](protected_func)
        
        # Apply chaos engineering
        if chaos_service_name:
            @wraps(protected_func)
            async def chaos_wrapper(*args, **kwargs):
                # Inject latency before execution
                await self.chaos_engineer.inject_latency(chaos_service_name)
                
                # Potentially inject errors
                self.chaos_engineer.inject_error(chaos_service_name)
                
                # Execute the protected function
                if inspect.iscoroutinefunction(protected_func):
                    return await protected_func(*args, **kwargs)
                else:
                    return protected_func(*args, **kwargs)
            
            protected_func = chaos_wrapper
        
        return protected_func
    
    def get_reliability_stats(self) -> Dict[str, Any]:
        circuit_breaker_stats = {
            name: cb.get_stats() 
            for name, cb in self.circuit_breakers.items()
        }
        
        bulkhead_stats = {
            name: bh.get_stats() 
            for name, bh in self.bulkheads.items()
        }
        
        return {
            "circuit_breakers": circuit_breaker_stats,
            "bulkheads": bulkhead_stats,
            "chaos_experiments": self.chaos_engineer.get_experiment_stats(),
            "total_protected_calls": self.total_protected_calls,
            "reliability_patterns_count": {
                "circuit_breakers": len(self.circuit_breakers),
                "bulkheads": len(self.bulkheads),
                "retry_policies": len(self.retry_policies)
            }
        }
    
    def setup_chaos_experiments(self) -> None:
        # Setup default chaos experiments
        default_experiments = [
            ChaosExperiment(
                experiment_id="latency_injection_default",
                experiment_type=ChaosExperimentType.LATENCY_INJECTION,
                target_service="pipeline_guard",
                parameters={"delay_ms": 500},
                probability=0.05  # 5% chance
            ),
            ChaosExperiment(
                experiment_id="error_injection_default",
                experiment_type=ChaosExperimentType.ERROR_INJECTION,
                target_service="model_profiler",
                parameters={"error_type": "timeout"},
                probability=0.02  # 2% chance
            )
        ]
        
        for experiment in default_experiments:
            self.chaos_engineer.register_experiment(experiment)


# Global reliability manager
_global_reliability_manager: Optional[ReliabilityManager] = None


def get_reliability_manager() -> ReliabilityManager:
    global _global_reliability_manager
    if _global_reliability_manager is None:
        _global_reliability_manager = ReliabilityManager()
        _global_reliability_manager.setup_chaos_experiments()
    return _global_reliability_manager


def circuit_breaker(name: str, 
                   failure_threshold: int = 5,
                   recovery_timeout: int = 60) -> Callable:
    manager = get_reliability_manager()
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    return manager.create_circuit_breaker(name, config)


def bulkhead(name: str, max_concurrent: int = 10) -> Callable:
    manager = get_reliability_manager()
    config = BulkheadConfig(max_concurrent_calls=max_concurrent)
    return manager.create_bulkhead(name, config)


def retry(max_attempts: int = 3, base_delay: float = 1.0) -> Callable:
    return RetryWithBackoff(max_attempts=max_attempts, base_delay=base_delay)


def get_reliability_status() -> Dict[str, Any]:
    manager = get_reliability_manager()
    return manager.get_reliability_stats()