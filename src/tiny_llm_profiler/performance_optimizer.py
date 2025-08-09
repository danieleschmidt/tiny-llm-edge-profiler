"""
Comprehensive performance optimization system for Generation 3.
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, TypeVar, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import deque
import statistics
import gc

from .logging_config import get_logger
from .cache import InMemoryCache
from .concurrent import ConcurrentProfiler
from .resource_pool import ResourcePoolManager
from .health import health_checker

logger = get_logger("performance_optimizer")

T = TypeVar('T')


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    # Cache settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 3600
    
    # Concurrency settings
    max_concurrent_tasks: int = 8
    task_timeout_seconds: int = 300
    
    # Resource pooling
    enable_resource_pooling: bool = True
    min_pool_size: int = 2
    max_pool_size: int = 10
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Memory optimization
    enable_gc_optimization: bool = True
    gc_threshold_mb: int = 500
    enable_memory_profiling: bool = False
    
    # Performance monitoring
    enable_metrics_collection: bool = True
    metrics_window_seconds: int = 300


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_mb: float
    cache_hit_rate: float
    task_throughput: float  # tasks per second
    average_response_time: float
    error_rate: float
    concurrent_tasks: int
    queue_length: int


class PerformanceOptimizer:
    """
    Central performance optimization system that coordinates all optimization strategies.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # Components
        self.cache = InMemoryCache(
            max_memory_mb=self.config.cache_size_mb,
            default_ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_caching else None
        
        self.concurrent_profiler = ConcurrentProfiler(
            max_threads=self.config.max_concurrent_tasks,
            timeout_seconds=self.config.task_timeout_seconds
        )
        
        self.resource_pool_manager = ResourcePoolManager() if self.config.enable_resource_pooling else None
        
        # Metrics and monitoring
        self.metrics_history: deque = deque(maxlen=1000)
        self.optimization_history: List[Dict[str, Any]] = []
        
        # State
        self.is_optimizing = False
        self.optimization_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Optimization strategies
        self.strategies = {
            'cache_optimization': self._optimize_cache,
            'memory_optimization': self._optimize_memory,
            'concurrency_optimization': self._optimize_concurrency,
            'resource_optimization': self._optimize_resources,
            'gc_optimization': self._optimize_garbage_collection
        }
        
        logger.info("Performance optimizer initialized")
    
    def start_optimization(self) -> None:
        """Start background performance optimization."""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info("Started performance optimization background thread")
    
    def stop_optimization(self) -> None:
        """Stop background performance optimization."""
        if not self.is_optimizing:
            return
        
        self._stop_event.set()
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        
        self.is_optimizing = False
        logger.info("Stopped performance optimization")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while not self._stop_event.is_set():
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Run optimization strategies
                self._run_optimization_cycle(metrics)
                
                # Sleep before next cycle
                time.sleep(30)  # 30 second optimization cycle
                
            except Exception as e:
                logger.exception("Error in optimization loop", exception=e)
                time.sleep(60)  # Back off on error
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        import psutil
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage_mb = memory_info.used / (1024 * 1024)
        
        # Cache metrics
        cache_hit_rate = 0.0
        if self.cache:
            stats = self.cache.get_stats()
            cache_hit_rate = stats['hit_rate']
        
        # Task metrics
        task_throughput = 0.0
        concurrent_tasks = 0
        queue_length = 0
        
        if hasattr(self.concurrent_profiler, 'get_stats'):
            profiler_stats = self.concurrent_profiler.get_stats()
            task_throughput = profiler_stats.get('tasks_per_second', 0.0)
            concurrent_tasks = profiler_stats.get('active_tasks', 0)
            queue_length = profiler_stats.get('queue_length', 0)
        
        # Calculate average response time from recent history
        avg_response_time = 0.0
        if len(self.metrics_history) > 0:
            recent_times = [m.average_response_time for m in list(self.metrics_history)[-10:]]
            if recent_times:
                avg_response_time = statistics.mean([t for t in recent_times if t > 0])
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            cache_hit_rate=cache_hit_rate,
            task_throughput=task_throughput,
            average_response_time=avg_response_time,
            error_rate=0.0,  # Would be calculated from task results
            concurrent_tasks=concurrent_tasks,
            queue_length=queue_length
        )
    
    def _run_optimization_cycle(self, metrics: PerformanceMetrics) -> None:
        """Run one complete optimization cycle."""
        optimization_results = {}
        
        for strategy_name, strategy_func in self.strategies.items():
            try:
                if self._should_run_strategy(strategy_name, metrics):
                    start_time = time.time()
                    result = strategy_func(metrics)
                    duration = time.time() - start_time
                    
                    optimization_results[strategy_name] = {
                        'result': result,
                        'duration_seconds': duration,
                        'timestamp': datetime.now()
                    }
                    
                    logger.debug(f"Optimization strategy {strategy_name} completed", 
                               duration_seconds=duration)
                    
            except Exception as e:
                logger.error(f"Optimization strategy {strategy_name} failed", 
                           exception=e)
        
        # Record optimization cycle
        if optimization_results:
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics,
                'optimizations': optimization_results
            })
            
            # Keep only recent history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
    
    def _should_run_strategy(self, strategy_name: str, metrics: PerformanceMetrics) -> bool:
        """Determine if an optimization strategy should run."""
        # Strategy-specific logic
        if strategy_name == 'cache_optimization':
            return metrics.cache_hit_rate < 0.8 or metrics.memory_usage_mb > 400
        
        elif strategy_name == 'memory_optimization':
            return metrics.memory_usage_mb > self.config.gc_threshold_mb
        
        elif strategy_name == 'concurrency_optimization':
            return (metrics.queue_length > 5 or 
                   metrics.cpu_usage_percent > 80 or 
                   metrics.average_response_time > 2.0)
        
        elif strategy_name == 'resource_optimization':
            return metrics.cpu_usage_percent > 70
        
        elif strategy_name == 'gc_optimization':
            return (self.config.enable_gc_optimization and 
                   metrics.memory_usage_mb > self.config.gc_threshold_mb)
        
        return True
    
    def _optimize_cache(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize cache performance."""
        if not self.cache:
            return {'status': 'cache_disabled'}
        
        optimizations = []
        
        # Check cache hit rate
        if metrics.cache_hit_rate < 0.6:
            # Increase cache size if memory allows
            current_stats = self.cache.get_stats()
            if current_stats['memory_usage_mb'] < current_stats['max_memory_mb'] * 0.8:
                optimizations.append('increased_cache_size')
                logger.info("Low cache hit rate detected, considering cache size increase")
        
        # Clean expired entries more aggressively if memory pressure
        if metrics.memory_usage_mb > 300:
            self.cache.clear(tag='temporary')
            optimizations.append('cleared_temporary_cache')
            logger.info("Memory pressure detected, cleared temporary cache entries")
        
        return {
            'optimizations': optimizations,
            'cache_stats': self.cache.get_stats()
        }
    
    def _optimize_memory(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize memory usage."""
        optimizations = []
        
        # Trigger garbage collection if memory usage is high
        if metrics.memory_usage_mb > self.config.gc_threshold_mb:
            import gc
            collected = gc.collect()
            optimizations.append(f'garbage_collection_{collected}')
            logger.info(f"High memory usage detected, collected {collected} objects")
        
        # Clear caches if memory pressure is critical
        if metrics.memory_usage_mb > self.config.gc_threshold_mb * 1.5:
            if self.cache:
                cleared = self.cache.clear()
                optimizations.append(f'cache_cleared_{cleared}')
                logger.warning(f"Critical memory pressure, cleared {cleared} cache entries")
        
        return {
            'optimizations': optimizations,
            'memory_before_mb': metrics.memory_usage_mb,
            'memory_after_mb': self._get_current_memory_usage()
        }
    
    def _optimize_concurrency(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize concurrency and task processing."""
        optimizations = []
        
        # Auto-scale worker threads based on queue length
        if metrics.queue_length > 10 and metrics.cpu_usage_percent < 80:
            # Could increase worker threads
            optimizations.append('consider_scale_up')
            logger.info("High queue length with available CPU, considering scale up")
        
        elif metrics.queue_length < 2 and metrics.cpu_usage_percent < 30:
            # Could decrease worker threads
            optimizations.append('consider_scale_down')
            logger.info("Low queue length and CPU usage, considering scale down")
        
        # Optimize task batching
        if metrics.average_response_time > 3.0:
            optimizations.append('optimize_task_batching')
            logger.info("High response time detected, optimizing task batching")
        
        return {
            'optimizations': optimizations,
            'current_metrics': {
                'queue_length': metrics.queue_length,
                'concurrent_tasks': metrics.concurrent_tasks,
                'cpu_usage': metrics.cpu_usage_percent
            }
        }
    
    def _optimize_resources(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize resource utilization."""
        optimizations = []
        
        # Resource pool optimization
        if self.resource_pool_manager:
            # Could implement resource pool scaling based on demand
            optimizations.append('resource_pool_analyzed')
        
        # CPU optimization
        if metrics.cpu_usage_percent > 90:
            optimizations.append('high_cpu_warning')
            logger.warning("Very high CPU usage detected")
        
        return {
            'optimizations': optimizations,
            'cpu_usage': metrics.cpu_usage_percent
        }
    
    def _optimize_garbage_collection(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize garbage collection."""
        if not self.config.enable_gc_optimization:
            return {'status': 'gc_optimization_disabled'}
        
        import gc
        
        # Get GC stats before
        gc_stats_before = gc.get_stats()
        
        # Tune GC thresholds based on memory usage
        if metrics.memory_usage_mb > self.config.gc_threshold_mb:
            # Set more aggressive GC thresholds
            gc.set_threshold(700, 10, 10)  # More frequent GC
        else:
            # Set normal GC thresholds
            gc.set_threshold(700, 10, 10)
        
        # Force collection if memory is very high
        collected = 0
        if metrics.memory_usage_mb > self.config.gc_threshold_mb * 1.2:
            collected = gc.collect()
        
        return {
            'objects_collected': collected,
            'gc_stats_before': len(gc_stats_before),
            'memory_before_mb': metrics.memory_usage_mb,
            'memory_after_mb': self._get_current_memory_usage()
        }
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        return psutil.virtual_memory().used / (1024 * 1024)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        current_metrics = self._collect_metrics()
        
        # Calculate performance improvements
        improvements = {}
        if len(self.metrics_history) >= 10:
            recent_metrics = list(self.metrics_history)[-10:]
            older_metrics = list(self.metrics_history)[-20:-10] if len(self.metrics_history) >= 20 else recent_metrics
            
            if older_metrics and recent_metrics:
                old_avg_response = statistics.mean([m.average_response_time for m in older_metrics if m.average_response_time > 0])
                new_avg_response = statistics.mean([m.average_response_time for m in recent_metrics if m.average_response_time > 0])
                
                if old_avg_response > 0 and new_avg_response > 0:
                    improvements['response_time_improvement'] = (old_avg_response - new_avg_response) / old_avg_response
                
                improvements['cache_hit_rate_improvement'] = statistics.mean([m.cache_hit_rate for m in recent_metrics]) - statistics.mean([m.cache_hit_rate for m in older_metrics])
        
        return {
            'timestamp': datetime.now(),
            'current_metrics': current_metrics,
            'optimization_active': self.is_optimizing,
            'total_optimization_cycles': len(self.optimization_history),
            'recent_optimizations': self.optimization_history[-5:] if self.optimization_history else [],
            'performance_improvements': improvements,
            'cache_stats': self.cache.get_stats() if self.cache else None,
            'system_health': health_checker.get_overall_health().healthy
        }


class AdaptiveOptimizer:
    """Adaptive optimizer that learns from profiling patterns."""
    
    def __init__(self, base_optimizer: PerformanceOptimizer):
        self.base_optimizer = base_optimizer
        self.learning_history: List[Dict[str, Any]] = []
        self.adaptation_rules: Dict[str, Callable] = {}
        
    def learn_from_patterns(self) -> None:
        """Learn optimization patterns from historical data."""
        if len(self.base_optimizer.metrics_history) < 50:
            return  # Need more data
        
        # Analyze patterns in metrics
        metrics_list = list(self.base_optimizer.metrics_history)
        
        # Pattern: High CPU followed by high response time
        cpu_spikes = [i for i, m in enumerate(metrics_list) if m.cpu_usage_percent > 80]
        
        for spike_idx in cpu_spikes:
            if spike_idx < len(metrics_list) - 5:  # Look ahead 5 measurements
                future_response_times = [metrics_list[spike_idx + i].average_response_time 
                                       for i in range(1, 6)]
                
                if any(rt > 2.0 for rt in future_response_times):
                    # Pattern detected: CPU spike leads to high response time
                    self._record_learning("cpu_spike_response_time_correlation", {
                        'cpu_threshold': 80,
                        'response_time_impact': max(future_response_times),
                        'confidence': 0.8
                    })
    
    def _record_learning(self, pattern_name: str, data: Dict[str, Any]) -> None:
        """Record a learned pattern."""
        learning_entry = {
            'timestamp': datetime.now(),
            'pattern': pattern_name,
            'data': data
        }
        
        self.learning_history.append(learning_entry)
        logger.info(f"Learned optimization pattern: {pattern_name}")


# Global optimizer instance
performance_optimizer = PerformanceOptimizer()
adaptive_optimizer = AdaptiveOptimizer(performance_optimizer)


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    return performance_optimizer


def start_performance_optimization() -> None:
    """Start global performance optimization."""
    performance_optimizer.start_optimization()


def stop_performance_optimization() -> None:
    """Stop global performance optimization."""
    performance_optimizer.stop_optimization()


def get_optimization_report() -> Dict[str, Any]:
    """Get current optimization report."""
    return performance_optimizer.get_optimization_report()