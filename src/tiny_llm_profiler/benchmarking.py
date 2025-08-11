"""
Comprehensive Benchmarking and Performance Comparison Tools - Generation 3
Provides comprehensive benchmarking capabilities including:
- Systematic profiler performance benchmarking
- Cross-configuration performance comparisons
- Benchmark result analysis and visualization
- Performance regression testing in benchmarks
- Standardized benchmark suites for different scenarios
- Automated benchmark execution and reporting
- Historical performance tracking and trending
- Competitive analysis and baseline comparisons
"""

import time
import threading
import asyncio
import statistics
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod
import json
import hashlib
import pickle
import gzip
import psutil
import resource
import gc
from pathlib import Path
import tracemalloc
import cProfile
import pstats
import io

from .exceptions import TinyLLMProfilerError, BenchmarkError, ValidationError
from .logging_config import get_logger
from .profiler import EdgeProfiler
from .results import ProfileResults
from .performance_analytics import PerformanceMetric, record_performance_metric

logger = get_logger("benchmarking")


class BenchmarkType(str, Enum):
    """Types of benchmarks."""
    LATENCY = "latency"                    # Response time benchmarks
    THROUGHPUT = "throughput"              # Requests per second benchmarks
    MEMORY_EFFICIENCY = "memory_efficiency" # Memory usage benchmarks
    CPU_EFFICIENCY = "cpu_efficiency"      # CPU utilization benchmarks
    SCALABILITY = "scalability"            # Load scaling benchmarks
    ACCURACY = "accuracy"                  # Result accuracy benchmarks
    STABILITY = "stability"                # Long-term stability benchmarks
    STRESS = "stress"                      # High-load stress tests


class BenchmarkStatus(str, Enum):
    """Benchmark execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComparisonMethod(str, Enum):
    """Methods for comparing benchmark results."""
    STATISTICAL = "statistical"           # Statistical significance testing
    PERCENTAGE = "percentage"             # Percentage difference
    ABSOLUTE = "absolute"                 # Absolute difference
    REGRESSION = "regression"             # Regression analysis
    DISTRIBUTION = "distribution"         # Distribution comparison


@dataclass
class BenchmarkConfiguration:
    """Configuration for a benchmark test."""
    benchmark_id: str
    benchmark_type: BenchmarkType
    name: str
    description: str
    
    # Test parameters
    duration_seconds: float = 60.0
    warmup_seconds: float = 10.0
    iterations: int = 100
    concurrency: int = 1
    
    # Profiler configuration
    profiler_config: Dict[str, Any] = field(default_factory=dict)
    
    # Test data configuration
    test_data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Resource limits
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    
    # Validation criteria
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # Tags and metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics."""
    # Timing metrics
    total_duration_seconds: float = 0.0
    warmup_duration_seconds: float = 0.0
    execution_duration_seconds: float = 0.0
    
    # Performance metrics
    operations_completed: int = 0
    operations_per_second: float = 0.0
    average_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Resource usage metrics
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    average_cpu_percent: float = 0.0
    
    # Throughput metrics
    bytes_processed: int = 0
    bytes_per_second: float = 0.0
    
    # Error metrics
    total_errors: int = 0
    error_rate_percent: float = 0.0
    timeout_count: int = 0
    
    # Statistical metrics
    latency_std_dev: float = 0.0
    latency_variance: float = 0.0
    coefficient_of_variation: float = 0.0
    
    def calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (0-100)."""
        # Combine throughput, resource efficiency, and reliability
        throughput_score = min(100.0, self.operations_per_second)
        
        # Memory efficiency (lower usage = higher score)
        memory_efficiency = max(0, 100 - self.average_memory_mb / 10)  # -10 per 100MB
        
        # CPU efficiency (lower usage = higher score) 
        cpu_efficiency = max(0, 100 - self.average_cpu_percent)
        
        # Reliability (lower error rate = higher score)
        reliability_score = max(0, 100 - self.error_rate_percent * 10)
        
        # Consistency (lower variance = higher score)
        consistency_score = max(0, 100 - self.coefficient_of_variation * 10)
        
        return (throughput_score + memory_efficiency + cpu_efficiency + reliability_score + consistency_score) / 5


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    benchmark_id: str
    configuration: BenchmarkConfiguration
    status: BenchmarkStatus
    
    # Execution info
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_host: str = field(default_factory=lambda: psutil.os.uname().node)
    
    # Metrics
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    
    # Raw measurements
    latency_measurements: List[float] = field(default_factory=list)
    memory_measurements: List[Tuple[datetime, float]] = field(default_factory=list)
    cpu_measurements: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Error information
    errors: List[Dict[str, Any]] = field(default_factory=list)
    failure_reason: Optional[str] = None
    
    # Validation results
    validation_passed: bool = True
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for reporting."""
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_type": self.configuration.benchmark_type.value,
            "name": self.configuration.name,
            "status": self.status.value,
            "duration": self.metrics.execution_duration_seconds,
            "operations_per_second": self.metrics.operations_per_second,
            "average_latency_ms": self.metrics.average_latency_ms,
            "p95_latency_ms": self.metrics.p95_latency_ms,
            "peak_memory_mb": self.metrics.peak_memory_mb,
            "average_cpu_percent": self.metrics.average_cpu_percent,
            "error_rate_percent": self.metrics.error_rate_percent,
            "efficiency_score": self.metrics.calculate_efficiency_score(),
            "start_time": self.start_time.isoformat(),
            "validation_passed": self.validation_passed
        }


@dataclass
class BenchmarkComparison:
    """Comparison between two benchmark results."""
    baseline_result: BenchmarkResult
    comparison_result: BenchmarkResult
    comparison_method: ComparisonMethod
    
    # Comparison metrics
    ops_per_second_change_percent: float = 0.0
    latency_change_percent: float = 0.0
    memory_change_percent: float = 0.0
    cpu_change_percent: float = 0.0
    
    # Statistical significance
    statistical_significance: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Overall assessment
    performance_improvement: bool = False
    regression_detected: bool = False
    significance_level: str = "none"  # none, low, medium, high
    
    def generate_summary(self) -> str:
        """Generate human-readable comparison summary."""
        direction = "improvement" if self.performance_improvement else "regression"
        
        summary_parts = [
            f"Performance {direction} detected:",
            f"  • Throughput: {self.ops_per_second_change_percent:+.1f}%",
            f"  • Latency: {self.latency_change_percent:+.1f}%",
            f"  • Memory: {self.memory_change_percent:+.1f}%",
            f"  • CPU: {self.cpu_change_percent:+.1f}%",
            f"  • Significance: {self.significance_level}"
        ]
        
        return "\n".join(summary_parts)


class BenchmarkWorkload(ABC):
    """Abstract base class for benchmark workloads."""
    
    @abstractmethod
    def setup(self, config: Dict[str, Any]) -> None:
        """Setup the workload with given configuration."""
        pass
    
    @abstractmethod
    async def execute_operation(self, profiler: EdgeProfiler) -> Tuple[float, Any]:
        """Execute a single operation and return (duration_ms, result)."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources after benchmark."""
        pass
    
    @abstractmethod
    def validate_result(self, result: Any) -> bool:
        """Validate that a result is correct."""
        pass


class StandardLLMWorkload(BenchmarkWorkload):
    """Standard LLM profiling workload."""
    
    def __init__(self):
        self.test_model_sizes = [100, 500, 1000, 5000]  # MB
        self.test_sequences = []
    
    def setup(self, config: Dict[str, Any]) -> None:
        """Setup test data and configurations."""
        model_count = config.get("model_count", 10)
        sequence_length = config.get("sequence_length", 512)
        
        # Generate test sequences
        self.test_sequences = []
        for i in range(model_count):
            # Simulate different model characteristics
            model_size = np.random.choice(self.test_model_sizes)
            sequence = {
                "model_id": f"test_model_{i}",
                "model_size_mb": model_size,
                "sequence_length": sequence_length,
                "parameters": {
                    "temperature": np.random.uniform(0.1, 1.0),
                    "top_p": np.random.uniform(0.8, 1.0)
                }
            }
            self.test_sequences.append(sequence)
    
    async def execute_operation(self, profiler: EdgeProfiler) -> Tuple[float, Any]:
        """Execute a profiling operation."""
        # Select random test sequence
        sequence = np.random.choice(self.test_sequences)
        
        start_time = time.time()
        
        # Simulate model profiling
        try:
            # Create mock profiling data
            profile_data = {
                "model_id": sequence["model_id"],
                "input_tokens": sequence["sequence_length"],
                "inference_time_ms": np.random.normal(100, 20),
                "memory_usage_mb": sequence["model_size_mb"] * np.random.uniform(0.8, 1.2),
                "compute_utilization": np.random.uniform(0.6, 0.9)
            }
            
            # Simulate processing delay
            processing_time = max(0.001, np.random.gamma(2, 0.01))  # Gamma distribution
            await asyncio.sleep(processing_time)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return duration_ms, profile_data
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            raise BenchmarkError(f"Operation failed: {e}")
    
    def cleanup(self) -> None:
        """Clean up test data."""
        self.test_sequences.clear()
    
    def validate_result(self, result: Any) -> bool:
        """Validate profiling result."""
        if not isinstance(result, dict):
            return False
        
        required_fields = ["model_id", "input_tokens", "inference_time_ms", "memory_usage_mb"]
        return all(field in result for field in required_fields)


class MemoryStressWorkload(BenchmarkWorkload):
    """Memory stress testing workload."""
    
    def __init__(self):
        self.memory_blocks = []
        self.block_size_mb = 10
    
    def setup(self, config: Dict[str, Any]) -> None:
        """Setup memory stress configuration."""
        self.block_size_mb = config.get("block_size_mb", 10)
        self.max_blocks = config.get("max_memory_blocks", 100)
    
    async def execute_operation(self, profiler: EdgeProfiler) -> Tuple[float, Any]:
        """Execute memory-intensive operation."""
        start_time = time.time()
        
        try:
            # Allocate memory block
            block_size = self.block_size_mb * 1024 * 1024
            memory_block = bytearray(block_size)
            
            # Fill with random data
            for i in range(0, len(memory_block), 1024):
                memory_block[i:i+4] = (i // 1024).to_bytes(4, 'little')
            
            self.memory_blocks.append(memory_block)
            
            # Limit total memory usage
            if len(self.memory_blocks) > self.max_blocks:
                self.memory_blocks.pop(0)
            
            # Simulate profiling work
            result = {
                "blocks_allocated": len(self.memory_blocks),
                "total_memory_mb": len(self.memory_blocks) * self.block_size_mb,
                "allocation_time_ms": (time.time() - start_time) * 1000
            }
            
            duration_ms = (time.time() - start_time) * 1000
            return duration_ms, result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            raise BenchmarkError(f"Memory operation failed: {e}")
    
    def cleanup(self) -> None:
        """Clean up allocated memory."""
        self.memory_blocks.clear()
        gc.collect()
    
    def validate_result(self, result: Any) -> bool:
        """Validate memory operation result."""
        return isinstance(result, dict) and "blocks_allocated" in result


class ConcurrencyWorkload(BenchmarkWorkload):
    """Concurrency stress testing workload."""
    
    def __init__(self):
        self.shared_counter = 0
        self.lock = threading.Lock()
    
    def setup(self, config: Dict[str, Any]) -> None:
        """Setup concurrency test."""
        self.shared_counter = 0
        self.operations_per_task = config.get("operations_per_task", 10)
    
    async def execute_operation(self, profiler: EdgeProfiler) -> Tuple[float, Any]:
        """Execute concurrent operation."""
        start_time = time.time()
        
        try:
            # Simulate concurrent work with shared resource
            local_operations = 0
            
            for _ in range(self.operations_per_task):
                # Simulate some computation
                computation_result = sum(range(100))
                
                # Update shared counter
                with self.lock:
                    self.shared_counter += 1
                    local_operations += 1
                
                # Small delay to simulate I/O
                await asyncio.sleep(0.001)
            
            result = {
                "local_operations": local_operations,
                "shared_counter_value": self.shared_counter,
                "computation_result": computation_result
            }
            
            duration_ms = (time.time() - start_time) * 1000
            return duration_ms, result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            raise BenchmarkError(f"Concurrency operation failed: {e}")
    
    def cleanup(self) -> None:
        """Reset shared state."""
        with self.lock:
            self.shared_counter = 0
    
    def validate_result(self, result: Any) -> bool:
        """Validate concurrency result."""
        return (isinstance(result, dict) and 
                "local_operations" in result and 
                result["local_operations"] > 0)


class ResourceMonitor:
    """Monitors system resources during benchmark execution."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        
        # Measurements
        self.memory_measurements: List[Tuple[datetime, float]] = []
        self.cpu_measurements: List[Tuple[datetime, float]] = []
        
        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.memory_measurements.clear()
        self.cpu_measurements.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics."""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # Calculate statistics
        if self.memory_measurements:
            memory_values = [mem for _, mem in self.memory_measurements]
            peak_memory = max(memory_values)
            avg_memory = statistics.mean(memory_values)
        else:
            peak_memory = avg_memory = 0.0
        
        if self.cpu_measurements:
            cpu_values = [cpu for _, cpu in self.cpu_measurements]
            peak_cpu = max(cpu_values)
            avg_cpu = statistics.mean(cpu_values)
        else:
            peak_cpu = avg_cpu = 0.0
        
        return {
            "peak_memory_mb": peak_memory,
            "average_memory_mb": avg_memory,
            "peak_cpu_percent": peak_cpu,
            "average_cpu_percent": avg_cpu,
            "memory_measurements": self.memory_measurements.copy(),
            "cpu_measurements": self.cpu_measurements.copy()
        }
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                timestamp = datetime.now()
                
                # Memory usage in MB
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_measurements.append((timestamp, memory_mb))
                
                # CPU usage percentage
                cpu_percent = process.cpu_percent()
                self.cpu_measurements.append((timestamp, cpu_percent))
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                break


class BenchmarkRunner:
    """Executes benchmark tests with comprehensive monitoring."""
    
    def __init__(self):
        self.workloads: Dict[str, BenchmarkWorkload] = {
            "standard_llm": StandardLLMWorkload(),
            "memory_stress": MemoryStressWorkload(),
            "concurrency": ConcurrencyWorkload()
        }
        
        self.resource_monitor = ResourceMonitor()
    
    async def run_benchmark(self, config: BenchmarkConfiguration) -> BenchmarkResult:
        """Run a single benchmark test."""
        logger.info(f"Starting benchmark: {config.name}")
        
        # Create result object
        result = BenchmarkResult(
            benchmark_id=config.benchmark_id,
            configuration=config,
            status=BenchmarkStatus.PENDING,
            start_time=datetime.now()
        )
        
        try:
            # Get workload
            workload_type = config.test_data_config.get("workload_type", "standard_llm")
            if workload_type not in self.workloads:
                raise BenchmarkError(f"Unknown workload type: {workload_type}")
            
            workload = self.workloads[workload_type]
            
            # Setup workload
            workload.setup(config.test_data_config)
            
            # Create profiler instance
            profiler = EdgeProfiler(**config.profiler_config)
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            result.status = BenchmarkStatus.RUNNING
            
            # Warmup phase
            if config.warmup_seconds > 0:
                await self._run_warmup(workload, profiler, config.warmup_seconds)
            
            # Main benchmark execution
            latency_measurements = await self._run_benchmark_phase(
                workload, profiler, config
            )
            
            # Stop monitoring and get resource stats
            resource_stats = self.resource_monitor.stop_monitoring()
            
            # Calculate metrics
            result.metrics = self._calculate_metrics(
                latency_measurements, resource_stats, config
            )
            result.latency_measurements = latency_measurements
            result.memory_measurements = resource_stats.get("memory_measurements", [])
            result.cpu_measurements = resource_stats.get("cpu_measurements", [])
            
            # Validate results
            result.validation_passed, result.validation_details = self._validate_benchmark_result(
                result, config
            )
            
            # Cleanup
            workload.cleanup()
            
            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.now()
            
            logger.info(f"Benchmark completed: {config.name} - {result.metrics.operations_per_second:.2f} ops/sec")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {config.name} - {e}")
            result.status = BenchmarkStatus.FAILED
            result.failure_reason = str(e)
            result.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            result.end_time = datetime.now()
        
        return result
    
    async def _run_warmup(self, workload: BenchmarkWorkload, profiler: EdgeProfiler, duration: float):
        """Run warmup phase."""
        logger.debug(f"Running warmup for {duration} seconds")
        
        start_time = time.time()
        warmup_operations = 0
        
        while (time.time() - start_time) < duration:
            try:
                await workload.execute_operation(profiler)
                warmup_operations += 1
            except Exception as e:
                logger.warning(f"Warmup operation failed: {e}")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.001)
        
        logger.debug(f"Warmup completed: {warmup_operations} operations")
    
    async def _run_benchmark_phase(
        self,
        workload: BenchmarkWorkload,
        profiler: EdgeProfiler,
        config: BenchmarkConfiguration
    ) -> List[float]:
        """Run main benchmark phase."""
        logger.debug(f"Running benchmark phase: {config.iterations} iterations, {config.concurrency} concurrency")
        
        latency_measurements = []
        completed_operations = 0
        errors = []
        
        # Determine execution mode
        if config.duration_seconds > 0:
            # Duration-based execution
            latency_measurements = await self._run_duration_based(
                workload, profiler, config.duration_seconds, config.concurrency
            )
        else:
            # Iteration-based execution
            latency_measurements = await self._run_iteration_based(
                workload, profiler, config.iterations, config.concurrency
            )
        
        return latency_measurements
    
    async def _run_duration_based(
        self,
        workload: BenchmarkWorkload,
        profiler: EdgeProfiler,
        duration: float,
        concurrency: int
    ) -> List[float]:
        """Run benchmark for specified duration."""
        latency_measurements = []
        start_time = time.time()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        async def execute_with_semaphore():
            async with semaphore:
                try:
                    latency, result = await workload.execute_operation(profiler)
                    return latency
                except Exception as e:
                    logger.debug(f"Operation error: {e}")
                    return None
        
        # Launch concurrent tasks
        tasks = []
        while (time.time() - start_time) < duration:
            task = asyncio.create_task(execute_with_semaphore())
            tasks.append(task)
            
            # Prevent task queue from growing too large
            if len(tasks) > concurrency * 10:
                # Wait for some tasks to complete
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                # Collect results
                for task in done:
                    result = await task
                    if result is not None:
                        latency_measurements.append(result)
                
                tasks = list(pending)
        
        # Wait for remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, (int, float)) and result is not None:
                    latency_measurements.append(result)
        
        return latency_measurements
    
    async def _run_iteration_based(
        self,
        workload: BenchmarkWorkload,
        profiler: EdgeProfiler,
        iterations: int,
        concurrency: int
    ) -> List[float]:
        """Run benchmark for specified number of iterations."""
        latency_measurements = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        async def execute_with_semaphore():
            async with semaphore:
                try:
                    latency, result = await workload.execute_operation(profiler)
                    return latency
                except Exception as e:
                    logger.debug(f"Operation error: {e}")
                    return None
        
        # Create all tasks
        tasks = [asyncio.create_task(execute_with_semaphore()) for _ in range(iterations)]
        
        # Execute tasks and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, (int, float)) and result is not None:
                latency_measurements.append(result)
        
        return latency_measurements
    
    def _calculate_metrics(
        self,
        latency_measurements: List[float],
        resource_stats: Dict[str, Any],
        config: BenchmarkConfiguration
    ) -> BenchmarkMetrics:
        """Calculate comprehensive benchmark metrics."""
        metrics = BenchmarkMetrics()
        
        if latency_measurements:
            # Timing metrics
            metrics.operations_completed = len(latency_measurements)
            metrics.execution_duration_seconds = config.duration_seconds if config.duration_seconds > 0 else sum(latency_measurements) / 1000
            metrics.operations_per_second = metrics.operations_completed / max(metrics.execution_duration_seconds, 0.001)
            
            # Latency metrics
            metrics.average_latency_ms = statistics.mean(latency_measurements)
            metrics.median_latency_ms = statistics.median(latency_measurements)
            metrics.min_latency_ms = min(latency_measurements)
            metrics.max_latency_ms = max(latency_measurements)
            metrics.latency_std_dev = statistics.stdev(latency_measurements) if len(latency_measurements) > 1 else 0.0
            metrics.latency_variance = statistics.variance(latency_measurements) if len(latency_measurements) > 1 else 0.0
            
            # Percentiles
            sorted_latencies = sorted(latency_measurements)
            metrics.p95_latency_ms = np.percentile(sorted_latencies, 95)
            metrics.p99_latency_ms = np.percentile(sorted_latencies, 99)
            
            # Coefficient of variation
            if metrics.average_latency_ms > 0:
                metrics.coefficient_of_variation = metrics.latency_std_dev / metrics.average_latency_ms
        
        # Resource metrics
        metrics.peak_memory_mb = resource_stats.get("peak_memory_mb", 0.0)
        metrics.average_memory_mb = resource_stats.get("average_memory_mb", 0.0)
        metrics.peak_cpu_percent = resource_stats.get("peak_cpu_percent", 0.0)
        metrics.average_cpu_percent = resource_stats.get("average_cpu_percent", 0.0)
        
        # Throughput estimation (approximate)
        if metrics.operations_per_second > 0:
            # Estimate bytes processed (rough approximation)
            avg_data_size_bytes = 1024  # Approximate data size per operation
            metrics.bytes_processed = int(metrics.operations_completed * avg_data_size_bytes)
            metrics.bytes_per_second = metrics.bytes_processed / max(metrics.execution_duration_seconds, 0.001)
        
        return metrics
    
    def _validate_benchmark_result(
        self,
        result: BenchmarkResult,
        config: BenchmarkConfiguration
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate benchmark result against success criteria."""
        validation_details = {}
        passed = True
        
        success_criteria = config.success_criteria
        metrics = result.metrics
        
        # Check minimum operations
        min_operations = success_criteria.get("min_operations", 1)
        if metrics.operations_completed < min_operations:
            validation_details["min_operations"] = {
                "expected": min_operations,
                "actual": metrics.operations_completed,
                "passed": False
            }
            passed = False
        
        # Check maximum error rate
        max_error_rate = success_criteria.get("max_error_rate_percent", 100.0)
        if metrics.error_rate_percent > max_error_rate:
            validation_details["max_error_rate"] = {
                "expected": max_error_rate,
                "actual": metrics.error_rate_percent,
                "passed": False
            }
            passed = False
        
        # Check minimum throughput
        min_ops_per_second = success_criteria.get("min_ops_per_second", 0.0)
        if metrics.operations_per_second < min_ops_per_second:
            validation_details["min_ops_per_second"] = {
                "expected": min_ops_per_second,
                "actual": metrics.operations_per_second,
                "passed": False
            }
            passed = False
        
        # Check maximum latency
        max_latency_ms = success_criteria.get("max_average_latency_ms", float('inf'))
        if metrics.average_latency_ms > max_latency_ms:
            validation_details["max_average_latency"] = {
                "expected": max_latency_ms,
                "actual": metrics.average_latency_ms,
                "passed": False
            }
            passed = False
        
        # Check maximum memory usage
        max_memory_mb = success_criteria.get("max_memory_mb", float('inf'))
        if metrics.peak_memory_mb > max_memory_mb:
            validation_details["max_memory"] = {
                "expected": max_memory_mb,
                "actual": metrics.peak_memory_mb,
                "passed": False
            }
            passed = False
        
        return passed, validation_details


class BenchmarkComparator:
    """Compares benchmark results and detects performance changes."""
    
    def __init__(self):
        self.comparison_methods = {
            ComparisonMethod.STATISTICAL: self._statistical_comparison,
            ComparisonMethod.PERCENTAGE: self._percentage_comparison,
            ComparisonMethod.ABSOLUTE: self._absolute_comparison
        }
    
    def compare_results(
        self,
        baseline: BenchmarkResult,
        comparison: BenchmarkResult,
        method: ComparisonMethod = ComparisonMethod.STATISTICAL
    ) -> BenchmarkComparison:
        """Compare two benchmark results."""
        
        if baseline.benchmark_id != comparison.benchmark_id:
            logger.warning("Comparing results from different benchmark types")
        
        # Perform comparison using specified method
        comparison_func = self.comparison_methods.get(method, self._statistical_comparison)
        comparison_result = comparison_func(baseline, comparison)
        
        return comparison_result
    
    def _statistical_comparison(self, baseline: BenchmarkResult, comparison: BenchmarkResult) -> BenchmarkComparison:
        """Statistical significance comparison."""
        result = BenchmarkComparison(
            baseline_result=baseline,
            comparison_result=comparison,
            comparison_method=ComparisonMethod.STATISTICAL
        )
        
        # Calculate percentage changes
        result.ops_per_second_change_percent = self._calculate_percentage_change(
            baseline.metrics.operations_per_second, comparison.metrics.operations_per_second
        )
        result.latency_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_latency_ms, comparison.metrics.average_latency_ms
        )
        result.memory_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_memory_mb, comparison.metrics.average_memory_mb
        )
        result.cpu_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_cpu_percent, comparison.metrics.average_cpu_percent
        )
        
        # Statistical significance testing (simplified t-test approximation)
        if baseline.latency_measurements and comparison.latency_measurements:
            baseline_latencies = baseline.latency_measurements
            comparison_latencies = comparison.latency_measurements
            
            # Perform Welch's t-test approximation
            significance, confidence_interval = self._welch_t_test(
                baseline_latencies, comparison_latencies
            )
            
            result.statistical_significance = significance
            result.confidence_interval = confidence_interval
        
        # Determine overall performance change
        # Weight different metrics (throughput most important for performance)
        throughput_weight = 0.4
        latency_weight = 0.3
        memory_weight = 0.2
        cpu_weight = 0.1
        
        performance_score = (
            result.ops_per_second_change_percent * throughput_weight +
            (-result.latency_change_percent) * latency_weight +  # Lower latency is better
            (-result.memory_change_percent) * memory_weight +    # Lower memory is better  
            (-result.cpu_change_percent) * cpu_weight            # Lower CPU is better
        )
        
        result.performance_improvement = performance_score > 0
        result.regression_detected = performance_score < -5  # >5% degradation
        
        # Determine significance level
        if result.statistical_significance > 0.95:
            result.significance_level = "high"
        elif result.statistical_significance > 0.8:
            result.significance_level = "medium"
        elif result.statistical_significance > 0.6:
            result.significance_level = "low"
        else:
            result.significance_level = "none"
        
        return result
    
    def _percentage_comparison(self, baseline: BenchmarkResult, comparison: BenchmarkResult) -> BenchmarkComparison:
        """Simple percentage-based comparison."""
        result = BenchmarkComparison(
            baseline_result=baseline,
            comparison_result=comparison,
            comparison_method=ComparisonMethod.PERCENTAGE
        )
        
        # Calculate percentage changes
        result.ops_per_second_change_percent = self._calculate_percentage_change(
            baseline.metrics.operations_per_second, comparison.metrics.operations_per_second
        )
        result.latency_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_latency_ms, comparison.metrics.average_latency_ms
        )
        result.memory_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_memory_mb, comparison.metrics.average_memory_mb
        )
        result.cpu_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_cpu_percent, comparison.metrics.average_cpu_percent
        )
        
        # Simple threshold-based assessment
        improvement_threshold = 5.0  # 5% improvement
        regression_threshold = -5.0  # 5% regression
        
        avg_change = (result.ops_per_second_change_percent + 
                     (-result.latency_change_percent) + 
                     (-result.memory_change_percent) + 
                     (-result.cpu_change_percent)) / 4
        
        result.performance_improvement = avg_change > improvement_threshold
        result.regression_detected = avg_change < regression_threshold
        result.significance_level = "medium" if abs(avg_change) > 10 else "low"
        
        return result
    
    def _absolute_comparison(self, baseline: BenchmarkResult, comparison: BenchmarkResult) -> BenchmarkComparison:
        """Absolute difference comparison."""
        result = BenchmarkComparison(
            baseline_result=baseline,
            comparison_result=comparison,
            comparison_method=ComparisonMethod.ABSOLUTE
        )
        
        # Calculate absolute changes as percentages for consistency
        result.ops_per_second_change_percent = self._calculate_percentage_change(
            baseline.metrics.operations_per_second, comparison.metrics.operations_per_second
        )
        result.latency_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_latency_ms, comparison.metrics.average_latency_ms
        )
        result.memory_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_memory_mb, comparison.metrics.average_memory_mb
        )
        result.cpu_change_percent = self._calculate_percentage_change(
            baseline.metrics.average_cpu_percent, comparison.metrics.average_cpu_percent
        )
        
        # Use absolute thresholds
        significant_ops_change = abs(comparison.metrics.operations_per_second - baseline.metrics.operations_per_second) > 10
        significant_latency_change = abs(comparison.metrics.average_latency_ms - baseline.metrics.average_latency_ms) > 50
        
        result.performance_improvement = (
            comparison.metrics.operations_per_second > baseline.metrics.operations_per_second and
            comparison.metrics.average_latency_ms <= baseline.metrics.average_latency_ms
        )
        
        result.regression_detected = (
            comparison.metrics.operations_per_second < baseline.metrics.operations_per_second * 0.9 or
            comparison.metrics.average_latency_ms > baseline.metrics.average_latency_ms * 1.2
        )
        
        result.significance_level = "high" if (significant_ops_change or significant_latency_change) else "low"
        
        return result
    
    def _calculate_percentage_change(self, baseline: float, comparison: float) -> float:
        """Calculate percentage change from baseline to comparison."""
        if baseline == 0:
            return 0.0 if comparison == 0 else 100.0
        
        return ((comparison - baseline) / baseline) * 100
    
    def _welch_t_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, Tuple[float, float]]:
        """Simplified Welch's t-test for unequal variances."""
        if len(sample1) < 2 or len(sample2) < 2:
            return 0.5, (0.0, 0.0)
        
        mean1 = statistics.mean(sample1)
        mean2 = statistics.mean(sample2)
        var1 = statistics.variance(sample1)
        var2 = statistics.variance(sample2)
        n1 = len(sample1)
        n2 = len(sample2)
        
        # Calculate t-statistic
        standard_error = np.sqrt(var1/n1 + var2/n2)
        if standard_error == 0:
            return 0.5, (0.0, 0.0)
        
        t_stat = (mean1 - mean2) / standard_error
        
        # Approximate degrees of freedom
        df = (var1/n1 + var2/n2)**2 / (var1**2/(n1**2*(n1-1)) + var2**2/(n2**2*(n2-1)))
        
        # Approximate p-value (very simplified)
        # In practice, would use proper statistical libraries
        significance = min(1.0, 1.0 - abs(t_stat) / 3.0)  # Rough approximation
        
        # Confidence interval (simplified)
        margin_of_error = 1.96 * standard_error  # 95% CI approximation
        ci_lower = (mean2 - mean1) - margin_of_error
        ci_upper = (mean2 - mean1) + margin_of_error
        
        return significance, (ci_lower, ci_upper)


class BenchmarkSuite:
    """Manages and executes benchmark suites."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path.home() / ".cache" / "tiny_llm_profiler" / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.runner = BenchmarkRunner()
        self.comparator = BenchmarkComparator()
        
        # Results storage
        self.results: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def create_standard_benchmark_suite(self) -> List[BenchmarkConfiguration]:
        """Create a standard suite of benchmark configurations."""
        
        suite = []
        
        # Basic performance benchmark
        suite.append(BenchmarkConfiguration(
            benchmark_id="standard_performance",
            benchmark_type=BenchmarkType.THROUGHPUT,
            name="Standard Performance Test",
            description="Basic throughput and latency test",
            duration_seconds=60.0,
            warmup_seconds=10.0,
            concurrency=1,
            test_data_config={"workload_type": "standard_llm", "model_count": 50},
            success_criteria={
                "min_ops_per_second": 10.0,
                "max_average_latency_ms": 500.0,
                "max_error_rate_percent": 1.0
            },
            tags={"standard", "performance"}
        ))
        
        # Memory stress test
        suite.append(BenchmarkConfiguration(
            benchmark_id="memory_stress",
            benchmark_type=BenchmarkType.MEMORY_EFFICIENCY,
            name="Memory Stress Test",
            description="Test memory usage under load",
            duration_seconds=120.0,
            warmup_seconds=10.0,
            concurrency=4,
            test_data_config={"workload_type": "memory_stress", "block_size_mb": 50},
            success_criteria={
                "max_memory_mb": 2000.0,
                "max_error_rate_percent": 5.0
            },
            tags={"stress", "memory"}
        ))
        
        # Concurrency test
        suite.append(BenchmarkConfiguration(
            benchmark_id="concurrency_test",
            benchmark_type=BenchmarkType.SCALABILITY,
            name="Concurrency Scalability Test",
            description="Test performance under concurrent load",
            iterations=1000,
            warmup_seconds=5.0,
            concurrency=10,
            test_data_config={"workload_type": "concurrency", "operations_per_task": 5},
            success_criteria={
                "min_ops_per_second": 50.0,
                "max_error_rate_percent": 2.0
            },
            tags={"concurrency", "scalability"}
        ))
        
        # Stability test
        suite.append(BenchmarkConfiguration(
            benchmark_id="stability_test",
            benchmark_type=BenchmarkType.STABILITY,
            name="Long-term Stability Test",
            description="Test stability over extended duration",
            duration_seconds=300.0,  # 5 minutes
            warmup_seconds=30.0,
            concurrency=2,
            test_data_config={"workload_type": "standard_llm", "model_count": 100},
            success_criteria={
                "max_error_rate_percent": 0.5,
                "max_memory_mb": 1500.0
            },
            tags={"stability", "endurance"}
        ))
        
        return suite
    
    async def run_benchmark_suite(
        self,
        configurations: List[BenchmarkConfiguration],
        save_results: bool = True
    ) -> List[BenchmarkResult]:
        """Run a complete benchmark suite."""
        logger.info(f"Running benchmark suite with {len(configurations)} tests")
        
        results = []
        
        for config in configurations:
            try:
                result = await self.runner.run_benchmark(config)
                results.append(result)
                
                # Store result
                with self._lock:
                    self.results[config.benchmark_id].append(result)
                
                # Save to disk if requested
                if save_results:
                    self._save_result(result)
                
                # Record performance metrics
                record_performance_metric(
                    f"benchmark_{config.benchmark_id}_ops_per_second",
                    result.metrics.operations_per_second,
                    tags={"benchmark_type", config.benchmark_type.value}
                )
                
                logger.info(f"Completed benchmark: {config.name} - Status: {result.status.value}")
                
            except Exception as e:
                logger.error(f"Failed to run benchmark {config.name}: {e}")
                
                # Create failed result
                failed_result = BenchmarkResult(
                    benchmark_id=config.benchmark_id,
                    configuration=config,
                    status=BenchmarkStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    failure_reason=str(e)
                )
                results.append(failed_result)
        
        logger.info(f"Benchmark suite completed: {len(results)} tests")
        return results
    
    def compare_benchmark_runs(
        self,
        benchmark_id: str,
        baseline_index: int = -2,  # Second to last
        comparison_index: int = -1,  # Last
        method: ComparisonMethod = ComparisonMethod.STATISTICAL
    ) -> Optional[BenchmarkComparison]:
        """Compare two benchmark runs."""
        with self._lock:
            if benchmark_id not in self.results:
                return None
            
            results_list = self.results[benchmark_id]
            
            if len(results_list) < 2:
                return None
            
            try:
                baseline = results_list[baseline_index]
                comparison = results_list[comparison_index]
                
                return self.comparator.compare_results(baseline, comparison, method)
                
            except IndexError:
                return None
    
    def generate_benchmark_report(
        self,
        include_historical: bool = True,
        include_comparisons: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "benchmark_results": {},
            "comparisons": {},
            "trends": {}
        }
        
        with self._lock:
            # Summary statistics
            total_benchmarks = sum(len(results) for results in self.results.values())
            successful_benchmarks = sum(
                len([r for r in results if r.status == BenchmarkStatus.COMPLETED])
                for results in self.results.values()
            )
            
            report["summary"] = {
                "total_benchmarks": total_benchmarks,
                "successful_benchmarks": successful_benchmarks,
                "success_rate": (successful_benchmarks / max(total_benchmarks, 1)) * 100,
                "benchmark_types": list(self.results.keys()),
                "execution_hosts": list(set(
                    result.execution_host 
                    for results in self.results.values() 
                    for result in results
                ))
            }
            
            # Individual benchmark results
            for benchmark_id, results_list in self.results.items():
                if not results_list:
                    continue
                
                latest_result = results_list[-1]
                
                benchmark_summary = {
                    "latest_result": latest_result.to_summary_dict(),
                    "total_runs": len(results_list),
                    "successful_runs": len([r for r in results_list if r.status == BenchmarkStatus.COMPLETED])
                }
                
                if include_historical and len(results_list) > 1:
                    # Historical performance
                    historical_metrics = []
                    for result in results_list:
                        if result.status == BenchmarkStatus.COMPLETED:
                            historical_metrics.append({
                                "timestamp": result.start_time.isoformat(),
                                "ops_per_second": result.metrics.operations_per_second,
                                "avg_latency_ms": result.metrics.average_latency_ms,
                                "efficiency_score": result.metrics.calculate_efficiency_score()
                            })
                    
                    benchmark_summary["historical_performance"] = historical_metrics
                
                report["benchmark_results"][benchmark_id] = benchmark_summary
            
            # Performance comparisons
            if include_comparisons:
                for benchmark_id, results_list in self.results.items():
                    if len(results_list) >= 2:
                        comparison = self.compare_benchmark_runs(benchmark_id)
                        if comparison:
                            report["comparisons"][benchmark_id] = {
                                "summary": comparison.generate_summary(),
                                "performance_improvement": comparison.performance_improvement,
                                "regression_detected": comparison.regression_detected,
                                "significance_level": comparison.significance_level,
                                "changes": {
                                    "ops_per_second": comparison.ops_per_second_change_percent,
                                    "latency": comparison.latency_change_percent,
                                    "memory": comparison.memory_change_percent,
                                    "cpu": comparison.cpu_change_percent
                                }
                            }
        
        return report
    
    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to disk."""
        try:
            timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{result.benchmark_id}_{timestamp}_{result.execution_host}.json"
            filepath = self.results_dir / filename
            
            # Convert result to serializable format
            result_data = {
                "benchmark_id": result.benchmark_id,
                "configuration": asdict(result.configuration),
                "status": result.status.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "execution_host": result.execution_host,
                "metrics": asdict(result.metrics),
                "validation_passed": result.validation_passed,
                "validation_details": result.validation_details,
                "failure_reason": result.failure_reason,
                "errors": result.errors
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            logger.debug(f"Saved benchmark result: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")
    
    def load_historical_results(self, benchmark_id: Optional[str] = None):
        """Load historical benchmark results from disk."""
        try:
            pattern = f"{benchmark_id}_*.json" if benchmark_id else "*.json"
            
            for filepath in self.results_dir.glob(pattern):
                try:
                    with open(filepath, 'r') as f:
                        result_data = json.load(f)
                    
                    # Reconstruct BenchmarkResult object (simplified)
                    config = BenchmarkConfiguration(**result_data["configuration"])
                    metrics = BenchmarkMetrics(**result_data["metrics"])
                    
                    result = BenchmarkResult(
                        benchmark_id=result_data["benchmark_id"],
                        configuration=config,
                        status=BenchmarkStatus(result_data["status"]),
                        start_time=datetime.fromisoformat(result_data["start_time"]),
                        execution_host=result_data["execution_host"],
                        metrics=metrics,
                        validation_passed=result_data["validation_passed"],
                        validation_details=result_data["validation_details"],
                        failure_reason=result_data.get("failure_reason")
                    )
                    
                    if result_data["end_time"]:
                        result.end_time = datetime.fromisoformat(result_data["end_time"])
                    
                    with self._lock:
                        self.results[result.benchmark_id].append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to load result from {filepath}: {e}")
            
            logger.info(f"Loaded historical benchmark results from {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load historical results: {e}")
    
    def get_benchmark_trends(self, benchmark_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze benchmark trends over specified time period."""
        with self._lock:
            if benchmark_id not in self.results:
                return {}
            
            # Filter results by time period
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_results = [
                r for r in self.results[benchmark_id]
                if r.start_time > cutoff_date and r.status == BenchmarkStatus.COMPLETED
            ]
            
            if len(recent_results) < 2:
                return {"message": "Insufficient data for trend analysis"}
            
            # Extract time series data
            timestamps = [r.start_time for r in recent_results]
            ops_per_second = [r.metrics.operations_per_second for r in recent_results]
            latencies = [r.metrics.average_latency_ms for r in recent_results]
            memory_usage = [r.metrics.average_memory_mb for r in recent_results]
            
            # Calculate trends (simplified linear regression)
            def calculate_trend(values):
                if len(values) < 2:
                    return 0.0
                x = list(range(len(values)))
                slope = np.polyfit(x, values, 1)[0]
                return slope
            
            trends = {
                "period_days": days,
                "total_runs": len(recent_results),
                "trends": {
                    "ops_per_second_trend": calculate_trend(ops_per_second),
                    "latency_trend": calculate_trend(latencies),
                    "memory_trend": calculate_trend(memory_usage)
                },
                "statistics": {
                    "ops_per_second": {
                        "mean": statistics.mean(ops_per_second),
                        "min": min(ops_per_second),
                        "max": max(ops_per_second),
                        "std": statistics.stdev(ops_per_second) if len(ops_per_second) > 1 else 0.0
                    },
                    "latency_ms": {
                        "mean": statistics.mean(latencies),
                        "min": min(latencies),
                        "max": max(latencies),
                        "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                    },
                    "memory_mb": {
                        "mean": statistics.mean(memory_usage),
                        "min": min(memory_usage),
                        "max": max(memory_usage),
                        "std": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0.0
                    }
                }
            }
            
            return trends


# Global benchmark suite instance
_global_benchmark_suite: Optional[BenchmarkSuite] = None


def get_benchmark_suite(**kwargs) -> BenchmarkSuite:
    """Get or create the global benchmark suite."""
    global _global_benchmark_suite
    
    if _global_benchmark_suite is None:
        _global_benchmark_suite = BenchmarkSuite(**kwargs)
        _global_benchmark_suite.load_historical_results()
    
    return _global_benchmark_suite


async def run_standard_benchmarks() -> List[BenchmarkResult]:
    """Run the standard benchmark suite."""
    suite = get_benchmark_suite()
    configurations = suite.create_standard_benchmark_suite()
    return await suite.run_benchmark_suite(configurations)


def compare_benchmark_performance(
    benchmark_id: str,
    method: ComparisonMethod = ComparisonMethod.STATISTICAL
) -> Optional[BenchmarkComparison]:
    """Compare latest benchmark performance with previous run."""
    suite = get_benchmark_suite()
    return suite.compare_benchmark_runs(benchmark_id, method=method)


def generate_performance_report() -> Dict[str, Any]:
    """Generate comprehensive performance report."""
    suite = get_benchmark_suite()
    return suite.generate_benchmark_report()


def get_benchmark_trends(benchmark_id: str, days: int = 30) -> Dict[str, Any]:
    """Get benchmark performance trends."""
    suite = get_benchmark_suite()
    return suite.get_benchmark_trends(benchmark_id, days)