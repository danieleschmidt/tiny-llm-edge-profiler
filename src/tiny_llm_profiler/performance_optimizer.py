"""
Advanced Performance Optimization Module for Generation 3
Provides comprehensive optimization capabilities including:
- Algorithmic optimizations for profiling operations
- Memory usage optimization with efficient data structures
- CPU optimization with vectorization and parallelization
- I/O optimization for device communication
- Caching strategies for expensive operations
"""

import time
import threading
import multiprocessing as mp
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import psutil
import math
import functools
from contextlib import contextmanager

from .exceptions import TinyLLMProfilerError, ResourceError
from .logging_config import get_logger, PerformanceLogger
from .models import QuantizedModel
from .results import ProfileResults

logger = get_logger("performance_optimizer")
perf_logger = PerformanceLogger()


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""

    # CPU Optimization
    enable_vectorization: bool = True
    enable_parallelization: bool = True
    max_threads: int = mp.cpu_count()
    cpu_affinity: Optional[List[int]] = None

    # Memory Optimization
    enable_memory_pooling: bool = True
    memory_pool_size_mb: int = 256
    enable_zero_copy: bool = True
    prefetch_size: int = 64  # KB

    # I/O Optimization
    enable_io_optimization: bool = True
    batch_size: int = 32
    buffer_size: int = 8192  # bytes
    async_io: bool = True

    # Caching Strategy
    enable_aggressive_caching: bool = True
    cache_warm_up: bool = True
    precompute_common_operations: bool = True

    # Algorithmic Optimization
    enable_algorithm_selection: bool = True
    prefer_approximate_algorithms: bool = False
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive, 3=extreme


class AlgorithmicOptimizer:
    """Optimizes algorithms used in profiling operations."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.algorithm_cache = {}
        self._performance_history = {}

    def optimize_latency_calculation(
        self, measurements: np.ndarray, method: str = "auto"
    ) -> Dict[str, float]:
        """
        Optimize latency calculations using vectorized operations.

        Args:
            measurements: Array of latency measurements
            method: Calculation method (auto, fast, accurate)

        Returns:
            Dictionary of optimized latency statistics
        """
        if method == "auto":
            method = "fast" if len(measurements) > 1000 else "accurate"

        # Use vectorized NumPy operations for performance
        if self.config.enable_vectorization and len(measurements) > 100:
            return self._vectorized_latency_stats(measurements)
        else:
            return self._standard_latency_stats(measurements)

    def _vectorized_latency_stats(self, measurements: np.ndarray) -> Dict[str, float]:
        """Vectorized latency statistics calculation."""
        # Sort once for multiple percentile calculations
        sorted_measurements = np.sort(measurements)

        # Batch percentile calculation
        percentiles = np.percentile(sorted_measurements, [50, 90, 95, 99])

        # Vectorized statistical operations
        stats = {
            "mean": np.mean(measurements),
            "std": np.std(measurements),
            "min": np.min(measurements),
            "max": np.max(measurements),
            "median": percentiles[0],
            "p90": percentiles[1],
            "p95": percentiles[2],
            "p99": percentiles[3],
            "count": len(measurements),
        }

        # Optimized outlier detection using vectorized operations
        q1, q3 = np.percentile(sorted_measurements, [25, 75])
        iqr = q3 - q1
        outlier_mask = (measurements < q1 - 1.5 * iqr) | (measurements > q3 + 1.5 * iqr)
        stats["outliers"] = np.sum(outlier_mask)
        stats["outlier_percentage"] = (stats["outliers"] / len(measurements)) * 100

        return stats

    def _standard_latency_stats(self, measurements: np.ndarray) -> Dict[str, float]:
        """Standard latency statistics for smaller datasets."""
        return {
            "mean": float(np.mean(measurements)),
            "std": float(np.std(measurements)),
            "min": float(np.min(measurements)),
            "max": float(np.max(measurements)),
            "median": float(np.median(measurements)),
            "p90": float(np.percentile(measurements, 90)),
            "p95": float(np.percentile(measurements, 95)),
            "p99": float(np.percentile(measurements, 99)),
            "count": len(measurements),
            "outliers": 0,
            "outlier_percentage": 0.0,
        }

    def optimize_memory_analysis(
        self, memory_samples: List[float], timestamps: List[float]
    ) -> Dict[str, Any]:
        """
        Optimize memory usage analysis with efficient algorithms.

        Args:
            memory_samples: Memory usage samples
            timestamps: Corresponding timestamps

        Returns:
            Optimized memory analysis results
        """
        if len(memory_samples) != len(timestamps):
            raise ValueError("Memory samples and timestamps must have same length")

        # Convert to NumPy arrays for vectorized operations
        memory_array = np.array(memory_samples)
        time_array = np.array(timestamps)

        # Vectorized peak detection
        peaks = self._find_memory_peaks_vectorized(memory_array)

        # Efficient trend analysis
        trend = self._calculate_trend_vectorized(memory_array, time_array)

        # Memory pattern recognition
        patterns = self._detect_memory_patterns(memory_array)

        return {
            "peak_memory": float(np.max(memory_array)),
            "average_memory": float(np.mean(memory_array)),
            "memory_growth_rate": trend["slope"],
            "memory_stability": trend["r_squared"],
            "peaks_detected": len(peaks),
            "peak_positions": peaks.tolist(),
            "patterns": patterns,
            "memory_efficiency_score": self._calculate_memory_efficiency(memory_array),
        }

    def _find_memory_peaks_vectorized(self, memory_array: np.ndarray) -> np.ndarray:
        """Vectorized peak detection in memory usage."""
        # Use scipy if available, fallback to simple method
        try:
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(
                memory_array, height=np.mean(memory_array) + np.std(memory_array)
            )
            return peaks
        except ImportError:
            # Simple peak detection using diff
            diff = np.diff(memory_array)
            peaks = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
            # Filter significant peaks
            threshold = np.mean(memory_array) + 0.5 * np.std(memory_array)
            significant_peaks = peaks[memory_array[peaks] > threshold]
            return significant_peaks

    def _calculate_trend_vectorized(
        self, values: np.ndarray, times: np.ndarray
    ) -> Dict[str, float]:
        """Vectorized trend calculation using least squares."""
        n = len(values)

        # Normalize time to start from 0
        times_norm = times - times[0]

        # Vectorized least squares calculation
        sum_x = np.sum(times_norm)
        sum_y = np.sum(values)
        sum_xy = np.sum(times_norm * values)
        sum_x2 = np.sum(times_norm * times_norm)

        # Calculate slope and intercept
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            slope = 0.0
            intercept = np.mean(values)
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_pred = slope * times_norm + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {"slope": slope, "intercept": intercept, "r_squared": r_squared}

    def _detect_memory_patterns(self, memory_array: np.ndarray) -> List[str]:
        """Detect memory usage patterns."""
        patterns = []

        # Check for memory leaks (consistent upward trend)
        if len(memory_array) > 10:
            recent_slope = self._calculate_trend_vectorized(
                memory_array[-10:], np.arange(10)
            )["slope"]

            if recent_slope > np.std(memory_array) * 0.1:
                patterns.append("potential_memory_leak")

        # Check for periodic patterns (simple autocorrelation)
        if len(memory_array) > 50:
            autocorr = np.correlate(memory_array, memory_array, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]

            # Look for periodic spikes in autocorrelation
            if np.max(autocorr[1:]) > 0.7 * autocorr[0]:
                patterns.append("periodic_memory_usage")

        # Check for sudden spikes
        if len(memory_array) > 5:
            diff = np.diff(memory_array)
            threshold = np.mean(diff) + 2 * np.std(diff)
            if np.any(diff > threshold):
                patterns.append("memory_spikes")

        return patterns

    def _calculate_memory_efficiency(self, memory_array: np.ndarray) -> float:
        """Calculate memory efficiency score (0-100)."""
        # Factors: low variance, no leaks, stable usage
        variance_score = max(
            0, 100 - (np.std(memory_array) / np.mean(memory_array)) * 100
        )

        # Trend score (penalize growth)
        trend = self._calculate_trend_vectorized(
            memory_array, np.arange(len(memory_array))
        )
        trend_score = max(0, 100 - abs(trend["slope"]) * 1000)

        # Stability score (R-squared of flat line fit)
        flat_line = np.full_like(memory_array, np.mean(memory_array))
        stability_score = 100 * (
            1 - np.mean((memory_array - flat_line) ** 2) / np.var(memory_array)
        )

        return (variance_score + trend_score + stability_score) / 3


class CPUOptimizer:
    """Optimizes CPU usage and enables vectorization/parallelization."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.thread_pool = None
        self.process_pool = None
        self._setup_cpu_optimization()

    def _setup_cpu_optimization(self):
        """Setup CPU optimization configurations."""
        # Set CPU affinity if specified
        if self.config.cpu_affinity:
            try:
                import psutil

                current_process = psutil.Process()
                current_process.cpu_affinity(self.config.cpu_affinity)
                logger.info(f"Set CPU affinity to cores: {self.config.cpu_affinity}")
            except Exception as e:
                logger.warning(f"Failed to set CPU affinity: {e}")

        # Initialize thread pools
        if self.config.enable_parallelization:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_threads)
            self.process_pool = ProcessPoolExecutor(max_workers=min(4, mp.cpu_count()))

    def parallel_profile_execution(
        self, profiling_tasks: List[Callable], use_processes: bool = False
    ) -> List[Any]:
        """
        Execute profiling tasks in parallel.

        Args:
            profiling_tasks: List of callable profiling tasks
            use_processes: Whether to use process pool (for CPU-bound tasks)

        Returns:
            List of results from parallel execution
        """
        if not self.config.enable_parallelization:
            return [task() for task in profiling_tasks]

        executor = self.process_pool if use_processes else self.thread_pool

        if not executor:
            return [task() for task in profiling_tasks]

        try:
            # Submit all tasks
            futures = [executor.submit(task) for task in profiling_tasks]

            # Collect results
            results = []
            for future in as_completed(futures, timeout=300):  # 5 minute timeout
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel task failed: {e}")
                    results.append(None)

            return results

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            return [task() for task in profiling_tasks]

    def vectorized_batch_processing(
        self,
        data_batches: List[np.ndarray],
        processing_func: Callable[[np.ndarray], np.ndarray],
    ) -> List[np.ndarray]:
        """
        Process data batches using vectorized operations.

        Args:
            data_batches: List of data batches to process
            processing_func: Vectorized processing function

        Returns:
            List of processed results
        """
        if not self.config.enable_vectorization:
            return [processing_func(batch) for batch in data_batches]

        try:
            # Stack batches for vectorized processing
            if all(batch.shape == data_batches[0].shape for batch in data_batches):
                # All batches same shape - can stack
                stacked_data = np.stack(data_batches)
                processed_stack = processing_func(stacked_data)
                return [processed_stack[i] for i in range(len(data_batches))]
            else:
                # Different shapes - process individually but with vectorization
                return [processing_func(batch) for batch in data_batches]

        except Exception as e:
            logger.warning(f"Vectorized processing failed: {e}")
            return [processing_func(batch) for batch in data_batches]

    def optimize_cpu_intensive_operation(
        self, operation: Callable, data: Any, chunk_size: Optional[int] = None
    ) -> Any:
        """
        Optimize CPU-intensive operations by chunking and parallelization.

        Args:
            operation: CPU-intensive operation to optimize
            data: Input data for the operation
            chunk_size: Size of data chunks for parallel processing

        Returns:
            Result of optimized operation
        """
        if not self.config.enable_parallelization:
            return operation(data)

        # Try to chunk data if it's array-like
        if hasattr(data, "__len__") and len(data) > 1000:
            chunk_size = chunk_size or max(100, len(data) // self.config.max_threads)

            # Create chunks
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i : i + chunk_size]
                chunks.append(chunk)

            # Process chunks in parallel
            chunk_operations = [functools.partial(operation, chunk) for chunk in chunks]
            chunk_results = self.parallel_profile_execution(
                chunk_operations, use_processes=True
            )

            # Combine results (assume they can be concatenated)
            try:
                if isinstance(chunk_results[0], np.ndarray):
                    return np.concatenate([r for r in chunk_results if r is not None])
                elif isinstance(chunk_results[0], list):
                    result = []
                    for r in chunk_results:
                        if r is not None:
                            result.extend(r)
                    return result
                else:
                    return chunk_results
            except Exception as e:
                logger.warning(f"Failed to combine chunk results: {e}")
                return operation(data)

        return operation(data)

    def shutdown(self):
        """Shutdown thread and process pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class MemoryOptimizer:
    """Advanced memory optimization and management."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_pools = {}
        self.allocation_tracker = {}
        self._setup_memory_optimization()

    def _setup_memory_optimization(self):
        """Setup memory optimization configurations."""
        if self.config.enable_memory_pooling:
            self._initialize_memory_pools()

    def _initialize_memory_pools(self):
        """Initialize memory pools for different data types."""
        pool_size_bytes = self.config.memory_pool_size_mb * 1024 * 1024

        # Pool for different array types
        self.memory_pools = {
            "float32": {
                "pool": [],
                "allocated": 0,
                "max_size": pool_size_bytes // 4,  # 4 bytes per float32
            },
            "float64": {
                "pool": [],
                "allocated": 0,
                "max_size": pool_size_bytes // 8,  # 8 bytes per float64
            },
            "int32": {"pool": [], "allocated": 0, "max_size": pool_size_bytes // 4},
        }

    @contextmanager
    def optimized_memory_allocation(self, dtype: str, size: int):
        """
        Context manager for optimized memory allocation.

        Args:
            dtype: Data type (float32, float64, int32)
            size: Number of elements needed

        Yields:
            Pre-allocated array from memory pool
        """
        if not self.config.enable_memory_pooling or dtype not in self.memory_pools:
            # Fallback to regular allocation
            if dtype == "float32":
                array = np.empty(size, dtype=np.float32)
            elif dtype == "float64":
                array = np.empty(size, dtype=np.float64)
            elif dtype == "int32":
                array = np.empty(size, dtype=np.int32)
            else:
                array = np.empty(size)

            yield array
            return

        pool_info = self.memory_pools[dtype]

        # Try to get from pool
        allocated_array = None
        for i, (pool_array, pool_size) in enumerate(pool_info["pool"]):
            if pool_size >= size:
                # Found suitable array in pool
                allocated_array = pool_array[:size]
                pool_info["pool"].pop(i)
                break

        # Create new if not found in pool
        if allocated_array is None:
            if dtype == "float32":
                allocated_array = np.empty(size, dtype=np.float32)
            elif dtype == "float64":
                allocated_array = np.empty(size, dtype=np.float64)
            elif dtype == "int32":
                allocated_array = np.empty(size, dtype=np.int32)

        try:
            yield allocated_array
        finally:
            # Return to pool if under limit
            if (
                pool_info["allocated"] + size < pool_info["max_size"]
                and len(pool_info["pool"]) < 10
            ):  # Limit pool size
                pool_info["pool"].append((allocated_array, size))
                pool_info["allocated"] += size

    def optimize_data_structures(self, data: Any) -> Any:
        """
        Optimize data structures for memory efficiency.

        Args:
            data: Input data to optimize

        Returns:
            Memory-optimized version of data
        """
        if isinstance(data, np.ndarray):
            return self._optimize_numpy_array(data)
        elif isinstance(data, list):
            return self._optimize_list(data)
        elif isinstance(data, dict):
            return self._optimize_dict(data)
        else:
            return data

    def _optimize_numpy_array(self, array: np.ndarray) -> np.ndarray:
        """Optimize NumPy array for memory efficiency."""
        # Check if we can use smaller dtype without losing precision
        if array.dtype == np.float64:
            # Check if values fit in float32 range
            if np.all(np.isfinite(array)) and np.all(np.abs(array) < 1e38):
                # Check if precision loss is acceptable
                float32_array = array.astype(np.float32)
                if np.allclose(array, float32_array, rtol=1e-6):
                    logger.debug(
                        f"Optimized float64 array to float32, saved {array.nbytes - float32_array.nbytes} bytes"
                    )
                    return float32_array

        elif array.dtype == np.int64:
            # Check if values fit in smaller int types
            if np.all(array >= np.iinfo(np.int32).min) and np.all(
                array <= np.iinfo(np.int32).max
            ):
                int32_array = array.astype(np.int32)
                logger.debug(
                    f"Optimized int64 array to int32, saved {array.nbytes - int32_array.nbytes} bytes"
                )
                return int32_array

        # Check for sparsity
        if array.size > 1000:
            zero_ratio = np.count_nonzero(array == 0) / array.size
            if zero_ratio > 0.7:  # More than 70% zeros
                try:
                    from scipy import sparse

                    sparse_array = sparse.csr_matrix(array)
                    logger.debug(
                        f"Converted to sparse matrix, density: {sparse_array.nnz / sparse_array.size:.3f}"
                    )
                    return sparse_array
                except ImportError:
                    pass

        return array

    def _optimize_list(self, lst: List[Any]) -> Union[List[Any], np.ndarray]:
        """Optimize list data structure."""
        if not lst:
            return lst

        # Check if list contains homogeneous numeric data
        if all(isinstance(x, (int, float)) for x in lst):
            # Convert to NumPy array for better memory efficiency
            if all(isinstance(x, int) for x in lst):
                return np.array(lst, dtype=np.int32)
            else:
                return np.array(lst, dtype=np.float32)

        return lst

    def _optimize_dict(self, dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize dictionary data structure."""
        optimized = {}
        for key, value in dictionary.items():
            optimized[key] = self.optimize_data_structures(value)
        return optimized

    def get_memory_usage_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        process = psutil.Process()
        memory_info = process.memory_info()

        pool_info = {}
        if self.config.enable_memory_pooling:
            for dtype, pool_data in self.memory_pools.items():
                pool_info[dtype] = {
                    "pool_size": len(pool_data["pool"]),
                    "allocated_mb": pool_data["allocated"]
                    * 4
                    / (1024 * 1024),  # Approximate
                    "max_size_mb": pool_data["max_size"] * 4 / (1024 * 1024),
                }

        return {
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "virtual_memory_mb": memory_info.vms / (1024 * 1024),
            "memory_pools": pool_info,
            "system_memory_available_mb": psutil.virtual_memory().available
            / (1024 * 1024),
        }


class IOOptimizer:
    """Optimizes I/O operations for device communication."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.connection_cache = {}
        self.buffer_pool = []
        self._io_stats = {
            "bytes_read": 0,
            "bytes_written": 0,
            "operations": 0,
            "cache_hits": 0,
        }

    def optimize_serial_communication(
        self, connection, commands: List[bytes], batch_size: Optional[int] = None
    ) -> List[bytes]:
        """
        Optimize serial communication with batching and buffering.

        Args:
            connection: Serial connection object
            commands: List of commands to send
            batch_size: Number of commands to batch together

        Returns:
            List of responses
        """
        batch_size = batch_size or self.config.batch_size
        responses = []

        # Process commands in batches
        for i in range(0, len(commands), batch_size):
            batch = commands[i : i + batch_size]
            batch_responses = self._process_command_batch(connection, batch)
            responses.extend(batch_responses)

        return responses

    def _process_command_batch(self, connection, batch: List[bytes]) -> List[bytes]:
        """Process a batch of commands efficiently."""
        responses = []

        # Pre-allocate buffer
        buffer = bytearray(self.config.buffer_size)

        for command in batch:
            # Write with optimized buffering
            connection.write(command)
            self._io_stats["bytes_written"] += len(command)

            # Read response with timeout
            response = self._read_response_optimized(connection, buffer)
            responses.append(response)

            self._io_stats["operations"] += 1

        return responses

    def _read_response_optimized(self, connection, buffer: bytearray) -> bytes:
        """Optimized response reading with buffering."""
        bytes_read = 0
        response_data = bytearray()

        while True:
            if connection.in_waiting > 0:
                chunk_size = min(connection.in_waiting, len(buffer))
                chunk = connection.read(chunk_size)
                response_data.extend(chunk)
                bytes_read += len(chunk)

                # Check for termination condition (e.g., newline)
                if b"\n" in chunk:
                    break
            else:
                # Brief wait to avoid busy polling
                time.sleep(0.001)

            # Safety timeout
            if bytes_read > 10000:  # 10KB limit
                break

        self._io_stats["bytes_read"] += bytes_read
        return bytes(response_data)

    def cache_connection_settings(
        self, device_path: str, settings: Dict[str, Any]
    ) -> None:
        """Cache connection settings for reuse."""
        self.connection_cache[device_path] = {
            "settings": settings,
            "timestamp": time.time(),
        }

    def get_cached_settings(self, device_path: str) -> Optional[Dict[str, Any]]:
        """Get cached connection settings."""
        if device_path in self.connection_cache:
            cached = self.connection_cache[device_path]
            # Cache valid for 1 hour
            if time.time() - cached["timestamp"] < 3600:
                self._io_stats["cache_hits"] += 1
                return cached["settings"]
        return None

    def get_io_stats(self) -> Dict[str, Any]:
        """Get I/O optimization statistics."""
        return self._io_stats.copy()


class PerformanceOptimizer:
    """Main performance optimizer coordinating all optimization components."""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

        # Initialize optimization components
        self.algorithmic_optimizer = AlgorithmicOptimizer(self.config)
        self.cpu_optimizer = CPUOptimizer(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.io_optimizer = IOOptimizer(self.config)

        # Performance monitoring
        self.performance_metrics = {
            "optimization_start_time": time.time(),
            "operations_optimized": 0,
            "total_time_saved": 0.0,
            "memory_saved_mb": 0.0,
        }

        logger.info("Performance optimizer initialized with advanced optimizations")

    def optimize_profiling_session(
        self, profiling_function: Callable, *args, **kwargs
    ) -> Any:
        """
        Optimize a complete profiling session.

        Args:
            profiling_function: Function to optimize
            *args: Arguments for the profiling function
            **kwargs: Keyword arguments for the profiling function

        Returns:
            Optimized profiling results
        """
        start_time = time.time()

        # Pre-optimization setup
        original_memory = self.memory_optimizer.get_memory_usage_info()

        # Apply optimizations
        try:
            # Memory optimization
            optimized_args = []
            for arg in args:
                optimized_args.append(
                    self.memory_optimizer.optimize_data_structures(arg)
                )

            optimized_kwargs = {}
            for key, value in kwargs.items():
                optimized_kwargs[key] = self.memory_optimizer.optimize_data_structures(
                    value
                )

            # Execute with optimizations
            if self.config.enable_parallelization and hasattr(
                profiling_function, "__call__"
            ):
                # Wrap in CPU optimization if applicable
                result = self.cpu_optimizer.optimize_cpu_intensive_operation(
                    profiling_function, (optimized_args, optimized_kwargs)
                )
            else:
                result = profiling_function(*optimized_args, **optimized_kwargs)

            # Post-optimization cleanup
            end_time = time.time()
            optimization_time = end_time - start_time

            # Update metrics
            self.performance_metrics["operations_optimized"] += 1

            # Calculate memory savings
            final_memory = self.memory_optimizer.get_memory_usage_info()
            memory_diff = original_memory.get(
                "process_memory_mb", 0
            ) - final_memory.get("process_memory_mb", 0)
            if memory_diff > 0:
                self.performance_metrics["memory_saved_mb"] += memory_diff

            logger.debug(f"Profiling session optimized in {optimization_time:.3f}s")

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fallback to unoptimized execution
            return profiling_function(*args, **kwargs)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        runtime = time.time() - self.performance_metrics["optimization_start_time"]

        stats = {
            "optimizer_runtime_seconds": runtime,
            "operations_optimized": self.performance_metrics["operations_optimized"],
            "total_memory_saved_mb": self.performance_metrics["memory_saved_mb"],
            "algorithmic_optimizations": True,
            "cpu_optimizations": self.config.enable_parallelization,
            "memory_optimizations": self.config.enable_memory_pooling,
            "io_optimizations": self.config.enable_io_optimization,
            "memory_usage": self.memory_optimizer.get_memory_usage_info(),
            "io_stats": self.io_optimizer.get_io_stats(),
        }

        return stats

    def shutdown(self):
        """Shutdown optimizer and clean up resources."""
        try:
            self.cpu_optimizer.shutdown()
            logger.info("Performance optimizer shutdown complete")
        except Exception as e:
            logger.error(f"Error during optimizer shutdown: {e}")


# Global performance optimizer instance
_global_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer(
    config: Optional[OptimizationConfig] = None,
) -> PerformanceOptimizer:
    """Get or create the global performance optimizer instance."""
    global _global_performance_optimizer

    if _global_performance_optimizer is None:
        _global_performance_optimizer = PerformanceOptimizer(config)

    return _global_performance_optimizer


def optimize_performance(config: Optional[OptimizationConfig] = None):
    """
    Decorator for optimizing function performance.

    Args:
        config: Optional optimization configuration

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        optimizer = get_performance_optimizer(config)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return optimizer.optimize_profiling_session(func, *args, **kwargs)

        return wrapper

    return decorator


# Performance monitoring context manager
@contextmanager
def performance_monitoring(name: str = "operation"):
    """Context manager for monitoring performance of operations."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        duration = end_time - start_time
        memory_diff = end_memory - start_memory

        perf_logger.log_performance_metric(
            operation=name, duration_seconds=duration, memory_delta_mb=memory_diff
        )

        logger.debug(
            f"Performance monitor - {name}: "
            f"{duration:.3f}s, {memory_diff:+.2f}MB memory"
        )
