"""Performance benchmarking utilities for testing."""

import time
import psutil
import threading
from contextlib import contextmanager
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import statistics
import sys


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    duration_seconds: float
    iterations: int
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float
    memory_peak_mb: float
    memory_avg_mb: float
    cpu_percent: float
    measurements: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmark:
    """Performance benchmarking utility for profiler components."""
    
    def __init__(self, name: str = "benchmark"):
        self.name = name
        self.results: List[BenchmarkResult] = []
        self._memory_monitor = None
        self._cpu_monitor = None
        self._monitoring = False
        
    def benchmark(
        self,
        func: Callable,
        iterations: int = 100,
        warmup_iterations: int = 10,
        name: str = None,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a function with multiple iterations."""
        test_name = name or func.__name__
        
        # Warmup runs
        for _ in range(warmup_iterations):
            func(**kwargs)
        
        # Monitoring setup
        memory_samples = []
        cpu_samples = []
        self._start_monitoring(memory_samples, cpu_samples)
        
        # Actual benchmark runs
        measurements = []
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            iter_start = time.perf_counter()
            func(**kwargs)
            iter_end = time.perf_counter()
            measurements.append(iter_end - iter_start)
        
        end_time = time.perf_counter()
        self._stop_monitoring()
        
        # Calculate statistics
        total_duration = end_time - start_time
        mean_time = statistics.mean(measurements)
        median_time = statistics.median(measurements)
        min_time = min(measurements)
        max_time = max(measurements)
        std_dev = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
        throughput = iterations / total_duration
        
        # Memory and CPU statistics
        memory_peak = max(memory_samples) if memory_samples else 0.0
        memory_avg = statistics.mean(memory_samples) if memory_samples else 0.0
        cpu_avg = statistics.mean(cpu_samples) if cpu_samples else 0.0
        
        result = BenchmarkResult(
            name=test_name,
            duration_seconds=total_duration,
            iterations=iterations,
            mean_time=mean_time,
            median_time=median_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_peak_mb=memory_peak,
            memory_avg_mb=memory_avg,
            cpu_percent=cpu_avg,
            measurements=measurements,
            metadata={
                "warmup_iterations": warmup_iterations,
                "python_version": sys.version,
                "platform": sys.platform
            }
        )
        
        self.results.append(result)
        return result
    
    @contextmanager
    def measure_time(self, name: str = "measurement"):
        """Context manager for measuring execution time."""
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        result = BenchmarkResult(
            name=name,
            duration_seconds=duration,
            iterations=1,
            mean_time=duration,
            median_time=duration,
            min_time=duration,
            max_time=duration,
            std_dev=0.0,
            throughput=1.0 / duration,
            memory_peak_mb=0.0,
            memory_avg_mb=0.0,
            cpu_percent=0.0,
            measurements=[duration]
        )
        
        self.results.append(result)
    
    def compare_functions(
        self,
        functions: Dict[str, Callable],
        iterations: int = 100,
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """Compare performance of multiple functions."""
        results = {}
        
        for name, func in functions.items():
            result = self.benchmark(
                func=func,
                iterations=iterations,
                name=name,
                **kwargs
            )
            results[name] = result
        
        return results
    
    def profile_memory_usage(
        self,
        func: Callable,
        duration_seconds: int = 60,
        sample_interval: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """Profile memory usage over time."""
        memory_samples = []
        start_time = time.time()
        
        def memory_sampler():
            while time.time() - start_time < duration_seconds:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append({
                    "timestamp": time.time() - start_time,
                    "memory_mb": memory_mb
                })
                time.sleep(sample_interval)
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=memory_sampler)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run function
        func_start = time.perf_counter()
        result = func(**kwargs)
        func_end = time.perf_counter()
        
        # Wait for monitoring to complete
        monitor_thread.join(timeout=duration_seconds + 1)
        
        return {
            "function_result": result,
            "execution_time": func_end - func_start,
            "memory_profile": memory_samples,
            "peak_memory_mb": max([s["memory_mb"] for s in memory_samples]),
            "avg_memory_mb": statistics.mean([s["memory_mb"] for s in memory_samples]),
            "memory_growth_mb": memory_samples[-1]["memory_mb"] - memory_samples[0]["memory_mb"]
        }
    
    def stress_test(
        self,
        func: Callable,
        duration_seconds: int = 300,
        concurrent_calls: int = 4,
        **kwargs
    ) -> Dict[str, Any]:
        """Stress test a function with concurrent execution."""
        results = []
        exceptions = []
        
        def worker():
            try:
                start_time = time.perf_counter()
                while time.perf_counter() - start_time < duration_seconds:
                    iter_start = time.perf_counter()
                    func(**kwargs)
                    iter_end = time.perf_counter()
                    results.append(iter_end - iter_start)
            except Exception as e:
                exceptions.append(e)
        
        # Start worker threads
        threads = []
        for _ in range(concurrent_calls):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        if not results:
            return {
                "error": "No successful executions",
                "exceptions": [str(e) for e in exceptions]
            }
        
        return {
            "total_executions": len(results),
            "successful_executions": len(results),
            "failed_executions": len(exceptions),
            "success_rate": len(results) / (len(results) + len(exceptions)),
            "mean_time": statistics.mean(results),
            "median_time": statistics.median(results),
            "throughput_per_second": len(results) / duration_seconds,
            "exceptions": [str(e) for e in exceptions[:10]]  # Limit exception list
        }
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a benchmark report."""
        if not self.results:
            return "No benchmark results available"
        
        report_lines = [
            f"# Benchmark Report: {self.name}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"Total benchmarks: {len(self.results)}",
            ""
        ]
        
        for result in self.results:
            report_lines.extend([
                f"### {result.name}",
                f"- **Iterations**: {result.iterations}",
                f"- **Duration**: {result.duration_seconds:.3f}s",
                f"- **Mean time**: {result.mean_time*1000:.3f}ms",
                f"- **Median time**: {result.median_time*1000:.3f}ms",
                f"- **Min time**: {result.min_time*1000:.3f}ms",
                f"- **Max time**: {result.max_time*1000:.3f}ms",
                f"- **Std deviation**: {result.std_dev*1000:.3f}ms",
                f"- **Throughput**: {result.throughput:.2f} ops/sec",
                f"- **Peak memory**: {result.memory_peak_mb:.2f} MB",
                f"- **Avg memory**: {result.memory_avg_mb:.2f} MB",
                f"- **CPU usage**: {result.cpu_percent:.1f}%",
                ""
            ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
        
        return report
    
    def export_json(self, output_path: Path) -> None:
        """Export results to JSON format."""
        data = {
            "benchmark_name": self.name,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": []
        }
        
        for result in self.results:
            data["results"].append({
                "name": result.name,
                "duration_seconds": result.duration_seconds,
                "iterations": result.iterations,
                "statistics": {
                    "mean_time": result.mean_time,
                    "median_time": result.median_time,
                    "min_time": result.min_time,
                    "max_time": result.max_time,
                    "std_dev": result.std_dev,
                    "throughput": result.throughput
                },
                "resources": {
                    "memory_peak_mb": result.memory_peak_mb,
                    "memory_avg_mb": result.memory_avg_mb,
                    "cpu_percent": result.cpu_percent
                },
                "measurements": result.measurements,
                "metadata": result.metadata
            })
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _start_monitoring(self, memory_samples: List, cpu_samples: List):
        """Start resource monitoring."""
        self._monitoring = True
        
        def monitor():
            process = psutil.Process()
            while self._monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    memory_samples.append(memory_mb)
                    cpu_samples.append(cpu_percent)
                    time.sleep(0.1)  # Sample every 100ms
                except psutil.NoSuchProcess:
                    break
        
        self._monitor_thread = threading.Thread(target=monitor)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def _stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)


# Utility functions for common benchmarking scenarios

def benchmark_model_loading(model_path: Path, iterations: int = 10) -> BenchmarkResult:
    """Benchmark model loading performance."""
    benchmark = PerformanceBenchmark("model_loading")
    
    def load_model():
        # Mock model loading - replace with actual implementation
        time.sleep(0.1)  # Simulate loading time
        return {"size_mb": 5.0, "loaded": True}
    
    return benchmark.benchmark(
        func=load_model,
        iterations=iterations,
        name="model_loading"
    )


def benchmark_device_communication(device_mock, iterations: int = 100) -> BenchmarkResult:
    """Benchmark device communication performance."""
    benchmark = PerformanceBenchmark("device_communication")
    
    def communicate():
        device_mock.write(b"TEST")
        return device_mock.read()
    
    return benchmark.benchmark(
        func=communicate,
        iterations=iterations,
        name="device_communication"
    )


def benchmark_quantization(model_data: bytes, iterations: int = 5) -> BenchmarkResult:
    """Benchmark model quantization performance."""
    benchmark = PerformanceBenchmark("quantization")
    
    def quantize():
        # Mock quantization - replace with actual implementation
        time.sleep(0.5)  # Simulate quantization time
        return model_data[:len(model_data)//2]  # Mock quantized model
    
    return benchmark.benchmark(
        func=quantize,
        iterations=iterations,
        name="quantization"
    )


class ProfilerBenchmarkSuite:
    """Complete benchmark suite for the profiler."""
    
    def __init__(self):
        self.benchmarks = {}
    
    def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run all performance benchmarks."""
        results = {}
        
        # Model operations benchmarks
        results["model_loading"] = benchmark_model_loading(
            Path("test_model.bin"), iterations=20
        )
        
        # Device communication benchmarks
        from unittest.mock import Mock
        mock_device = Mock()
        mock_device.write.return_value = None
        mock_device.read.return_value = b"OK"
        
        results["device_comm"] = benchmark_device_communication(
            mock_device, iterations=200
        )
        
        # Quantization benchmarks
        results["quantization"] = benchmark_quantization(
            b"dummy_model_data" * 1000, iterations=10
        )
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate comparative benchmark report."""
        report_lines = [
            "# Profiler Performance Benchmark Suite",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance Summary",
            ""
        ]
        
        # Create comparison table
        report_lines.extend([
            "| Operation | Mean Time (ms) | Throughput (ops/s) | Peak Memory (MB) |",
            "|-----------|----------------|-------------------|------------------|"
        ])
        
        for name, result in results.items():
            report_lines.append(
                f"| {name} | {result.mean_time*1000:.2f} | "
                f"{result.throughput:.2f} | {result.memory_peak_mb:.2f} |"
            )
        
        report_lines.extend([
            "",
            "## Detailed Results",
            ""
        ])
        
        for name, result in results.items():
            report_lines.extend([
                f"### {name}",
                f"- Iterations: {result.iterations}",
                f"- Total duration: {result.duration_seconds:.3f}s",
                f"- Mean ± Std: {result.mean_time*1000:.3f} ± {result.std_dev*1000:.3f}ms",
                f"- Range: {result.min_time*1000:.3f} - {result.max_time*1000:.3f}ms",
                f"- Resource usage: {result.memory_avg_mb:.2f}MB RAM, {result.cpu_percent:.1f}% CPU",
                ""
            ])
        
        return "\n".join(report_lines)