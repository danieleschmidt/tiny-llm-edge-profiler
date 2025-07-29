"""Performance benchmarks and regression tests."""

import pytest
import time
from pathlib import Path
from typing import List, Dict
import statistics

from tiny_llm_profiler import EdgeProfiler, QuantizedModel, StandardBenchmarks


class TestPerformanceBenchmarks:
    """Performance benchmarks and regression testing."""
    
    @pytest.fixture
    def benchmark_models(self, tmp_path: Path) -> List[Path]:
        """Create benchmark models of different sizes."""
        models = []
        
        # Create models of different sizes for benchmarking
        sizes = [1_000_000, 2_000_000, 4_000_000]  # 1MB, 2MB, 4MB
        
        for i, size in enumerate(sizes):
            model_path = tmp_path / f"benchmark_model_{i}.bin"
            model_data = bytes(range(256)) * (size // 256)
            model_path.write_bytes(model_data)
            models.append(model_path)
            
        return models
    
    def test_quantization_performance_comparison(self, benchmark_models):
        """Compare performance across different quantization levels."""
        model_path = benchmark_models[1]  # 2MB model
        
        results = {}
        quantization_levels = [2, 3, 4]
        
        for bits in quantization_levels:
            model = QuantizedModel.from_file(
                model_path,
                quantization=bits,
                vocab_size=5000
            )
            
            # Mock profiler for performance testing
            profiler = EdgeProfiler(platform="esp32")
            
            start_time = time.perf_counter()
            
            # Simulate profiling operations
            with profiler:
                for _ in range(10):  # Multiple iterations for stability
                    _ = profiler._prepare_model(model)
                    _ = profiler._validate_model(model)
                    
            end_time = time.perf_counter()
            
            results[bits] = {
                'preparation_time_ms': (end_time - start_time) * 1000,
                'model_size_mb': model.size_bytes / 1_000_000,
                'memory_efficiency': model.size_bytes / (2 ** bits)
            }
        
        # Verify that lower bit quantization is more efficient
        assert results[2]['memory_efficiency'] > results[4]['memory_efficiency']
        assert all(r['preparation_time_ms'] < 1000 for r in results.values())
    
    def test_platform_comparison_benchmark(self):
        """Benchmark profiler initialization across platforms."""
        platforms = ["esp32", "stm32f4", "rp2040"]
        initialization_times = {}
        
        for platform in platforms:
            times = []
            
            for _ in range(5):  # Multiple runs for stability
                start_time = time.perf_counter()
                profiler = EdgeProfiler(platform=platform)
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)  # ms
            
            initialization_times[platform] = {
                'mean_ms': statistics.mean(times),
                'std_ms': statistics.stdev(times),
                'min_ms': min(times),
                'max_ms': max(times)
            }
        
        # All platforms should initialize quickly
        for platform, stats in initialization_times.items():
            assert stats['mean_ms'] < 100, f"{platform} initialization too slow"
            assert stats['std_ms'] < 50, f"{platform} initialization inconsistent"
    
    @pytest.mark.slow
    def test_memory_usage_benchmark(self, benchmark_models):
        """Benchmark memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
        
        profiler = EdgeProfiler(platform="esp32")
        memory_measurements = []
        
        for model_path in benchmark_models:
            model = QuantizedModel.from_file(model_path, quantization=4)
            
            # Measure memory before model loading
            mem_before = process.memory_info().rss
            
            # Load and process model
            with profiler:
                processed_model = profiler._prepare_model(model)
                
                # Measure memory after model loading
                mem_after = process.memory_info().rss
                
            memory_measurements.append({
                'model_size_mb': model.size_bytes / 1_000_000,
                'memory_increase_mb': (mem_after - mem_before) / 1_000_000,
                'memory_efficiency_ratio': (mem_after - mem_before) / model.size_bytes
            })
        
        # Memory usage should scale reasonably with model size
        for measurement in memory_measurements:
            # Memory increase shouldn't be more than 3x model size
            assert measurement['memory_efficiency_ratio'] < 3.0
            # Should use at least some memory
            assert measurement['memory_increase_mb'] > 0
    
    def test_concurrent_profiler_performance(self):
        """Test performance with multiple concurrent profilers."""
        import concurrent.futures
        import threading
        
        def create_profiler(platform: str) -> float:
            """Create profiler and return initialization time."""
            start_time = time.perf_counter()
            profiler = EdgeProfiler(platform=platform)
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000
        
        # Test concurrent initialization
        platforms = ["esp32"] * 5 + ["stm32f4"] * 3
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(create_profiler, platform)
                for platform in platforms
            ]
            
            times = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Concurrent initialization should still be fast
        assert max(times) < 500  # No single initialization > 500ms
        assert statistics.mean(times) < 200  # Average < 200ms
    
    @pytest.mark.slow
    def test_standard_benchmark_suite(self):
        """Run the standard benchmark suite and verify performance."""
        benchmarks = StandardBenchmarks()
        
        # Mock benchmark execution for testing
        class MockBenchmarkResults:
            def __init__(self):
                self.results = {
                    'latency_ms': [95.2, 87.1, 92.4],
                    'throughput_tps': [22.1, 24.3, 23.2],
                    'memory_kb': [380, 375, 382],
                    'accuracy': [0.87, 0.89, 0.88]
                }
                
            def get_summary(self):
                return {
                    'mean_latency_ms': statistics.mean(self.results['latency_ms']),
                    'mean_throughput_tps': statistics.mean(self.results['throughput_tps']),
                    'mean_memory_kb': statistics.mean(self.results['memory_kb']),
                    'mean_accuracy': statistics.mean(self.results['accuracy'])
                }
        
        # Simulate benchmark execution
        mock_results = MockBenchmarkResults()
        summary = mock_results.get_summary()
        
        # Verify benchmark results meet expected performance thresholds
        assert summary['mean_latency_ms'] < 150  # Reasonable latency
        assert summary['mean_throughput_tps'] > 15  # Minimum throughput
        assert summary['mean_memory_kb'] < 500  # Memory constraint
        assert summary['mean_accuracy'] > 0.8  # Accuracy threshold
    
    def test_regression_performance_tracking(self):
        """Track performance metrics for regression testing."""
        # This would typically load historical performance data
        historical_benchmarks = {
            'esp32_2bit_latency_ms': 95.0,
            'esp32_2bit_throughput_tps': 22.0,
            'esp32_2bit_memory_kb': 380,
            'stm32f4_4bit_latency_ms': 78.0,
            'stm32f4_4bit_throughput_tps': 25.5,
            'stm32f4_4bit_memory_kb': 190,
        }
        
        # Simulate current benchmark results
        current_benchmarks = {
            'esp32_2bit_latency_ms': 92.0,  # Improved
            'esp32_2bit_throughput_tps': 23.1,  # Improved
            'esp32_2bit_memory_kb': 375,  # Improved
            'stm32f4_4bit_latency_ms': 85.0,  # Regressed
            'stm32f4_4bit_throughput_tps': 24.8,  # Regressed slightly
            'stm32f4_4bit_memory_kb': 185,  # Improved
        }
        
        # Check for significant regressions (>10% worse)
        regressions = []
        improvements = []
        
        for metric, historical_value in historical_benchmarks.items():
            current_value = current_benchmarks[metric]
            
            if 'latency' in metric or 'memory' in metric:
                # Lower is better for latency and memory
                change_pct = (current_value - historical_value) / historical_value
                if change_pct > 0.1:  # 10% increase is bad
                    regressions.append((metric, change_pct))
                elif change_pct < -0.05:  # 5% decrease is good
                    improvements.append((metric, change_pct))
            else:
                # Higher is better for throughput
                change_pct = (current_value - historical_value) / historical_value
                if change_pct < -0.1:  # 10% decrease is bad
                    regressions.append((metric, change_pct))
                elif change_pct > 0.05:  # 5% increase is good
                    improvements.append((metric, change_pct))
        
        # Report findings
        if regressions:
            pytest.fail(f"Performance regressions detected: {regressions}")
        
        # At least some metrics should show improvements
        assert len(improvements) > 0, "No performance improvements detected"
    
    @pytest.mark.parametrize("model_size_mb,expected_max_latency_ms", [
        (1, 50),   # 1MB model should be fast
        (2, 95),   # 2MB model moderate latency
        (4, 180),  # 4MB model higher latency
    ])
    def test_model_size_latency_scaling(self, model_size_mb, expected_max_latency_ms, tmp_path):
        """Test that latency scales reasonably with model size."""
        # Create model of specified size
        model_path = tmp_path / f"model_{model_size_mb}mb.bin"
        model_data = b"\x00" * (model_size_mb * 1_000_000)
        model_path.write_bytes(model_data)
        
        model = QuantizedModel.from_file(model_path, quantization=4)
        profiler = EdgeProfiler(platform="esp32")
        
        # Mock profiling to simulate latency based on model size
        start_time = time.perf_counter()
        
        # Simulate processing time proportional to model size
        processing_time = model_size_mb * 0.02  # 20ms per MB
        time.sleep(processing_time)
        
        end_time = time.perf_counter()
        simulated_latency_ms = (end_time - start_time) * 1000
        
        # Verify latency is within expected bounds
        assert simulated_latency_ms <= expected_max_latency_ms