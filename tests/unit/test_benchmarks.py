"""
Unit tests for the benchmarks module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from pathlib import Path

from tiny_llm_profiler.benchmarks import (
    StandardBenchmarks,
    BenchmarkTask,
    BenchmarkResult
)
from tiny_llm_profiler.profiler import ProfilingConfig
from tiny_llm_profiler.results import ProfileResults, LatencyProfile, MemoryProfile, PowerProfile


class TestBenchmarkTask:
    """Test the BenchmarkTask dataclass."""
    
    def test_creation_minimal(self):
        """Test benchmark task creation with minimal parameters."""
        task = BenchmarkTask(
            name="test_task",
            prompts=["Test prompt"],
            expected_output_length=20
        )
        
        assert task.name == "test_task"
        assert task.prompts == ["Test prompt"]
        assert task.expected_output_length == 20
        assert task.timeout_seconds == 120  # Default
        assert task.description == ""       # Default
    
    def test_creation_full(self):
        """Test benchmark task creation with all parameters."""
        task = BenchmarkTask(
            name="full_task",
            prompts=["Prompt 1", "Prompt 2"],
            expected_output_length=50,
            timeout_seconds=300,
            description="Full test task"
        )
        
        assert task.name == "full_task"
        assert len(task.prompts) == 2
        assert task.expected_output_length == 50
        assert task.timeout_seconds == 300
        assert task.description == "Full test task"


class TestBenchmarkResult:
    """Test the BenchmarkResult dataclass."""
    
    def test_creation_minimal(self):
        """Test benchmark result creation with minimal parameters."""
        result = BenchmarkResult(
            task_name="test_task",
            model_name="test_model",
            platform="esp32",
            quantization="4bit",
            tokens_per_second=10.0,
            first_token_latency_ms=100.0,
            total_latency_ms=1000.0,
            peak_memory_kb=512.0,
            energy_per_token_mj=2.5
        )
        
        assert result.task_name == "test_task"
        assert result.model_name == "test_model"
        assert result.platform == "esp32"
        assert result.quantization == "4bit"
        assert result.tokens_per_second == 10.0
        assert result.accuracy_score is None  # Default
        assert result.success is True         # Default
        assert result.error_message is None   # Default
        assert result.timestamp > 0           # Should be set by post_init
    
    def test_creation_full(self):
        """Test benchmark result creation with all parameters."""
        timestamp = time.time()
        
        result = BenchmarkResult(
            task_name="full_task",
            model_name="full_model",
            platform="stm32f4",
            quantization="2bit",
            tokens_per_second=15.0,
            first_token_latency_ms=80.0,
            total_latency_ms=800.0,
            peak_memory_kb=256.0,
            energy_per_token_mj=1.8,
            accuracy_score=0.95,
            success=False,
            error_message="Test error",
            timestamp=timestamp
        )
        
        assert result.accuracy_score == 0.95
        assert result.success is False
        assert result.error_message == "Test error"
        assert result.timestamp == timestamp
    
    def test_post_init_sets_timestamp(self):
        """Test that post_init sets timestamp when not provided."""
        start_time = time.time()
        
        result = BenchmarkResult(
            task_name="test",
            model_name="test",
            platform="esp32",
            quantization="4bit",
            tokens_per_second=10.0,
            first_token_latency_ms=100.0,
            total_latency_ms=1000.0,
            peak_memory_kb=512.0,
            energy_per_token_mj=2.5,
            timestamp=0.0  # Should be replaced
        )
        
        end_time = time.time()
        
        assert start_time <= result.timestamp <= end_time


class TestStandardBenchmarks:
    """Test the StandardBenchmarks class."""
    
    def test_initialization(self):
        """Test standard benchmarks initialization."""
        benchmarks = StandardBenchmarks()
        
        assert len(benchmarks.tasks) > 0
        assert len(benchmarks.results) == 0
        
        # Check that standard tasks are present
        expected_tasks = [
            "text_generation",
            "summarization", 
            "qa",
            "code_generation",
            "reasoning"
        ]
        
        for task_name in expected_tasks:
            assert task_name in benchmarks.tasks
            task = benchmarks.tasks[task_name]
            assert isinstance(task, BenchmarkTask)
            assert len(task.prompts) > 0
            assert task.expected_output_length > 0
    
    def test_initialize_benchmark_tasks_structure(self):
        """Test that initialized benchmark tasks have correct structure."""
        benchmarks = StandardBenchmarks()
        
        # Test text generation task
        text_gen_task = benchmarks.tasks["text_generation"]
        assert text_gen_task.name == "text_generation"
        assert len(text_gen_task.prompts) >= 3
        assert text_gen_task.expected_output_length > 0
        assert "text generation" in text_gen_task.description.lower()
        
        # Test QA task
        qa_task = benchmarks.tasks["qa"]
        assert qa_task.name == "qa"
        assert all("Q:" in prompt and "A:" in prompt for prompt in qa_task.prompts)
        assert qa_task.expected_output_length < text_gen_task.expected_output_length
    
    def test_estimate_model_size(self):
        """Test model size estimation from name."""
        benchmarks = StandardBenchmarks()
        
        # Test different model sizes
        assert benchmarks._estimate_model_size("gpt2-125m") == 0.5
        assert benchmarks._estimate_model_size("model-350m") == 1.4
        assert benchmarks._estimate_model_size("llama-1b") == 2.8
        assert benchmarks._estimate_model_size("llama-1.1b") == 2.8
        assert benchmarks._estimate_model_size("large-7b") == 14.0
        assert benchmarks._estimate_model_size("unknown-model") == 2.0  # Default
    
    def test_extract_quantization(self):
        """Test quantization extraction from model name."""
        benchmarks = StandardBenchmarks()
        
        assert benchmarks._extract_quantization("model-2bit") == "2bit"
        assert benchmarks._extract_quantization("model-3bit") == "3bit"
        assert benchmarks._extract_quantization("model-4bit") == "4bit"
        assert benchmarks._extract_quantization("model-8bit") == "8bit"
        assert benchmarks._extract_quantization("model-fp16") == "4bit"  # Default
    
    def test_get_platform_device(self):
        """Test getting default device path for platforms."""
        benchmarks = StandardBenchmarks()
        
        assert benchmarks._get_platform_device("esp32") == "/dev/ttyUSB0"
        assert benchmarks._get_platform_device("stm32f4") == "/dev/ttyACM0"
        assert benchmarks._get_platform_device("stm32f7") == "/dev/ttyACM0"
        assert benchmarks._get_platform_device("rp2040") == "/dev/ttyACM1"
        assert benchmarks._get_platform_device("rpi_zero") is None
        assert benchmarks._get_platform_device("jetson_nano") is None
        assert benchmarks._get_platform_device("unknown") is None
    
    @patch('tiny_llm_profiler.benchmarks.EdgeProfiler')
    def test_run_single_benchmark_success(self, mock_profiler_class):
        """Test running a single benchmark successfully."""
        # Mock profiler and results
        mock_profiler = Mock()
        mock_profiler_class.return_value = mock_profiler
        
        # Create mock profile results
        mock_profile_results = Mock(spec=ProfileResults)
        mock_profile_results.latency_profile = LatencyProfile(
            first_token_latency_ms=120.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1200.0,
            tokens_per_second=12.5,
            latency_std_ms=10.0
        )
        mock_profile_results.memory_profile = MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=450.0,
            memory_usage_kb=350.0,
            memory_efficiency_tokens_per_kb=0.28
        )
        mock_profile_results.power_profile = PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=85.0,
            peak_power_mw=120.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6
        )
        
        mock_profiler.profile_model.return_value = mock_profile_results
        
        benchmarks = StandardBenchmarks()
        config = ProfilingConfig(measurement_iterations=2)
        
        result = benchmarks._run_single_benchmark(
            model_name="test-model-4bit",
            platform="esp32",
            task_name="text_generation",
            config=config
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.success is True
        assert result.task_name == "text_generation"
        assert result.model_name == "test-model-4bit"
        assert result.platform == "esp32"
        assert result.quantization == "4bit"
        assert result.tokens_per_second == 12.5
        assert result.first_token_latency_ms == 120.0
        assert result.total_latency_ms == 1200.0
        assert result.peak_memory_kb == 450.0
        assert result.energy_per_token_mj == 6.8
    
    @patch('tiny_llm_profiler.benchmarks.EdgeProfiler')
    def test_run_single_benchmark_failure(self, mock_profiler_class):
        """Test running a single benchmark with failure."""
        # Mock profiler to raise exception
        mock_profiler = Mock()
        mock_profiler_class.return_value = mock_profiler
        mock_profiler.profile_model.side_effect = Exception("Profiling failed")
        
        benchmarks = StandardBenchmarks()
        config = ProfilingConfig()
        
        result = benchmarks._run_single_benchmark(
            model_name="test-model",
            platform="esp32",
            task_name="text_generation",
            config=config
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.success is False
        assert result.error_message == "Profiling failed"
        assert result.tokens_per_second == 0.0
        assert result.peak_memory_kb == 0.0
    
    @patch('tiny_llm_profiler.benchmarks.StandardBenchmarks._run_single_benchmark')
    def test_run_tiny_ml_perf_sequential(self, mock_run_single):
        """Test running TinyML performance benchmarks sequentially."""
        # Mock successful benchmark results
        mock_result = BenchmarkResult(
            task_name="text_generation",
            model_name="test-model",
            platform="esp32",
            quantization="4bit",
            tokens_per_second=10.0,
            first_token_latency_ms=100.0,
            total_latency_ms=1000.0,
            peak_memory_kb=400.0,
            energy_per_token_mj=5.0,
            success=True
        )
        mock_run_single.return_value = mock_result
        
        benchmarks = StandardBenchmarks()
        
        results = benchmarks.run_tiny_ml_perf(
            models=["model1", "model2"],
            platforms=["esp32", "stm32f4"],
            tasks=["text_generation"],
            parallel=False
        )
        
        # Should run 2 models * 2 platforms * 1 task = 4 benchmarks
        assert len(results) == 4
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert mock_run_single.call_count == 4
        
        # Results should be added to benchmarks.results
        assert len(benchmarks.results) == 4
    
    @patch('tiny_llm_profiler.benchmarks.StandardBenchmarks._run_single_benchmark')
    def test_run_tiny_ml_perf_parallel(self, mock_run_single):
        """Test running TinyML performance benchmarks in parallel."""
        # Mock successful benchmark results
        mock_result = BenchmarkResult(
            task_name="text_generation",
            model_name="test-model",
            platform="esp32", 
            quantization="4bit",
            tokens_per_second=10.0,
            first_token_latency_ms=100.0,
            total_latency_ms=1000.0,
            peak_memory_kb=400.0,
            energy_per_token_mj=5.0,
            success=True
        )
        mock_run_single.return_value = mock_result
        
        benchmarks = StandardBenchmarks()
        
        results = benchmarks.run_tiny_ml_perf(
            models=["model1"],
            platforms=["esp32", "stm32f4"],
            tasks=["text_generation", "qa"],
            parallel=True,
            max_workers=2
        )
        
        # Should run 1 model * 2 platforms * 2 tasks = 4 benchmarks
        assert len(results) == 4
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert mock_run_single.call_count == 4
    
    @patch('tiny_llm_profiler.benchmarks.StandardBenchmarks._run_single_benchmark')
    def test_run_tiny_ml_perf_with_failures(self, mock_run_single):
        """Test running benchmarks with some failures."""
        def side_effect(*args, **kwargs):
            # First call succeeds, second fails
            if mock_run_single.call_count == 1:
                return BenchmarkResult(
                    task_name="text_generation",
                    model_name="test-model",
                    platform="esp32",
                    quantization="4bit",
                    tokens_per_second=10.0,
                    first_token_latency_ms=100.0,
                    total_latency_ms=1000.0,
                    peak_memory_kb=400.0,
                    energy_per_token_mj=5.0,
                    success=True
                )
            else:
                raise Exception("Benchmark failed")
        
        mock_run_single.side_effect = side_effect
        
        benchmarks = StandardBenchmarks()
        
        results = benchmarks.run_tiny_ml_perf(
            models=["model1"],
            platforms=["esp32", "stm32f4"],
            tasks=["text_generation"],
            parallel=False
        )
        
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert "Benchmark failed" in results[1].error_message
    
    def test_calculate_performance_score(self):
        """Test performance score calculation."""
        benchmarks = StandardBenchmarks()
        
        # Good performance result
        good_result = BenchmarkResult(
            task_name="test",
            model_name="test",
            platform="esp32",
            quantization="4bit",
            tokens_per_second=20.0,     # High throughput
            first_token_latency_ms=50.0, # Low latency
            total_latency_ms=500.0,
            peak_memory_kb=200.0,       # Low memory
            energy_per_token_mj=1.0,    # Low energy
            success=True
        )
        
        # Poor performance result
        poor_result = BenchmarkResult(
            task_name="test",
            model_name="test",
            platform="esp32",
            quantization="4bit",
            tokens_per_second=2.0,       # Low throughput
            first_token_latency_ms=500.0, # High latency
            total_latency_ms=5000.0,
            peak_memory_kb=1000.0,       # High memory
            energy_per_token_mj=10.0,    # High energy
            success=True
        )
        
        # Failed result
        failed_result = BenchmarkResult(
            task_name="test",
            model_name="test",
            platform="esp32",
            quantization="4bit",
            tokens_per_second=0.0,
            first_token_latency_ms=0.0,
            total_latency_ms=0.0,
            peak_memory_kb=0.0,
            energy_per_token_mj=0.0,
            success=False
        )
        
        good_score = benchmarks._calculate_performance_score(good_result)
        poor_score = benchmarks._calculate_performance_score(poor_result)
        failed_score = benchmarks._calculate_performance_score(failed_result)
        
        assert good_score > poor_score
        assert poor_score > failed_score
        assert failed_score == 0.0
    
    def test_create_leaderboard_empty_results(self):
        """Test creating leaderboard with empty results."""
        benchmarks = StandardBenchmarks()
        
        # Should not raise exception with empty results
        benchmarks.create_leaderboard([], "test_leaderboard.md")
    
    @patch('builtins.open', new_callable=lambda: Mock())
    def test_create_leaderboard_with_results(self, mock_open):
        """Test creating leaderboard with benchmark results."""
        benchmarks = StandardBenchmarks()
        
        results = [
            BenchmarkResult(
                task_name="text_generation",
                model_name="model1",
                platform="esp32",
                quantization="4bit",
                tokens_per_second=15.0,
                first_token_latency_ms=100.0,
                total_latency_ms=1000.0,
                peak_memory_kb=400.0,
                energy_per_token_mj=5.0,
                success=True
            ),
            BenchmarkResult(
                task_name="text_generation",
                model_name="model2",
                platform="esp32",
                quantization="2bit",
                tokens_per_second=20.0,
                first_token_latency_ms=80.0,
                total_latency_ms=800.0,
                peak_memory_kb=300.0,
                energy_per_token_mj=4.0,
                success=True
            )
        ]
        
        mock_file = Mock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=False)
        
        benchmarks.create_leaderboard(results, "test_leaderboard.md")
        
        # Should write markdown content
        mock_open.assert_called_once_with("test_leaderboard.md", 'w')
        mock_file.write.assert_called_once()
        
        # Check that markdown content contains expected elements
        written_content = mock_file.write.call_args[0][0]
        assert "# TinyML LLM Benchmark Leaderboard" in written_content
        assert "Text Generation" in written_content
        assert "model1" in written_content
        assert "model2" in written_content
    
    def test_export_results_json(self):
        """Test exporting results to JSON."""
        benchmarks = StandardBenchmarks()
        benchmarks.results = [
            BenchmarkResult(
                task_name="test_task",
                model_name="test_model",
                platform="esp32",
                quantization="4bit",
                tokens_per_second=10.0,
                first_token_latency_ms=100.0,
                total_latency_ms=1000.0,
                peak_memory_kb=400.0,
                energy_per_token_mj=5.0
            )
        ]
        
        with patch('builtins.open', Mock()) as mock_open:
            with patch('json.dump') as mock_json_dump:
                mock_file = Mock()
                mock_open.return_value.__enter__ = Mock(return_value=mock_file)
                mock_open.return_value.__exit__ = Mock(return_value=False)
                
                benchmarks.export_results("results.json", format="json")
                
                mock_open.assert_called_once_with("results.json", 'w')
                mock_json_dump.assert_called_once()
                
                # Check that the data passed to json.dump is correct
                exported_data = mock_json_dump.call_args[0][0]
                assert len(exported_data) == 1
                assert exported_data[0]["task_name"] == "test_task"
    
    def test_export_results_csv(self):
        """Test exporting results to CSV."""
        benchmarks = StandardBenchmarks()
        benchmarks.results = [
            BenchmarkResult(
                task_name="test_task",
                model_name="test_model",
                platform="esp32",
                quantization="4bit",
                tokens_per_second=10.0,
                first_token_latency_ms=100.0,
                total_latency_ms=1000.0,
                peak_memory_kb=400.0,
                energy_per_token_mj=5.0
            )
        ]
        
        with patch('builtins.open', Mock()) as mock_open:
            with patch('csv.DictWriter') as mock_csv_writer:
                mock_file = Mock()
                mock_open.return_value.__enter__ = Mock(return_value=mock_file)
                mock_open.return_value.__exit__ = Mock(return_value=False)
                
                mock_writer_instance = Mock()
                mock_csv_writer.return_value = mock_writer_instance
                
                benchmarks.export_results("results.csv", format="csv")
                
                mock_open.assert_called_once_with("results.csv", 'w', newline='')
                mock_writer_instance.writeheader.assert_called_once()
                mock_writer_instance.writerow.assert_called_once()
    
    def test_analyze_results_empty(self):
        """Test analyzing empty results."""
        benchmarks = StandardBenchmarks()
        
        analysis = benchmarks.analyze_results()
        
        assert "error" in analysis
        assert "No results to analyze" in analysis["error"]
    
    def test_analyze_results_no_successful(self):
        """Test analyzing results with no successful runs."""
        benchmarks = StandardBenchmarks()
        benchmarks.results = [
            BenchmarkResult(
                task_name="test",
                model_name="test",
                platform="esp32",
                quantization="4bit",
                tokens_per_second=0.0,
                first_token_latency_ms=0.0,
                total_latency_ms=0.0,
                peak_memory_kb=0.0,
                energy_per_token_mj=0.0,
                success=False
            )
        ]
        
        analysis = benchmarks.analyze_results()
        
        assert "error" in analysis
        assert "No successful benchmark runs" in analysis["error"]
    
    def test_analyze_results_with_data(self):
        """Test analyzing results with successful data."""
        benchmarks = StandardBenchmarks()
        benchmarks.results = [
            BenchmarkResult(
                task_name="text_generation",
                model_name="model1",
                platform="esp32",
                quantization="4bit",
                tokens_per_second=15.0,
                first_token_latency_ms=100.0,
                total_latency_ms=1000.0,
                peak_memory_kb=400.0,
                energy_per_token_mj=5.0,
                success=True
            ),
            BenchmarkResult(
                task_name="qa",
                model_name="model2",
                platform="stm32f4",
                quantization="2bit",
                tokens_per_second=20.0,
                first_token_latency_ms=80.0,
                total_latency_ms=800.0,
                peak_memory_kb=300.0,
                energy_per_token_mj=4.0,
                success=True
            ),
            BenchmarkResult(
                task_name="failed_test",
                model_name="model3",
                platform="rp2040",
                quantization="8bit",
                tokens_per_second=0.0,
                first_token_latency_ms=0.0,
                total_latency_ms=0.0,
                peak_memory_kb=0.0,
                energy_per_token_mj=0.0,
                success=False
            )
        ]
        
        analysis = benchmarks.analyze_results()
        
        # Check basic statistics
        assert analysis["total_benchmarks"] == 3
        assert analysis["successful_benchmarks"] == 2
        assert analysis["success_rate"] == 2/3
        
        # Check categorization
        assert set(analysis["platforms_tested"]) == {"esp32", "stm32f4"}
        assert set(analysis["models_tested"]) == {"model1", "model2"}
        assert set(analysis["tasks_tested"]) == {"text_generation", "qa"}
        
        # Check performance statistics
        assert "performance_stats" in analysis
        perf_stats = analysis["performance_stats"]
        assert perf_stats["avg_tokens_per_second"] == 17.5  # (15 + 20) / 2
        assert perf_stats["max_tokens_per_second"] == 20.0
        assert perf_stats["min_tokens_per_second"] == 15.0
        
        # Check memory statistics
        assert "memory_stats" in analysis
        memory_stats = analysis["memory_stats"]
        assert memory_stats["avg_peak_memory_kb"] == 350.0  # (400 + 300) / 2
        
        # Check best performer
        assert "best_performer" in analysis
        best = analysis["best_performer"]
        assert best["model"] == "model2"  # Should have better performance score
        assert best["platform"] == "stm32f4"


if __name__ == "__main__":
    pytest.main([__file__])