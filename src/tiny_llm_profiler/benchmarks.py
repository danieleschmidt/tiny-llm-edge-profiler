"""
Standardized benchmarking suite for edge LLM performance comparison.
"""

import time
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .profiler import EdgeProfiler, ProfilingConfig
from .models import QuantizedModel
from .results import ProfileResults


@dataclass
class BenchmarkTask:
    """Definition of a benchmark task."""

    name: str
    prompts: List[str]
    expected_output_length: int
    timeout_seconds: int = 120
    description: str = ""


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark."""

    task_name: str
    model_name: str
    platform: str
    quantization: str

    # Performance metrics
    tokens_per_second: float
    first_token_latency_ms: float
    total_latency_ms: float
    peak_memory_kb: float
    energy_per_token_mj: float

    # Quality metrics (if available)
    accuracy_score: Optional[float] = None

    # Additional metadata
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class StandardBenchmarks:
    """Standard benchmark suite for TinyML LLM performance."""

    def __init__(self):
        self.tasks = self._initialize_benchmark_tasks()
        self.results: List[BenchmarkResult] = []

    def _initialize_benchmark_tasks(self) -> Dict[str, BenchmarkTask]:
        """Initialize standard benchmark tasks."""
        return {
            "text_generation": BenchmarkTask(
                name="text_generation",
                prompts=[
                    "The future of artificial intelligence is",
                    "Write a short story about a robot",
                    "Explain quantum computing in simple terms",
                    "List the benefits of renewable energy",
                    "Describe the process of photosynthesis",
                ],
                expected_output_length=50,
                description="General text generation capability",
            ),
            "summarization": BenchmarkTask(
                name="summarization",
                prompts=[
                    "Summarize: Climate change refers to long-term shifts in global temperatures and weather patterns. While climate changes are natural, since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas.",
                    "Summarize: Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents.",
                    "Summarize: The Internet is a global system of interconnected computer networks that uses the Internet protocol suite to communicate between networks and devices.",
                ],
                expected_output_length=30,
                description="Text summarization task",
            ),
            "qa": BenchmarkTask(
                name="qa",
                prompts=[
                    "Q: What is the capital of France? A:",
                    "Q: How many legs does a spider have? A:",
                    "Q: What is 2 + 2? A:",
                    "Q: What is the largest ocean on Earth? A:",
                    "Q: Who wrote Romeo and Juliet? A:",
                ],
                expected_output_length=10,
                description="Question answering task",
            ),
            "code_generation": BenchmarkTask(
                name="code_generation",
                prompts=[
                    "Write a Python function to calculate factorial",
                    "Create a JavaScript function to reverse a string",
                    "Write a C function to find the maximum in an array",
                    "Python code to sort a list of numbers",
                    "Function to check if a number is prime",
                ],
                expected_output_length=80,
                description="Code generation capability",
            ),
            "reasoning": BenchmarkTask(
                name="reasoning",
                prompts=[
                    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                    "A farmer has 17 sheep and all but 9 die. How many are left?",
                    "If a red house is made of red bricks and a yellow house is made of yellow bricks, what is a greenhouse made of?",
                    "What comes next in the sequence: 2, 4, 8, 16, ?",
                    "If you're running in a race and pass the person in second place, what place are you in?",
                ],
                expected_output_length=40,
                description="Logical reasoning tasks",
            ),
        }

    def run_tiny_ml_perf(
        self,
        models: List[str],
        platforms: List[str],
        tasks: Optional[List[str]] = None,
        config: Optional[ProfilingConfig] = None,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> List[BenchmarkResult]:
        """
        Run TinyML performance benchmarks.

        Args:
            models: List of model names/paths to benchmark
            platforms: List of platforms to test on
            tasks: List of task names (if None, runs all tasks)
            config: Profiling configuration
            parallel: Whether to run benchmarks in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            List of BenchmarkResult objects
        """
        tasks = tasks or list(self.tasks.keys())
        config = config or ProfilingConfig(
            measurement_iterations=3, warmup_iterations=1
        )

        benchmark_configs = []
        for model_name in models:
            for platform in platforms:
                for task_name in tasks:
                    benchmark_configs.append((model_name, platform, task_name))

        results = []

        if parallel and len(benchmark_configs) > 1:
            # Run benchmarks in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {
                    executor.submit(
                        self._run_single_benchmark,
                        model_name,
                        platform,
                        task_name,
                        config,
                    ): (model_name, platform, task_name)
                    for model_name, platform, task_name in benchmark_configs
                }

                for future in as_completed(future_to_config):
                    config_info = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"Completed benchmark: {config_info}")
                    except Exception as exc:
                        print(f"Benchmark {config_info} generated exception: {exc}")
                        # Create failed result
                        results.append(
                            BenchmarkResult(
                                task_name=config_info[2],
                                model_name=config_info[0],
                                platform=config_info[1],
                                quantization="unknown",
                                tokens_per_second=0.0,
                                first_token_latency_ms=0.0,
                                total_latency_ms=0.0,
                                peak_memory_kb=0.0,
                                energy_per_token_mj=0.0,
                                success=False,
                                error_message=str(exc),
                            )
                        )
        else:
            # Run benchmarks sequentially
            for model_name, platform, task_name in benchmark_configs:
                try:
                    result = self._run_single_benchmark(
                        model_name, platform, task_name, config
                    )
                    results.append(result)
                    print(
                        f"Completed benchmark: {model_name} on {platform} for {task_name}"
                    )
                except Exception as exc:
                    print(f"Benchmark failed: {exc}")
                    results.append(
                        BenchmarkResult(
                            task_name=task_name,
                            model_name=model_name,
                            platform=platform,
                            quantization="unknown",
                            tokens_per_second=0.0,
                            first_token_latency_ms=0.0,
                            total_latency_ms=0.0,
                            peak_memory_kb=0.0,
                            energy_per_token_mj=0.0,
                            success=False,
                            error_message=str(exc),
                        )
                    )

        self.results.extend(results)
        return results

    def _run_single_benchmark(
        self, model_name: str, platform: str, task_name: str, config: ProfilingConfig
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        task = self.tasks[task_name]

        # Create model (simplified - in real implementation would load actual model)
        model = QuantizedModel(
            name=model_name,
            size_mb=self._estimate_model_size(model_name),
            quantization=self._extract_quantization(model_name),
        )

        # Initialize profiler
        profiler = EdgeProfiler(
            platform=platform,
            device=self._get_platform_device(platform),
            connection="local" if platform in ["rpi_zero", "jetson_nano"] else "serial",
        )

        try:
            # Run profiling
            profile_results = profiler.profile_model(
                model=model,
                test_prompts=task.prompts,
                metrics=["latency", "memory", "power"],
                config=config,
            )

            # Extract metrics for benchmark result
            tokens_per_second = 0.0
            first_token_latency = 0.0
            total_latency = 0.0
            peak_memory = 0.0
            energy_per_token = 0.0

            if profile_results.latency_profile:
                tokens_per_second = profile_results.latency_profile.tokens_per_second
                first_token_latency = (
                    profile_results.latency_profile.first_token_latency_ms
                )
                total_latency = profile_results.latency_profile.total_latency_ms

            if profile_results.memory_profile:
                peak_memory = profile_results.memory_profile.peak_memory_kb

            if profile_results.power_profile:
                energy_per_token = profile_results.power_profile.energy_per_token_mj

            return BenchmarkResult(
                task_name=task_name,
                model_name=model_name,
                platform=platform,
                quantization=model.quantization,
                tokens_per_second=tokens_per_second,
                first_token_latency_ms=first_token_latency,
                total_latency_ms=total_latency,
                peak_memory_kb=peak_memory,
                energy_per_token_mj=energy_per_token,
                success=True,
            )

        except Exception as e:
            return BenchmarkResult(
                task_name=task_name,
                model_name=model_name,
                platform=platform,
                quantization=model.quantization,
                tokens_per_second=0.0,
                first_token_latency_ms=0.0,
                total_latency_ms=0.0,
                peak_memory_kb=0.0,
                energy_per_token_mj=0.0,
                success=False,
                error_message=str(e),
            )

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size based on name."""
        # Simple heuristics based on model name
        if "125m" in model_name.lower():
            return 0.5
        elif "350m" in model_name.lower():
            return 1.4
        elif "1b" in model_name.lower() or "1.1b" in model_name.lower():
            return 2.8
        elif "7b" in model_name.lower():
            return 14.0
        else:
            return 2.0  # Default size

    def _extract_quantization(self, model_name: str) -> str:
        """Extract quantization info from model name."""
        if "2bit" in model_name.lower():
            return "2bit"
        elif "3bit" in model_name.lower():
            return "3bit"
        elif "4bit" in model_name.lower():
            return "4bit"
        elif "8bit" in model_name.lower():
            return "8bit"
        else:
            return "4bit"  # Default quantization

    def _get_platform_device(self, platform: str) -> Optional[str]:
        """Get default device path for platform."""
        device_map = {
            "esp32": "/dev/ttyUSB0",
            "stm32f4": "/dev/ttyACM0",
            "stm32f7": "/dev/ttyACM0",
            "rp2040": "/dev/ttyACM1",
            "rpi_zero": None,
            "jetson_nano": None,
        }
        return device_map.get(platform)

    def create_leaderboard(self, results: List[BenchmarkResult], output_path: str):
        """Create performance leaderboard from benchmark results."""
        if not results:
            return

        # Group results by task
        task_results = {}
        for result in results:
            if result.success:
                if result.task_name not in task_results:
                    task_results[result.task_name] = []
                task_results[result.task_name].append(result)

        # Generate markdown leaderboard
        markdown_content = ["# TinyML LLM Benchmark Leaderboard\n"]
        markdown_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for task_name, task_results_list in task_results.items():
            markdown_content.append(f"## {task_name.replace('_', ' ').title()}\n")

            # Sort by performance score
            sorted_results = sorted(
                task_results_list,
                key=lambda x: self._calculate_performance_score(x),
                reverse=True,
            )

            # Create table
            headers = [
                "Rank",
                "Model",
                "Platform",
                "Quantization",
                "Tokens/sec",
                "First Token (ms)",
                "Memory (KB)",
                "Energy (mJ/token)",
                "Score",
            ]

            markdown_content.append("| " + " | ".join(headers) + " |")
            markdown_content.append("|" + "|".join([" --- " for _ in headers]) + "|")

            for i, result in enumerate(sorted_results, 1):
                score = self._calculate_performance_score(result)
                row = [
                    str(i),
                    result.model_name,
                    result.platform,
                    result.quantization,
                    f"{result.tokens_per_second:.2f}",
                    f"{result.first_token_latency_ms:.1f}",
                    f"{result.peak_memory_kb:.1f}",
                    f"{result.energy_per_token_mj:.2f}",
                    f"{score:.2f}",
                ]
                markdown_content.append("| " + " | ".join(row) + " |")

            markdown_content.append("\n")

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(markdown_content))

    def _calculate_performance_score(self, result: BenchmarkResult) -> float:
        """Calculate composite performance score."""
        if not result.success:
            return 0.0

        score = 0.0
        weight_sum = 0.0

        # Throughput component (higher is better)
        if result.tokens_per_second > 0:
            score += result.tokens_per_second * 0.4
            weight_sum += 0.4

        # Latency component (lower is better, invert)
        if result.first_token_latency_ms > 0:
            latency_score = 1000.0 / result.first_token_latency_ms
            score += latency_score * 0.3
            weight_sum += 0.3

        # Memory efficiency (lower is better, invert)
        if result.peak_memory_kb > 0:
            memory_score = 1000.0 / result.peak_memory_kb
            score += memory_score * 0.2
            weight_sum += 0.2

        # Energy efficiency (lower is better, invert)
        if result.energy_per_token_mj > 0:
            energy_score = 10.0 / result.energy_per_token_mj
            score += energy_score * 0.1
            weight_sum += 0.1

        return score / weight_sum if weight_sum > 0 else 0.0

    def export_results(self, output_path: str, format: str = "json"):
        """Export benchmark results to file."""
        if format == "json":
            results_data = [asdict(result) for result in self.results]
            with open(output_path, "w") as f:
                json.dump(results_data, f, indent=2)

        elif format == "csv":
            import csv

            if self.results:
                fieldnames = list(asdict(self.results[0]).keys())
                with open(output_path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in self.results:
                        writer.writerow(asdict(result))

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and provide insights."""
        if not self.results:
            return {"error": "No results to analyze"}

        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            return {"error": "No successful benchmark runs"}

        analysis = {
            "total_benchmarks": len(self.results),
            "successful_benchmarks": len(successful_results),
            "success_rate": len(successful_results) / len(self.results),
            "platforms_tested": list(set(r.platform for r in successful_results)),
            "models_tested": list(set(r.model_name for r in successful_results)),
            "tasks_tested": list(set(r.task_name for r in successful_results)),
        }

        # Performance statistics
        tokens_per_sec = [
            r.tokens_per_second for r in successful_results if r.tokens_per_second > 0
        ]
        if tokens_per_sec:
            analysis["performance_stats"] = {
                "avg_tokens_per_second": statistics.mean(tokens_per_sec),
                "max_tokens_per_second": max(tokens_per_sec),
                "min_tokens_per_second": min(tokens_per_sec),
                "median_tokens_per_second": statistics.median(tokens_per_sec),
            }

        # Memory statistics
        memory_usage = [
            r.peak_memory_kb for r in successful_results if r.peak_memory_kb > 0
        ]
        if memory_usage:
            analysis["memory_stats"] = {
                "avg_peak_memory_kb": statistics.mean(memory_usage),
                "max_peak_memory_kb": max(memory_usage),
                "min_peak_memory_kb": min(memory_usage),
                "median_peak_memory_kb": statistics.median(memory_usage),
            }

        # Find best performers
        if successful_results:
            best_performer = max(
                successful_results, key=self._calculate_performance_score
            )
            analysis["best_performer"] = {
                "model": best_performer.model_name,
                "platform": best_performer.platform,
                "task": best_performer.task_name,
                "score": self._calculate_performance_score(best_performer),
            }

        return analysis
