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


class StandardBenchmarks:
    """Standard benchmark suite for edge LLM evaluation."""
    
    def __init__(self):
        """Initialize with standard benchmark tasks."""
        self.benchmark_tasks = self._create_standard_tasks()
        self.results_history: List[BenchmarkResult] = []
    
    def _create_standard_tasks(self) -> List[BenchmarkTask]:
        """Create standard benchmark tasks."""
        return [
            BenchmarkTask(
                name="text_generation",
                prompts=[
                    "Write a simple Python function to calculate fibonacci numbers",
                    "Explain how neural networks work in simple terms",
                    "Create a shopping list for a healthy dinner",
                    "Write a short poem about artificial intelligence",
                    "Describe the process of photosynthesis"
                ],
                expected_output_length=100,
                description="General text generation benchmark"
            ),
            
            BenchmarkTask(
                name="code_generation", 
                prompts=[
                    "def binary_search(arr, target):",
                    "class LinkedList:",
                    "# Sort a list of numbers\ndef sort_numbers(numbers):",
                    "import json\n# Parse JSON data",
                    "# Calculate factorial recursively\ndef factorial(n):"
                ],
                expected_output_length=50,
                description="Code generation and completion"
            ),
            
            BenchmarkTask(
                name="question_answering",
                prompts=[
                    "What is the capital of France?",
                    "How do computers process information?",
                    "What are the benefits of renewable energy?", 
                    "Explain the water cycle",
                    "What is machine learning?"
                ],
                expected_output_length=75,
                description="Question answering benchmark"
            ),
            
            BenchmarkTask(
                name="summarization",
                prompts=[
                    "Summarize this text: Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of \"intelligent agents\": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
                    "Summarize: Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is a natural phenomenon, scientific evidence shows that human activities have been the main driver since the 1800s.",
                    "Summarize: The Internet is a global system of interconnected computer networks that use the Internet protocol suite to link devices worldwide."
                ],
                expected_output_length=30,
                description="Text summarization benchmark"
            ),
            
            BenchmarkTask(
                name="edge_specific",
                prompts=[
                    "Optimize this code for low memory usage",
                    "How to reduce power consumption in embedded systems?",
                    "List 3 ways to make this algorithm faster on microcontrollers",
                    "Explain quantization in neural networks",
                    "What are the challenges of running AI on edge devices?"
                ],
                expected_output_length=60,
                description="Edge computing and optimization specific tasks"
            )
        ]
    
    def run_tiny_ml_perf(
        self,
        models: List[str],
        platforms: List[str], 
        tasks: Optional[List[str]] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run TinyML performance benchmark suite.
        
        Args:
            models: List of model paths or names to benchmark
            platforms: List of target platforms
            tasks: List of task names to run (default: all tasks)
            output_file: Output file for results
            
        Returns:
            Dictionary with benchmark results
        """
        if tasks is None:
            tasks = [task.name for task in self.benchmark_tasks]
        
        benchmark_results = []
        total_combinations = len(models) * len(platforms) * len(tasks)
        current = 0
        
        print(f"Running {total_combinations} benchmark combinations...")
        
        for model_path in models:
            for platform in platforms:
                for task_name in tasks:
                    current += 1
                    print(f"[{current}/{total_combinations}] {task_name} - {platform} - {Path(model_path).name}")
                    
                    try:
                        result = self._run_single_benchmark(model_path, platform, task_name)
                        benchmark_results.append(result)
                        self.results_history.append(result)
                    
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
                        error_result = BenchmarkResult(
                            task_name=task_name,
                            model_name=Path(model_path).name,
                            platform=platform,
                            quantization="unknown",
                            tokens_per_second=0.0,
                            first_token_latency_ms=0.0,
                            total_latency_ms=0.0,
                            peak_memory_kb=0.0,
                            energy_per_token_mj=0.0,
                            success=False,
                            error_message=str(e),
                            timestamp=time.time()
                        )
                        benchmark_results.append(error_result)
        
        # Create comprehensive results
        results = {
            "benchmark_suite": "TinyML Performance v1.0",
            "timestamp": time.time(),
            "total_tests": len(benchmark_results),
            "successful_tests": sum(1 for r in benchmark_results if r.success),
            "results": [asdict(r) for r in benchmark_results],
            "summary": self._generate_summary(benchmark_results)
        }
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results
    
    def _run_single_benchmark(
        self, 
        model_path: str, 
        platform: str, 
        task_name: str
    ) -> BenchmarkResult:
        """Run a single benchmark combination."""
        # Find the benchmark task
        task = next((t for t in self.benchmark_tasks if t.name == task_name), None)
        if not task:
            raise ValueError(f"Unknown benchmark task: {task_name}")
        
        # Load model
        model = QuantizedModel.from_file(model_path)
        
        # Initialize profiler
        profiler = EdgeProfiler(platform=platform, connection="local")
        
        # Configure profiling
        config = ProfilingConfig(
            measurement_iterations=3,  # Fewer iterations for benchmarking
            warmup_iterations=1,
            timeout_seconds=task.timeout_seconds
        )
        
        # Run profiling
        start_time = time.time()
        
        try:
            with profiler:
                results = profiler.profile_model(
                    model=model,
                    test_prompts=task.prompts[:3],  # Use first 3 prompts
                    metrics=["latency", "memory", "power"],
                    config=config
                )
            
            # Extract metrics
            tokens_per_second = 0.0
            first_token_latency = 0.0
            total_latency = 0.0
            peak_memory = 0.0
            energy_per_token = 0.0
            
            if results.latency_profile:
                tokens_per_second = results.latency_profile.tokens_per_second
                first_token_latency = results.latency_profile.first_token_latency_ms
                total_latency = results.latency_profile.total_latency_ms
            
            if results.memory_profile:
                peak_memory = results.memory_profile.peak_memory_kb
            
            if results.power_profile:
                energy_per_token = results.power_profile.energy_per_token_mj
            
            return BenchmarkResult(
                task_name=task_name,
                model_name=model.name,
                platform=platform,
                quantization=model.quantization.value,
                tokens_per_second=tokens_per_second,
                first_token_latency_ms=first_token_latency,
                total_latency_ms=total_latency,
                peak_memory_kb=peak_memory,
                energy_per_token_mj=energy_per_token,
                success=True,
                timestamp=start_time
            )
            
        except Exception as e:
            return BenchmarkResult(
                task_name=task_name,
                model_name=Path(model_path).name,
                platform=platform,
                quantization="unknown",
                tokens_per_second=0.0,
                first_token_latency_ms=0.0,
                total_latency_ms=0.0,
                peak_memory_kb=0.0,
                energy_per_token_mj=0.0,
                success=False,
                error_message=str(e),
                timestamp=start_time
            )
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful benchmark runs"}
        
        # Platform performance summary
        platform_stats = {}
        for platform in set(r.platform for r in successful_results):
            platform_results = [r for r in successful_results if r.platform == platform]
            
            platform_stats[platform] = {
                "count": len(platform_results),
                "avg_tokens_per_second": statistics.mean(r.tokens_per_second for r in platform_results),
                "avg_first_token_latency": statistics.mean(r.first_token_latency_ms for r in platform_results),
                "avg_peak_memory_kb": statistics.mean(r.peak_memory_kb for r in platform_results),
                "avg_energy_per_token": statistics.mean(r.energy_per_token_mj for r in platform_results)
            }
        
        # Model performance summary
        model_stats = {}
        for model in set(r.model_name for r in successful_results):
            model_results = [r for r in successful_results if r.model_name == model]
            
            model_stats[model] = {
                "count": len(model_results),
                "avg_tokens_per_second": statistics.mean(r.tokens_per_second for r in model_results),
                "avg_peak_memory_kb": statistics.mean(r.peak_memory_kb for r in model_results),
                "quantization": model_results[0].quantization if model_results else "unknown"
            }
        
        # Overall statistics
        overall_stats = {
            "total_successful_runs": len(successful_results),
            "avg_tokens_per_second": statistics.mean(r.tokens_per_second for r in successful_results),
            "median_tokens_per_second": statistics.median(r.tokens_per_second for r in successful_results),
            "best_performance": {
                "tokens_per_second": max(r.tokens_per_second for r in successful_results),
                "model": max(successful_results, key=lambda r: r.tokens_per_second).model_name,
                "platform": max(successful_results, key=lambda r: r.tokens_per_second).platform
            },
            "most_memory_efficient": {
                "peak_memory_kb": min(r.peak_memory_kb for r in successful_results if r.peak_memory_kb > 0),
                "model": min(successful_results, key=lambda r: r.peak_memory_kb if r.peak_memory_kb > 0 else float('inf')).model_name,
                "platform": min(successful_results, key=lambda r: r.peak_memory_kb if r.peak_memory_kb > 0 else float('inf')).platform
            }
        }
        
        return {
            "overall": overall_stats,
            "by_platform": platform_stats,
            "by_model": model_stats
        }
    
    def create_leaderboard(self, results: Dict[str, Any], output_file: str):
        """Create markdown leaderboard from benchmark results."""
        if "summary" not in results:
            print("No summary data available for leaderboard")
            return
        
        summary = results["summary"]
        
        # Create markdown content
        markdown_content = [
            "# TinyML Edge Profiler Benchmark Results",
            "",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Tests: {results['total_tests']}",
            f"Successful Tests: {results['successful_tests']}",
            "",
            "## Overall Performance Leaders",
            ""
        ]
        
        if "overall" in summary:
            overall = summary["overall"]
            markdown_content.extend([
                f"**Best Performance:** {overall['best_performance']['tokens_per_second']:.1f} tok/s",
                f"- Model: {overall['best_performance']['model']}",
                f"- Platform: {overall['best_performance']['platform']}",
                "",
                f"**Most Memory Efficient:** {overall['most_memory_efficient']['peak_memory_kb']:.0f} KB peak",
                f"- Model: {overall['most_memory_efficient']['model']}",
                f"- Platform: {overall['most_memory_efficient']['platform']}",
                ""
            ])
        
        # Platform comparison table
        if "by_platform" in summary:
            markdown_content.extend([
                "## Platform Performance Comparison",
                "",
                "| Platform | Avg Tok/s | Avg First Token (ms) | Avg Memory (KB) | Avg Energy (mJ/tok) |",
                "|----------|-----------|----------------------|-----------------|-------------------|"
            ])
            
            for platform, stats in summary["by_platform"].items():
                markdown_content.append(
                    f"| {platform} | {stats['avg_tokens_per_second']:.1f} | "
                    f"{stats['avg_first_token_latency']:.1f} | "
                    f"{stats['avg_peak_memory_kb']:.0f} | "
                    f"{stats['avg_energy_per_token']:.2f} |"
                )
            
            markdown_content.append("")
        
        # Model comparison table
        if "by_model" in summary:
            markdown_content.extend([
                "## Model Performance Comparison",
                "",
                "| Model | Quantization | Avg Tok/s | Avg Memory (KB) |",
                "|-------|--------------|-----------|-----------------|"
            ])
            
            for model, stats in summary["by_model"].items():
                markdown_content.append(
                    f"| {model} | {stats['quantization']} | "
                    f"{stats['avg_tokens_per_second']:.1f} | "
                    f"{stats['avg_peak_memory_kb']:.0f} |"
                )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(markdown_content))
        
        print(f"Leaderboard saved to {output_file}")
    
    def compare_models(
        self,
        model_paths: List[str],
        platform: str,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple models on a single platform."""
        if tasks is None:
            tasks = [task.name for task in self.benchmark_tasks]
        
        comparison_results = []
        
        for model_path in model_paths:
            model_results = {}
            
            for task_name in tasks:
                try:
                    result = self._run_single_benchmark(model_path, platform, task_name)
                    model_results[task_name] = result
                except Exception as e:
                    print(f"Failed to benchmark {model_path} on {task_name}: {e}")
            
            comparison_results.append({
                "model_path": model_path,
                "model_name": Path(model_path).name,
                "results": model_results
            })
        
        return {
            "platform": platform,
            "models": comparison_results,
            "timestamp": time.time()
        }
    
    def get_task_names(self) -> List[str]:
        """Get list of available benchmark task names."""
        return [task.name for task in self.benchmark_tasks]
    
    def get_task_description(self, task_name: str) -> str:
        """Get description of a specific benchmark task."""
        task = next((t for t in self.benchmark_tasks if t.name == task_name), None)
        return task.description if task else "Task not found"