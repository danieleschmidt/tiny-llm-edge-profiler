"""
Analysis tools for profiling results and performance optimization.
"""

import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .results import ProfileResults, LatencyProfile, MemoryProfile, PowerProfile


@dataclass 
class BottleneckInfo:
    """Information about a performance bottleneck."""
    component: str
    impact: float  # Impact percentage (0.0 to 1.0)
    description: str
    recommendation: str


@dataclass
class ComparisonResult:
    """Result of comparing multiple profiling runs."""
    model_name: str
    platform: str
    profile: ProfileResults
    relative_performance: float  # Relative to best performer


class MetricsAnalyzer:
    """Analyze profiling results and identify performance characteristics."""
    
    def __init__(self):
        self.results_history: List[ProfileResults] = []
    
    def analyze(self, results: ProfileResults) -> Dict[str, Any]:
        """
        Comprehensive analysis of profiling results.
        
        Args:
            results: ProfileResults to analyze
            
        Returns:
            Dictionary containing analysis metrics
        """
        self.results_history.append(results)
        
        analysis = {
            "platform": results.platform,
            "model_name": results.model_name,
            "model_size_mb": results.model_size_mb,
            "quantization": results.quantization
        }
        
        # Latency analysis
        if results.latency_profile:
            latency = results.latency_profile
            analysis.update({
                "first_token_ms": latency.first_token_latency_ms,
                "inter_token_ms": latency.inter_token_latency_ms,
                "tokens_per_second": latency.tokens_per_second,
                "total_latency_ms": latency.total_latency_ms,
                "latency_variability": latency.latency_std_ms / latency.total_latency_ms if latency.total_latency_ms > 0 else 0.0
            })
        
        # Memory analysis
        if results.memory_profile:
            memory = results.memory_profile
            analysis.update({
                "peak_memory_kb": memory.peak_memory_kb,
                "memory_usage_kb": memory.memory_usage_kb,
                "memory_overhead_kb": memory.peak_memory_kb - memory.baseline_memory_kb,
                "memory_efficiency": memory.memory_efficiency_tokens_per_kb
            })
            
            # Calculate tokens per KB if we have latency data
            if results.latency_profile and results.latency_profile.tokens_per_second > 0:
                tokens_per_kb = results.latency_profile.tokens_per_second / (memory.memory_usage_kb / 1000)
                analysis["tokens_per_kb"] = tokens_per_kb
        
        # Power analysis
        if results.power_profile:
            power = results.power_profile
            analysis.update({
                "idle_power_mw": power.idle_power_mw,
                "active_power_mw": power.active_power_mw,
                "peak_power_mw": power.peak_power_mw,
                "energy_per_token_mj": power.energy_per_token_mj,
                "power_efficiency": 1000.0 / power.energy_per_token_mj if power.energy_per_token_mj > 0 else 0.0  # tokens per joule
            })
        
        return analysis
    
    def find_bottlenecks(self, results: ProfileResults) -> List[BottleneckInfo]:
        """
        Identify performance bottlenecks in the profiling results.
        
        Args:
            results: ProfileResults to analyze
            
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Latency bottlenecks
        if results.latency_profile:
            latency = results.latency_profile
            
            # First token latency is high
            if latency.first_token_latency_ms > 200:
                bottlenecks.append(BottleneckInfo(
                    component="First Token Generation",
                    impact=0.8,
                    description=f"First token latency is {latency.first_token_latency_ms:.1f}ms, indicating slow model loading or initialization",
                    recommendation="Optimize model loading, reduce initialization overhead, or implement model caching"
                ))
            
            # Inter-token latency is high
            if latency.inter_token_latency_ms > 50:
                bottlenecks.append(BottleneckInfo(
                    component="Token Generation Speed",
                    impact=0.7,
                    description=f"Inter-token latency is {latency.inter_token_latency_ms:.1f}ms, limiting throughput",
                    recommendation="Optimize inference kernel, use hardware acceleration, or increase CPU frequency"
                ))
                
            # High latency variability
            if latency.latency_std_ms / latency.total_latency_ms > 0.2:
                bottlenecks.append(BottleneckInfo(
                    component="Latency Consistency",
                    impact=0.5,
                    description="High latency variability indicates inconsistent performance",
                    recommendation="Stabilize system load, disable power management, or optimize memory allocation"
                ))
        
        # Memory bottlenecks
        if results.memory_profile:
            memory = results.memory_profile
            memory_overhead = memory.peak_memory_kb - memory.baseline_memory_kb
            
            # High memory usage
            if memory.peak_memory_kb > 400:  # >400KB is high for microcontrollers
                bottlenecks.append(BottleneckInfo(
                    component="Memory Usage",
                    impact=0.9,
                    description=f"Peak memory usage is {memory.peak_memory_kb:.1f}KB, may exceed device limits",
                    recommendation="Use more aggressive quantization, reduce batch size, or implement memory pooling"
                ))
            
            # Low memory efficiency
            if memory.memory_efficiency_tokens_per_kb < 0.1:
                bottlenecks.append(BottleneckInfo(
                    component="Memory Efficiency",
                    impact=0.6,
                    description="Low memory efficiency indicates poor memory utilization",
                    recommendation="Optimize memory layout, use activation checkpointing, or reduce model parameters"
                ))
        
        # Power bottlenecks
        if results.power_profile:
            power = results.power_profile
            
            # High power consumption
            if power.active_power_mw > 200:
                bottlenecks.append(BottleneckInfo(
                    component="Power Consumption",
                    impact=0.7,
                    description=f"Active power consumption is {power.active_power_mw:.1f}mW, limiting battery life",
                    recommendation="Reduce CPU frequency, use power-efficient algorithms, or implement dynamic power scaling"
                ))
                
            # Poor energy efficiency
            if power.energy_per_token_mj > 5.0:
                bottlenecks.append(BottleneckInfo(
                    component="Energy Efficiency", 
                    impact=0.8,
                    description=f"Energy per token is {power.energy_per_token_mj:.1f}mJ, indicating poor efficiency",
                    recommendation="Optimize inference algorithm, use lower precision, or implement energy-aware scheduling"
                ))
        
        # Sort by impact (highest first)
        bottlenecks.sort(key=lambda x: x.impact, reverse=True)
        return bottlenecks
    
    def generate_report(self, results: ProfileResults, output_path: str):
        """Generate HTML report with analysis and visualizations."""
        analysis = self.analyze(results)
        bottlenecks = self.find_bottlenecks(results)
        
        if not HAS_PLOTLY:
            print("Warning: Plotly not available, skipping visualization")
            return
        
        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Latency Breakdown", "Memory Usage", "Power Profile", "Performance Summary"),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Latency breakdown
        if results.latency_profile:
            latency = results.latency_profile
            fig.add_trace(
                go.Bar(
                    x=["First Token", "Inter-token (avg)", "Total"],
                    y=[latency.first_token_latency_ms, latency.inter_token_latency_ms, latency.total_latency_ms],
                    name="Latency (ms)"
                ),
                row=1, col=1
            )
        
        # Memory usage
        if results.memory_profile:
            memory = results.memory_profile
            fig.add_trace(
                go.Bar(
                    x=["Baseline", "Peak", "Average Usage"],
                    y=[memory.baseline_memory_kb, memory.peak_memory_kb, memory.memory_usage_kb],
                    name="Memory (KB)"
                ),
                row=1, col=2
            )
        
        # Power profile
        if results.power_profile:
            power = results.power_profile
            fig.add_trace(
                go.Bar(
                    x=["Idle", "Active", "Peak"],
                    y=[power.idle_power_mw, power.active_power_mw, power.peak_power_mw],
                    name="Power (mW)"
                ),
                row=2, col=1
            )
        
        # Performance summary scatter
        if results.latency_profile and results.memory_profile:
            fig.add_trace(
                go.Scatter(
                    x=[results.memory_profile.memory_usage_kb],
                    y=[results.latency_profile.tokens_per_second],
                    mode="markers",
                    marker=dict(size=20, color="red"),
                    name=f"{results.model_name} ({results.platform})"
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text=f"Performance Analysis: {results.model_name}")
        fig.write_html(output_path)


class ComparativeAnalyzer:
    """Compare performance across multiple models and platforms."""
    
    def __init__(self):
        self.comparisons: List[ComparisonResult] = []
    
    def compare(self, comparison_data: List[Tuple[str, str, ProfileResults]]) -> List[ComparisonResult]:
        """
        Compare multiple profiling results.
        
        Args:
            comparison_data: List of (model_name, platform, ProfileResults) tuples
            
        Returns:
            List of ComparisonResult objects with relative performance scores
        """
        if not comparison_data:
            return []
        
        # Calculate performance scores for each result
        scores = []
        for model_name, platform, results in comparison_data:
            score = self._calculate_performance_score(results)
            scores.append((model_name, platform, results, score))
        
        # Find best performer for relative scoring
        best_score = max(scores, key=lambda x: x[3])[3]
        
        # Create comparison results
        comparison_results = []
        for model_name, platform, results, score in scores:
            relative_performance = score / best_score if best_score > 0 else 0.0
            comparison_results.append(ComparisonResult(
                model_name=model_name,
                platform=platform,
                profile=results,
                relative_performance=relative_performance
            ))
        
        # Sort by performance (best first)
        comparison_results.sort(key=lambda x: x.relative_performance, reverse=True)
        self.comparisons = comparison_results
        
        return comparison_results
    
    def _calculate_performance_score(self, results: ProfileResults) -> float:
        """Calculate composite performance score."""
        score = 0.0
        weight_sum = 0.0
        
        # Latency component (higher tokens/sec is better)
        if results.latency_profile and results.latency_profile.tokens_per_second > 0:
            score += results.latency_profile.tokens_per_second * 0.4
            weight_sum += 0.4
        
        # Memory efficiency component (lower memory usage is better)
        if results.memory_profile and results.memory_profile.memory_usage_kb > 0:
            memory_efficiency = 1000.0 / results.memory_profile.memory_usage_kb  # Inverse of memory usage
            score += memory_efficiency * 0.3
            weight_sum += 0.3
        
        # Energy efficiency component (higher tokens per joule is better)
        if results.power_profile and results.power_profile.energy_per_token_mj > 0:
            energy_efficiency = 1000.0 / results.power_profile.energy_per_token_mj
            score += energy_efficiency * 0.3
            weight_sum += 0.3
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def plot_comparison(self, results: List[ComparisonResult], metrics: List[str], save_to: str):
        """Create comparison visualization."""
        if not results:
            return
            
        if not HAS_PLOTLY:
            print("Warning: Plotly not available, skipping visualization")
            return
        
        # Prepare data for plotting
        models = [f"{r.model_name} ({r.platform})" for r in results]
        
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics, 1):
            values = []
            
            for result in results:
                if metric == "latency" and result.profile.latency_profile:
                    values.append(result.profile.latency_profile.tokens_per_second)
                elif metric == "memory" and result.profile.memory_profile:
                    values.append(result.profile.memory_profile.memory_usage_kb)
                elif metric == "energy" and result.profile.power_profile:
                    values.append(result.profile.power_profile.energy_per_token_mj)
                else:
                    values.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.title(),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
        
        fig.update_layout(height=300 * len(metrics), title_text="Model Comparison")
        fig.write_html(save_to)
    
    def recommend_deployment(self, constraints: Dict[str, float]) -> List[ComparisonResult]:
        """
        Recommend best model-platform combinations based on constraints.
        
        Args:
            constraints: Dictionary of constraint limits (max_latency_ms, max_memory_kb, max_power_mw)
            
        Returns:
            List of ComparisonResult objects that meet constraints, sorted by performance
        """
        if not self.comparisons:
            return []
        
        recommended = []
        
        for result in self.comparisons:
            meets_constraints = True
            
            # Check latency constraint
            if "max_latency_ms" in constraints and result.profile.latency_profile:
                if result.profile.latency_profile.total_latency_ms > constraints["max_latency_ms"]:
                    meets_constraints = False
            
            # Check memory constraint
            if "max_memory_kb" in constraints and result.profile.memory_profile:
                if result.profile.memory_profile.peak_memory_kb > constraints["max_memory_kb"]:
                    meets_constraints = False
            
            # Check power constraint
            if "max_power_mw" in constraints and result.profile.power_profile:
                if result.profile.power_profile.active_power_mw > constraints["max_power_mw"]:
                    meets_constraints = False
            
            if meets_constraints:
                recommended.append(result)
        
        return recommended