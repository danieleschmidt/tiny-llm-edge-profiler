"""
Data structures and classes for storing and analyzing profiling results.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


@dataclass
class LatencyProfile:
    """Latency profiling results."""
    first_token_latency_ms: float
    inter_token_latency_ms: float
    total_latency_ms: float
    tokens_per_second: float
    latency_std_ms: float
    percentile_50_ms: float = 0.0
    percentile_90_ms: float = 0.0
    percentile_95_ms: float = 0.0
    percentile_99_ms: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.tokens_per_second == 0 and self.inter_token_latency_ms > 0:
            self.tokens_per_second = 1000.0 / self.inter_token_latency_ms


@dataclass
class MemoryProfile:
    """Memory usage profiling results."""
    baseline_memory_kb: float
    peak_memory_kb: float
    memory_usage_kb: float
    memory_efficiency_tokens_per_kb: float
    fragmentation_percent: float = 0.0
    gc_overhead_percent: float = 0.0
    stack_usage_kb: float = 0.0
    heap_usage_kb: float = 0.0


@dataclass
class PowerProfile:
    """Power consumption profiling results."""
    idle_power_mw: float
    active_power_mw: float
    peak_power_mw: float
    energy_per_token_mj: float
    total_energy_mj: float
    average_current_ma: float = 0.0
    voltage_v: float = 3.3
    thermal_info: Optional[Dict[str, float]] = None
    
    # Additional fields that may be added by power profiler
    avg_voltage_v: float = 3.3
    avg_current_ma: float = 0.0
    power_std_mw: float = 0.0
    sample_count: int = 0
    duration_s: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.average_current_ma == 0 and self.active_power_mw > 0:
            self.average_current_ma = self.active_power_mw / self.voltage_v


@dataclass
class AccuracyProfile:
    """Model accuracy and quality metrics."""
    perplexity: float
    bleu_score: float = 0.0
    rouge_score: Dict[str, float] = field(default_factory=dict)
    semantic_similarity: float = 0.0
    task_specific_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ThroughputProfile:
    """Throughput and performance metrics."""
    tokens_per_second: float
    requests_per_second: float
    batch_processing_efficiency: float = 1.0
    queue_time_ms: float = 0.0
    processing_time_ms: float = 0.0


class ProfileResults:
    """
    Comprehensive profiling results container.
    
    Aggregates all profiling metrics and provides analysis methods.
    """
    
    def __init__(
        self,
        platform: str,
        model_name: str,
        model_size_mb: float,
        quantization: str,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize ProfileResults.
        
        Args:
            platform: Target platform name
            model_name: Name of the profiled model
            model_size_mb: Model size in megabytes
            quantization: Quantization level used
            timestamp: Profiling timestamp
        """
        self.platform = platform
        self.model_name = model_name
        self.model_size_mb = model_size_mb
        self.quantization = quantization
        self.timestamp = timestamp or datetime.now(timezone.utc)
        
        # Profile data
        self.latency_profile: Optional[LatencyProfile] = None
        self.memory_profile: Optional[MemoryProfile] = None
        self.power_profile: Optional[PowerProfile] = None
        self.accuracy_profile: Optional[AccuracyProfile] = None
        self.throughput_profile: Optional[ThroughputProfile] = None
        
        # Additional metadata
        self.test_configuration: Dict[str, Any] = {}
        self.environment_info: Dict[str, Any] = {}
        self.raw_measurements: Dict[str, List[float]] = {}
        
    def add_latency_profile(self, profile: LatencyProfile) -> None:
        """Add latency profiling results."""
        self.latency_profile = profile
        
        # Update throughput profile
        if not self.throughput_profile:
            self.throughput_profile = ThroughputProfile(
                tokens_per_second=profile.tokens_per_second,
                requests_per_second=0.0
            )
        else:
            self.throughput_profile.tokens_per_second = profile.tokens_per_second
    
    def add_memory_profile(self, profile: MemoryProfile) -> None:
        """Add memory profiling results."""
        self.memory_profile = profile
    
    def add_power_profile(self, profile: PowerProfile) -> None:
        """Add power profiling results."""
        self.power_profile = profile
    
    def add_accuracy_profile(self, profile: AccuracyProfile) -> None:
        """Add accuracy profiling results."""
        self.accuracy_profile = profile
    
    def add_raw_measurements(self, metric_name: str, measurements: List[float]) -> None:
        """Add raw measurement data for analysis."""
        self.raw_measurements[metric_name] = measurements
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all profiling results."""
        summary = {
            "platform": self.platform,
            "model_name": self.model_name,
            "model_size_mb": self.model_size_mb,
            "quantization": self.quantization,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.latency_profile:
            summary["latency"] = {
                "first_token_ms": self.latency_profile.first_token_latency_ms,
                "inter_token_ms": self.latency_profile.inter_token_latency_ms,
                "tokens_per_second": self.latency_profile.tokens_per_second,
                "total_latency_ms": self.latency_profile.total_latency_ms
            }
        
        if self.memory_profile:
            summary["memory"] = {
                "peak_memory_kb": self.memory_profile.peak_memory_kb,
                "memory_usage_kb": self.memory_profile.memory_usage_kb,
                "efficiency_tokens_per_kb": self.memory_profile.memory_efficiency_tokens_per_kb
            }
        
        if self.power_profile:
            summary["power"] = {
                "active_power_mw": self.power_profile.active_power_mw,
                "energy_per_token_mj": self.power_profile.energy_per_token_mj,
                "total_energy_mj": self.power_profile.total_energy_mj
            }
        
        if self.accuracy_profile:
            summary["accuracy"] = {
                "perplexity": self.accuracy_profile.perplexity,
                "bleu_score": self.accuracy_profile.bleu_score
            }
        
        return summary
    
    def calculate_efficiency_score(self) -> float:
        """
        Calculate overall efficiency score (0-100).
        
        Combines latency, memory, and power efficiency into a single score.
        """
        scores = []
        
        # Latency efficiency (higher tokens/sec is better)
        if self.latency_profile:
            # Normalize to typical edge device performance (1-20 tokens/sec)
            latency_score = min(100, (self.latency_profile.tokens_per_second / 20.0) * 100)
            scores.append(latency_score)
        
        # Memory efficiency (lower usage is better)
        if self.memory_profile:
            # Penalize high memory usage (>2MB is concerning for edge)
            memory_usage_mb = self.memory_profile.peak_memory_kb / 1024
            memory_score = max(0, 100 - (memory_usage_mb / 2.0) * 50)
            scores.append(memory_score)
        
        # Power efficiency (lower power is better)
        if self.power_profile:
            # Penalize high power consumption (>200mW is concerning)
            power_score = max(0, 100 - (self.power_profile.active_power_mw / 200.0) * 50)
            scores.append(power_score)
        
        return np.mean(scores) if scores else 0.0
    
    def compare_with(self, other: "ProfileResults") -> Dict[str, float]:
        """
        Compare this profile with another and return relative performance ratios.
        
        Args:
            other: Another ProfileResults to compare against
            
        Returns:
            Dictionary with performance ratios (>1 means this is better)
        """
        comparison = {}
        
        # Compare latency
        if self.latency_profile and other.latency_profile:
            comparison["tokens_per_second_ratio"] = (
                self.latency_profile.tokens_per_second / other.latency_profile.tokens_per_second
            )
            comparison["latency_improvement"] = (
                other.latency_profile.total_latency_ms / self.latency_profile.total_latency_ms
            )
        
        # Compare memory usage
        if self.memory_profile and other.memory_profile:
            comparison["memory_efficiency_ratio"] = (
                other.memory_profile.peak_memory_kb / self.memory_profile.peak_memory_kb
            )
        
        # Compare power consumption
        if self.power_profile and other.power_profile:
            comparison["power_efficiency_ratio"] = (
                other.power_profile.active_power_mw / self.power_profile.active_power_mw
            )
            comparison["energy_efficiency_ratio"] = (
                other.power_profile.energy_per_token_mj / self.power_profile.energy_per_token_mj
            )
        
        return comparison
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on profiling results."""
        recommendations = []
        
        # Latency recommendations
        if self.latency_profile:
            if self.latency_profile.tokens_per_second < 5:
                recommendations.append("Consider more aggressive quantization (2-bit) to improve latency")
            
            if self.latency_profile.first_token_latency_ms > 200:
                recommendations.append("First token latency is high - optimize model loading or use model caching")
        
        # Memory recommendations
        if self.memory_profile:
            memory_usage_mb = self.memory_profile.peak_memory_kb / 1024
            
            if memory_usage_mb > 1.5:
                recommendations.append("High memory usage - consider smaller context length or 2-bit quantization")
            
            if self.memory_profile.fragmentation_percent > 20:
                recommendations.append("High memory fragmentation - consider memory pool optimization")
        
        # Power recommendations
        if self.power_profile:
            if self.power_profile.active_power_mw > 150:
                recommendations.append("High power consumption - consider CPU frequency scaling or sleep modes")
            
            if self.power_profile.energy_per_token_mj > 5:
                recommendations.append("High energy per token - optimize inference loop or reduce model complexity")
        
        # Platform-specific recommendations
        if self.platform in ["esp32", "esp32s3"] and self.memory_profile:
            if self.memory_profile.peak_memory_kb > 400:
                recommendations.append("ESP32: Enable PSRAM usage to handle larger models")
        
        elif self.platform.startswith("stm32") and self.memory_profile:
            if self.memory_profile.peak_memory_kb > 150:
                recommendations.append("STM32: Model too large for typical SRAM - consider flash execution or smaller model")
        
        return recommendations
    
    def export_json(self, output_path: Union[str, Path]) -> None:
        """Export results to JSON file."""
        output_path = Path(output_path)
        
        data = {
            "metadata": {
                "platform": self.platform,
                "model_name": self.model_name,
                "model_size_mb": self.model_size_mb,
                "quantization": self.quantization,
                "timestamp": self.timestamp.isoformat(),
                "test_configuration": self.test_configuration,
                "environment_info": self.environment_info
            },
            "profiles": {},
            "raw_measurements": self.raw_measurements,
            "summary": self.get_summary(),
            "efficiency_score": self.calculate_efficiency_score(),
            "recommendations": self.get_recommendations()
        }
        
        # Add profile data
        if self.latency_profile:
            data["profiles"]["latency"] = self.latency_profile.__dict__
        
        if self.memory_profile:
            data["profiles"]["memory"] = self.memory_profile.__dict__
        
        if self.power_profile:
            data["profiles"]["power"] = self.power_profile.__dict__
        
        if self.accuracy_profile:
            data["profiles"]["accuracy"] = self.accuracy_profile.__dict__
        
        if self.throughput_profile:
            data["profiles"]["throughput"] = self.throughput_profile.__dict__
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, input_path: Union[str, Path]) -> "ProfileResults":
        """Load results from JSON file."""
        input_path = Path(input_path)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        metadata = data["metadata"]
        timestamp = datetime.fromisoformat(metadata["timestamp"])
        
        # Create ProfileResults instance
        results = cls(
            platform=metadata["platform"],
            model_name=metadata["model_name"],
            model_size_mb=metadata["model_size_mb"],
            quantization=metadata["quantization"],
            timestamp=timestamp
        )
        
        results.test_configuration = metadata.get("test_configuration", {})
        results.environment_info = metadata.get("environment_info", {})
        results.raw_measurements = data.get("raw_measurements", {})
        
        # Load profile data
        profiles = data.get("profiles", {})
        
        if "latency" in profiles:
            results.latency_profile = LatencyProfile(**profiles["latency"])
        
        if "memory" in profiles:
            results.memory_profile = MemoryProfile(**profiles["memory"])
        
        if "power" in profiles:
            results.power_profile = PowerProfile(**profiles["power"])
        
        if "accuracy" in profiles:
            results.accuracy_profile = AccuracyProfile(**profiles["accuracy"])
        
        if "throughput" in profiles:
            results.throughput_profile = ThroughputProfile(**profiles["throughput"])
        
        return results
    
    def export_csv(self, output_path: Union[str, Path]) -> None:
        """Export summary results to CSV format."""
        import csv
        
        output_path = Path(output_path)
        summary = self.get_summary()
        
        # Flatten nested dictionaries
        flattened = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{key}_{subkey}"] = subvalue
            else:
                flattened[key] = value
        
        # Add efficiency score and recommendations count
        flattened["efficiency_score"] = self.calculate_efficiency_score()
        flattened["num_recommendations"] = len(self.get_recommendations())
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flattened.keys())
            writer.writeheader()
            writer.writerow(flattened)
    
    def __repr__(self) -> str:
        """String representation of ProfileResults."""
        summary = self.get_summary()
        
        lines = [
            f"ProfileResults({self.model_name} on {self.platform})",
            f"  Model: {self.model_size_mb:.1f}MB {self.quantization} quantization"
        ]
        
        if "latency" in summary:
            lat = summary["latency"]
            lines.append(f"  Latency: {lat['tokens_per_second']:.1f} tok/s, {lat['first_token_ms']:.0f}ms first token")
        
        if "memory" in summary:
            mem = summary["memory"]
            lines.append(f"  Memory: {mem['peak_memory_kb']:.0f}KB peak")
        
        if "power" in summary:
            pwr = summary["power"]
            lines.append(f"  Power: {pwr['active_power_mw']:.0f}mW, {pwr['energy_per_token_mj']:.1f}mJ/token")
        
        lines.append(f"  Efficiency Score: {self.calculate_efficiency_score():.1f}/100")
        
        return "\n".join(lines)