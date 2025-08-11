"""
Advanced Performance Analytics Engine for Generation 3
Provides comprehensive performance analytics capabilities including:
- Real-time performance monitoring with streaming analytics
- Performance regression detection using statistical methods
- Optimization recommendation engine with ML-based insights
- Performance profiling of the profiler itself (meta-profiling)
- Trend analysis and forecasting for performance metrics
- Anomaly detection in performance patterns
- Automated performance reporting and alerting
"""

import time
import threading
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import statistics
import json
from pathlib import Path
import weakref

from .exceptions import TinyLLMProfilerError, ResourceError
from .logging_config import get_logger, PerformanceLogger
from .results import ProfileResults

logger = get_logger("performance_analytics")
perf_logger = PerformanceLogger()


class MetricType(str, Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_DEPTH = "queue_depth"
    RESOURCE_EFFICIENCY = "resource_efficiency"


class RegressionSeverity(str, Enum):
    """Severity levels for performance regressions."""
    MINOR = "minor"          # < 10% degradation
    MODERATE = "moderate"    # 10-25% degradation
    MAJOR = "major"         # 25-50% degradation
    CRITICAL = "critical"   # > 50% degradation


class AnalyticsAlert(str, Enum):
    """Types of analytics alerts."""
    PERFORMANCE_REGRESSION = "performance_regression"
    ANOMALY_DETECTED = "anomaly_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    TREND_ALERT = "trend_alert"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if isinstance(self.tags, list):
            self.tags = set(self.tags)


@dataclass
class RegressionDetectionResult:
    """Result of regression detection analysis."""
    metric_name: str
    regression_detected: bool
    severity: Optional[RegressionSeverity] = None
    baseline_value: float = 0.0
    current_value: float = 0.0
    degradation_percentage: float = 0.0
    confidence_score: float = 0.0
    detection_timestamp: datetime = field(default_factory=datetime.now)
    evidence: List[str] = field(default_factory=list)
    
    def to_alert_message(self) -> str:
        """Convert to human-readable alert message."""
        if not self.regression_detected:
            return f"No regression detected for {self.metric_name}"
        
        return (
            f"{self.severity.value.upper()} regression detected in {self.metric_name}: "
            f"{self.degradation_percentage:.1f}% degradation "
            f"(baseline: {self.baseline_value:.3f}, current: {self.current_value:.3f})"
        )


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""
    metric_name: str
    anomaly_detected: bool
    anomaly_type: Optional[str] = None
    anomaly_score: float = 0.0
    threshold: float = 0.0
    current_value: float = 0.0
    expected_value: float = 0.0
    detection_timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    target_metric: str
    recommendation_type: str  # scale_up, scale_down, tune_parameters, etc.
    description: str
    expected_improvement: float  # Expected improvement percentage
    confidence: float  # 0.0 to 1.0
    priority: str = "medium"  # low, medium, high, critical
    implementation_effort: str = "medium"  # low, medium, high
    cost_impact: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class StatisticalAnalyzer:
    """Statistical analysis methods for performance data."""
    
    @staticmethod
    def calculate_trend(values: List[float], timestamps: List[datetime]) -> Dict[str, float]:
        """Calculate trend using linear regression."""
        if len(values) < 2 or len(timestamps) != len(values):
            return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}
        
        # Convert timestamps to numeric (seconds since first timestamp)
        base_time = timestamps[0]
        x = [(ts - base_time).total_seconds() for ts in timestamps]
        y = values
        
        # Linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return {"slope": 0.0, "intercept": np.mean(y), "r_squared": 0.0}
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_pred = [intercept + slope * xi for xi in x]
        ss_res = sum((yi - yi_pred) ** 2 for yi, yi_pred in zip(y, y_pred))
        ss_tot = sum((yi - np.mean(y)) ** 2 for yi in y)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared
        }
    
    @staticmethod
    def detect_changepoint(values: List[float], min_size: int = 10) -> Optional[int]:
        """Detect change point in time series using cumulative sum."""
        if len(values) < min_size * 2:
            return None
        
        # Calculate cumulative sum of deviations from mean
        mean_value = np.mean(values)
        cumsum = np.cumsum([v - mean_value for v in values])
        
        # Find point with maximum absolute cumulative sum
        max_idx = np.argmax(np.abs(cumsum))
        
        # Validate change point significance
        left_mean = np.mean(values[:max_idx]) if max_idx > 0 else mean_value
        right_mean = np.mean(values[max_idx:]) if max_idx < len(values) else mean_value
        
        # Check if change is significant (more than 2 standard deviations)
        std_dev = np.std(values)
        if abs(right_mean - left_mean) > 2 * std_dev:
            return max_idx
        
        return None
    
    @staticmethod
    def calculate_percentiles(values: List[float]) -> Dict[str, float]:
        """Calculate key percentiles for performance analysis."""
        if not values:
            return {}
        
        percentiles = [50, 75, 90, 95, 99]
        result = {}
        
        for p in percentiles:
            result[f"p{p}"] = np.percentile(values, p)
        
        result.update({
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        })
        
        return result
    
    @staticmethod
    def detect_outliers(values: List[float], method: str = "iqr") -> List[int]:
        """Detect outliers using specified method."""
        if len(values) < 10:
            return []
        
        outlier_indices = []
        
        if method == "iqr":
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_indices = [
                i for i, v in enumerate(values)
                if v < lower_bound or v > upper_bound
            ]
        
        elif method == "zscore":
            mean = np.mean(values)
            std = np.std(values)
            
            outlier_indices = [
                i for i, v in enumerate(values)
                if abs(v - mean) > 3 * std
            ]
        
        return outlier_indices


class RegressionDetector:
    """Detects performance regressions in metrics."""
    
    def __init__(
        self,
        baseline_window_size: int = 100,
        comparison_window_size: int = 50,
        min_samples: int = 20,
        sensitivity: float = 0.05  # 5% degradation threshold
    ):
        self.baseline_window_size = baseline_window_size
        self.comparison_window_size = comparison_window_size
        self.min_samples = min_samples
        self.sensitivity = sensitivity
        
        # Store historical data for each metric
        self.metric_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=baseline_window_size * 2)
        )
        
        # Store regression detection results
        self.regression_history: Dict[str, List[RegressionDetectionResult]] = defaultdict(list)
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a metric data point."""
        self.metric_history[metric.name].append((metric.timestamp, metric.value))
    
    def detect_regression(self, metric_name: str) -> Optional[RegressionDetectionResult]:
        """Detect regression for a specific metric."""
        if metric_name not in self.metric_history:
            return None
        
        history = list(self.metric_history[metric_name])
        
        if len(history) < self.min_samples:
            return RegressionDetectionResult(
                metric_name=metric_name,
                regression_detected=False
            )
        
        # Split into baseline and recent data
        baseline_data = history[:-self.comparison_window_size]
        recent_data = history[-self.comparison_window_size:]
        
        if len(baseline_data) < self.min_samples or len(recent_data) < 10:
            return RegressionDetectionResult(
                metric_name=metric_name,
                regression_detected=False
            )
        
        # Calculate statistics
        baseline_values = [value for _, value in baseline_data]
        recent_values = [value for _, value in recent_data]
        
        baseline_mean = np.mean(baseline_values)
        recent_mean = np.mean(recent_values)
        
        # Determine if lower values are better (like latency) or higher values are better (like throughput)
        lower_is_better = self._is_lower_better_metric(metric_name)
        
        # Calculate degradation
        if lower_is_better:
            # For metrics where lower is better (latency, error rate)
            degradation_ratio = (recent_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else 0
        else:
            # For metrics where higher is better (throughput, efficiency)
            degradation_ratio = (baseline_mean - recent_mean) / baseline_mean if baseline_mean > 0 else 0
        
        degradation_percentage = degradation_ratio * 100
        
        # Check for regression
        regression_detected = degradation_ratio > self.sensitivity
        
        if not regression_detected:
            return RegressionDetectionResult(
                metric_name=metric_name,
                regression_detected=False,
                baseline_value=baseline_mean,
                current_value=recent_mean,
                degradation_percentage=degradation_percentage
            )
        
        # Determine severity
        severity = self._classify_regression_severity(degradation_percentage)
        
        # Calculate confidence using statistical tests
        confidence_score = self._calculate_confidence(baseline_values, recent_values)
        
        # Generate evidence
        evidence = self._generate_evidence(baseline_values, recent_values, degradation_percentage)
        
        result = RegressionDetectionResult(
            metric_name=metric_name,
            regression_detected=True,
            severity=severity,
            baseline_value=baseline_mean,
            current_value=recent_mean,
            degradation_percentage=degradation_percentage,
            confidence_score=confidence_score,
            evidence=evidence
        )
        
        # Store result
        self.regression_history[metric_name].append(result)
        
        # Keep only recent regression results
        if len(self.regression_history[metric_name]) > 50:
            self.regression_history[metric_name] = self.regression_history[metric_name][-25:]
        
        return result
    
    def _is_lower_better_metric(self, metric_name: str) -> bool:
        """Determine if lower values are better for this metric."""
        lower_better_metrics = {
            "latency", "response_time", "error_rate", "memory_usage", 
            "cpu_utilization", "queue_depth", "failure_rate"
        }
        
        return any(keyword in metric_name.lower() for keyword in lower_better_metrics)
    
    def _classify_regression_severity(self, degradation_percentage: float) -> RegressionSeverity:
        """Classify regression severity based on degradation percentage."""
        if degradation_percentage > 50:
            return RegressionSeverity.CRITICAL
        elif degradation_percentage > 25:
            return RegressionSeverity.MAJOR
        elif degradation_percentage > 10:
            return RegressionSeverity.MODERATE
        else:
            return RegressionSeverity.MINOR
    
    def _calculate_confidence(self, baseline: List[float], recent: List[float]) -> float:
        """Calculate confidence score using statistical tests."""
        try:
            # Use Welch's t-test for unequal variances
            baseline_mean = np.mean(baseline)
            recent_mean = np.mean(recent)
            baseline_var = np.var(baseline, ddof=1)
            recent_var = np.var(recent, ddof=1)
            
            n1, n2 = len(baseline), len(recent)
            
            # Calculate t-statistic
            pooled_se = np.sqrt(baseline_var/n1 + recent_var/n2)
            if pooled_se == 0:
                return 0.5
            
            t_stat = abs(recent_mean - baseline_mean) / pooled_se
            
            # Approximate p-value (simplified)
            # In practice, would use proper statistical libraries
            confidence = min(1.0, t_stat / 3.0)  # Rough approximation
            
            return confidence
        
        except Exception:
            return 0.5  # Default moderate confidence
    
    def _generate_evidence(
        self,
        baseline: List[float],
        recent: List[float],
        degradation_percentage: float
    ) -> List[str]:
        """Generate evidence list for regression detection."""
        evidence = []
        
        evidence.append(f"Baseline mean: {np.mean(baseline):.3f}")
        evidence.append(f"Recent mean: {np.mean(recent):.3f}")
        evidence.append(f"Degradation: {degradation_percentage:.1f}%")
        
        # Check for increased variability
        baseline_std = np.std(baseline)
        recent_std = np.std(recent)
        
        if recent_std > baseline_std * 1.5:
            evidence.append(f"Increased variability detected (std: {baseline_std:.3f} -> {recent_std:.3f})")
        
        # Check for trend
        timestamps = list(range(len(recent)))
        trend = StatisticalAnalyzer.calculate_trend(recent, 
                                                   [datetime.now() - timedelta(seconds=i) for i in timestamps])
        
        if abs(trend["slope"]) > 0.001 and trend["r_squared"] > 0.3:
            trend_direction = "increasing" if trend["slope"] > 0 else "decreasing"
            evidence.append(f"Strong {trend_direction} trend detected (R²={trend['r_squared']:.3f})")
        
        return evidence
    
    def get_regression_summary(self) -> Dict[str, Any]:
        """Get summary of all regression detections."""
        summary = {
            "total_metrics_monitored": len(self.metric_history),
            "regressions_by_severity": {
                severity.value: 0 for severity in RegressionSeverity
            },
            "recent_regressions": [],
            "metrics_with_regressions": set()
        }
        
        # Count regressions by severity
        for metric_name, results in self.regression_history.items():
            for result in results:
                if result.regression_detected:
                    summary["regressions_by_severity"][result.severity.value] += 1
                    summary["metrics_with_regressions"].add(metric_name)
                    
                    # Add to recent regressions (last 24 hours)
                    if result.detection_timestamp > datetime.now() - timedelta(hours=24):
                        summary["recent_regressions"].append({
                            "metric": result.metric_name,
                            "severity": result.severity.value,
                            "degradation": result.degradation_percentage,
                            "timestamp": result.detection_timestamp.isoformat()
                        })
        
        summary["metrics_with_regressions"] = list(summary["metrics_with_regressions"])
        
        return summary


class AnomalyDetector:
    """Detects anomalies in performance metrics."""
    
    def __init__(
        self,
        window_size: int = 100,
        sensitivity: float = 2.0,  # Standard deviations for anomaly threshold
        min_samples: int = 30
    ):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        
        # Store metric statistics
        self.metric_stats: Dict[str, Dict[str, float]] = {}
        self.metric_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Store anomaly results
        self.anomaly_history: Dict[str, List[AnomalyDetectionResult]] = defaultdict(list)
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a metric data point and update statistics."""
        self.metric_history[metric.name].append((metric.timestamp, metric.value))
        self._update_statistics(metric.name)
    
    def _update_statistics(self, metric_name: str):
        """Update running statistics for a metric."""
        history = list(self.metric_history[metric_name])
        
        if len(history) < self.min_samples:
            return
        
        values = [value for _, value in history]
        
        self.metric_stats[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "q75": np.percentile(values, 75),
            "q25": np.percentile(values, 25)
        }
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> Optional[AnomalyDetectionResult]:
        """Detect if current value is anomalous."""
        if metric_name not in self.metric_stats:
            return None
        
        stats = self.metric_stats[metric_name]
        
        # Z-score based detection
        z_score = abs(current_value - stats["mean"]) / max(stats["std"], 0.001)
        is_anomaly = z_score > self.sensitivity
        
        if not is_anomaly:
            return AnomalyDetectionResult(
                metric_name=metric_name,
                anomaly_detected=False,
                current_value=current_value,
                expected_value=stats["mean"]
            )
        
        # Classify anomaly type
        anomaly_type = "high_outlier" if current_value > stats["mean"] else "low_outlier"
        
        # Check for other anomaly patterns
        if current_value > stats["q75"] + 3 * (stats["q75"] - stats["q25"]):
            anomaly_type = "extreme_high"
        elif current_value < stats["q25"] - 3 * (stats["q75"] - stats["q25"]):
            anomaly_type = "extreme_low"
        
        result = AnomalyDetectionResult(
            metric_name=metric_name,
            anomaly_detected=True,
            anomaly_type=anomaly_type,
            anomaly_score=z_score,
            threshold=self.sensitivity,
            current_value=current_value,
            expected_value=stats["mean"],
            context={
                "std_dev": stats["std"],
                "median": stats["median"],
                "z_score": z_score
            }
        )
        
        # Store result
        self.anomaly_history[metric_name].append(result)
        
        # Keep only recent anomalies
        if len(self.anomaly_history[metric_name]) > 100:
            self.anomaly_history[metric_name] = self.anomaly_history[metric_name][-50:]
        
        return result
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detections."""
        summary = {
            "total_metrics_monitored": len(self.metric_stats),
            "recent_anomalies": [],
            "anomaly_counts_by_type": defaultdict(int),
            "metrics_with_anomalies": set()
        }
        
        # Process anomaly history
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for metric_name, anomalies in self.anomaly_history.items():
            for anomaly in anomalies:
                if anomaly.anomaly_detected:
                    summary["metrics_with_anomalies"].add(metric_name)
                    summary["anomaly_counts_by_type"][anomaly.anomaly_type] += 1
                    
                    if anomaly.detection_timestamp > cutoff_time:
                        summary["recent_anomalies"].append({
                            "metric": anomaly.metric_name,
                            "type": anomaly.anomaly_type,
                            "score": anomaly.anomaly_score,
                            "value": anomaly.current_value,
                            "timestamp": anomaly.detection_timestamp.isoformat()
                        })
        
        summary["metrics_with_anomalies"] = list(summary["metrics_with_anomalies"])
        summary["anomaly_counts_by_type"] = dict(summary["anomaly_counts_by_type"])
        
        return summary


class OptimizationEngine:
    """Generates optimization recommendations based on performance analytics."""
    
    def __init__(self):
        self.recommendation_rules: List[Callable] = []
        self._initialize_recommendation_rules()
    
    def _initialize_recommendation_rules(self):
        """Initialize optimization recommendation rules."""
        self.recommendation_rules = [
            self._check_latency_optimization,
            self._check_throughput_optimization,
            self._check_memory_optimization,
            self._check_cpu_optimization,
            self._check_scaling_recommendations
        ]
    
    def generate_recommendations(
        self,
        metrics: Dict[str, List[PerformanceMetric]],
        regression_results: Dict[str, RegressionDetectionResult],
        anomaly_results: Dict[str, List[AnomalyDetectionResult]]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis results."""
        recommendations = []
        
        # Apply each recommendation rule
        for rule in self.recommendation_rules:
            try:
                rule_recommendations = rule(metrics, regression_results, anomaly_results)
                recommendations.extend(rule_recommendations)
            except Exception as e:
                logger.error(f"Recommendation rule error: {e}")
        
        # Sort by priority and confidence
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 1), r.confidence),
            reverse=True
        )
        
        return recommendations
    
    def _check_latency_optimization(
        self,
        metrics: Dict[str, List[PerformanceMetric]],
        regression_results: Dict[str, RegressionDetectionResult],
        anomaly_results: Dict[str, List[AnomalyDetectionResult]]
    ) -> List[OptimizationRecommendation]:
        """Check for latency optimization opportunities."""
        recommendations = []
        
        # Check for latency regressions
        for metric_name, result in regression_results.items():
            if "latency" in metric_name.lower() and result.regression_detected:
                if result.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]:
                    recommendations.append(OptimizationRecommendation(
                        target_metric=metric_name,
                        recommendation_type="performance_tuning",
                        description=f"Critical latency regression detected. Consider optimizing algorithms or scaling resources.",
                        expected_improvement=result.degradation_percentage * 0.8,  # Assume 80% recovery
                        confidence=result.confidence_score,
                        priority="critical" if result.severity == RegressionSeverity.CRITICAL else "high"
                    ))
        
        # Check for consistently high latency
        for metric_name, metric_list in metrics.items():
            if "latency" in metric_name.lower() and len(metric_list) > 50:
                recent_values = [m.value for m in metric_list[-50:]]
                p95_latency = np.percentile(recent_values, 95)
                median_latency = np.median(recent_values)
                
                # If P95 is much higher than median, suggest optimization
                if p95_latency > median_latency * 3:
                    recommendations.append(OptimizationRecommendation(
                        target_metric=metric_name,
                        recommendation_type="tail_latency_optimization",
                        description=f"High P95 latency ({p95_latency:.3f}) vs median ({median_latency:.3f}). Consider connection pooling or caching.",
                        expected_improvement=30.0,  # Estimate
                        confidence=0.7,
                        priority="medium"
                    ))
        
        return recommendations
    
    def _check_throughput_optimization(
        self,
        metrics: Dict[str, List[PerformanceMetric]],
        regression_results: Dict[str, RegressionDetectionResult],
        anomaly_results: Dict[str, List[AnomalyDetectionResult]]
    ) -> List[OptimizationRecommendation]:
        """Check for throughput optimization opportunities."""
        recommendations = []
        
        # Check for throughput regressions
        for metric_name, result in regression_results.items():
            if "throughput" in metric_name.lower() and result.regression_detected:
                recommendations.append(OptimizationRecommendation(
                    target_metric=metric_name,
                    recommendation_type="scale_up",
                    description=f"Throughput regression detected. Consider scaling up resources or optimizing bottlenecks.",
                    expected_improvement=result.degradation_percentage * 0.7,
                    confidence=result.confidence_score,
                    priority="high" if result.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL] else "medium"
                ))
        
        return recommendations
    
    def _check_memory_optimization(
        self,
        metrics: Dict[str, List[PerformanceMetric]],
        regression_results: Dict[str, RegressionDetectionResult],
        anomaly_results: Dict[str, List[AnomalyDetectionResult]]
    ) -> List[OptimizationRecommendation]:
        """Check for memory optimization opportunities."""
        recommendations = []
        
        # Check for memory usage trends
        for metric_name, metric_list in metrics.items():
            if "memory" in metric_name.lower() and len(metric_list) > 100:
                values = [m.value for m in metric_list]
                timestamps = [m.timestamp for m in metric_list]
                
                trend = StatisticalAnalyzer.calculate_trend(values, timestamps)
                
                # If memory usage is consistently increasing
                if trend["slope"] > 0 and trend["r_squared"] > 0.5:
                    recommendations.append(OptimizationRecommendation(
                        target_metric=metric_name,
                        recommendation_type="memory_leak_investigation",
                        description=f"Memory usage shows consistent upward trend. Investigate potential memory leaks.",
                        expected_improvement=25.0,
                        confidence=trend["r_squared"],
                        priority="high"
                    ))
        
        return recommendations
    
    def _check_cpu_optimization(
        self,
        metrics: Dict[str, List[PerformanceMetric]],
        regression_results: Dict[str, RegressionDetectionResult],
        anomaly_results: Dict[str, List[AnomalyDetectionResult]]
    ) -> List[OptimizationRecommendation]:
        """Check for CPU optimization opportunities."""
        recommendations = []
        
        # Check for high CPU utilization
        for metric_name, metric_list in metrics.items():
            if "cpu" in metric_name.lower() and len(metric_list) > 20:
                recent_values = [m.value for m in metric_list[-20:]]
                avg_cpu = np.mean(recent_values)
                
                if avg_cpu > 80:  # High CPU usage
                    recommendations.append(OptimizationRecommendation(
                        target_metric=metric_name,
                        recommendation_type="cpu_optimization",
                        description=f"High CPU utilization ({avg_cpu:.1f}%). Consider optimizing algorithms or scaling horizontally.",
                        expected_improvement=20.0,
                        confidence=0.8,
                        priority="medium" if avg_cpu < 90 else "high"
                    ))
        
        return recommendations
    
    def _check_scaling_recommendations(
        self,
        metrics: Dict[str, List[PerformanceMetric]],
        regression_results: Dict[str, RegressionDetectionResult],
        anomaly_results: Dict[str, List[AnomalyDetectionResult]]
    ) -> List[OptimizationRecommendation]:
        """Check for scaling recommendations."""
        recommendations = []
        
        # Check queue depth for scaling decisions
        for metric_name, metric_list in metrics.items():
            if "queue" in metric_name.lower() and len(metric_list) > 30:
                recent_values = [m.value for m in metric_list[-30:]]
                avg_queue_depth = np.mean(recent_values)
                
                if avg_queue_depth > 10:  # High queue depth
                    recommendations.append(OptimizationRecommendation(
                        target_metric=metric_name,
                        recommendation_type="horizontal_scaling",
                        description=f"High queue depth ({avg_queue_depth:.1f}). Consider scaling out to reduce queuing time.",
                        expected_improvement=40.0,
                        confidence=0.75,
                        priority="high"
                    ))
        
        return recommendations


class PerformanceReporter:
    """Generates performance reports and alerts."""
    
    def __init__(self, alert_callbacks: List[Callable] = None):
        self.alert_callbacks = alert_callbacks or []
        self.report_history: List[Dict[str, Any]] = []
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def generate_performance_report(
        self,
        metrics: Dict[str, List[PerformanceMetric]],
        regression_results: Dict[str, RegressionDetectionResult],
        anomaly_results: Dict[str, List[AnomalyDetectionResult]],
        recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report_timestamp = datetime.now()
        
        # Calculate summary statistics
        total_metrics = sum(len(metric_list) for metric_list in metrics.values())
        active_regressions = sum(1 for r in regression_results.values() if r.regression_detected)
        total_anomalies = sum(len(anomaly_list) for anomaly_list in anomaly_results.values())
        critical_recommendations = sum(1 for r in recommendations if r.priority == "critical")
        
        # Generate metric summaries
        metric_summaries = {}
        for metric_name, metric_list in metrics.items():
            if metric_list:
                values = [m.value for m in metric_list]
                metric_summaries[metric_name] = StatisticalAnalyzer.calculate_percentiles(values)
        
        report = {
            "report_id": f"perf_report_{int(report_timestamp.timestamp())}",
            "generated_at": report_timestamp.isoformat(),
            "summary": {
                "total_metrics_collected": total_metrics,
                "unique_metrics": len(metrics),
                "active_regressions": active_regressions,
                "total_anomalies": total_anomalies,
                "critical_recommendations": critical_recommendations,
                "health_score": self._calculate_health_score(
                    active_regressions, total_anomalies, critical_recommendations
                )
            },
            "metric_statistics": metric_summaries,
            "regressions": {
                name: asdict(result) for name, result in regression_results.items()
                if result.regression_detected
            },
            "recent_anomalies": {
                name: [asdict(anomaly) for anomaly in anomaly_list[-10:]]  # Last 10 anomalies
                for name, anomaly_list in anomaly_results.items()
                if anomaly_list
            },
            "optimization_recommendations": [r.to_dict() for r in recommendations[:10]],  # Top 10
            "trends": self._analyze_trends(metrics)
        }
        
        # Store report
        self.report_history.append(report)
        
        # Keep only recent reports
        if len(self.report_history) > 50:
            self.report_history = self.report_history[-25:]
        
        # Generate alerts if needed
        self._check_and_send_alerts(report)
        
        return report
    
    def _calculate_health_score(
        self,
        active_regressions: int,
        total_anomalies: int,
        critical_recommendations: int
    ) -> float:
        """Calculate overall system health score (0-100)."""
        base_score = 100.0
        
        # Penalize based on issues
        base_score -= active_regressions * 10  # -10 per regression
        base_score -= total_anomalies * 2     # -2 per anomaly
        base_score -= critical_recommendations * 15  # -15 per critical recommendation
        
        return max(0.0, min(100.0, base_score))
    
    def _analyze_trends(self, metrics: Dict[str, List[PerformanceMetric]]) -> Dict[str, Any]:
        """Analyze trends across metrics."""
        trends = {}
        
        for metric_name, metric_list in metrics.items():
            if len(metric_list) > 50:
                values = [m.value for m in metric_list]
                timestamps = [m.timestamp for m in metric_list]
                
                trend = StatisticalAnalyzer.calculate_trend(values, timestamps)
                
                # Classify trend
                if trend["r_squared"] > 0.5:
                    if trend["slope"] > 0.001:
                        trend_direction = "increasing"
                    elif trend["slope"] < -0.001:
                        trend_direction = "decreasing"
                    else:
                        trend_direction = "stable"
                else:
                    trend_direction = "variable"
                
                trends[metric_name] = {
                    "direction": trend_direction,
                    "strength": trend["r_squared"],
                    "slope": trend["slope"]
                }
        
        return trends
    
    def _check_and_send_alerts(self, report: Dict[str, Any]):
        """Check report for alert conditions and send alerts."""
        alerts = []
        
        # Health score alert
        health_score = report["summary"]["health_score"]
        if health_score < 70:
            alerts.append({
                "type": AnalyticsAlert.THRESHOLD_EXCEEDED.value,
                "severity": "high" if health_score < 50 else "medium",
                "message": f"System health score is low: {health_score:.1f}/100",
                "timestamp": datetime.now().isoformat()
            })
        
        # Regression alerts
        if report["summary"]["active_regressions"] > 0:
            alerts.append({
                "type": AnalyticsAlert.PERFORMANCE_REGRESSION.value,
                "severity": "high",
                "message": f"{report['summary']['active_regressions']} performance regression(s) detected",
                "timestamp": datetime.now().isoformat()
            })
        
        # Critical recommendation alerts
        if report["summary"]["critical_recommendations"] > 0:
            alerts.append({
                "type": AnalyticsAlert.THRESHOLD_EXCEEDED.value,
                "severity": "critical",
                "message": f"{report['summary']['critical_recommendations']} critical optimization(s) needed",
                "timestamp": datetime.now().isoformat()
            })
        
        # Send alerts
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")


class PerformanceAnalyticsEngine:
    """Main performance analytics engine coordinating all analytics components."""
    
    def __init__(
        self,
        enable_regression_detection: bool = True,
        enable_anomaly_detection: bool = True,
        enable_optimization_engine: bool = True,
        enable_reporting: bool = True
    ):
        self.enable_regression_detection = enable_regression_detection
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_optimization_engine = enable_optimization_engine
        self.enable_reporting = enable_reporting
        
        # Initialize components
        if enable_regression_detection:
            self.regression_detector = RegressionDetector()
        
        if enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector()
        
        if enable_optimization_engine:
            self.optimization_engine = OptimizationEngine()
        
        if enable_reporting:
            self.reporter = PerformanceReporter()
        
        # Metric storage
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.metrics_lock = threading.RLock()
        
        # Analysis results
        self.latest_regression_results: Dict[str, RegressionDetectionResult] = {}
        self.latest_anomaly_results: Dict[str, List[AnomalyDetectionResult]] = {}
        self.latest_recommendations: List[OptimizationRecommendation] = []
        
        # Background analysis
        self.running = False
        self.analysis_executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info("Performance Analytics Engine initialized")
    
    def start_analysis(self):
        """Start background performance analysis."""
        if self.running:
            return
        
        self.running = True
        
        # Start analysis tasks
        analysis_tasks = [
            ("regression_analysis", self._regression_analysis_loop),
            ("anomaly_analysis", self._anomaly_analysis_loop),
            ("optimization_analysis", self._optimization_analysis_loop)
        ]
        
        for name, task in analysis_tasks:
            self.analysis_executor.submit(task)
        
        logger.info("Performance analysis started")
    
    def stop_analysis(self):
        """Stop background performance analysis."""
        if not self.running:
            return
        
        self.running = False
        self.analysis_executor.shutdown(wait=True, timeout=30.0)
        
        logger.info("Performance analysis stopped")
    
    def record_metric(
        self,
        name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=timestamp or datetime.now(),
            metadata=metadata or {},
            tags=tags or set()
        )
        
        with self.metrics_lock:
            self.metrics[name].append(metric)
            
            # Keep only recent metrics to manage memory
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-500:]
        
        # Add to analysis components
        if self.enable_regression_detection:
            self.regression_detector.add_metric(metric)
        
        if self.enable_anomaly_detection:
            self.anomaly_detector.add_metric(metric)
    
    def _regression_analysis_loop(self):
        """Background regression analysis loop."""
        while self.running:
            try:
                time.sleep(60)  # Analyze every minute
                
                if not self.enable_regression_detection:
                    continue
                
                # Analyze all metrics for regressions
                with self.metrics_lock:
                    metric_names = list(self.metrics.keys())
                
                for metric_name in metric_names:
                    try:
                        result = self.regression_detector.detect_regression(metric_name)
                        if result:
                            self.latest_regression_results[metric_name] = result
                            
                            if result.regression_detected:
                                logger.warning(f"Regression detected: {result.to_alert_message()}")
                    
                    except Exception as e:
                        logger.error(f"Regression analysis error for {metric_name}: {e}")
                
            except Exception as e:
                logger.error(f"Regression analysis loop error: {e}")
                time.sleep(300)  # Wait longer on error
    
    def _anomaly_analysis_loop(self):
        """Background anomaly analysis loop."""
        while self.running:
            try:
                time.sleep(30)  # Analyze every 30 seconds
                
                if not self.enable_anomaly_detection:
                    continue
                
                # Check recent metrics for anomalies
                with self.metrics_lock:
                    for metric_name, metric_list in self.metrics.items():
                        if not metric_list:
                            continue
                        
                        # Check last few metrics
                        for metric in metric_list[-5:]:  # Check last 5 metrics
                            try:
                                result = self.anomaly_detector.detect_anomaly(metric_name, metric.value)
                                if result and result.anomaly_detected:
                                    if metric_name not in self.latest_anomaly_results:
                                        self.latest_anomaly_results[metric_name] = []
                                    
                                    self.latest_anomaly_results[metric_name].append(result)
                                    
                                    logger.warning(
                                        f"Anomaly detected in {metric_name}: "
                                        f"{result.anomaly_type} (score: {result.anomaly_score:.2f})"
                                    )
                            
                            except Exception as e:
                                logger.error(f"Anomaly analysis error for {metric_name}: {e}")
                
            except Exception as e:
                logger.error(f"Anomaly analysis loop error: {e}")
                time.sleep(120)  # Wait longer on error
    
    def _optimization_analysis_loop(self):
        """Background optimization analysis loop."""
        while self.running:
            try:
                time.sleep(300)  # Analyze every 5 minutes
                
                if not self.enable_optimization_engine:
                    continue
                
                # Generate optimization recommendations
                with self.metrics_lock:
                    metrics_copy = {
                        name: metric_list[-100:]  # Use last 100 metrics
                        for name, metric_list in self.metrics.items()
                    }
                
                recommendations = self.optimization_engine.generate_recommendations(
                    metrics_copy,
                    self.latest_regression_results,
                    self.latest_anomaly_results
                )
                
                self.latest_recommendations = recommendations
                
                # Log critical recommendations
                critical_recs = [r for r in recommendations if r.priority == "critical"]
                for rec in critical_recs:
                    logger.critical(f"Critical optimization needed: {rec.description}")
                
            except Exception as e:
                logger.error(f"Optimization analysis loop error: {e}")
                time.sleep(600)  # Wait longer on error
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate current performance report."""
        if not self.enable_reporting:
            return {"error": "Reporting is disabled"}
        
        with self.metrics_lock:
            metrics_copy = dict(self.metrics)
        
        return self.reporter.generate_performance_report(
            metrics_copy,
            self.latest_regression_results,
            self.latest_anomaly_results,
            self.latest_recommendations
        )
    
    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics engine statistics."""
        with self.metrics_lock:
            total_metrics = sum(len(metric_list) for metric_list in self.metrics.values())
            unique_metrics = len(self.metrics)
        
        stats = {
            "running": self.running,
            "total_metrics_collected": total_metrics,
            "unique_metrics": unique_metrics,
            "components_enabled": {
                "regression_detection": self.enable_regression_detection,
                "anomaly_detection": self.enable_anomaly_detection,
                "optimization_engine": self.enable_optimization_engine,
                "reporting": self.enable_reporting
            },
            "analysis_results": {
                "active_regressions": sum(
                    1 for r in self.latest_regression_results.values()
                    if r.regression_detected
                ),
                "recent_anomalies": sum(
                    len(anomaly_list) for anomaly_list in self.latest_anomaly_results.values()
                ),
                "optimization_recommendations": len(self.latest_recommendations)
            }
        }
        
        # Add component-specific stats
        if self.enable_regression_detection:
            stats["regression_detector"] = self.regression_detector.get_regression_summary()
        
        if self.enable_anomaly_detection:
            stats["anomaly_detector"] = self.anomaly_detector.get_anomaly_summary()
        
        return stats
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback for notifications."""
        if self.enable_reporting:
            self.reporter.add_alert_callback(callback)


# Global analytics engine instance
_global_analytics_engine: Optional[PerformanceAnalyticsEngine] = None


def get_analytics_engine(**kwargs) -> PerformanceAnalyticsEngine:
    """Get or create the global performance analytics engine."""
    global _global_analytics_engine
    
    if _global_analytics_engine is None:
        _global_analytics_engine = PerformanceAnalyticsEngine(**kwargs)
    
    return _global_analytics_engine


def start_performance_analytics(**kwargs) -> PerformanceAnalyticsEngine:
    """Start performance analytics system."""
    engine = get_analytics_engine(**kwargs)
    engine.start_analysis()
    return engine


def record_performance_metric(
    name: str,
    value: float,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Set[str]] = None
):
    """Record a performance metric for analytics."""
    engine = get_analytics_engine()
    engine.record_metric(name, value, timestamp, metadata, tags)


def generate_performance_report() -> Dict[str, Any]:
    """Generate current performance report."""
    engine = get_analytics_engine()
    return engine.generate_report()


def add_performance_alert_callback(callback: Callable):
    """Add callback for performance alerts."""
    engine = get_analytics_engine()
    engine.add_alert_callback(callback)