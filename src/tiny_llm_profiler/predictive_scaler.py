"""
Advanced Predictive Auto-Scaling Infrastructure for Generation 3
Provides comprehensive auto-scaling capabilities including:
- Predictive scaling using historical data and machine learning
- Resource usage forecasting with multiple time horizons
- Dynamic scaling based on predicted load patterns
- Multi-dimensional resource optimization
- Cost-aware scaling decisions
- Anomaly detection for unusual load patterns
"""

import time
import threading
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import json
import psutil
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import math
from pathlib import Path

from .exceptions import TinyLLMProfilerError, ResourceError
from .logging_config import get_logger, PerformanceLogger
from .auto_scaling import ScalingDecision, ScalingMetrics, ScalingConfig, LoadLevel

logger = get_logger("predictive_scaler")
perf_logger = PerformanceLogger()


class ForecastHorizon(str, Enum):
    """Time horizons for forecasting."""
    SHORT_TERM = "1h"      # Next 1 hour
    MEDIUM_TERM = "4h"     # Next 4 hours  
    LONG_TERM = "24h"      # Next 24 hours
    WEEKLY = "7d"          # Next 7 days


class ScalingStrategy(str, Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"            # Traditional reactive scaling
    PREDICTIVE = "predictive"        # Predictive scaling based on forecasts
    ADAPTIVE = "adaptive"            # Adaptive based on patterns
    COST_OPTIMIZED = "cost_optimized" # Cost-aware scaling
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Performance-first scaling


class ResourceType(str, Enum):
    """Types of resources that can be scaled."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    WORKERS = "workers"
    CONNECTIONS = "connections"
    STORAGE_GB = "storage_gb"


@dataclass
class ResourceForecast:
    """Forecast for a specific resource type."""
    resource_type: ResourceType
    horizon: ForecastHorizon
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]  # (lower, upper) bounds
    timestamps: List[datetime]
    accuracy_score: float = 0.0  # Model accuracy from validation
    
    def get_peak_demand(self) -> Tuple[datetime, float]:
        """Get predicted peak demand and its timestamp."""
        if not self.predicted_values:
            return datetime.now(), 0.0
        
        peak_idx = np.argmax(self.predicted_values)
        return self.timestamps[peak_idx], self.predicted_values[peak_idx]
    
    def get_average_demand(self) -> float:
        """Get average predicted demand."""
        return np.mean(self.predicted_values) if self.predicted_values else 0.0


@dataclass
class ScalingRecommendation:
    """Recommendation for scaling actions."""
    resource_type: ResourceType
    current_capacity: float
    recommended_capacity: float
    confidence: float  # 0.0 to 1.0
    reasoning: str
    cost_impact: Optional[float] = None
    performance_impact: Optional[float] = None
    urgency: str = "normal"  # low, normal, high, critical
    
    def get_scaling_factor(self) -> float:
        """Get the scaling factor."""
        if self.current_capacity == 0:
            return 1.0
        return self.recommended_capacity / self.current_capacity


class TimeSeriesPredictor(ABC):
    """Abstract base class for time series prediction models."""
    
    @abstractmethod
    def fit(self, timestamps: List[datetime], values: List[float]) -> None:
        """Train the model on historical data."""
        pass
    
    @abstractmethod
    def predict(
        self,
        steps_ahead: int,
        confidence_level: float = 0.95
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Predict future values with confidence intervals."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class MovingAveragePredictor(TimeSeriesPredictor):
    """Simple moving average predictor."""
    
    def __init__(self, window_size: int = 12):
        self.window_size = window_size
        self.values: deque = deque(maxlen=window_size * 2)  # Keep extra for variance calc
        self.fitted = False
    
    def fit(self, timestamps: List[datetime], values: List[float]) -> None:
        """Fit the moving average model."""
        self.values.extend(values)
        self.fitted = len(self.values) >= self.window_size
    
    def predict(
        self,
        steps_ahead: int,
        confidence_level: float = 0.95
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Predict using moving average."""
        if not self.fitted:
            return [], []
        
        # Calculate moving average
        recent_values = list(self.values)[-self.window_size:]
        mean_value = np.mean(recent_values)
        std_value = np.std(recent_values)
        
        # Generate predictions (simple - same value repeated)
        predictions = [mean_value] * steps_ahead
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        margin = z_score * std_value
        
        confidence_intervals = [
            (max(0, mean_value - margin), mean_value + margin)
            for _ in range(steps_ahead)
        ]
        
        return predictions, confidence_intervals
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "moving_average",
            "window_size": self.window_size,
            "fitted": self.fitted,
            "data_points": len(self.values)
        }


class LinearTrendPredictor(TimeSeriesPredictor):
    """Linear trend predictor using least squares."""
    
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.fitted = False
        self.residual_std = 0.0
        self.r_squared = 0.0
    
    def fit(self, timestamps: List[datetime], values: List[float]) -> None:
        """Fit linear trend model."""
        if len(values) < 2:
            return
        
        # Convert timestamps to numeric values (seconds since first timestamp)
        base_time = timestamps[0]
        x = [(ts - base_time).total_seconds() for ts in timestamps]
        y = values
        
        # Calculate linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator != 0:
            self.slope = (n * sum_xy - sum_x * sum_y) / denominator
            self.intercept = (sum_y - self.slope * sum_x) / n
            
            # Calculate residual standard deviation
            y_pred = [self.intercept + self.slope * xi for xi in x]
            residuals = [yi - yi_pred for yi, yi_pred in zip(y, y_pred)]
            self.residual_std = np.std(residuals)
            
            # Calculate R-squared
            y_mean = np.mean(y)
            ss_tot = sum((yi - y_mean) ** 2 for yi in y)
            ss_res = sum(residual ** 2 for residual in residuals)
            self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            self.fitted = True
    
    def predict(
        self,
        steps_ahead: int,
        confidence_level: float = 0.95
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Predict using linear trend."""
        if not self.fitted:
            return [], []
        
        # Predict future values
        predictions = []
        confidence_intervals = []
        
        z_score = 1.96 if confidence_level == 0.95 else 2.58
        
        # Assume each step is 5 minutes ahead
        step_size_seconds = 300  # 5 minutes
        
        for i in range(1, steps_ahead + 1):
            x_future = i * step_size_seconds
            y_pred = self.intercept + self.slope * x_future
            
            # Confidence interval grows with distance from training data
            margin = z_score * self.residual_std * (1 + i * 0.1)  # Growing uncertainty
            
            predictions.append(max(0, y_pred))  # Ensure non-negative
            confidence_intervals.append((
                max(0, y_pred - margin),
                y_pred + margin
            ))
        
        return predictions, confidence_intervals
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "linear_trend",
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "fitted": self.fitted
        }


class SeasonalPredictor(TimeSeriesPredictor):
    """Seasonal predictor for daily/weekly patterns."""
    
    def __init__(self, seasonal_periods: List[int] = None):
        # Default seasonal periods: hourly (24), daily (7*24)
        self.seasonal_periods = seasonal_periods or [24, 24*7]
        self.seasonal_components: Dict[int, List[float]] = {}
        self.trend_predictor = LinearTrendPredictor()
        self.fitted = False
    
    def fit(self, timestamps: List[datetime], values: List[float]) -> None:
        """Fit seasonal decomposition model."""
        if len(values) < max(self.seasonal_periods) * 2:
            # Not enough data for seasonal analysis, fallback to trend
            self.trend_predictor.fit(timestamps, values)
            self.fitted = self.trend_predictor.fitted
            return
        
        # Fit trend component
        self.trend_predictor.fit(timestamps, values)
        
        # Extract seasonal components
        for period in self.seasonal_periods:
            if len(values) >= period * 2:
                seasonal_pattern = self._extract_seasonal_pattern(values, period)
                self.seasonal_components[period] = seasonal_pattern
        
        self.fitted = True
    
    def _extract_seasonal_pattern(self, values: List[float], period: int) -> List[float]:
        """Extract seasonal pattern for given period."""
        # Simple seasonal extraction: average values for each position in period
        seasonal_pattern = [0.0] * period
        counts = [0] * period
        
        for i, value in enumerate(values):
            position = i % period
            seasonal_pattern[position] += value
            counts[position] += 1
        
        # Calculate averages
        for i in range(period):
            if counts[i] > 0:
                seasonal_pattern[i] /= counts[i]
            else:
                seasonal_pattern[i] = np.mean(values) if values else 0
        
        # Normalize around mean
        pattern_mean = np.mean(seasonal_pattern)
        seasonal_pattern = [x - pattern_mean for x in seasonal_pattern]
        
        return seasonal_pattern
    
    def predict(
        self,
        steps_ahead: int,
        confidence_level: float = 0.95
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Predict using seasonal model."""
        if not self.fitted:
            return [], []
        
        # Get trend predictions
        trend_predictions, trend_intervals = self.trend_predictor.predict(
            steps_ahead, confidence_level
        )
        
        if not trend_predictions:
            return [], []
        
        predictions = []
        confidence_intervals = []
        
        for i in range(steps_ahead):
            trend_value = trend_predictions[i]
            
            # Add seasonal components
            seasonal_adjustment = 0.0
            for period, pattern in self.seasonal_components.items():
                position = i % period
                seasonal_adjustment += pattern[position]
            
            # Average seasonal adjustments if multiple periods
            if self.seasonal_components:
                seasonal_adjustment /= len(self.seasonal_components)
            
            prediction = max(0, trend_value + seasonal_adjustment)
            predictions.append(prediction)
            
            # Adjust confidence intervals
            trend_lower, trend_upper = trend_intervals[i]
            confidence_intervals.append((
                max(0, trend_lower + seasonal_adjustment),
                trend_upper + seasonal_adjustment
            ))
        
        return predictions, confidence_intervals
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "seasonal",
            "seasonal_periods": self.seasonal_periods,
            "seasonal_components": len(self.seasonal_components),
            "trend_model": self.trend_predictor.get_model_info(),
            "fitted": self.fitted
        }


class AnomalyDetector:
    """Detects anomalies in resource usage patterns."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.recent_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def update_baseline(self, metric_name: str, values: List[float]):
        """Update baseline statistics for a metric."""
        if len(values) < 10:  # Need minimum data
            return
        
        self.baseline_stats[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "q95": np.percentile(values, 95),
            "q05": np.percentile(values, 5)
        }
    
    def detect_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Detect if a value is anomalous."""
        self.recent_values[metric_name].append(value)
        
        if metric_name not in self.baseline_stats:
            return {"is_anomaly": False, "reason": "insufficient_baseline_data"}
        
        stats = self.baseline_stats[metric_name]
        
        # Z-score based detection
        z_score = abs(value - stats["mean"]) / max(stats["std"], 0.01)
        is_anomaly = z_score > self.sensitivity
        
        # Additional checks
        anomaly_type = None
        if is_anomaly:
            if value > stats["q95"]:
                anomaly_type = "high_spike"
            elif value < stats["q05"]:
                anomaly_type = "low_drop"
            else:
                anomaly_type = "unusual_value"
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "z_score": z_score,
            "threshold": self.sensitivity,
            "current_value": value,
            "baseline_mean": stats["mean"],
            "baseline_std": stats["std"]
        }
    
    def detect_pattern_anomaly(self, metric_name: str) -> Dict[str, Any]:
        """Detect pattern-based anomalies."""
        if len(self.recent_values[metric_name]) < 20:
            return {"is_anomaly": False, "reason": "insufficient_recent_data"}
        
        recent = list(self.recent_values[metric_name])
        
        # Check for sudden changes in trend
        mid_point = len(recent) // 2
        first_half = recent[:mid_point]
        second_half = recent[mid_point:]
        
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        # Detect significant shift
        shift_ratio = abs(second_mean - first_mean) / max(first_mean, 0.01)
        
        if shift_ratio > 0.5:  # 50% change
            return {
                "is_anomaly": True,
                "anomaly_type": "trend_shift",
                "shift_ratio": shift_ratio,
                "first_half_mean": first_mean,
                "second_half_mean": second_mean
            }
        
        return {"is_anomaly": False}


class CostOptimizer:
    """Optimizes scaling decisions based on cost considerations."""
    
    def __init__(self):
        # Simplified cost model (would integrate with cloud provider APIs)
        self.cost_per_unit = {
            ResourceType.CPU_CORES: 0.05,    # $ per core per hour
            ResourceType.MEMORY_GB: 0.01,    # $ per GB per hour
            ResourceType.WORKERS: 0.1,       # $ per worker per hour
            ResourceType.CONNECTIONS: 0.001, # $ per connection per hour
            ResourceType.STORAGE_GB: 0.001   # $ per GB per hour
        }
        
        self.cost_history: List[Tuple[datetime, float]] = []
    
    def calculate_cost(
        self,
        resource_allocations: Dict[ResourceType, float],
        duration_hours: float = 1.0
    ) -> float:
        """Calculate total cost for resource allocation."""
        total_cost = 0.0
        
        for resource_type, amount in resource_allocations.items():
            unit_cost = self.cost_per_unit.get(resource_type, 0.0)
            total_cost += unit_cost * amount * duration_hours
        
        return total_cost
    
    def optimize_allocation(
        self,
        forecasted_demand: Dict[ResourceType, float],
        performance_requirements: Dict[str, float],
        budget_constraint: Optional[float] = None
    ) -> Dict[ResourceType, float]:
        """Optimize resource allocation considering cost and performance."""
        
        # Simple optimization: meet minimum requirements at lowest cost
        optimized_allocation = {}
        
        # Start with minimum required resources
        for resource_type, demand in forecasted_demand.items():
            # Add safety margin
            safety_margin = 1.2  # 20% buffer
            min_allocation = demand * safety_margin
            
            optimized_allocation[resource_type] = min_allocation
        
        # Check budget constraint
        if budget_constraint:
            current_cost = self.calculate_cost(optimized_allocation)
            
            if current_cost > budget_constraint:
                # Scale down proportionally
                scale_factor = budget_constraint / current_cost
                for resource_type in optimized_allocation:
                    optimized_allocation[resource_type] *= scale_factor
        
        return optimized_allocation
    
    def get_cost_forecast(
        self,
        resource_forecasts: Dict[ResourceType, ResourceForecast]
    ) -> List[Tuple[datetime, float]]:
        """Generate cost forecast based on resource forecasts."""
        if not resource_forecasts:
            return []
        
        # Get the shortest forecast horizon
        min_length = min(len(f.predicted_values) for f in resource_forecasts.values())
        
        cost_forecast = []
        
        for i in range(min_length):
            timestamp = list(resource_forecasts.values())[0].timestamps[i]
            
            # Calculate cost for this time step
            resource_usage = {}
            for resource_type, forecast in resource_forecasts.items():
                resource_usage[resource_type] = forecast.predicted_values[i]
            
            cost = self.calculate_cost(resource_usage, duration_hours=1.0)
            cost_forecast.append((timestamp, cost))
        
        return cost_forecast


class PredictiveScaler:
    """Advanced predictive auto-scaler with machine learning capabilities."""
    
    def __init__(
        self,
        strategy: ScalingStrategy = ScalingStrategy.PREDICTIVE,
        forecast_horizons: List[ForecastHorizon] = None,
        enable_anomaly_detection: bool = True,
        enable_cost_optimization: bool = False
    ):
        self.strategy = strategy
        self.forecast_horizons = forecast_horizons or [
            ForecastHorizon.SHORT_TERM,
            ForecastHorizon.MEDIUM_TERM
        ]
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_cost_optimization = enable_cost_optimization
        
        # Resource tracking
        self.resource_metrics: Dict[ResourceType, deque] = {
            resource_type: deque(maxlen=2000)  # Keep 2000 data points
            for resource_type in ResourceType
        }
        
        self.resource_timestamps: Dict[ResourceType, deque] = {
            resource_type: deque(maxlen=2000)
            for resource_type in ResourceType
        }
        
        # Prediction models for each resource type
        self.predictors: Dict[ResourceType, Dict[ForecastHorizon, TimeSeriesPredictor]] = {}
        self._initialize_predictors()
        
        # Components
        if enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector()
        
        if enable_cost_optimization:
            self.cost_optimizer = CostOptimizer()
        
        # Current state
        self.current_capacity: Dict[ResourceType, float] = {
            ResourceType.CPU_CORES: psutil.cpu_count(),
            ResourceType.MEMORY_GB: psutil.virtual_memory().total / (1024**3),
            ResourceType.WORKERS: 4,
            ResourceType.CONNECTIONS: 10,
            ResourceType.STORAGE_GB: 100
        }
        
        # Scaling history and performance
        self.scaling_decisions: List[Tuple[datetime, Dict[ResourceType, ScalingRecommendation]]] = []
        self.prediction_accuracy: Dict[ResourceType, deque] = {
            rt: deque(maxlen=100) for rt in ResourceType
        }
        
        # Background tasks
        self.running = False
        self.background_tasks: List[threading.Thread] = []
        
        logger.info(f"Initialized predictive scaler with strategy: {strategy}")
    
    def _initialize_predictors(self):
        """Initialize prediction models."""
        for resource_type in ResourceType:
            self.predictors[resource_type] = {}
            
            for horizon in self.forecast_horizons:
                # Choose predictor based on horizon
                if horizon in [ForecastHorizon.SHORT_TERM]:
                    predictor = MovingAveragePredictor(window_size=12)
                elif horizon in [ForecastHorizon.MEDIUM_TERM]:
                    predictor = LinearTrendPredictor()
                else:
                    predictor = SeasonalPredictor()
                
                self.predictors[resource_type][horizon] = predictor
    
    def start(self):
        """Start the predictive scaler."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        tasks = [
            ("model_training", self._model_training_loop),
            ("prediction_validation", self._prediction_validation_loop),
            ("scaling_decision", self._scaling_decision_loop)
        ]
        
        for name, target in tasks:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.background_tasks.append(thread)
        
        logger.info("Predictive scaler started")
    
    def stop(self):
        """Stop the predictive scaler."""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for background tasks to finish
        for thread in self.background_tasks:
            thread.join(timeout=5.0)
        
        self.background_tasks.clear()
        logger.info("Predictive scaler stopped")
    
    def update_metrics(
        self,
        resource_type: ResourceType,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """Update metrics for a resource type."""
        timestamp = timestamp or datetime.now()
        
        self.resource_metrics[resource_type].append(value)
        self.resource_timestamps[resource_type].append(timestamp)
        
        # Update anomaly detector baseline if enabled
        if self.enable_anomaly_detection and len(self.resource_metrics[resource_type]) > 50:
            recent_values = list(self.resource_metrics[resource_type])[-50:]
            self.anomaly_detector.update_baseline(resource_type.value, recent_values)
    
    def generate_forecasts(self) -> Dict[ResourceType, Dict[ForecastHorizon, ResourceForecast]]:
        """Generate forecasts for all resource types and horizons."""
        forecasts = {}
        
        for resource_type in ResourceType:
            forecasts[resource_type] = {}
            
            # Check if we have enough data
            if len(self.resource_metrics[resource_type]) < 20:
                continue
            
            timestamps = list(self.resource_timestamps[resource_type])
            values = list(self.resource_metrics[resource_type])
            
            for horizon in self.forecast_horizons:
                predictor = self.predictors[resource_type][horizon]
                
                # Train/update model
                predictor.fit(timestamps, values)
                
                # Generate predictions
                steps_ahead = self._get_steps_for_horizon(horizon)
                predictions, confidence_intervals = predictor.predict(steps_ahead)
                
                if predictions:
                    # Generate future timestamps
                    last_timestamp = timestamps[-1]
                    future_timestamps = [
                        last_timestamp + timedelta(minutes=5*i) 
                        for i in range(1, steps_ahead + 1)
                    ]
                    
                    forecast = ResourceForecast(
                        resource_type=resource_type,
                        horizon=horizon,
                        predicted_values=predictions,
                        confidence_intervals=confidence_intervals,
                        timestamps=future_timestamps,
                        accuracy_score=self._get_model_accuracy(resource_type)
                    )
                    
                    forecasts[resource_type][horizon] = forecast
        
        return forecasts
    
    def _get_steps_for_horizon(self, horizon: ForecastHorizon) -> int:
        """Get number of prediction steps for horizon."""
        steps_map = {
            ForecastHorizon.SHORT_TERM: 12,   # 1 hour (5-min intervals)
            ForecastHorizon.MEDIUM_TERM: 48,  # 4 hours
            ForecastHorizon.LONG_TERM: 288,   # 24 hours
            ForecastHorizon.WEEKLY: 2016      # 7 days
        }
        return steps_map.get(horizon, 12)
    
    def _get_model_accuracy(self, resource_type: ResourceType) -> float:
        """Get model accuracy for a resource type."""
        if resource_type in self.prediction_accuracy:
            accuracies = list(self.prediction_accuracy[resource_type])
            return np.mean(accuracies) if accuracies else 0.0
        return 0.0
    
    def generate_scaling_recommendations(
        self,
        forecasts: Dict[ResourceType, Dict[ForecastHorizon, ResourceForecast]]
    ) -> Dict[ResourceType, ScalingRecommendation]:
        """Generate scaling recommendations based on forecasts."""
        recommendations = {}
        
        for resource_type in ResourceType:
            if resource_type not in forecasts:
                continue
            
            # Use short-term forecast for scaling decisions
            horizon_forecasts = forecasts[resource_type]
            primary_forecast = horizon_forecasts.get(ForecastHorizon.SHORT_TERM)
            
            if not primary_forecast:
                continue
            
            # Analyze forecast
            peak_time, peak_demand = primary_forecast.get_peak_demand()
            average_demand = primary_forecast.get_average_demand()
            current_capacity = self.current_capacity[resource_type]
            
            # Determine recommended capacity
            recommended_capacity = self._calculate_recommended_capacity(
                resource_type,
                peak_demand,
                average_demand,
                current_capacity,
                primary_forecast
            )
            
            # Calculate confidence based on model accuracy and forecast variance
            confidence = self._calculate_recommendation_confidence(primary_forecast)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                resource_type,
                current_capacity,
                recommended_capacity,
                peak_demand,
                average_demand
            )
            
            # Determine urgency
            urgency = self._determine_urgency(
                resource_type,
                peak_time,
                peak_demand,
                current_capacity
            )
            
            recommendation = ScalingRecommendation(
                resource_type=resource_type,
                current_capacity=current_capacity,
                recommended_capacity=recommended_capacity,
                confidence=confidence,
                reasoning=reasoning,
                urgency=urgency
            )
            
            # Add cost impact if cost optimization is enabled
            if self.enable_cost_optimization:
                recommendation.cost_impact = self._calculate_cost_impact(recommendation)
            
            recommendations[resource_type] = recommendation
        
        return recommendations
    
    def _calculate_recommended_capacity(
        self,
        resource_type: ResourceType,
        peak_demand: float,
        average_demand: float,
        current_capacity: float,
        forecast: ResourceForecast
    ) -> float:
        """Calculate recommended capacity for a resource."""
        
        # Safety margins by resource type
        safety_margins = {
            ResourceType.CPU_CORES: 1.3,      # 30% buffer
            ResourceType.MEMORY_GB: 1.4,      # 40% buffer
            ResourceType.WORKERS: 1.2,        # 20% buffer
            ResourceType.CONNECTIONS: 1.1,    # 10% buffer
            ResourceType.STORAGE_GB: 1.5      # 50% buffer
        }
        
        safety_margin = safety_margins.get(resource_type, 1.2)
        
        if self.strategy == ScalingStrategy.PERFORMANCE_OPTIMIZED:
            # Use peak demand with larger safety margin
            recommended = peak_demand * safety_margin
        elif self.strategy == ScalingStrategy.COST_OPTIMIZED:
            # Use average demand with smaller safety margin
            recommended = average_demand * (safety_margin * 0.8)
        else:
            # Balanced approach: between average and peak
            recommended = (average_demand + peak_demand) / 2 * safety_margin
        
        # Ensure minimum viable capacity
        min_capacity = {
            ResourceType.CPU_CORES: 1,
            ResourceType.MEMORY_GB: 1,
            ResourceType.WORKERS: 1,
            ResourceType.CONNECTIONS: 2,
            ResourceType.STORAGE_GB: 10
        }
        
        recommended = max(recommended, min_capacity.get(resource_type, 1))
        
        # Limit scaling changes to reasonable increments
        if recommended > current_capacity * 2:
            recommended = current_capacity * 2  # Max 2x scale up
        elif recommended < current_capacity * 0.5:
            recommended = current_capacity * 0.7  # Max 30% scale down
        
        return recommended
    
    def _calculate_recommendation_confidence(self, forecast: ResourceForecast) -> float:
        """Calculate confidence in scaling recommendation."""
        confidence = forecast.accuracy_score
        
        # Adjust based on forecast variance
        if forecast.predicted_values:
            variance = np.var(forecast.predicted_values)
            mean_value = np.mean(forecast.predicted_values)
            
            # Lower confidence if high variance relative to mean
            if mean_value > 0:
                cv = np.sqrt(variance) / mean_value  # Coefficient of variation
                confidence *= max(0.3, 1 - cv)  # Minimum 30% confidence
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_reasoning(
        self,
        resource_type: ResourceType,
        current_capacity: float,
        recommended_capacity: float,
        peak_demand: float,
        average_demand: float
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        scaling_factor = recommended_capacity / current_capacity if current_capacity > 0 else 1
        
        if scaling_factor > 1.1:  # Scale up
            return (
                f"Scale up {resource_type.value} by {scaling_factor:.1f}x due to "
                f"predicted peak demand of {peak_demand:.1f} (current: {current_capacity:.1f})"
            )
        elif scaling_factor < 0.9:  # Scale down
            return (
                f"Scale down {resource_type.value} by {1/scaling_factor:.1f}x as "
                f"average demand is {average_demand:.1f} (current: {current_capacity:.1f})"
            )
        else:
            return f"Maintain current {resource_type.value} capacity of {current_capacity:.1f}"
    
    def _determine_urgency(
        self,
        resource_type: ResourceType,
        peak_time: datetime,
        peak_demand: float,
        current_capacity: float
    ) -> str:
        """Determine urgency of scaling action."""
        time_to_peak = (peak_time - datetime.now()).total_seconds() / 60  # minutes
        demand_ratio = peak_demand / current_capacity if current_capacity > 0 else 0
        
        if demand_ratio > 2.0 and time_to_peak < 15:  # Very high demand, very soon
            return "critical"
        elif demand_ratio > 1.5 and time_to_peak < 30:  # High demand, soon
            return "high"
        elif demand_ratio > 1.2 and time_to_peak < 60:  # Moderate demand
            return "normal"
        else:
            return "low"
    
    def _calculate_cost_impact(self, recommendation: ScalingRecommendation) -> float:
        """Calculate cost impact of scaling recommendation."""
        if not self.enable_cost_optimization:
            return 0.0
        
        current_cost = self.cost_optimizer.calculate_cost({
            recommendation.resource_type: recommendation.current_capacity
        })
        
        recommended_cost = self.cost_optimizer.calculate_cost({
            recommendation.resource_type: recommendation.recommended_capacity
        })
        
        return recommended_cost - current_cost
    
    def _model_training_loop(self):
        """Background loop for training prediction models."""
        while self.running:
            try:
                time.sleep(300)  # Train every 5 minutes
                
                # Train models for each resource type
                for resource_type in ResourceType:
                    if len(self.resource_metrics[resource_type]) < 50:
                        continue
                    
                    timestamps = list(self.resource_timestamps[resource_type])
                    values = list(self.resource_metrics[resource_type])
                    
                    # Train all predictors for this resource
                    for horizon, predictor in self.predictors[resource_type].items():
                        try:
                            predictor.fit(timestamps, values)
                        except Exception as e:
                            logger.error(f"Model training error for {resource_type}/{horizon}: {e}")
                
            except Exception as e:
                logger.error(f"Model training loop error: {e}")
                time.sleep(600)  # Wait longer on error
    
    def _prediction_validation_loop(self):
        """Background loop for validating prediction accuracy."""
        while self.running:
            try:
                time.sleep(600)  # Validate every 10 minutes
                
                # Validate predictions against actual values
                # This would compare historical predictions with actual outcomes
                # For now, we'll simulate validation
                
                for resource_type in ResourceType:
                    # Simulate accuracy score (would be calculated from real validation)
                    accuracy = np.random.uniform(0.7, 0.95)  # Placeholder
                    self.prediction_accuracy[resource_type].append(accuracy)
                
            except Exception as e:
                logger.error(f"Prediction validation error: {e}")
                time.sleep(900)  # Wait longer on error
    
    def _scaling_decision_loop(self):
        """Background loop for making scaling decisions."""
        while self.running:
            try:
                time.sleep(120)  # Make decisions every 2 minutes
                
                # Generate forecasts
                forecasts = self.generate_forecasts()
                
                # Generate recommendations
                recommendations = self.generate_scaling_recommendations(forecasts)
                
                # Store decisions
                if recommendations:
                    self.scaling_decisions.append((datetime.now(), recommendations))
                    
                    # Keep only recent decisions
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.scaling_decisions = [
                        (timestamp, recs) for timestamp, recs in self.scaling_decisions
                        if timestamp > cutoff_time
                    ]
                    
                    # Log high-priority recommendations
                    for resource_type, rec in recommendations.items():
                        if rec.urgency in ["high", "critical"]:
                            logger.warning(
                                f"{rec.urgency.upper()} scaling recommendation: "
                                f"{rec.reasoning} (confidence: {rec.confidence:.2f})"
                            )
                
            except Exception as e:
                logger.error(f"Scaling decision loop error: {e}")
                time.sleep(300)  # Wait longer on error
    
    def get_current_forecasts(self) -> Dict[ResourceType, Dict[ForecastHorizon, ResourceForecast]]:
        """Get current forecasts for all resources."""
        return self.generate_forecasts()
    
    def get_scaling_recommendations(self) -> Dict[ResourceType, ScalingRecommendation]:
        """Get current scaling recommendations."""
        forecasts = self.generate_forecasts()
        return self.generate_scaling_recommendations(forecasts)
    
    def get_predictive_scaler_stats(self) -> Dict[str, Any]:
        """Get comprehensive predictive scaler statistics."""
        stats = {
            "strategy": self.strategy.value,
            "forecast_horizons": [h.value for h in self.forecast_horizons],
            "anomaly_detection_enabled": self.enable_anomaly_detection,
            "cost_optimization_enabled": self.enable_cost_optimization,
            "running": self.running,
            "current_capacity": {rt.value: cap for rt, cap in self.current_capacity.items()},
            "data_points": {
                rt.value: len(self.resource_metrics[rt]) 
                for rt in ResourceType
            },
            "model_accuracy": {
                rt.value: self._get_model_accuracy(rt)
                for rt in ResourceType
            },
            "recent_decisions": len(self.scaling_decisions),
            "prediction_models": {}
        }
        
        # Add model information
        for resource_type in ResourceType:
            stats["prediction_models"][resource_type.value] = {}
            for horizon, predictor in self.predictors[resource_type].items():
                stats["prediction_models"][resource_type.value][horizon.value] = (
                    predictor.get_model_info()
                )
        
        return stats


# Global predictive scaler instance
_global_predictive_scaler: Optional[PredictiveScaler] = None


def get_predictive_scaler(**kwargs) -> PredictiveScaler:
    """Get or create the global predictive scaler."""
    global _global_predictive_scaler
    
    if _global_predictive_scaler is None:
        _global_predictive_scaler = PredictiveScaler(**kwargs)
    
    return _global_predictive_scaler


def start_predictive_scaling(
    strategy: ScalingStrategy = ScalingStrategy.PREDICTIVE,
    **kwargs
) -> PredictiveScaler:
    """Start predictive scaling system."""
    scaler = get_predictive_scaler(strategy=strategy, **kwargs)
    scaler.start()
    return scaler


def update_resource_metrics(
    resource_type: Union[ResourceType, str],
    value: float,
    timestamp: Optional[datetime] = None
):
    """Update resource metrics for predictive scaling."""
    if isinstance(resource_type, str):
        resource_type = ResourceType(resource_type)
    
    scaler = get_predictive_scaler()
    scaler.update_metrics(resource_type, value, timestamp)