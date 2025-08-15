"""
Adaptive Scaling Guard System
Intelligent auto-scaling with predictive algorithms and resource optimization
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    INSTANCES = "instances"
    GPU = "gpu"


class ScalingStrategy(Enum):
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    ML_BASED = "ml_based"


@dataclass
class ResourceMetrics:
    resource_type: ResourceType
    current_usage: float
    capacity: float
    utilization_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    resource_type: ResourceType
    direction: ScalingDirection
    from_value: float
    to_value: float
    reason: str
    strategy_used: ScalingStrategy
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    execution_time_ms: float = 0.0


@dataclass
class ScalingRule:
    resource_type: ResourceType
    scale_up_threshold: float
    scale_down_threshold: float
    min_capacity: float
    max_capacity: float
    step_size: float
    cooldown_seconds: int = 300


@dataclass
class WorkloadPattern:
    pattern_id: str
    name: str
    description: str
    time_ranges: List[Tuple[int, int]]  # (start_hour, end_hour)
    expected_load_multiplier: float
    resource_requirements: Dict[ResourceType, float]


class PredictiveModel(ABC):
    @abstractmethod
    def predict_demand(self, current_metrics: List[ResourceMetrics], 
                      horizon_minutes: int = 15) -> Dict[ResourceType, float]:
        pass
    
    @abstractmethod
    def train(self, historical_data: List[ResourceMetrics]) -> None:
        pass


class LinearPredictiveModel(PredictiveModel):
    def __init__(self):
        self.models: Dict[ResourceType, LinearRegression] = {}
        self.scalers: Dict[ResourceType, StandardScaler] = {}
        self.is_trained = False
        self.feature_window = 10  # Use last 10 data points for prediction
        
    def train(self, historical_data: List[ResourceMetrics]) -> None:
        if len(historical_data) < 20:
            logger.warning("Insufficient data for training predictive model")
            return
            
        # Group data by resource type
        resource_data: Dict[ResourceType, List[Tuple[float, float]]] = {}
        
        for metric in historical_data:
            if metric.resource_type not in resource_data:
                resource_data[metric.resource_type] = []
            
            timestamp_numeric = metric.timestamp.timestamp()
            resource_data[metric.resource_type].append(
                (timestamp_numeric, metric.utilization_percent)
            )
        
        # Train model for each resource type
        for resource_type, data in resource_data.items():
            if len(data) < self.feature_window + 1:
                continue
                
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if len(X) > 0:
                # Initialize and train model
                self.scalers[resource_type] = StandardScaler()
                X_scaled = self.scalers[resource_type].fit_transform(X)
                
                self.models[resource_type] = LinearRegression()
                self.models[resource_type].fit(X_scaled, y)
                
                logger.info(f"Trained predictive model for {resource_type.value}")
        
        self.is_trained = len(self.models) > 0
    
    def _prepare_training_data(self, data: List[Tuple[float, float]]) -> Tuple[List, List]:
        X, y = [], []
        
        # Sort by timestamp
        data.sort(key=lambda x: x[0])
        
        # Create sliding windows
        for i in range(self.feature_window, len(data)):
            # Features: previous utilization values and time features
            features = []
            
            # Add previous utilization values
            for j in range(i - self.feature_window, i):
                features.append(data[j][1])
            
            # Add time-based features
            timestamp = data[i][0]
            dt = datetime.fromtimestamp(timestamp)
            features.extend([
                dt.hour,  # Hour of day
                dt.weekday(),  # Day of week
                dt.day,  # Day of month
            ])
            
            X.append(features)
            y.append(data[i][1])  # Target: current utilization
        
        return X, y
    
    def predict_demand(self, current_metrics: List[ResourceMetrics], 
                      horizon_minutes: int = 15) -> Dict[ResourceType, float]:
        predictions = {}
        
        if not self.is_trained:
            return predictions
        
        # Group current metrics by resource type
        current_data: Dict[ResourceType, List[ResourceMetrics]] = {}
        for metric in current_metrics:
            if metric.resource_type not in current_data:
                current_data[metric.resource_type] = []
            current_data[metric.resource_type].append(metric)
        
        # Make predictions for each resource type
        for resource_type, metrics in current_data.items():
            if resource_type not in self.models:
                continue
                
            try:
                prediction = self._predict_resource_demand(
                    resource_type, metrics, horizon_minutes
                )
                predictions[resource_type] = prediction
                
            except Exception as e:
                logger.error(f"Error predicting demand for {resource_type.value}: {str(e)}")
        
        return predictions
    
    def _predict_resource_demand(self, resource_type: ResourceType, 
                               metrics: List[ResourceMetrics], 
                               horizon_minutes: int) -> float:
        # Sort metrics by timestamp
        metrics.sort(key=lambda x: x.timestamp)
        
        # Take last window_size metrics
        recent_metrics = metrics[-self.feature_window:]
        
        if len(recent_metrics) < self.feature_window:
            # Not enough data, return current utilization
            return metrics[-1].utilization_percent if metrics else 0.0
        
        # Prepare features
        features = []
        
        # Add recent utilization values
        for metric in recent_metrics:
            features.append(metric.utilization_percent)
        
        # Add future time features (horizon minutes from now)
        future_time = datetime.now() + timedelta(minutes=horizon_minutes)
        features.extend([
            future_time.hour,
            future_time.weekday(),
            future_time.day
        ])
        
        # Scale features and predict
        X = np.array([features])
        X_scaled = self.scalers[resource_type].transform(X)
        prediction = self.models[resource_type].predict(X_scaled)[0]
        
        # Ensure prediction is within reasonable bounds
        return max(0.0, min(100.0, prediction))


class MLPredictiveModel(PredictiveModel):
    def __init__(self):
        self.models: Dict[ResourceType, RandomForestRegressor] = {}
        self.scalers: Dict[ResourceType, StandardScaler] = {}
        self.is_trained = False
        self.feature_window = 15
        
    def train(self, historical_data: List[ResourceMetrics]) -> None:
        if len(historical_data) < 50:
            logger.warning("Insufficient data for ML model training")
            return
            
        # Group and prepare data
        resource_data: Dict[ResourceType, List[Tuple[float, float, Dict[str, Any]]]] = {}
        
        for metric in historical_data:
            if metric.resource_type not in resource_data:
                resource_data[metric.resource_type] = []
            
            timestamp_numeric = metric.timestamp.timestamp()
            resource_data[metric.resource_type].append(
                (timestamp_numeric, metric.utilization_percent, metric.metadata)
            )
        
        # Train advanced model for each resource type
        for resource_type, data in resource_data.items():
            if len(data) < self.feature_window + 1:
                continue
                
            X, y = self._prepare_ml_training_data(data)
            
            if len(X) > 10:  # Need more data for RF
                self.scalers[resource_type] = StandardScaler()
                X_scaled = self.scalers[resource_type].fit_transform(X)
                
                self.models[resource_type] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.models[resource_type].fit(X_scaled, y)
                
                logger.info(f"Trained ML model for {resource_type.value}")
        
        self.is_trained = len(self.models) > 0
    
    def _prepare_ml_training_data(self, data: List[Tuple[float, float, Dict[str, Any]]]) -> Tuple[List, List]:
        X, y = [], []
        
        # Sort by timestamp
        data.sort(key=lambda x: x[0])
        
        for i in range(self.feature_window, len(data)):
            features = []
            
            # Previous utilization values
            for j in range(i - self.feature_window, i):
                features.append(data[j][1])
            
            # Statistical features from recent window
            recent_values = [data[j][1] for j in range(i - self.feature_window, i)]
            features.extend([
                np.mean(recent_values),
                np.std(recent_values),
                np.min(recent_values),
                np.max(recent_values),
                np.percentile(recent_values, 75) - np.percentile(recent_values, 25)  # IQR
            ])
            
            # Time features
            timestamp = data[i][0]
            dt = datetime.fromtimestamp(timestamp)
            features.extend([
                dt.hour,
                dt.weekday(),
                dt.day,
                dt.month,
                1 if dt.weekday() < 5 else 0,  # Is weekday
                np.sin(2 * np.pi * dt.hour / 24),  # Cyclical hour
                np.cos(2 * np.pi * dt.hour / 24)
            ])
            
            # Metadata features (if available)
            metadata = data[i][2]
            features.extend([
                metadata.get('concurrent_requests', 0),
                metadata.get('queue_length', 0),
                metadata.get('error_rate', 0)
            ])
            
            X.append(features)
            y.append(data[i][1])
        
        return X, y
    
    def predict_demand(self, current_metrics: List[ResourceMetrics], 
                      horizon_minutes: int = 15) -> Dict[ResourceType, float]:
        predictions = {}
        
        if not self.is_trained:
            return predictions
        
        # Similar to LinearPredictiveModel but with enhanced features
        current_data: Dict[ResourceType, List[ResourceMetrics]] = {}
        for metric in current_metrics:
            if metric.resource_type not in current_data:
                current_data[metric.resource_type] = []
            current_data[metric.resource_type].append(metric)
        
        for resource_type, metrics in current_data.items():
            if resource_type not in self.models:
                continue
                
            try:
                prediction = self._predict_ml_resource_demand(
                    resource_type, metrics, horizon_minutes
                )
                predictions[resource_type] = prediction
                
            except Exception as e:
                logger.error(f"Error in ML prediction for {resource_type.value}: {str(e)}")
        
        return predictions
    
    def _predict_ml_resource_demand(self, resource_type: ResourceType, 
                                   metrics: List[ResourceMetrics], 
                                   horizon_minutes: int) -> float:
        metrics.sort(key=lambda x: x.timestamp)
        recent_metrics = metrics[-self.feature_window:]
        
        if len(recent_metrics) < self.feature_window:
            return metrics[-1].utilization_percent if metrics else 0.0
        
        # Build enhanced feature vector
        features = []
        
        # Utilization values
        for metric in recent_metrics:
            features.append(metric.utilization_percent)
        
        # Statistical features
        values = [m.utilization_percent for m in recent_metrics]
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.percentile(values, 75) - np.percentile(values, 25)
        ])
        
        # Future time features
        future_time = datetime.now() + timedelta(minutes=horizon_minutes)
        features.extend([
            future_time.hour,
            future_time.weekday(),
            future_time.day,
            future_time.month,
            1 if future_time.weekday() < 5 else 0,
            np.sin(2 * np.pi * future_time.hour / 24),
            np.cos(2 * np.pi * future_time.hour / 24)
        ])
        
        # Current metadata
        latest_metadata = recent_metrics[-1].metadata
        features.extend([
            latest_metadata.get('concurrent_requests', 0),
            latest_metadata.get('queue_length', 0),
            latest_metadata.get('error_rate', 0)
        ])
        
        # Scale and predict
        X = np.array([features])
        X_scaled = self.scalers[resource_type].transform(X)
        prediction = self.models[resource_type].predict(X_scaled)[0]
        
        return max(0.0, min(100.0, prediction))


class AdaptiveScalingController:
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.scaling_rules: Dict[ResourceType, ScalingRule] = {}
        self.workload_patterns: List[WorkloadPattern] = []
        
        # Predictive models
        self.linear_model = LinearPredictiveModel()
        self.ml_model = MLPredictiveModel()
        self.current_model = self.linear_model
        
        # Metrics and events
        self.metrics_history: List[ResourceMetrics] = []
        self.scaling_events: List[ScalingEvent] = []
        self.last_scaling_time: Dict[ResourceType, datetime] = {}
        
        # Configuration
        self.history_window_hours = 24
        self.prediction_horizon_minutes = 15
        self.max_history_size = 10000
        
        # Statistics
        self.total_scaling_events = 0
        self.successful_scaling_events = 0
        
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        self.scaling_rules[rule.resource_type] = rule
        logger.info(f"Added scaling rule for {rule.resource_type.value}")
    
    def add_workload_pattern(self, pattern: WorkloadPattern) -> None:
        self.workload_patterns.append(pattern)
        logger.info(f"Added workload pattern: {pattern.name}")
    
    def add_metrics(self, metrics: List[ResourceMetrics]) -> None:
        for metric in metrics:
            self.metrics_history.append(metric)
        
        # Maintain history size
        if len(self.metrics_history) > self.max_history_size:
            excess = len(self.metrics_history) - self.max_history_size
            self.metrics_history = self.metrics_history[excess:]
        
        # Retrain models periodically
        if len(self.metrics_history) % 100 == 0:
            self._retrain_models()
    
    def _retrain_models(self) -> None:
        if len(self.metrics_history) < 50:
            return
            
        try:
            # Train both models
            self.linear_model.train(self.metrics_history)
            self.ml_model.train(self.metrics_history)
            
            # Choose best model based on strategy
            if self.strategy == ScalingStrategy.ML_BASED and self.ml_model.is_trained:
                self.current_model = self.ml_model
            elif self.linear_model.is_trained:
                self.current_model = self.linear_model
                
            logger.info("Predictive models retrained")
            
        except Exception as e:
            logger.error(f"Error retraining models: {str(e)}")
    
    async def evaluate_scaling_needs(self, current_metrics: List[ResourceMetrics]) -> List[ScalingEvent]:
        scaling_events = []
        
        # Add current metrics
        self.add_metrics(current_metrics)
        
        # Evaluate each resource type
        for metric in current_metrics:
            if metric.resource_type not in self.scaling_rules:
                continue
                
            scaling_event = await self._evaluate_resource_scaling(metric)
            if scaling_event:
                scaling_events.append(scaling_event)
        
        return scaling_events
    
    async def _evaluate_resource_scaling(self, metric: ResourceMetrics) -> Optional[ScalingEvent]:
        rule = self.scaling_rules[metric.resource_type]
        
        # Check cooldown period
        if self._is_in_cooldown(metric.resource_type, rule):
            return None
        
        # Current utilization
        current_utilization = metric.utilization_percent
        
        # Get prediction if using predictive strategies
        predicted_utilization = current_utilization
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID, ScalingStrategy.ML_BASED]:
            predictions = self.current_model.predict_demand(
                [metric], self.prediction_horizon_minutes
            )
            predicted_utilization = predictions.get(metric.resource_type, current_utilization)
        
        # Apply workload pattern adjustments
        pattern_multiplier = self._get_workload_pattern_multiplier(metric.resource_type)
        predicted_utilization *= pattern_multiplier
        
        # Determine scaling direction
        scaling_direction = self._determine_scaling_direction(
            current_utilization, predicted_utilization, rule
        )
        
        if scaling_direction == ScalingDirection.MAINTAIN:
            return None
        
        # Calculate new capacity
        new_capacity = self._calculate_new_capacity(
            metric.capacity, scaling_direction, rule
        )
        
        # Create scaling event
        strategy_used = self.strategy
        reason = self._build_scaling_reason(
            current_utilization, predicted_utilization, 
            scaling_direction, pattern_multiplier
        )
        
        scaling_event = ScalingEvent(
            resource_type=metric.resource_type,
            direction=scaling_direction,
            from_value=metric.capacity,
            to_value=new_capacity,
            reason=reason,
            strategy_used=strategy_used
        )
        
        # Execute scaling
        success, execution_time = await self._execute_scaling(scaling_event)
        scaling_event.success = success
        scaling_event.execution_time_ms = execution_time
        
        # Record event
        self.scaling_events.append(scaling_event)
        self.total_scaling_events += 1
        if success:
            self.successful_scaling_events += 1
            self.last_scaling_time[metric.resource_type] = datetime.now()
        
        return scaling_event
    
    def _is_in_cooldown(self, resource_type: ResourceType, rule: ScalingRule) -> bool:
        last_scaling = self.last_scaling_time.get(resource_type)
        if not last_scaling:
            return False
            
        cooldown_end = last_scaling + timedelta(seconds=rule.cooldown_seconds)
        return datetime.now() < cooldown_end
    
    def _get_workload_pattern_multiplier(self, resource_type: ResourceType) -> float:
        current_hour = datetime.now().hour
        
        for pattern in self.workload_patterns:
            for start_hour, end_hour in pattern.time_ranges:
                if start_hour <= current_hour <= end_hour:
                    if resource_type in pattern.resource_requirements:
                        return pattern.expected_load_multiplier
        
        return 1.0  # No pattern match, no adjustment
    
    def _determine_scaling_direction(self, current: float, predicted: float, 
                                   rule: ScalingRule) -> ScalingDirection:
        # Use the higher of current or predicted for scale-up decisions
        max_utilization = max(current, predicted)
        # Use the lower for scale-down decisions (more conservative)
        min_utilization = min(current, predicted)
        
        if max_utilization >= rule.scale_up_threshold:
            return ScalingDirection.UP
        elif min_utilization <= rule.scale_down_threshold:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.MAINTAIN
    
    def _calculate_new_capacity(self, current_capacity: float, 
                              direction: ScalingDirection, 
                              rule: ScalingRule) -> float:
        if direction == ScalingDirection.UP:
            new_capacity = current_capacity + rule.step_size
            return min(new_capacity, rule.max_capacity)
        elif direction == ScalingDirection.DOWN:
            new_capacity = current_capacity - rule.step_size
            return max(new_capacity, rule.min_capacity)
        else:
            return current_capacity
    
    def _build_scaling_reason(self, current: float, predicted: float, 
                            direction: ScalingDirection, pattern_multiplier: float) -> str:
        reason_parts = [
            f"Current utilization: {current:.1f}%",
            f"Predicted utilization: {predicted:.1f}%",
            f"Direction: {direction.value}"
        ]
        
        if pattern_multiplier != 1.0:
            reason_parts.append(f"Workload pattern multiplier: {pattern_multiplier:.2f}")
        
        return ", ".join(reason_parts)
    
    async def _execute_scaling(self, event: ScalingEvent) -> Tuple[bool, float]:
        start_time = time.time()
        
        try:
            # Simulate scaling execution
            logger.info(f"Executing scaling: {event.resource_type.value} "
                       f"{event.direction.value} from {event.from_value} to {event.to_value}")
            
            # Simulate execution time
            await asyncio.sleep(0.1)
            
            execution_time = (time.time() - start_time) * 1000
            return True, execution_time
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {str(e)}")
            execution_time = (time.time() - start_time) * 1000
            return False, execution_time
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        recent_events = [
            e for e in self.scaling_events 
            if e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_scaling_events": self.total_scaling_events,
            "successful_scaling_events": self.successful_scaling_events,
            "success_rate": self.successful_scaling_events / max(1, self.total_scaling_events),
            "recent_events_count": len(recent_events),
            "scaling_by_direction": {
                direction.value: len([
                    e for e in recent_events if e.direction == direction
                ])
                for direction in ScalingDirection
            },
            "scaling_by_resource": {
                resource.value: len([
                    e for e in recent_events if e.resource_type == resource
                ])
                for resource in ResourceType
            },
            "model_status": {
                "current_model": "ML" if self.current_model == self.ml_model else "Linear",
                "linear_trained": self.linear_model.is_trained,
                "ml_trained": self.ml_model.is_trained,
                "metrics_history_size": len(self.metrics_history)
            }
        }
    
    def get_recent_scaling_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        recent_events = self.scaling_events[-limit:]
        
        return [
            {
                "resource_type": event.resource_type.value,
                "direction": event.direction.value,
                "from_value": event.from_value,
                "to_value": event.to_value,
                "reason": event.reason,
                "strategy": event.strategy_used.value,
                "timestamp": event.timestamp.isoformat(),
                "success": event.success,
                "execution_time_ms": event.execution_time_ms
            }
            for event in recent_events
        ]


# Global adaptive scaling controller
_global_scaling_controller: Optional[AdaptiveScalingController] = None


def get_scaling_controller() -> AdaptiveScalingController:
    global _global_scaling_controller
    if _global_scaling_controller is None:
        _global_scaling_controller = AdaptiveScalingController()
        _setup_default_scaling_rules(_global_scaling_controller)
    return _global_scaling_controller


def _setup_default_scaling_rules(controller: AdaptiveScalingController) -> None:
    # Default scaling rules
    default_rules = [
        ScalingRule(
            resource_type=ResourceType.CPU,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            min_capacity=1.0,
            max_capacity=16.0,
            step_size=1.0,
            cooldown_seconds=300
        ),
        ScalingRule(
            resource_type=ResourceType.MEMORY,
            scale_up_threshold=85.0,
            scale_down_threshold=40.0,
            min_capacity=2.0,
            max_capacity=32.0,
            step_size=2.0,
            cooldown_seconds=300
        ),
        ScalingRule(
            resource_type=ResourceType.INSTANCES,
            scale_up_threshold=75.0,
            scale_down_threshold=25.0,
            min_capacity=1.0,
            max_capacity=10.0,
            step_size=1.0,
            cooldown_seconds=600
        )
    ]
    
    for rule in default_rules:
        controller.add_scaling_rule(rule)
    
    # Default workload patterns
    business_hours_pattern = WorkloadPattern(
        pattern_id="business_hours",
        name="Business Hours High Load",
        description="Higher resource requirements during business hours",
        time_ranges=[(9, 17)],  # 9 AM to 5 PM
        expected_load_multiplier=1.5,
        resource_requirements={
            ResourceType.CPU: 1.5,
            ResourceType.MEMORY: 1.3,
            ResourceType.INSTANCES: 1.4
        }
    )
    
    controller.add_workload_pattern(business_hours_pattern)


async def evaluate_auto_scaling(metrics: List[ResourceMetrics]) -> List[Dict[str, Any]]:
    controller = get_scaling_controller()
    events = await controller.evaluate_scaling_needs(metrics)
    
    return [
        {
            "resource_type": event.resource_type.value,
            "direction": event.direction.value,
            "from_value": event.from_value,
            "to_value": event.to_value,
            "success": event.success
        }
        for event in events
    ]


def get_scaling_status() -> Dict[str, Any]:
    controller = get_scaling_controller()
    return controller.get_scaling_stats()