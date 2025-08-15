"""
Self-Healing Pipeline Guard System
Advanced CI/CD pipeline monitoring, failure prediction, and autonomous healing
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"  
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"


class PipelineStage(Enum):
    BUILD = "build"
    TEST = "test"
    LINT = "lint"
    SECURITY = "security"
    DEPLOY = "deploy"
    MONITOR = "monitor"


class FailureType(Enum):
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    CASCADING = "cascading"
    INFRASTRUCTURE = "infrastructure"
    MODEL_DRIFT = "model_drift"


@dataclass
class PipelineMetrics:
    stage: PipelineStage
    status: HealthStatus
    duration_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureEvent:
    stage: PipelineStage
    failure_type: FailureType
    error_message: str
    stack_trace: Optional[str]
    metrics: PipelineMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    recovery_attempts: int = 0
    resolved: bool = False


@dataclass
class HealingAction:
    action_id: str
    name: str
    description: str
    target_stage: PipelineStage
    priority: int
    estimated_duration: timedelta
    success_probability: float
    
    
class PipelineMonitor(ABC):
    @abstractmethod
    async def collect_metrics(self) -> List[PipelineMetrics]:
        pass
    
    @abstractmethod
    async def detect_failures(self, metrics: List[PipelineMetrics]) -> List[FailureEvent]:
        pass


class FailurePredictor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: List[PipelineMetrics] = []
        self.failure_detector = IsolationForest(contamination=0.1, random_state=42)
        self.trend_analyzer = LinearRegression()
        self.is_trained = False
        
    def add_metrics(self, metrics: PipelineMetrics) -> None:
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
            
        if len(self.metrics_history) >= 20:
            self._retrain_models()
    
    def _extract_features(self, metrics: PipelineMetrics) -> List[float]:
        return [
            metrics.duration_seconds,
            metrics.memory_usage_mb,
            metrics.cpu_usage_percent,
            metrics.success_rate,
            metrics.error_count,
            time.mktime(metrics.timestamp.timetuple()) % 86400  # Time of day
        ]
    
    def _retrain_models(self) -> None:
        if len(self.metrics_history) < 20:
            return
            
        features = [self._extract_features(m) for m in self.metrics_history[-50:]]
        X = np.array(features)
        
        # Train anomaly detector
        self.failure_detector.fit(X)
        
        # Train trend analyzer
        y = [m.success_rate for m in self.metrics_history[-50:]]
        time_features = [[i] for i in range(len(y))]
        self.trend_analyzer.fit(time_features, y)
        
        self.is_trained = True
        logger.info("Failure prediction models retrained")
    
    def predict_failure_probability(self, metrics: PipelineMetrics) -> float:
        if not self.is_trained:
            return 0.0
            
        features = np.array([self._extract_features(metrics)])
        
        # Anomaly score
        anomaly_score = self.failure_detector.decision_function(features)[0]
        anomaly_prob = max(0, -anomaly_score / 2)  # Convert to probability
        
        # Trend degradation
        if len(self.metrics_history) >= 5:
            recent_trend = [m.success_rate for m in self.metrics_history[-5:]]
            time_points = [[i] for i in range(len(recent_trend))]
            trend_slope = self.trend_analyzer.coef_[0] if hasattr(self.trend_analyzer, 'coef_') else 0
            trend_prob = max(0, -trend_slope * 10)  # Negative slope indicates degradation
        else:
            trend_prob = 0
            
        # Combined probability
        return min(1.0, (anomaly_prob + trend_prob) / 2)


class AutoHealer(ABC):
    @abstractmethod
    async def can_heal(self, failure: FailureEvent) -> bool:
        pass
    
    @abstractmethod
    async def heal(self, failure: FailureEvent) -> bool:
        pass
    
    @abstractmethod
    def get_healing_actions(self) -> List[HealingAction]:
        pass


class BasicAutoHealer(AutoHealer):
    def __init__(self):
        self.healing_actions = [
            HealingAction(
                action_id="restart_stage",
                name="Restart Failed Stage",
                description="Restart the failed pipeline stage",
                target_stage=PipelineStage.BUILD,
                priority=1,
                estimated_duration=timedelta(minutes=5),
                success_probability=0.7
            ),
            HealingAction(
                action_id="clear_cache",
                name="Clear Build Cache",
                description="Clear build caches and temporary files",
                target_stage=PipelineStage.BUILD,
                priority=2,
                estimated_duration=timedelta(minutes=2),
                success_probability=0.6
            ),
            HealingAction(
                action_id="increase_resources",
                name="Scale Up Resources",
                description="Increase CPU and memory allocation",
                target_stage=PipelineStage.BUILD,
                priority=3,
                estimated_duration=timedelta(minutes=3),
                success_probability=0.8
            ),
            HealingAction(
                action_id="rollback_dependencies",
                name="Rollback Dependencies",
                description="Rollback to last known good dependency versions",
                target_stage=PipelineStage.BUILD,
                priority=4,
                estimated_duration=timedelta(minutes=10),
                success_probability=0.9
            )
        ]
    
    async def can_heal(self, failure: FailureEvent) -> bool:
        return failure.recovery_attempts < 3 and failure.failure_type != FailureType.PERSISTENT
    
    async def heal(self, failure: FailureEvent) -> bool:
        try:
            logger.info(f"Attempting to heal failure in {failure.stage.value}")
            
            # Select healing action based on failure type and attempts
            action = self._select_healing_action(failure)
            if not action:
                return False
            
            # Simulate healing action
            logger.info(f"Executing healing action: {action.name}")
            await asyncio.sleep(1)  # Simulate action execution
            
            # Simulate success probability
            import random
            success = random.random() < action.success_probability
            
            if success:
                logger.info(f"Healing action {action.name} succeeded")
                failure.resolved = True
            else:
                logger.warning(f"Healing action {action.name} failed")
                failure.recovery_attempts += 1
                
            return success
            
        except Exception as e:
            logger.error(f"Error during healing: {str(e)}")
            return False
    
    def _select_healing_action(self, failure: FailureEvent) -> Optional[HealingAction]:
        # Select action based on failure stage and attempts
        suitable_actions = [
            action for action in self.healing_actions
            if action.target_stage == failure.stage or action.target_stage == PipelineStage.BUILD
        ]
        
        if not suitable_actions:
            return None
            
        # Sort by priority and select based on attempt count
        suitable_actions.sort(key=lambda x: x.priority)
        attempt_index = min(failure.recovery_attempts, len(suitable_actions) - 1)
        return suitable_actions[attempt_index]
    
    def get_healing_actions(self) -> List[HealingAction]:
        return self.healing_actions.copy()


class CIPipelineMonitor(PipelineMonitor):
    def __init__(self):
        self.current_metrics: Dict[PipelineStage, PipelineMetrics] = {}
        
    async def collect_metrics(self) -> List[PipelineMetrics]:
        # Simulate collecting metrics from CI/CD system
        metrics = []
        
        for stage in PipelineStage:
            # Simulate realistic metrics with some variation
            import random
            base_duration = {
                PipelineStage.BUILD: 120,
                PipelineStage.TEST: 180,
                PipelineStage.LINT: 30,
                PipelineStage.SECURITY: 60,
                PipelineStage.DEPLOY: 90,
                PipelineStage.MONITOR: 15
            }
            
            duration = base_duration[stage] + random.uniform(-30, 30)
            memory = random.uniform(500, 2000)
            cpu = random.uniform(20, 90)
            success_rate = random.uniform(0.85, 1.0)
            error_count = random.randint(0, 5) if success_rate < 0.95 else 0
            
            # Introduce occasional failures
            if random.random() < 0.05:  # 5% chance of issues
                success_rate = random.uniform(0.3, 0.8)
                error_count = random.randint(3, 10)
                cpu = random.uniform(70, 100)
                
            status = HealthStatus.HEALTHY
            if success_rate < 0.7:
                status = HealthStatus.CRITICAL
            elif success_rate < 0.9:
                status = HealthStatus.WARNING
                
            metric = PipelineMetrics(
                stage=stage,
                status=status,
                duration_seconds=duration,
                memory_usage_mb=memory,
                cpu_usage_percent=cpu,
                success_rate=success_rate,
                error_count=error_count
            )
            
            metrics.append(metric)
            self.current_metrics[stage] = metric
            
        return metrics
    
    async def detect_failures(self, metrics: List[PipelineMetrics]) -> List[FailureEvent]:
        failures = []
        
        for metric in metrics:
            if metric.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                failure_type = FailureType.TRANSIENT
                if metric.error_count > 5:
                    failure_type = FailureType.PERSISTENT
                elif metric.cpu_usage_percent > 95:
                    failure_type = FailureType.INFRASTRUCTURE
                    
                failure = FailureEvent(
                    stage=metric.stage,
                    failure_type=failure_type,
                    error_message=f"Stage {metric.stage.value} failed with {metric.error_count} errors",
                    stack_trace=None,
                    metrics=metric
                )
                failures.append(failure)
                
        return failures


class SelfHealingPipelineGuard:
    def __init__(
        self,
        monitor: Optional[PipelineMonitor] = None,
        predictor: Optional[FailurePredictor] = None,
        healer: Optional[AutoHealer] = None
    ):
        self.monitor = monitor or CIPipelineMonitor()
        self.predictor = predictor or FailurePredictor()
        self.healer = healer or BasicAutoHealer()
        
        self.active_failures: List[FailureEvent] = []
        self.healing_history: List[FailureEvent] = []
        self.running = False
        self.monitoring_interval = 30  # seconds
        
        # Statistics
        self.total_failures_detected = 0
        self.total_failures_healed = 0
        self.total_healing_attempts = 0
        
    async def start_monitoring(self) -> None:
        if self.running:
            logger.warning("Pipeline guard already running")
            return
            
        self.running = True
        logger.info("Starting self-healing pipeline guard")
        
        try:
            while self.running:
                await self._monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            self.running = False
            
    async def stop_monitoring(self) -> None:
        self.running = False
        logger.info("Stopping self-healing pipeline guard")
        
    async def _monitoring_cycle(self) -> None:
        try:
            # Collect current metrics
            metrics = await self.monitor.collect_metrics()
            
            # Update predictor with new metrics
            for metric in metrics:
                self.predictor.add_metrics(metric)
                
                # Check for failure prediction
                failure_prob = self.predictor.predict_failure_probability(metric)
                if failure_prob > 0.8:
                    logger.warning(
                        f"High failure probability ({failure_prob:.2f}) detected for {metric.stage.value}"
                    )
            
            # Detect actual failures
            new_failures = await self.monitor.detect_failures(metrics)
            
            # Process new failures
            for failure in new_failures:
                self.total_failures_detected += 1
                await self._handle_failure(failure)
                
            # Check on active failures
            await self._check_active_failures()
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
    
    async def _handle_failure(self, failure: FailureEvent) -> None:
        logger.warning(f"Failure detected in {failure.stage.value}: {failure.error_message}")
        
        # Check if we can heal this failure
        if await self.healer.can_heal(failure):
            self.active_failures.append(failure)
            await self._attempt_healing(failure)
        else:
            logger.error(f"Cannot heal failure in {failure.stage.value} - manual intervention required")
            self.healing_history.append(failure)
    
    async def _attempt_healing(self, failure: FailureEvent) -> None:
        self.total_healing_attempts += 1
        logger.info(f"Attempting to heal failure in {failure.stage.value}")
        
        success = await self.healer.heal(failure)
        
        if success:
            self.total_failures_healed += 1
            logger.info(f"Successfully healed failure in {failure.stage.value}")
            failure.resolved = True
            self.healing_history.append(failure)
            if failure in self.active_failures:
                self.active_failures.remove(failure)
        else:
            logger.warning(f"Failed to heal failure in {failure.stage.value}")
            failure.recovery_attempts += 1
    
    async def _check_active_failures(self) -> None:
        # Check if active failures need retry or should be abandoned
        for failure in self.active_failures.copy():
            if failure.resolved:
                continue
                
            # Retry healing if possible
            if await self.healer.can_heal(failure):
                await self._attempt_healing(failure)
            else:
                # Move to history if can't heal anymore
                logger.error(f"Abandoning healing attempts for {failure.stage.value}")
                self.healing_history.append(failure)
                self.active_failures.remove(failure)
    
    def get_health_status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "active_failures": len(self.active_failures),
            "total_failures_detected": self.total_failures_detected,
            "total_failures_healed": self.total_failures_healed,
            "total_healing_attempts": self.total_healing_attempts,
            "healing_success_rate": (
                self.total_failures_healed / max(1, self.total_healing_attempts)
            ),
            "active_failure_stages": [f.stage.value for f in self.active_failures],
            "predictor_trained": self.predictor.is_trained
        }
    
    def get_healing_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        recent_history = self.healing_history[-limit:]
        return [
            {
                "stage": failure.stage.value,
                "failure_type": failure.failure_type.value,
                "error_message": failure.error_message,
                "recovery_attempts": failure.recovery_attempts,
                "resolved": failure.resolved,
                "timestamp": failure.timestamp.isoformat()
            }
            for failure in recent_history
        ]


# Factory function
def create_pipeline_guard(
    monitor_type: str = "ci",
    predictor_config: Optional[Dict[str, Any]] = None,
    healer_type: str = "basic"
) -> SelfHealingPipelineGuard:
    
    # Create monitor
    if monitor_type == "ci":
        monitor = CIPipelineMonitor()
    else:
        raise ValueError(f"Unknown monitor type: {monitor_type}")
    
    # Create predictor
    predictor_config = predictor_config or {}
    predictor = FailurePredictor(**predictor_config)
    
    # Create healer
    if healer_type == "basic":
        healer = BasicAutoHealer()
    else:
        raise ValueError(f"Unknown healer type: {healer_type}")
    
    return SelfHealingPipelineGuard(monitor, predictor, healer)


# Global instance
_global_pipeline_guard: Optional[SelfHealingPipelineGuard] = None


def get_pipeline_guard() -> SelfHealingPipelineGuard:
    global _global_pipeline_guard
    if _global_pipeline_guard is None:
        _global_pipeline_guard = create_pipeline_guard()
    return _global_pipeline_guard


async def start_pipeline_guard() -> None:
    guard = get_pipeline_guard()
    await guard.start_monitoring()


async def stop_pipeline_guard() -> None:
    guard = get_pipeline_guard()
    await guard.stop_monitoring()


def get_pipeline_health() -> Dict[str, Any]:
    guard = get_pipeline_guard()
    return guard.get_health_status()