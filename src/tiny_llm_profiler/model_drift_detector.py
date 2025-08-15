"""
Model Drift Detection System
Detect and prevent LLM performance degradation over time
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class DriftType(Enum):
    PERFORMANCE = "performance"
    DISTRIBUTION = "distribution"
    CONCEPT = "concept"
    FEATURE = "feature"


class DriftSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelPerformanceMetrics:
    accuracy: float
    f1_score: float
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    sample_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAlert:
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    current_value: float
    baseline_value: float
    drift_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    recommendations: List[str] = field(default_factory=list)


class DriftDetector(ABC):
    @abstractmethod
    def detect_drift(
        self,
        current_metrics: ModelPerformanceMetrics,
        baseline_metrics: ModelPerformanceMetrics
    ) -> List[DriftAlert]:
        pass


class StatisticalDriftDetector(DriftDetector):
    def __init__(self, 
                 performance_threshold: float = 0.05,
                 distribution_threshold: float = 0.1):
        self.performance_threshold = performance_threshold
        self.distribution_threshold = distribution_threshold
        
    def detect_drift(
        self,
        current_metrics: ModelPerformanceMetrics,
        baseline_metrics: ModelPerformanceMetrics
    ) -> List[DriftAlert]:
        alerts = []
        
        # Performance drift detection
        perf_alerts = self._detect_performance_drift(current_metrics, baseline_metrics)
        alerts.extend(perf_alerts)
        
        # Latency drift detection
        latency_alerts = self._detect_latency_drift(current_metrics, baseline_metrics)
        alerts.extend(latency_alerts)
        
        return alerts
    
    def _detect_performance_drift(
        self,
        current: ModelPerformanceMetrics,
        baseline: ModelPerformanceMetrics
    ) -> List[DriftAlert]:
        alerts = []
        
        # Check accuracy drift
        accuracy_drift = baseline.accuracy - current.accuracy
        if accuracy_drift > self.performance_threshold:
            severity = self._calculate_severity(accuracy_drift, self.performance_threshold)
            alerts.append(DriftAlert(
                drift_type=DriftType.PERFORMANCE,
                severity=severity,
                metric_name="accuracy",
                current_value=current.accuracy,
                baseline_value=baseline.accuracy,
                drift_score=accuracy_drift,
                description=f"Model accuracy dropped by {accuracy_drift:.3f}",
                recommendations=[
                    "Check for data distribution changes",
                    "Consider model retraining",
                    "Verify input data quality"
                ]
            ))
        
        # Check F1 score drift
        f1_drift = baseline.f1_score - current.f1_score
        if f1_drift > self.performance_threshold:
            severity = self._calculate_severity(f1_drift, self.performance_threshold)
            alerts.append(DriftAlert(
                drift_type=DriftType.PERFORMANCE,
                severity=severity,
                metric_name="f1_score",
                current_value=current.f1_score,
                baseline_value=baseline.f1_score,
                drift_score=f1_drift,
                description=f"Model F1 score dropped by {f1_drift:.3f}",
                recommendations=[
                    "Analyze class distribution changes",
                    "Review model performance on edge cases",
                    "Consider ensemble methods"
                ]
            ))
        
        return alerts
    
    def _detect_latency_drift(
        self,
        current: ModelPerformanceMetrics,
        baseline: ModelPerformanceMetrics
    ) -> List[DriftAlert]:
        alerts = []
        
        # Check latency increase
        latency_increase = (current.latency_ms - baseline.latency_ms) / baseline.latency_ms
        if latency_increase > 0.2:  # 20% increase threshold
            severity = self._calculate_severity(latency_increase, 0.2)
            alerts.append(DriftAlert(
                drift_type=DriftType.PERFORMANCE,
                severity=severity,
                metric_name="latency",
                current_value=current.latency_ms,
                baseline_value=baseline.latency_ms,
                drift_score=latency_increase,
                description=f"Model latency increased by {latency_increase*100:.1f}%",
                recommendations=[
                    "Check system resource usage",
                    "Optimize model inference pipeline",
                    "Scale infrastructure resources"
                ]
            ))
        
        return alerts
    
    def _calculate_severity(self, drift_value: float, threshold: float) -> DriftSeverity:
        ratio = drift_value / threshold
        if ratio < 1.5:
            return DriftSeverity.LOW
        elif ratio < 3.0:
            return DriftSeverity.MEDIUM
        elif ratio < 5.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


class DataDistributionDriftDetector(DriftDetector):
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.baseline_features: Optional[np.ndarray] = None
        
    def fit_baseline(self, baseline_texts: List[str]) -> None:
        self.baseline_features = self.vectorizer.fit_transform(baseline_texts)
        logger.info(f"Fitted baseline distribution with {len(baseline_texts)} samples")
    
    def detect_drift(
        self,
        current_metrics: ModelPerformanceMetrics,
        baseline_metrics: ModelPerformanceMetrics
    ) -> List[DriftAlert]:
        # This detector needs text data, which would come from metadata
        if "input_texts" not in current_metrics.metadata:
            return []
            
        current_texts = current_metrics.metadata["input_texts"]
        if not current_texts or self.baseline_features is None:
            return []
            
        try:
            current_features = self.vectorizer.transform(current_texts)
            drift_score = self._calculate_distribution_drift(current_features)
            
            if drift_score > self.threshold:
                severity = self._calculate_severity(drift_score)
                return [DriftAlert(
                    drift_type=DriftType.DISTRIBUTION,
                    severity=severity,
                    metric_name="input_distribution",
                    current_value=drift_score,
                    baseline_value=0.0,
                    drift_score=drift_score,
                    description=f"Input data distribution drift detected (score: {drift_score:.3f})",
                    recommendations=[
                        "Analyze input data patterns",
                        "Update training data with recent samples",
                        "Consider domain adaptation techniques"
                    ]
                )]
            
        except Exception as e:
            logger.error(f"Error in distribution drift detection: {str(e)}")
            
        return []
    
    def _calculate_distribution_drift(self, current_features: np.ndarray) -> float:
        # Calculate average cosine similarity between baseline and current features
        baseline_centroid = np.mean(self.baseline_features.toarray(), axis=0)
        current_centroid = np.mean(current_features.toarray(), axis=0)
        
        similarity = cosine_similarity(
            baseline_centroid.reshape(1, -1),
            current_centroid.reshape(1, -1)
        )[0, 0]
        
        # Convert similarity to drift score (1 - similarity)
        return 1.0 - similarity
    
    def _calculate_severity(self, drift_score: float) -> DriftSeverity:
        if drift_score < 0.2:
            return DriftSeverity.LOW
        elif drift_score < 0.4:
            return DriftSeverity.MEDIUM
        elif drift_score < 0.6:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


class ModelDriftMonitor:
    def __init__(self, 
                 window_size: int = 100,
                 baseline_update_interval: timedelta = timedelta(days=7)):
        self.window_size = window_size
        self.baseline_update_interval = baseline_update_interval
        
        self.metrics_history: List[ModelPerformanceMetrics] = []
        self.baseline_metrics: Optional[ModelPerformanceMetrics] = None
        self.last_baseline_update: Optional[datetime] = None
        
        # Detectors
        self.statistical_detector = StatisticalDriftDetector()
        self.distribution_detector = DataDistributionDriftDetector()
        
        # Alerts
        self.active_alerts: List[DriftAlert] = []
        self.alert_history: List[DriftAlert] = []
        
    def add_metrics(self, metrics: ModelPerformanceMetrics) -> List[DriftAlert]:
        self.metrics_history.append(metrics)
        
        # Maintain window size
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
        
        # Update baseline if needed
        self._update_baseline_if_needed()
        
        # Detect drift if we have a baseline
        alerts = []
        if self.baseline_metrics:
            alerts = self._detect_drift(metrics)
            
        # Update alert lists
        for alert in alerts:
            if alert not in self.active_alerts:
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
                logger.warning(f"Drift alert: {alert.description}")
        
        return alerts
    
    def _update_baseline_if_needed(self) -> None:
        now = datetime.now()
        
        # Initialize baseline with first metrics
        if self.baseline_metrics is None and self.metrics_history:
            self.baseline_metrics = self.metrics_history[0]
            self.last_baseline_update = now
            logger.info("Initialized baseline metrics")
            return
        
        # Update baseline periodically
        if (self.last_baseline_update and 
            now - self.last_baseline_update > self.baseline_update_interval):
            
            if len(self.metrics_history) >= 10:
                # Use recent stable period as new baseline
                recent_metrics = self.metrics_history[-10:]
                self.baseline_metrics = self._calculate_average_metrics(recent_metrics)
                self.last_baseline_update = now
                logger.info("Updated baseline metrics")
    
    def _calculate_average_metrics(self, metrics_list: List[ModelPerformanceMetrics]) -> ModelPerformanceMetrics:
        if not metrics_list:
            raise ValueError("Cannot calculate average of empty metrics list")
        
        avg_accuracy = np.mean([m.accuracy for m in metrics_list])
        avg_f1 = np.mean([m.f1_score for m in metrics_list])
        avg_latency = np.mean([m.latency_ms for m in metrics_list])
        avg_throughput = np.mean([m.throughput_tokens_per_sec for m in metrics_list])
        avg_memory = np.mean([m.memory_usage_mb for m in metrics_list])
        avg_error_rate = np.mean([m.error_rate for m in metrics_list])
        
        return ModelPerformanceMetrics(
            accuracy=avg_accuracy,
            f1_score=avg_f1,
            latency_ms=avg_latency,
            throughput_tokens_per_sec=avg_throughput,
            memory_usage_mb=avg_memory,
            error_rate=avg_error_rate,
            sample_count=sum(m.sample_count for m in metrics_list)
        )
    
    def _detect_drift(self, metrics: ModelPerformanceMetrics) -> List[DriftAlert]:
        alerts = []
        
        # Statistical drift detection
        stat_alerts = self.statistical_detector.detect_drift(metrics, self.baseline_metrics)
        alerts.extend(stat_alerts)
        
        # Distribution drift detection
        dist_alerts = self.distribution_detector.detect_drift(metrics, self.baseline_metrics)
        alerts.extend(dist_alerts)
        
        return alerts
    
    def clear_alerts(self, alert_types: Optional[List[DriftType]] = None) -> None:
        if alert_types is None:
            self.active_alerts.clear()
        else:
            self.active_alerts = [
                alert for alert in self.active_alerts
                if alert.drift_type not in alert_types
            ]
    
    def get_drift_summary(self) -> Dict[str, Any]:
        return {
            "baseline_timestamp": self.baseline_metrics.timestamp.isoformat() if self.baseline_metrics else None,
            "active_alerts_count": len(self.active_alerts),
            "total_alerts_count": len(self.alert_history),
            "metrics_history_length": len(self.metrics_history),
            "active_alerts_by_type": {
                drift_type.value: len([
                    alert for alert in self.active_alerts
                    if alert.drift_type == drift_type
                ])
                for drift_type in DriftType
            },
            "active_alerts_by_severity": {
                severity.value: len([
                    alert for alert in self.active_alerts
                    if alert.severity == severity
                ])
                for severity in DriftSeverity
            }
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        return [
            {
                "drift_type": alert.drift_type.value,
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "baseline_value": alert.baseline_value,
                "drift_score": alert.drift_score,
                "timestamp": alert.timestamp.isoformat(),
                "description": alert.description,
                "recommendations": alert.recommendations
            }
            for alert in self.active_alerts
        ]


# Global drift monitor instance
_global_drift_monitor: Optional[ModelDriftMonitor] = None


def get_drift_monitor() -> ModelDriftMonitor:
    global _global_drift_monitor
    if _global_drift_monitor is None:
        _global_drift_monitor = ModelDriftMonitor()
    return _global_drift_monitor


def monitor_model_performance(metrics: ModelPerformanceMetrics) -> List[DriftAlert]:
    monitor = get_drift_monitor()
    return monitor.add_metrics(metrics)


def get_drift_status() -> Dict[str, Any]:
    monitor = get_drift_monitor()
    return monitor.get_drift_summary()


def clear_drift_alerts(alert_types: Optional[List[str]] = None) -> None:
    monitor = get_drift_monitor()
    if alert_types:
        drift_types = [DriftType(t) for t in alert_types]
        monitor.clear_alerts(drift_types)
    else:
        monitor.clear_alerts()