"""
Intelligent Monitoring System
AI-powered monitoring with adaptive alerting and predictive analytics.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .advanced_monitoring import MonitoringSystem
from .exceptions import ProfilerError


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringPattern(str, Enum):
    """Types of monitoring patterns."""
    THRESHOLD_BREACH = "threshold_breach"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    CORRELATION_ANALYSIS = "correlation_analysis"
    PREDICTIVE_ALERT = "predictive_alert"


@dataclass
class MetricThreshold:
    """Adaptive threshold configuration."""
    metric_name: str
    baseline_value: float
    warning_deviation: float
    critical_deviation: float
    adaptive: bool = True
    learning_rate: float = 0.1
    confidence_interval: float = 0.95


@dataclass
class AlertRule:
    """Intelligent alert rule configuration."""
    rule_id: str
    metric_pattern: MonitoringPattern
    conditions: Dict[str, Any]
    severity: AlertSeverity
    action_callbacks: List[Callable] = field(default_factory=list)
    cooldown_seconds: float = 300.0  # 5 minutes
    adaptive_threshold: bool = True
    last_triggered: Optional[float] = None


@dataclass
class IntelligentAlert:
    """Intelligent alert with context and recommendations."""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    metric_name: str
    current_value: float
    expected_value: float
    deviation: float
    confidence: float
    pattern_type: MonitoringPattern
    context: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    correlation_insights: List[str] = field(default_factory=list)


class AdaptiveThresholdManager:
    """Manages adaptive thresholds that learn from historical data."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)
        
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_statistics: Dict[str, Dict[str, float]] = {}
    
    def register_metric(
        self,
        metric_name: str,
        initial_baseline: float,
        warning_deviation: float = 0.2,  # 20%
        critical_deviation: float = 0.5,  # 50%
        adaptive: bool = True
    ) -> None:
        """Register a metric for adaptive threshold monitoring."""
        threshold = MetricThreshold(
            metric_name=metric_name,
            baseline_value=initial_baseline,
            warning_deviation=warning_deviation,
            critical_deviation=critical_deviation,
            adaptive=adaptive,
            learning_rate=self.learning_rate
        )
        
        self.thresholds[metric_name] = threshold
        self.logger.info(f"Registered adaptive threshold for metric: {metric_name}")
    
    def update_metric_value(self, metric_name: str, value: float) -> None:
        """Update metric value and adapt thresholds if enabled."""
        if metric_name not in self.thresholds:
            return
        
        # Store metric value in history
        self.metric_history[metric_name].append({
            'value': value,
            'timestamp': time.time()
        })
        
        threshold = self.thresholds[metric_name]
        
        if threshold.adaptive:
            self._update_adaptive_threshold(metric_name, value)
    
    def _update_adaptive_threshold(self, metric_name: str, new_value: float) -> None:
        """Update adaptive threshold based on new metric value."""
        threshold = self.thresholds[metric_name]
        history = self.metric_history[metric_name]
        
        if len(history) < 10:  # Need minimum history for adaptation
            return
        
        # Calculate running statistics
        recent_values = [entry['value'] for entry in list(history)[-50:]]  # Last 50 values
        
        mean_value = np.mean(recent_values)
        std_value = np.std(recent_values)
        
        # Update baseline using exponential moving average
        alpha = threshold.learning_rate
        threshold.baseline_value = (
            alpha * mean_value + (1 - alpha) * threshold.baseline_value
        )
        
        # Adapt deviation thresholds based on observed variance
        if std_value > 0:
            # Scale thresholds based on observed variability
            variance_factor = min(2.0, std_value / threshold.baseline_value)
            threshold.warning_deviation = max(0.1, threshold.warning_deviation * variance_factor)
            threshold.critical_deviation = max(0.2, threshold.critical_deviation * variance_factor)
        
        # Update statistics
        self.baseline_statistics[metric_name] = {
            'mean': mean_value,
            'std': std_value,
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'variance_factor': variance_factor if std_value > 0 else 1.0
        }
    
    def check_threshold_violation(
        self,
        metric_name: str,
        current_value: float
    ) -> Optional[Tuple[AlertSeverity, float]]:
        """
        Check if metric value violates thresholds.
        
        Returns:
            Tuple of (severity, deviation) if violation detected, None otherwise
        """
        if metric_name not in self.thresholds:
            return None
        
        threshold = self.thresholds[metric_name]
        baseline = threshold.baseline_value
        
        if baseline == 0:
            return None
        
        deviation = abs(current_value - baseline) / baseline
        
        if deviation > threshold.critical_deviation:
            return AlertSeverity.CRITICAL, deviation
        elif deviation > threshold.warning_deviation:
            return AlertSeverity.WARNING, deviation
        
        return None
    
    def get_threshold_status(self, metric_name: str) -> Dict[str, Any]:
        """Get current threshold status for a metric."""
        if metric_name not in self.thresholds:
            return {"error": "Metric not registered"}
        
        threshold = self.thresholds[metric_name]
        statistics = self.baseline_statistics.get(metric_name, {})
        history_count = len(self.metric_history[metric_name])
        
        return {
            "metric_name": metric_name,
            "baseline_value": threshold.baseline_value,
            "warning_threshold": threshold.baseline_value * (1 + threshold.warning_deviation),
            "critical_threshold": threshold.baseline_value * (1 + threshold.critical_deviation),
            "adaptive": threshold.adaptive,
            "history_count": history_count,
            "statistics": statistics
        }


class AnomalyDetector:
    """Advanced anomaly detection using statistical methods."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Number of standard deviations for anomaly threshold
        self.logger = logging.getLogger(__name__)
        
        self.metric_models: Dict[str, Dict[str, Any]] = {}
        self.anomaly_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def train_model(self, metric_name: str, training_data: List[float]) -> None:
        """Train anomaly detection model for a metric."""
        if len(training_data) < 30:
            self.logger.warning(f"Insufficient training data for {metric_name}: {len(training_data)} samples")
            return
        
        # Calculate statistical parameters
        mean_value = np.mean(training_data)
        std_value = np.std(training_data)
        
        # Calculate percentiles for robust anomaly detection
        percentiles = np.percentile(training_data, [1, 5, 25, 50, 75, 95, 99])
        
        # Store model parameters
        self.metric_models[metric_name] = {
            'mean': mean_value,
            'std': std_value,
            'percentiles': percentiles,
            'min_value': np.min(training_data),
            'max_value': np.max(training_data),
            'training_size': len(training_data),
            'last_updated': time.time()
        }
        
        self.logger.info(f"Trained anomaly detection model for {metric_name}")
    
    def detect_anomaly(
        self,
        metric_name: str,
        current_value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Detect anomalies in metric values.
        
        Returns:
            Anomaly details if detected, None otherwise
        """
        if metric_name not in self.metric_models:
            return None
        
        model = self.metric_models[metric_name]
        
        # Z-score based detection
        z_score = abs(current_value - model['mean']) / model['std'] if model['std'] > 0 else 0
        
        # Percentile-based detection
        percentiles = model['percentiles']
        is_outlier_percentile = (
            current_value < percentiles[0] or  # Below 1st percentile
            current_value > percentiles[6]     # Above 99th percentile
        )
        
        # Determine if anomaly
        is_anomaly_zscore = z_score > self.sensitivity
        is_anomaly = is_anomaly_zscore or is_outlier_percentile
        
        if is_anomaly:
            anomaly_info = {
                'metric_name': metric_name,
                'current_value': current_value,
                'expected_range': (percentiles[1], percentiles[5]),  # 5th to 95th percentile
                'z_score': z_score,
                'anomaly_type': self._classify_anomaly_type(current_value, model),
                'severity': self._calculate_anomaly_severity(z_score, is_outlier_percentile),
                'confidence': self._calculate_confidence(z_score, percentiles, current_value),
                'timestamp': time.time(),
                'context': context or {}
            }
            
            # Store in history
            self.anomaly_history[metric_name].append(anomaly_info)
            
            return anomaly_info
        
        return None
    
    def _classify_anomaly_type(self, value: float, model: Dict[str, Any]) -> str:
        """Classify the type of anomaly detected."""
        mean_value = model['mean']
        percentiles = model['percentiles']
        
        if value > percentiles[6]:  # Above 99th percentile
            return "extreme_high"
        elif value > percentiles[5]:  # Above 95th percentile
            return "high"
        elif value < percentiles[0]:  # Below 1st percentile
            return "extreme_low"
        elif value < percentiles[1]:  # Below 5th percentile
            return "low"
        else:
            return "statistical_outlier"
    
    def _calculate_anomaly_severity(
        self,
        z_score: float,
        is_outlier_percentile: bool
    ) -> AlertSeverity:
        """Calculate severity of detected anomaly."""
        if z_score > 4.0 or is_outlier_percentile:
            return AlertSeverity.CRITICAL
        elif z_score > 3.0:
            return AlertSeverity.ERROR
        elif z_score > 2.0:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _calculate_confidence(
        self,
        z_score: float,
        percentiles: np.ndarray,
        current_value: float
    ) -> float:
        """Calculate confidence in anomaly detection."""
        # Base confidence on z-score magnitude
        confidence = min(1.0, z_score / 5.0)  # Normalize to 0-1
        
        # Boost confidence for extreme percentile violations
        if current_value < percentiles[0] or current_value > percentiles[6]:
            confidence = max(confidence, 0.9)
        
        return confidence


class CorrelationAnalyzer:
    """Analyzes correlations between different metrics."""
    
    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold
        self.logger = logging.getLogger(__name__)
        
        self.metric_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.correlation_patterns: List[Dict[str, Any]] = []
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update metric values for correlation analysis."""
        timestamp = time.time()
        
        for metric_name, value in metrics.items():
            self.metric_data[metric_name].append({
                'value': value,
                'timestamp': timestamp
            })
        
        # Update correlations if we have enough data
        if len(list(self.metric_data.values())[0]) >= 50:
            self._update_correlations()
    
    def _update_correlations(self) -> None:
        """Update correlation matrix between metrics."""
        metric_names = list(self.metric_data.keys())
        
        for i, metric1 in enumerate(metric_names):
            for j, metric2 in enumerate(metric_names[i+1:], i+1):
                correlation = self._calculate_correlation(metric1, metric2)
                self.correlation_matrix[(metric1, metric2)] = correlation
                
                # Detect strong correlations
                if abs(correlation) > self.correlation_threshold:
                    pattern = {
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': correlation,
                        'type': 'positive' if correlation > 0 else 'negative',
                        'strength': abs(correlation),
                        'discovered_at': time.time()
                    }
                    
                    # Add to patterns if not already present
                    if not any(p['metric1'] == metric1 and p['metric2'] == metric2 
                             for p in self.correlation_patterns):
                        self.correlation_patterns.append(pattern)
    
    def _calculate_correlation(self, metric1: str, metric2: str) -> float:
        """Calculate correlation between two metrics."""
        data1 = [entry['value'] for entry in self.metric_data[metric1]]
        data2 = [entry['value'] for entry in self.metric_data[metric2]]
        
        # Ensure same length
        min_length = min(len(data1), len(data2))
        data1 = data1[-min_length:]
        data2 = data2[-min_length:]
        
        if min_length < 10:
            return 0.0
        
        correlation_matrix = np.corrcoef(data1, data2)
        return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
    
    def analyze_alert_correlations(
        self,
        triggered_metric: str,
        current_metrics: Dict[str, float]
    ) -> List[str]:
        """Analyze correlations when an alert is triggered."""
        insights = []
        
        # Find correlated metrics
        for (metric1, metric2), correlation in self.correlation_matrix.items():
            if abs(correlation) > self.correlation_threshold:
                if metric1 == triggered_metric:
                    correlated_metric = metric2
                elif metric2 == triggered_metric:
                    correlated_metric = metric1
                else:
                    continue
                
                if correlated_metric in current_metrics:
                    current_value = current_metrics[correlated_metric]
                    correlation_type = "positively" if correlation > 0 else "negatively"
                    
                    insight = (
                        f"{correlated_metric} is {correlation_type} correlated "
                        f"(r={correlation:.2f}) with {triggered_metric}. "
                        f"Current value: {current_value:.2f}"
                    )
                    insights.append(insight)
        
        return insights


class IntelligentMonitoringSystem:
    """Advanced monitoring system with AI-powered analytics."""
    
    def __init__(self, base_monitoring: Optional[MonitoringSystem] = None):
        self.base_monitoring = base_monitoring or MonitoringSystem()
        self.logger = logging.getLogger(__name__)
        
        self.threshold_manager = AdaptiveThresholdManager()
        self.anomaly_detector = AnomalyDetector()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, IntelligentAlert] = {}
        self.alert_history: List[IntelligentAlert] = []
        
        self.monitoring_active = False
        self.monitoring_task = None
        
    async def start_intelligent_monitoring(
        self,
        monitoring_interval_seconds: float = 5.0
    ) -> None:
        """Start intelligent monitoring with AI analytics."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._intelligent_monitoring_loop(monitoring_interval_seconds)
        )
        
        self.logger.info("Started intelligent monitoring system")
    
    async def stop_intelligent_monitoring(self) -> None:
        """Stop intelligent monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped intelligent monitoring system")
    
    async def _intelligent_monitoring_loop(self, interval_seconds: float) -> None:
        """Main intelligent monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = await self._collect_comprehensive_metrics()
                
                # Update adaptive thresholds
                for metric_name, value in current_metrics.items():
                    self.threshold_manager.update_metric_value(metric_name, value)
                
                # Update correlation analysis
                self.correlation_analyzer.update_metrics(current_metrics)
                
                # Check for threshold violations
                await self._check_threshold_violations(current_metrics)
                
                # Detect anomalies
                await self._detect_anomalies(current_metrics)
                
                # Process alert rules
                await self._process_alert_rules(current_metrics)
                
                # Clean up expired alerts
                await self._cleanup_expired_alerts()
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in intelligent monitoring loop: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_comprehensive_metrics(self) -> Dict[str, float]:
        """Collect comprehensive system metrics."""
        # This would integrate with actual monitoring systems
        return {
            'cpu_usage_percent': np.random.normal(50, 10),
            'memory_usage_kb': np.random.normal(300, 50),
            'disk_io_ops_sec': np.random.normal(100, 20),
            'network_latency_ms': np.random.normal(50, 10),
            'profiling_latency_ms': np.random.normal(100, 25),
            'power_consumption_mw': np.random.normal(250, 30),
            'temperature_celsius': np.random.normal(45, 5),
            'error_rate_percent': np.random.exponential(1.0),
            'throughput_ops_sec': np.random.normal(75, 15),
            'queue_depth': np.random.poisson(5)
        }
    
    async def _check_threshold_violations(self, current_metrics: Dict[str, float]) -> None:
        """Check for threshold violations and generate alerts."""
        for metric_name, value in current_metrics.items():
            violation = self.threshold_manager.check_threshold_violation(metric_name, value)
            
            if violation:
                severity, deviation = violation
                await self._generate_threshold_alert(metric_name, value, severity, deviation)
    
    async def _detect_anomalies(self, current_metrics: Dict[str, float]) -> None:
        """Detect anomalies and generate alerts."""
        for metric_name, value in current_metrics.items():
            anomaly = self.anomaly_detector.detect_anomaly(metric_name, value)
            
            if anomaly:
                await self._generate_anomaly_alert(anomaly)
    
    async def _generate_threshold_alert(
        self,
        metric_name: str,
        current_value: float,
        severity: AlertSeverity,
        deviation: float
    ) -> None:
        """Generate alert for threshold violation."""
        alert_id = f"threshold_{metric_name}_{int(time.time())}"
        
        threshold_info = self.threshold_manager.get_threshold_status(metric_name)
        expected_value = threshold_info.get('baseline_value', 0)
        
        # Get correlation insights
        current_metrics = await self._collect_comprehensive_metrics()
        correlation_insights = self.correlation_analyzer.analyze_alert_correlations(
            metric_name, current_metrics
        )
        
        alert = IntelligentAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            expected_value=expected_value,
            deviation=deviation,
            confidence=0.8,  # High confidence for threshold violations
            pattern_type=MonitoringPattern.THRESHOLD_BREACH,
            context={
                'threshold_info': threshold_info,
                'violation_type': 'adaptive_threshold'
            },
            recommendations=self._generate_threshold_recommendations(metric_name, deviation),
            correlation_insights=correlation_insights
        )
        
        await self._process_alert(alert)
    
    async def _generate_anomaly_alert(self, anomaly_info: Dict[str, Any]) -> None:
        """Generate alert for detected anomaly."""
        alert_id = f"anomaly_{anomaly_info['metric_name']}_{int(time.time())}"
        
        alert = IntelligentAlert(
            alert_id=alert_id,
            timestamp=anomaly_info['timestamp'],
            severity=anomaly_info['severity'],
            metric_name=anomaly_info['metric_name'],
            current_value=anomaly_info['current_value'],
            expected_value=anomaly_info['expected_range'][0],  # Use lower bound as expected
            deviation=anomaly_info['z_score'],
            confidence=anomaly_info['confidence'],
            pattern_type=MonitoringPattern.ANOMALY_DETECTION,
            context={
                'anomaly_type': anomaly_info['anomaly_type'],
                'z_score': anomaly_info['z_score'],
                'expected_range': anomaly_info['expected_range']
            },
            recommendations=self._generate_anomaly_recommendations(anomaly_info),
            correlation_insights=[]
        )
        
        await self._process_alert(alert)
    
    def _generate_threshold_recommendations(
        self,
        metric_name: str,
        deviation: float
    ) -> List[str]:
        """Generate recommendations for threshold violations."""
        recommendations = []
        
        if metric_name == 'memory_usage_kb':
            if deviation > 0.5:
                recommendations.append("Consider implementing memory optimization")
            recommendations.append("Monitor for memory leaks")
            
        elif metric_name == 'cpu_usage_percent':
            if deviation > 0.3:
                recommendations.append("Investigate high CPU usage patterns")
            recommendations.append("Consider workload balancing")
            
        elif metric_name == 'profiling_latency_ms':
            recommendations.append("Analyze profiling pipeline for bottlenecks")
            recommendations.append("Consider algorithm optimization")
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, anomaly_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations for anomalies."""
        recommendations = []
        anomaly_type = anomaly_info['anomaly_type']
        
        if anomaly_type in ['extreme_high', 'high']:
            recommendations.append("Investigate cause of unusually high values")
            recommendations.append("Check for system overload conditions")
            
        elif anomaly_type in ['extreme_low', 'low']:
            recommendations.append("Verify system is functioning correctly")
            recommendations.append("Check for performance degradation")
            
        else:
            recommendations.append("Monitor for recurring statistical outliers")
            recommendations.append("Consider adjusting anomaly detection sensitivity")
        
        return recommendations
    
    async def _process_alert_rules(self, current_metrics: Dict[str, float]) -> None:
        """Process custom alert rules."""
        for rule_id, rule in self.alert_rules.items():
            if await self._evaluate_alert_rule(rule, current_metrics):
                await self._trigger_rule_alert(rule, current_metrics)
    
    async def _evaluate_alert_rule(
        self,
        rule: AlertRule,
        current_metrics: Dict[str, float]
    ) -> bool:
        """Evaluate if an alert rule should trigger."""
        # Check cooldown period
        if rule.last_triggered:
            time_since_last = time.time() - rule.last_triggered
            if time_since_last < rule.cooldown_seconds:
                return False
        
        # Evaluate rule conditions
        conditions = rule.conditions
        
        if rule.metric_pattern == MonitoringPattern.THRESHOLD_BREACH:
            metric_name = conditions.get('metric_name')
            threshold = conditions.get('threshold')
            
            if metric_name in current_metrics:
                return current_metrics[metric_name] > threshold
                
        elif rule.metric_pattern == MonitoringPattern.CORRELATION_ANALYSIS:
            # Complex correlation-based rules would be implemented here
            pass
        
        return False
    
    async def _trigger_rule_alert(
        self,
        rule: AlertRule,
        current_metrics: Dict[str, float]
    ) -> None:
        """Trigger alert for custom rule."""
        rule.last_triggered = time.time()
        
        # Execute rule callbacks
        for callback in rule.action_callbacks:
            try:
                await callback(rule, current_metrics)
            except Exception as e:
                self.logger.error(f"Error executing alert callback: {str(e)}")
    
    async def _process_alert(self, alert: IntelligentAlert) -> None:
        """Process and store alert."""
        # Check if similar alert is already active
        existing_alert_key = f"{alert.metric_name}_{alert.pattern_type.value}"
        
        if existing_alert_key in self.active_alerts:
            # Update existing alert
            existing_alert = self.active_alerts[existing_alert_key]
            if alert.severity.value > existing_alert.severity.value:
                self.active_alerts[existing_alert_key] = alert
        else:
            # Add new alert
            self.active_alerts[existing_alert_key] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(
            f"Alert generated: {alert.severity.value} - {alert.metric_name} - "
            f"Current: {alert.current_value:.2f}, Expected: {alert.expected_value:.2f}"
        )
    
    async def _cleanup_expired_alerts(self) -> None:
        """Clean up expired active alerts."""
        current_time = time.time()
        expired_keys = []
        
        for key, alert in self.active_alerts.items():
            # Consider alerts expired after 1 hour
            if current_time - alert.timestamp > 3600:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_alerts[key]
    
    def register_metric_threshold(
        self,
        metric_name: str,
        baseline_value: float,
        warning_deviation: float = 0.2,
        critical_deviation: float = 0.5
    ) -> None:
        """Register adaptive threshold for a metric."""
        self.threshold_manager.register_metric(
            metric_name, baseline_value, warning_deviation, critical_deviation
        )
    
    def train_anomaly_detector(
        self,
        metric_name: str,
        training_data: List[float]
    ) -> None:
        """Train anomaly detector for a metric."""
        self.anomaly_detector.train_model(metric_name, training_data)
    
    def register_alert_rule(
        self,
        rule_id: str,
        metric_pattern: MonitoringPattern,
        conditions: Dict[str, Any],
        severity: AlertSeverity,
        action_callbacks: Optional[List[Callable]] = None
    ) -> None:
        """Register custom alert rule."""
        rule = AlertRule(
            rule_id=rule_id,
            metric_pattern=metric_pattern,
            conditions=conditions,
            severity=severity,
            action_callbacks=action_callbacks or []
        )
        
        self.alert_rules[rule_id] = rule
        self.logger.info(f"Registered alert rule: {rule_id}")
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            'system_status': {
                'monitoring_active': self.monitoring_active,
                'total_metrics_tracked': len(self.threshold_manager.thresholds),
                'active_alerts': len(self.active_alerts),
                'total_alert_history': len(self.alert_history)
            },
            'threshold_manager': {
                'adaptive_thresholds': len([t for t in self.threshold_manager.thresholds.values() if t.adaptive]),
                'metrics_with_baseline': len(self.threshold_manager.baseline_statistics)
            },
            'anomaly_detector': {
                'trained_models': len(self.anomaly_detector.metric_models),
                'total_anomalies_detected': sum(len(history) for history in self.anomaly_detector.anomaly_history.values())
            },
            'correlation_analyzer': {
                'correlation_patterns': len(self.correlation_analyzer.correlation_patterns),
                'metrics_analyzed': len(self.correlation_analyzer.metric_data)
            },
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'metric_name': alert.metric_name,
                    'timestamp': alert.timestamp
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ]
        }


# Global intelligent monitoring instance
_global_intelligent_monitoring: Optional[IntelligentMonitoringSystem] = None

def get_intelligent_monitoring() -> IntelligentMonitoringSystem:
    """Get global intelligent monitoring instance."""
    global _global_intelligent_monitoring
    if _global_intelligent_monitoring is None:
        _global_intelligent_monitoring = IntelligentMonitoringSystem()
    return _global_intelligent_monitoring

async def start_intelligent_monitoring(
    monitoring_interval_seconds: float = 5.0
) -> None:
    """Start global intelligent monitoring."""
    monitoring = get_intelligent_monitoring()
    await monitoring.start_intelligent_monitoring(monitoring_interval_seconds)

async def stop_intelligent_monitoring() -> None:
    """Stop global intelligent monitoring."""
    monitoring = get_intelligent_monitoring()
    await monitoring.stop_intelligent_monitoring()

def register_smart_threshold(
    metric_name: str,
    baseline_value: float,
    warning_deviation: float = 0.2,
    critical_deviation: float = 0.5
) -> None:
    """Register adaptive threshold for intelligent monitoring."""
    monitoring = get_intelligent_monitoring()
    monitoring.register_metric_threshold(
        metric_name, baseline_value, warning_deviation, critical_deviation
    )