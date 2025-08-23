"""
Advanced monitoring and alerting system for the Tiny LLM Edge Profiler.

This module provides comprehensive monitoring capabilities including:
- Real-time metrics collection and aggregation
- Alert management with configurable thresholds
- Health monitoring with automated recovery
- Performance analytics and trend analysis
- Integration with external monitoring systems
"""

import time
import threading
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import defaultdict, deque

import numpy as np
import psutil

from .exceptions import TinyLLMProfilerError
from .logging_config import get_logger, PerformanceLogger
from .reliability import ReliabilityManager, CircuitBreaker
from .config import get_config


class MetricType(str, Enum):
    """Types of metrics to collect."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(str, Enum):
    """Alert states."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class Metric:
    """Individual metric data point."""

    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    severity: AlertSeverity
    duration_seconds: float = 60.0
    description: str = ""
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance."""

    id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    state: AlertState
    started_at: datetime
    resolved_at: Optional[datetime] = None
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        result = asdict(self)
        result["started_at"] = self.started_at.isoformat()
        if self.resolved_at:
            result["resolved_at"] = self.resolved_at.isoformat()
        return result


class MetricCollector:
    """Collects and stores metrics with efficient aggregation."""

    def __init__(self, max_retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.max_retention_hours = max_retention_hours
        self.lock = threading.RLock()
        self.logger = get_logger("metric_collector")

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_running = True
        self.cleanup_thread.start()

    def record(self, metric: Metric):
        """Record a metric data point."""
        with self.lock:
            metric_key = self._get_metric_key(metric.name, metric.labels)
            self.metrics[metric_key].append(metric)

            self.logger.debug(
                f"Recorded metric: {metric.name}={metric.value}",
                metric_name=metric.name,
                value=metric.value,
                labels=metric.labels,
            )

    def get_metrics(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Metric]:
        """Get metrics matching the criteria."""
        with self.lock:
            metric_key = self._get_metric_key(metric_name, labels or {})

            if metric_key not in self.metrics:
                return []

            metrics = list(self.metrics[metric_key])

            # Filter by time range
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_metrics.append(metric)
                metrics = filtered_metrics

            return metrics

    def get_latest(
        self, metric_name: str, labels: Optional[Dict[str, str]] = None
    ) -> Optional[Metric]:
        """Get the latest metric value."""
        metrics = self.get_metrics(metric_name, labels)
        return metrics[-1] if metrics else None

    def get_aggregated(
        self,
        metric_name: str,
        aggregation: str,  # "avg", "sum", "min", "max", "count"
        labels: Optional[Dict[str, str]] = None,
        duration_minutes: int = 10,
    ) -> Optional[float]:
        """Get aggregated metric value over a time period."""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration_minutes)

        metrics = self.get_metrics(metric_name, labels, start_time, end_time)

        if not metrics:
            return None

        values = [m.value for m in metrics]

        if aggregation == "avg":
            return np.mean(values)
        elif aggregation == "sum":
            return np.sum(values)
        elif aggregation == "min":
            return np.min(values)
        elif aggregation == "max":
            return np.max(values)
        elif aggregation == "count":
            return len(values)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name

        label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(label_parts)}}}"

    def _cleanup_loop(self):
        """Background cleanup of old metrics."""
        while self.cleanup_running:
            try:
                self._cleanup_old_metrics()
                time.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in metric cleanup: {e}")

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.max_retention_hours)

        with self.lock:
            for metric_key, metric_deque in self.metrics.items():
                # Remove old metrics from the beginning of deque
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()

    def stop(self):
        """Stop the metric collector."""
        self.cleanup_running = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)


class AlertManager:
    """Manages alerting rules and active alerts."""

    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []

        self.evaluation_interval = 30  # seconds
        self.evaluation_thread: Optional[threading.Thread] = None
        self.evaluation_running = False

        self.logger = get_logger("alert_manager")

        # Setup default alert rules
        self._setup_default_rules()

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)

    def start(self):
        """Start the alert evaluation loop."""
        if self.evaluation_running:
            return

        self.evaluation_running = True
        self.evaluation_thread = threading.Thread(
            target=self._evaluation_loop, daemon=True
        )
        self.evaluation_thread.start()
        self.logger.info("Alert manager started")

    def stop(self):
        """Stop the alert evaluation loop."""
        self.evaluation_running = False
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=5.0)
        self.logger.info("Alert manager stopped")

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get currently active alerts."""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alert_history if a.started_at >= cutoff_time]

    def silence_alert(self, alert_id: str, duration_minutes: int = 60):
        """Silence an active alert for a specified duration."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.state = AlertState.SILENCED

            # Schedule un-silencing
            def unsilence():
                time.sleep(duration_minutes * 60)
                if alert_id in self.active_alerts:
                    self.active_alerts[alert_id].state = AlertState.ACTIVE

            threading.Thread(target=unsilence, daemon=True).start()
            self.logger.info(
                f"Silenced alert {alert_id} for {duration_minutes} minutes"
            )

    def _setup_default_rules(self):
        """Setup default alerting rules."""
        default_rules = [
            AlertRule(
                name="high_memory_usage",
                metric_name="system_memory_percent",
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=120.0,
                description="System memory usage is above 85%",
            ),
            AlertRule(
                name="critical_memory_usage",
                metric_name="system_memory_percent",
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60.0,
                description="System memory usage is critically high (>95%)",
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="system_cpu_percent",
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=180.0,
                description="System CPU usage is above 90%",
            ),
            AlertRule(
                name="device_connection_failed",
                metric_name="device_connection_failures",
                condition="gt",
                threshold=3.0,
                severity=AlertSeverity.ERROR,
                duration_seconds=60.0,
                description="Multiple device connection failures detected",
            ),
            AlertRule(
                name="profiling_timeout",
                metric_name="profiling_timeouts",
                condition="gt",
                threshold=0.0,
                severity=AlertSeverity.ERROR,
                duration_seconds=0.0,  # Alert immediately
                description="Profiling operation timed out",
            ),
            AlertRule(
                name="circuit_breaker_open",
                metric_name="circuit_breaker_open",
                condition="gt",
                threshold=0.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=0.0,
                description="Circuit breaker is open - device communication failing",
            ),
        ]

        for rule in default_rules:
            self.add_rule(rule)

    def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self.evaluation_running:
            try:
                self._evaluate_rules()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                self.logger.error(f"Error in alert evaluation: {e}")
                time.sleep(10)  # Longer sleep on error

    def _evaluate_rules(self):
        """Evaluate all alert rules."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            try:
                self._evaluate_rule(rule)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule_name}: {e}")

    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        # Get current metric value
        current_metric = self.metric_collector.get_latest(rule.metric_name)

        if not current_metric:
            return  # No data available

        current_value = current_metric.value
        threshold = rule.threshold

        # Check condition
        condition_met = False
        if rule.condition == "gt":
            condition_met = current_value > threshold
        elif rule.condition == "gte":
            condition_met = current_value >= threshold
        elif rule.condition == "lt":
            condition_met = current_value < threshold
        elif rule.condition == "lte":
            condition_met = current_value <= threshold
        elif rule.condition == "eq":
            condition_met = current_value == threshold

        alert_id = f"{rule.name}_{rule.metric_name}"

        if condition_met:
            # Check if alert already exists
            if alert_id not in self.active_alerts:
                # Check duration requirement
                if rule.duration_seconds > 0:
                    # Check if condition has been met for required duration
                    duration_start = datetime.now() - timedelta(
                        seconds=rule.duration_seconds
                    )
                    historical_metrics = self.metric_collector.get_metrics(
                        rule.metric_name, start_time=duration_start
                    )

                    if not all(
                        self._check_condition(rule, m.value) for m in historical_metrics
                    ):
                        return  # Condition not met for required duration

                # Create new alert
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    severity=rule.severity,
                    state=AlertState.ACTIVE,
                    started_at=datetime.now(),
                    description=rule.description,
                    labels=rule.labels.copy(),
                )

                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)

                self.logger.warning(
                    f"Alert triggered: {rule.name}",
                    rule_name=rule.name,
                    current_value=current_value,
                    threshold=threshold,
                    severity=rule.severity.value,
                )

                # Notify handlers
                self._notify_handlers(alert)

        else:
            # Condition not met - resolve alert if active
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now()

                del self.active_alerts[alert_id]

                self.logger.info(
                    f"Alert resolved: {rule.name}",
                    rule_name=rule.name,
                    current_value=current_value,
                )

                # Notify handlers of resolution
                self._notify_handlers(alert)

    def _check_condition(self, rule: AlertRule, value: float) -> bool:
        """Check if a value meets the alert condition."""
        if rule.condition == "gt":
            return value > rule.threshold
        elif rule.condition == "gte":
            return value >= rule.threshold
        elif rule.condition == "lt":
            return value < rule.threshold
        elif rule.condition == "lte":
            return value <= rule.threshold
        elif rule.condition == "eq":
            return value == rule.threshold
        return False

    def _notify_handlers(self, alert: Alert):
        """Notify all alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")


class MonitoringSystem:
    """Comprehensive monitoring system for the profiler."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager(self.metric_collector)

        self.system_monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.monitor_interval = 10  # seconds

        self.logger = get_logger("monitoring_system")

        # Setup default alert handlers
        self._setup_alert_handlers()

    def start(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        # Start alert manager
        self.alert_manager.start()

        # Start system monitoring
        self.system_monitor_thread = threading.Thread(
            target=self._system_monitor_loop, daemon=True
        )
        self.system_monitor_thread.start()

        self.logger.info("Monitoring system started")

    def stop(self):
        """Stop the monitoring system."""
        self.monitoring_active = False

        # Stop alert manager
        self.alert_manager.stop()

        # Stop metric collector
        self.metric_collector.stop()

        # Stop system monitoring
        if self.system_monitor_thread and self.system_monitor_thread.is_alive():
            self.system_monitor_thread.join(timeout=5.0)

        self.logger.info("Monitoring system stopped")

    def record_metric(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        metric = Metric(
            name=name, value=value, timestamp=datetime.now(), labels=labels or {}
        )
        self.metric_collector.record(metric)

    def get_metrics(self, name: str, **kwargs) -> List[Metric]:
        """Get metrics by name."""
        return self.metric_collector.get_metrics(name, **kwargs)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [
            a for a in active_alerts if a.severity == AlertSeverity.CRITICAL
        ]
        warning_alerts = [
            a for a in active_alerts if a.severity == AlertSeverity.WARNING
        ]

        # Get latest system metrics
        memory_metric = self.metric_collector.get_latest("system_memory_percent")
        cpu_metric = self.metric_collector.get_latest("system_cpu_percent")

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": (
                "critical"
                if critical_alerts
                else ("warning" if warning_alerts else "healthy")
            ),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "system_metrics": {
                "memory_percent": memory_metric.value if memory_metric else None,
                "cpu_percent": cpu_metric.value if cpu_metric else None,
            },
            "monitoring_active": self.monitoring_active,
            "alert_rules_count": len(self.alert_manager.alert_rules),
        }

    def export_metrics(self, output_path: Path, format: str = "json"):
        """Export all metrics to file."""
        all_metrics = {}

        # Collect all metrics
        for metric_key, metric_deque in self.metric_collector.metrics.items():
            metrics_data = []
            for metric in metric_deque:
                metrics_data.append(
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "timestamp": metric.timestamp.isoformat(),
                        "labels": metric.labels,
                    }
                )
            all_metrics[metric_key] = metrics_data

        # Export to file
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Exported metrics to {output_path}")

    def _setup_alert_handlers(self):
        """Setup default alert handlers."""

        # Console handler
        def console_handler(alert: Alert):
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨",
            }

            emoji = severity_emoji.get(alert.severity, "")
            state_msg = "RESOLVED" if alert.state == AlertState.RESOLVED else "ACTIVE"

            print(
                f"{emoji} [{alert.severity.value.upper()}] {state_msg}: {alert.description}"
            )
            print(
                f"   Metric: {alert.metric_name} = {alert.current_value} (threshold: {alert.threshold})"
            )

        # Log handler
        def log_handler(alert: Alert):
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL,
            }.get(alert.severity, logging.WARNING)

            self.logger.log(
                log_level,
                f"Alert {alert.state.value}: {alert.description}",
                alert_id=alert.id,
                rule_name=alert.rule_name,
                current_value=alert.current_value,
                threshold=alert.threshold,
                severity=alert.severity.value,
            )

        self.alert_manager.add_alert_handler(console_handler)
        self.alert_manager.add_alert_handler(log_handler)

    def _system_monitor_loop(self):
        """Monitor system resources continuously."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()

                # Record metrics
                self.record_metric("system_memory_percent", memory.percent)
                self.record_metric(
                    "system_memory_available_mb", memory.available / (1024 * 1024)
                )
                self.record_metric("system_cpu_percent", cpu_percent)

                # Disk usage
                try:
                    disk = psutil.disk_usage("/")
                    disk_percent = (disk.used / disk.total) * 100
                    self.record_metric("system_disk_percent", disk_percent)
                    self.record_metric("system_disk_free_gb", disk.free / (1024**3))
                except Exception as e:
                    self.logger.debug(f"Error collecting disk metrics: {e}")

                # Temperature (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        all_temps = []
                        for sensor_name, sensors in temps.items():
                            for sensor in sensors:
                                if sensor.current:
                                    all_temps.append(sensor.current)

                        if all_temps:
                            avg_temp = np.mean(all_temps)
                            max_temp = np.max(all_temps)
                            self.record_metric("system_temperature_avg", avg_temp)
                            self.record_metric("system_temperature_max", max_temp)
                except Exception as e:
                    self.logger.debug(f"Error collecting temperature metrics: {e}")

                time.sleep(self.monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(30)  # Longer sleep on error


# Global monitoring system instance
monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> MonitoringSystem:
    """Get the global monitoring system instance."""
    global monitoring_system
    if monitoring_system is None:
        monitoring_system = MonitoringSystem()
    return monitoring_system


def start_monitoring():
    """Start the global monitoring system."""
    get_monitoring_system().start()


def stop_monitoring():
    """Stop the global monitoring system."""
    if monitoring_system:
        monitoring_system.stop()


def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Convenience function to record a metric."""
    get_monitoring_system().record_metric(name, value, labels)


def get_health_summary() -> Dict[str, Any]:
    """Get overall system health summary."""
    return get_monitoring_system().get_health_summary()
