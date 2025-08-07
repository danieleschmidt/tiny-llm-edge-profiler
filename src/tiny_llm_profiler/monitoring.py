"""
Advanced monitoring and health checking for profiling operations.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

import psutil
import numpy as np


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(str, Enum):
    """Types of metrics to monitor."""
    LATENCY = "latency"
    MEMORY = "memory"
    CPU = "cpu"
    POWER = "power"
    TEMPERATURE = "temperature"
    THROUGHPUT = "throughput"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_func: Callable[[], bool]
    critical: bool = False
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    description: str = ""


@dataclass
class Metric:
    """Single metric measurement."""
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert notification."""
    timestamp: datetime
    level: str  # "info", "warning", "error", "critical"
    source: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    auto_resolve: bool = True


class HealthMonitor:
    """Monitors system health during profiling operations."""
    
    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self.health_checks: List[HealthCheck] = []
        self.metrics: List[Metric] = []
        self.alerts: List[Alert] = []
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        def check_memory():
            """Check available memory."""
            memory = psutil.virtual_memory()
            return memory.available > 100 * 1024 * 1024  # > 100MB
        
        def check_disk_space():
            """Check available disk space."""
            disk = psutil.disk_usage('.')
            return disk.free > 500 * 1024 * 1024  # > 500MB
        
        def check_cpu_usage():
            """Check CPU usage isn't too high."""
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 90  # < 90% CPU usage
        
        self.health_checks = [
            HealthCheck(
                name="memory_availability",
                check_func=check_memory,
                critical=True,
                description="Ensure sufficient memory is available"
            ),
            HealthCheck(
                name="disk_space",
                check_func=check_disk_space,
                critical=False,
                description="Ensure sufficient disk space for results"
            ),
            HealthCheck(
                name="cpu_usage",
                check_func=check_cpu_usage,
                critical=False,
                description="Ensure CPU isn't overloaded"
            )
        ]
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a custom health check."""
        self.health_checks.append(health_check)
    
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Run health checks
                self._run_health_checks()
                
                # Collect metrics
                self._collect_system_metrics()
                
                # Check for alerts
                self._process_alerts()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _run_health_checks(self):
        """Execute all health checks."""
        for health_check in self.health_checks:
            try:
                start_time = time.time()
                is_healthy = health_check.check_func()
                duration = time.time() - start_time
                
                if duration > health_check.timeout_seconds:
                    self._create_alert(
                        level="warning",
                        source=health_check.name,
                        message=f"Health check '{health_check.name}' took {duration:.1f}s (timeout: {health_check.timeout_seconds}s)"
                    )
                
                if not is_healthy:
                    level = "critical" if health_check.critical else "warning"
                    self._create_alert(
                        level=level,
                        source=health_check.name,
                        message=f"Health check '{health_check.name}' failed: {health_check.description}"
                    )
                
            except Exception as e:
                self._create_alert(
                    level="error",
                    source=health_check.name,
                    message=f"Health check '{health_check.name}' raised exception: {e}"
                )
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        now = datetime.now()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.append(Metric(
            timestamp=now,
            metric_type=MetricType.MEMORY,
            value=memory.used / (1024**2),  # MB
            unit="MB",
            metadata={"total_mb": memory.total / (1024**2), "percent": memory.percent}
        ))
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self.metrics.append(Metric(
            timestamp=now,
            metric_type=MetricType.CPU,
            value=cpu_percent,
            unit="%",
            metadata={"cpu_count": psutil.cpu_count()}
        ))
        
        # Temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get average temperature
                all_temps = []
                for sensor_name, sensors in temps.items():
                    for sensor in sensors:
                        if sensor.current:
                            all_temps.append(sensor.current)
                
                if all_temps:
                    avg_temp = np.mean(all_temps)
                    self.metrics.append(Metric(
                        timestamp=now,
                        metric_type=MetricType.TEMPERATURE,
                        value=avg_temp,
                        unit="°C",
                        metadata={"sensor_count": len(all_temps)}
                    ))
        except Exception:
            # Temperature sensors not available
            pass
        
        # Cleanup old metrics (keep last hour)
        cutoff_time = now - timedelta(hours=1)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
    
    def _process_alerts(self):
        """Process and potentially resolve alerts."""
        current_time = datetime.now()
        
        # Auto-resolve alerts that are older than 5 minutes and marked for auto-resolve
        resolve_cutoff = current_time - timedelta(minutes=5)
        
        resolved_alerts = []
        for alert in self.alerts:
            if alert.auto_resolve and alert.timestamp < resolve_cutoff:
                resolved_alerts.append(alert)
        
        for alert in resolved_alerts:
            self.alerts.remove(alert)
            self.logger.info(f"Auto-resolved alert: {alert.message}")
    
    def _create_alert(self, level: str, source: str, message: str, **kwargs):
        """Create a new alert."""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            source=source,
            message=message,
            **kwargs
        )
        
        self.alerts.append(alert)
        
        # Log the alert
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"Alert [{level}] from {source}: {message}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        # Determine overall health
        critical_alerts = [a for a in self.alerts if a.level == "critical"]
        warning_alerts = [a for a in self.alerts if a.level in ["warning", "error"]]
        
        if critical_alerts:
            overall_status = HealthStatus.CRITICAL
        elif warning_alerts:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Get latest metrics
        latest_metrics = {}
        for metric_type in MetricType:
            type_metrics = [m for m in self.metrics if m.metric_type == metric_type]
            if type_metrics:
                latest = max(type_metrics, key=lambda x: x.timestamp)
                latest_metrics[metric_type.value] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp.isoformat()
                }
        
        return {
            "overall_status": overall_status.value,
            "last_check": datetime.now().isoformat(),
            "active_alerts": len(self.alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "latest_metrics": latest_metrics,
            "health_checks_count": len(self.health_checks)
        }
    
    def get_alerts(self, level: Optional[str] = None, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        alerts = self.alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        if last_n:
            alerts = alerts[:last_n]
        
        return [
            {
                "timestamp": a.timestamp.isoformat(),
                "level": a.level,
                "source": a.source,
                "message": a.message,
                "metric_value": a.metric_value,
                "threshold": a.threshold
            }
            for a in alerts
        ]
    
    def export_metrics(self, output_path: Path, format: str = "json"):
        """Export collected metrics to file."""
        if format == "json":
            metrics_data = []
            for metric in self.metrics:
                metrics_data.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "type": metric.metric_type.value,
                    "value": metric.value,
                    "unit": metric.unit,
                    "metadata": metric.metadata
                })
            
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        
        elif format == "csv":
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "type", "value", "unit"])
                
                for metric in self.metrics:
                    writer.writerow([
                        metric.timestamp.isoformat(),
                        metric.metric_type.value,
                        metric.value,
                        metric.unit
                    ])


class ProfilingMonitor:
    """Monitors profiling operations with detailed metrics."""
    
    def __init__(self, profiling_name: str = "default"):
        self.profiling_name = profiling_name
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.checkpoints: List[Dict[str, Any]] = []
        self.health_monitor = HealthMonitor()
        self.logger = logging.getLogger(__name__)
    
    def start_profiling(self):
        """Start profiling monitoring."""
        self.start_time = datetime.now()
        self.health_monitor.start_monitoring()
        self.logger.info(f"Started profiling monitoring: {self.profiling_name}")
        
        self._checkpoint("profiling_started", {"name": self.profiling_name})
    
    def stop_profiling(self):
        """Stop profiling monitoring."""
        self.end_time = datetime.now()
        self.health_monitor.stop_monitoring()
        
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        self.logger.info(f"Stopped profiling monitoring: {self.profiling_name} (duration: {duration:.1f}s)")
        
        self._checkpoint("profiling_completed", {
            "name": self.profiling_name,
            "duration_seconds": duration
        })
    
    def _checkpoint(self, event: str, data: Dict[str, Any] = None):
        """Record a profiling checkpoint."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data or {}
        }
        
        self.checkpoints.append(checkpoint)
        self.logger.info(f"Checkpoint: {event}")
    
    def checkpoint(self, event: str, **kwargs):
        """Public method to add checkpoints."""
        self._checkpoint(event, kwargs)
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get profiling session summary."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        health_status = self.health_monitor.get_health_status()
        
        return {
            "profiling_name": self.profiling_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "checkpoints_count": len(self.checkpoints),
            "health_status": health_status,
            "alerts_summary": {
                "total": health_status["active_alerts"],
                "critical": health_status["critical_alerts"],
                "warning": health_status["warning_alerts"]
            }
        }
    
    def export_session_data(self, output_dir: Path):
        """Export complete session data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Export summary
        summary = self.get_profiling_summary()
        with open(output_dir / "session_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Export checkpoints
        with open(output_dir / "checkpoints.json", 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
        
        # Export metrics
        self.health_monitor.export_metrics(output_dir / "metrics.json")
        
        # Export alerts
        alerts = self.health_monitor.get_alerts()
        with open(output_dir / "alerts.json", 'w') as f:
            json.dump(alerts, f, indent=2)
        
        self.logger.info(f"Session data exported to {output_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_profiling()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            self._checkpoint("profiling_error", {
                "error_type": exc_type.__name__,
                "error_message": str(exc_val)
            })
        
        self.stop_profiling()


class AlertManager:
    """Manages and routes alerts from profiling operations."""
    
    def __init__(self):
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def handle_alert(self, alert: Alert):
        """Process an alert through all handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    @staticmethod
    def console_handler(alert: Alert):
        """Simple console alert handler."""
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ALERT [{alert.level.upper()}] {alert.source}: {alert.message}")
    
    @staticmethod
    def file_handler(log_file: Path):
        """Create a file-based alert handler."""
        def handler(alert: Alert):
            timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a') as f:
                f.write(f"[{timestamp}] ALERT [{alert.level.upper()}] {alert.source}: {alert.message}\n")
        
        return handler