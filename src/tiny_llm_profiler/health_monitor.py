"""
Health monitoring and system checks for robust profiling operations.
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SystemHealthMonitor:
    """System-wide health monitoring for profiling operations."""
    
    def __init__(self, check_interval: int = 5):
        self.check_interval = check_interval
        self._running = False
        self._thread = None
        self._metrics = HealthMetrics()
        
    def start(self):
        """Start health monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()
        
    def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join()
            
    def get_current_metrics(self) -> 'HealthMetrics':
        """Get current health metrics."""
        return self._metrics
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        import psutil
        while self._running:
            try:
                self._metrics.cpu_usage_percent = psutil.cpu_percent()
                self._metrics.memory_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
                self._metrics.disk_usage_percent = psutil.disk_usage('/').percent
                self._metrics.timestamp = time.time()
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            time.sleep(self.check_interval)


@dataclass
class HealthMetrics:
    """Health and performance metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_percent: float = 0.0
    active_connections: int = 0
    profiling_errors: int = 0
    profiling_successes: int = 0
    avg_response_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = self.profiling_errors + self.profiling_successes
        if total == 0:
            return 100.0
        return (self.profiling_successes / total) * 100.0


@dataclass
class HealthCheck:
    """Definition of a health check."""
    name: str
    check_function: Callable[[], bool]
    critical: bool = False
    interval_seconds: int = 60
    timeout_seconds: int = 30
    last_run: Optional[datetime] = None
    last_result: bool = True
    consecutive_failures: int = 0


class HealthMonitor:
    """System health monitoring for profiling operations."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.metrics_history: List[HealthMetrics] = []
        self.health_checks: List[HealthCheck] = []
        self.alerts: List[str] = []
        
        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Initialize default health checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default health checks."""
        # System resource checks
        self.health_checks.extend([
            HealthCheck(
                name="memory_usage",
                check_function=self._check_memory_usage,
                critical=True,
                interval_seconds=30
            ),
            HealthCheck(
                name="disk_space",
                check_function=self._check_disk_space,
                critical=True,
                interval_seconds=60
            ),
            HealthCheck(
                name="system_load",
                check_function=self._check_system_load,
                critical=False,
                interval_seconds=30
            )
        ])
    
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metric entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Run health checks
                self._run_health_checks()
                
                # Check for alerts
                self._check_alert_conditions(metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        metrics = HealthMetrics()
        
        try:
            # Try to get system metrics if available
            import psutil
            
            # CPU usage
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.memory_usage_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('.')
            metrics.disk_usage_percent = (disk.used / disk.total) * 100
            
        except ImportError:
            # Fallback to simulated metrics
            metrics.cpu_usage_percent = 15.0
            metrics.memory_usage_mb = 256.0
            metrics.disk_usage_percent = 45.0
        
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _run_health_checks(self):
        """Run scheduled health checks."""
        current_time = datetime.now()
        
        for check in self.health_checks:
            # Check if it's time to run this check
            if check.last_run is None or \
               (current_time - check.last_run).total_seconds() >= check.interval_seconds:
                
                try:
                    logger.debug(f"Running health check: {check.name}")
                    result = check.check_function()
                    
                    check.last_run = current_time
                    check.last_result = result
                    
                    if result:
                        check.consecutive_failures = 0
                    else:
                        check.consecutive_failures += 1
                        
                        # Log warning for failed checks
                        logger.warning(
                            f"Health check '{check.name}' failed "
                            f"({check.consecutive_failures} consecutive failures)"
                        )
                        
                        # Critical checks trigger alerts
                        if check.critical and check.consecutive_failures >= 3:
                            alert_msg = f"CRITICAL: Health check '{check.name}' failed 3+ times"
                            self.alerts.append(alert_msg)
                            logger.error(alert_msg)
                
                except Exception as e:
                    logger.error(f"Error running health check '{check.name}': {e}")
                    check.consecutive_failures += 1
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage is within acceptable limits."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if >90% memory usage
        except ImportError:
            return True  # Can't check, assume OK
    
    def _check_disk_space(self) -> bool:
        """Check disk space is sufficient."""
        try:
            import psutil
            disk = psutil.disk_usage('.')
            percent_used = (disk.used / disk.total) * 100
            return percent_used < 95  # Alert if >95% disk usage
        except ImportError:
            return True  # Can't check, assume OK
    
    def _check_system_load(self) -> bool:
        """Check system load is reasonable."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 95  # Alert if >95% CPU usage for extended period
        except ImportError:
            return True  # Can't check, assume OK
    
    def _check_alert_conditions(self, metrics: HealthMetrics):
        """Check for alert conditions in current metrics."""
        # Check success rate
        if metrics.success_rate < 80:
            self.alerts.append(f"Low success rate: {metrics.success_rate:.1f}%")
        
        # Check response time
        if metrics.avg_response_time_ms > 5000:  # 5 seconds
            self.alerts.append(f"High response time: {metrics.avg_response_time_ms:.0f}ms")
    
    def get_current_health(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.metrics_history:
            return {"status": "unknown", "message": "No metrics collected yet"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Determine overall health
        critical_failures = sum(1 for check in self.health_checks 
                              if check.critical and not check.last_result)
        
        if critical_failures > 0:
            status = "critical"
        elif any(not check.last_result for check in self.health_checks):
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": latest_metrics.timestamp,
            "metrics": {
                "cpu_usage_percent": latest_metrics.cpu_usage_percent,
                "memory_usage_mb": latest_metrics.memory_usage_mb,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "success_rate": latest_metrics.success_rate,
                "avg_response_time_ms": latest_metrics.avg_response_time_ms
            },
            "health_checks": [
                {
                    "name": check.name,
                    "status": "pass" if check.last_result else "fail",
                    "critical": check.critical,
                    "consecutive_failures": check.consecutive_failures
                }
                for check in self.health_checks
            ],
            "recent_alerts": self.alerts[-10:] if self.alerts else []
        }
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        # Filter metrics for the specified time period
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": f"No metrics from last {hours} hours"}
        
        # Calculate statistics
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        memory_values = [m.memory_usage_mb for m in recent_metrics]
        response_times = [m.avg_response_time_ms for m in recent_metrics if m.avg_response_time_ms > 0]
        
        summary = {
            "period_hours": hours,
            "samples_count": len(recent_metrics),
            "cpu_usage": {
                "avg": statistics.mean(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0
            },
            "memory_usage": {
                "avg": statistics.mean(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0
            }
        }
        
        if response_times:
            summary["response_time"] = {
                "avg": statistics.mean(response_times),
                "max": max(response_times),
                "min": min(response_times)
            }
        
        return summary
    
    def add_custom_check(self, name: str, check_function: Callable[[], bool], 
                        critical: bool = False, interval_seconds: int = 60):
        """Add a custom health check."""
        custom_check = HealthCheck(
            name=name,
            check_function=check_function,
            critical=critical,
            interval_seconds=interval_seconds
        )
        self.health_checks.append(custom_check)
        logger.info(f"Added custom health check: {name}")
    
    def record_profiling_result(self, success: bool, response_time_ms: float = 0):
        """Record the result of a profiling operation."""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            if success:
                latest.profiling_successes += 1
            else:
                latest.profiling_errors += 1
            
            # Update average response time (simple moving average)
            if response_time_ms > 0:
                if latest.avg_response_time_ms == 0:
                    latest.avg_response_time_ms = response_time_ms
                else:
                    # Weighted average with new measurement
                    latest.avg_response_time_ms = (
                        latest.avg_response_time_ms * 0.9 + response_time_ms * 0.1
                    )


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def start_global_monitoring():
    """Start global health monitoring."""
    monitor = get_health_monitor()
    monitor.start_monitoring()


def stop_global_monitoring():
    """Stop global health monitoring."""
    monitor = get_health_monitor()
    monitor.stop_monitoring()


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    monitor = get_health_monitor()
    return monitor.get_current_health()