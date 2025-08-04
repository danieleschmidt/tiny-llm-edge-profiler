"""
Health checks and system monitoring for the Tiny LLM Edge Profiler.
"""

import time
import psutil
import platform
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from .exceptions import TinyLLMProfilerError
from .logging_config import get_logger
from .security import validate_environment

logger = get_logger("health")


@dataclass
class HealthStatus:
    """Health check status result."""
    name: str
    healthy: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_free_gb: float
    temperature_celsius: Optional[float] = None
    load_average: Optional[List[float]] = None


class HealthChecker:
    """
    Comprehensive health checking system for the profiler.
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthStatus]] = {}
        self.last_check_results: Dict[str, HealthStatus] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1000
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("python_environment", self._check_python_environment)
        self.register_check("security_environment", self._check_security_environment)
        self.register_check("dependencies", self._check_dependencies)
    
    def register_check(self, name: str, check_function: Callable[[], HealthStatus]):
        """
        Register a custom health check.
        
        Args:
            name: Name of the health check
            check_function: Function that returns HealthStatus
        """
        self.checks[name] = check_function
        logger.debug(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthStatus:
        """
        Run a specific health check.
        
        Args:
            name: Name of the health check to run
            
        Returns:
            HealthStatus result
            
        Raises:
            ValueError: If check name is not registered
        """
        if name not in self.checks:
            raise ValueError(f"Health check '{name}' not registered")
        
        start_time = time.time()
        
        try:
            status = self.checks[name]()
            status.response_time_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            status = HealthStatus(
                name=name,
                healthy=False,
                message=f"Health check failed: {e}",
                details={"exception": str(e)},
                response_time_ms=(time.time() - start_time) * 1000
            )
            logger.exception(f"Health check {name} failed", exception=e)
        
        self.last_check_results[name] = status
        return status
    
    def run_all_checks(self) -> Dict[str, HealthStatus]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary mapping check names to HealthStatus results
        """
        results = {}
        
        for check_name in self.checks:
            results[check_name] = self.run_check(check_name)
        
        logger.info(f"Completed {len(results)} health checks")
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health status.
        
        Returns:
            HealthStatus summarizing all checks
        """
        if not self.last_check_results:
            # Run all checks if none have been run
            self.run_all_checks()
        
        healthy_checks = sum(1 for status in self.last_check_results.values() if status.healthy)
        total_checks = len(self.last_check_results)
        
        overall_healthy = healthy_checks == total_checks
        
        failed_checks = [
            name for name, status in self.last_check_results.items() 
            if not status.healthy
        ]
        
        if overall_healthy:
            message = f"All {total_checks} health checks passed"
        else:
            message = f"{len(failed_checks)} of {total_checks} health checks failed: {', '.join(failed_checks)}"
        
        return HealthStatus(
            name="overall_health",
            healthy=overall_healthy,
            message=message,
            details={
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "failed_checks": failed_checks,
                "check_results": {name: status.healthy for name, status in self.last_check_results.items()}
            }
        )
    
    def collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system performance metrics.
        
        Returns:
            SystemMetrics with current system state
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Load average (Unix-like systems only)
        load_average = None
        try:
            load_average = list(psutil.getloadavg())
        except (AttributeError, OSError):
            pass  # Not available on all platforms
        
        # Temperature (if available)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get first available temperature reading
                for sensor_name, sensors in temps.items():
                    if sensors:
                        temperature = sensors[0].current
                        break
        except (AttributeError, OSError):
            pass  # Not available on all platforms
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            disk_free_gb=disk_free_gb,
            temperature_celsius=temperature,
            load_average=load_average
        )
        
        # Add to history
        self.system_metrics_history.append(metrics)
        
        # Trim history if too large
        if len(self.system_metrics_history) > self.max_history_size:
            self.system_metrics_history = self.system_metrics_history[-self.max_history_size:]
        
        return metrics
    
    def get_metrics_summary(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """
        Get summary of system metrics over a time period.
        
        Args:
            duration_minutes: Duration to analyze in minutes
            
        Returns:
            Dictionary with metrics summary
        """
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        # Filter metrics within time period
        recent_metrics = [
            m for m in self.system_metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No metrics data available"}
        
        # Calculate averages and extremes
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        summary = {
            "time_period_minutes": duration_minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "average_percent": sum(cpu_values) / len(cpu_values),
                "max_percent": max(cpu_values),
                "min_percent": min(cpu_values)
            },
            "memory": {
                "average_percent": sum(memory_values) / len(memory_values),
                "max_percent": max(memory_values),
                "min_percent": min(memory_values),
                "current_available_mb": recent_metrics[-1].memory_available_mb
            },
            "disk": {
                "free_gb": recent_metrics[-1].disk_free_gb
            }
        }
        
        # Add temperature if available
        temp_values = [m.temperature_celsius for m in recent_metrics if m.temperature_celsius is not None]
        if temp_values:
            summary["temperature"] = {
                "average_celsius": sum(temp_values) / len(temp_values),
                "max_celsius": max(temp_values),
                "current_celsius": recent_metrics[-1].temperature_celsius
            }
        
        return summary
    
    def _check_system_resources(self) -> HealthStatus:
        """Check system resource availability."""
        try:
            # Get current metrics
            metrics = self.collect_system_metrics()
            
            issues = []
            
            # Check CPU usage
            if metrics.cpu_percent > 90:
                issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
            # Check memory usage
            if metrics.memory_percent > 85:
                issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            
            # Check available memory
            if metrics.memory_available_mb < 100:  # Less than 100MB available
                issues.append(f"Low available memory: {metrics.memory_available_mb:.1f}MB")
            
            # Check disk space
            if metrics.disk_free_gb < 1:  # Less than 1GB free
                issues.append(f"Low disk space: {metrics.disk_free_gb:.1f}GB")
            
            # Check temperature if available
            if metrics.temperature_celsius and metrics.temperature_celsius > 80:
                issues.append(f"High temperature: {metrics.temperature_celsius:.1f}°C")
            
            healthy = len(issues) == 0
            message = "System resources OK" if healthy else f"Resource issues: {'; '.join(issues)}"
            
            return HealthStatus(
                name="system_resources",
                healthy=healthy,
                message=message,
                details={
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "memory_available_mb": metrics.memory_available_mb,
                    "disk_free_gb": metrics.disk_free_gb,
                    "temperature_celsius": metrics.temperature_celsius,
                    "issues": issues
                }
            )
            
        except Exception as e:
            return HealthStatus(
                name="system_resources",
                healthy=False,
                message=f"Failed to check system resources: {e}",
                details={"exception": str(e)}
            )
    
    def _check_disk_space(self) -> HealthStatus:
        """Check disk space for various directories."""
        try:
            checks = {
                "root": "/",
                "tmp": "/tmp",
                "home": str(Path.home())
            }
            
            results = {}
            issues = []
            
            for name, path in checks.items():
                if Path(path).exists():
                    try:
                        usage = psutil.disk_usage(path)
                        free_gb = usage.free / (1024 * 1024 * 1024)
                        used_percent = (usage.used / usage.total) * 100
                        
                        results[name] = {
                            "free_gb": free_gb,
                            "used_percent": used_percent
                        }
                        
                        if free_gb < 0.5:  # Less than 500MB
                            issues.append(f"{name}: only {free_gb:.2f}GB free")
                        elif used_percent > 95:
                            issues.append(f"{name}: {used_percent:.1f}% full")
                            
                    except OSError as e:
                        issues.append(f"{name}: cannot check disk usage - {e}")
            
            healthy = len(issues) == 0
            message = "Disk space OK" if healthy else f"Disk space issues: {'; '.join(issues)}"
            
            return HealthStatus(
                name="disk_space",
                healthy=healthy,
                message=message,
                details={"disk_usage": results, "issues": issues}
            )
            
        except Exception as e:
            return HealthStatus(
                name="disk_space",
                healthy=False,
                message=f"Failed to check disk space: {e}"
            )
    
    def _check_python_environment(self) -> HealthStatus:
        """Check Python environment and version."""
        try:
            import sys
            
            issues = []
            
            # Check Python version
            version = sys.version_info
            if version < (3, 8):
                issues.append(f"Python version {version.major}.{version.minor} is too old (minimum: 3.8)")
            
            # Check if running in virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            
            details = {
                "python_version": f"{version.major}.{version.minor}.{version.micro}",
                "platform": platform.platform(),
                "architecture": platform.architecture()[0],
                "in_virtual_env": in_venv,
                "executable": sys.executable
            }
            
            if not in_venv:
                issues.append("Not running in virtual environment (recommended for isolation)")
            
            healthy = len(issues) == 0
            message = "Python environment OK" if healthy else f"Environment issues: {'; '.join(issues)}"
            
            return HealthStatus(
                name="python_environment",
                healthy=healthy,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                name="python_environment",
                healthy=False,
                message=f"Failed to check Python environment: {e}"
            )
    
    def _check_security_environment(self) -> HealthStatus:
        """Check security-related environment settings."""
        try:
            security_results = validate_environment()
            
            healthy = security_results["secure"] and len(security_results["errors"]) == 0
            
            issues = security_results["errors"] + security_results["warnings"]
            message = "Security environment OK" if healthy else f"Security issues: {'; '.join(issues[:3])}"
            
            return HealthStatus(
                name="security_environment",
                healthy=healthy,
                message=message,
                details=security_results
            )
            
        except Exception as e:
            return HealthStatus(
                name="security_environment",
                healthy=False,
                message=f"Failed to check security environment: {e}"
            )
    
    def _check_dependencies(self) -> HealthStatus:
        """Check required dependencies are available."""
        try:
            required_modules = [
                'numpy', 'serial', 'psutil', 'click', 'rich', 
                'plotly', 'pandas', 'aiofiles', 'httpx', 'pydantic'
            ]
            
            missing_modules = []
            version_info = {}
            
            for module_name in required_modules:
                try:
                    module = __import__(module_name)
                    version = getattr(module, '__version__', None)
                    version_info[module_name] = version or "unknown"
                except ImportError:
                    missing_modules.append(module_name)
            
            healthy = len(missing_modules) == 0
            
            if healthy:
                message = f"All {len(required_modules)} required dependencies available"
            else:
                message = f"Missing dependencies: {', '.join(missing_modules)}"
            
            return HealthStatus(
                name="dependencies",
                healthy=healthy,
                message=message,
                details={
                    "required_modules": required_modules,
                    "missing_modules": missing_modules,
                    "version_info": version_info
                }
            )
            
        except Exception as e:
            return HealthStatus(
                name="dependencies",
                healthy=False,
                message=f"Failed to check dependencies: {e}"
            )
    
    def export_health_report(self, output_path: Path) -> None:
        """
        Export comprehensive health report to file.
        
        Args:
            output_path: Path to save the health report
        """
        # Run all health checks
        check_results = self.run_all_checks()
        overall_health = self.get_overall_health()
        
        # Get system metrics summary
        metrics_summary = self.get_metrics_summary(duration_minutes=60)
        
        # Prepare report data
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": {
                "healthy": overall_health.healthy,
                "message": overall_health.message,
                "details": overall_health.details
            },
            "individual_checks": {
                name: {
                    "healthy": status.healthy,
                    "message": status.message,
                    "response_time_ms": status.response_time_ms,
                    "details": status.details
                }
                for name, status in check_results.items()
            },
            "system_metrics": metrics_summary,
            "recommendations": self._generate_recommendations(check_results, metrics_summary)
        }
        
        # Write report to file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Health report exported to {output_path}")
    
    def _generate_recommendations(self, check_results: Dict[str, HealthStatus], metrics_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health check results."""
        recommendations = []
        
        # Check for high resource usage
        if "cpu" in metrics_summary and metrics_summary["cpu"]["average_percent"] > 70:
            recommendations.append("Consider reducing CPU-intensive operations or upgrading hardware")
        
        if "memory" in metrics_summary and metrics_summary["memory"]["average_percent"] > 80:
            recommendations.append("Monitor memory usage - consider optimizing memory allocation or adding more RAM")
        
        if "disk" in metrics_summary and metrics_summary["disk"]["free_gb"] < 2:
            recommendations.append("Free up disk space or move to larger storage")
        
        # Check individual health check failures
        for name, status in check_results.items():
            if not status.healthy:
                if name == "dependencies":
                    recommendations.append("Install missing dependencies with: pip install -r requirements.txt")
                elif name == "python_environment":
                    recommendations.append("Consider upgrading Python version or using a virtual environment")
                elif name == "security_environment":
                    recommendations.append("Review security warnings and apply recommended fixes")
        
        return recommendations


# Global health checker instance
health_checker = HealthChecker()


def get_health_status() -> Dict[str, Any]:
    """
    Get current health status.
    
    Returns:
        Dictionary with health status information
    """
    overall_health = health_checker.get_overall_health()
    
    return {
        "healthy": overall_health.healthy,
        "message": overall_health.message,
        "timestamp": overall_health.timestamp.isoformat(),
        "details": overall_health.details
    }


def run_health_checks() -> bool:
    """
    Run all health checks and return overall health status.
    
    Returns:
        True if all checks passed, False otherwise
    """
    overall_health = health_checker.get_overall_health()
    return overall_health.healthy