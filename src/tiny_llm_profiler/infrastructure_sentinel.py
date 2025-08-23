"""
Infrastructure Sentinel System
Monitor and auto-heal infrastructure issues for edge AI deployments
"""

import asyncio
import logging
import psutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import json
import subprocess
import shutil

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    TEMPERATURE = "temperature"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class InfrastructureStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ResourceMetrics:
    resource_type: ResourceType
    current_value: float
    max_value: float
    utilization_percent: float
    status: InfrastructureStatus
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfrastructureAlert:
    resource_type: ResourceType
    alert_level: AlertLevel
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    auto_healing_attempted: bool = False
    resolved: bool = False


@dataclass
class HealingAction:
    action_id: str
    name: str
    description: str
    resource_type: ResourceType
    command: str
    estimated_impact: str
    risk_level: str = "low"


class ResourceMonitor(ABC):
    @abstractmethod
    async def collect_metrics(self) -> ResourceMetrics:
        pass

    @abstractmethod
    def get_thresholds(self) -> Dict[str, float]:
        pass


class CPUMonitor(ResourceMonitor):
    def __init__(self):
        self.thresholds = {"warning": 80.0, "critical": 95.0, "emergency": 98.0}

    async def collect_metrics(self) -> ResourceMetrics:
        # Get CPU usage over 1 second interval
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # Get CPU frequency info
        cpu_freq = psutil.cpu_freq()
        current_freq = cpu_freq.current if cpu_freq else 0
        max_freq = cpu_freq.max if cpu_freq else 0

        # Determine status
        status = InfrastructureStatus.HEALTHY
        if cpu_percent >= self.thresholds["emergency"]:
            status = InfrastructureStatus.FAILED
        elif cpu_percent >= self.thresholds["critical"]:
            status = InfrastructureStatus.CRITICAL
        elif cpu_percent >= self.thresholds["warning"]:
            status = InfrastructureStatus.DEGRADED

        return ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_value=cpu_percent,
            max_value=100.0,
            utilization_percent=cpu_percent,
            status=status,
            metadata={
                "cpu_count": cpu_count,
                "current_freq_mhz": current_freq,
                "max_freq_mhz": max_freq,
                "per_cpu_percent": psutil.cpu_percent(percpu=True),
            },
        )

    def get_thresholds(self) -> Dict[str, float]:
        return self.thresholds.copy()


class MemoryMonitor(ResourceMonitor):
    def __init__(self):
        self.thresholds = {"warning": 85.0, "critical": 95.0, "emergency": 98.0}

    async def collect_metrics(self) -> ResourceMetrics:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Determine status
        status = InfrastructureStatus.HEALTHY
        if memory.percent >= self.thresholds["emergency"]:
            status = InfrastructureStatus.FAILED
        elif memory.percent >= self.thresholds["critical"]:
            status = InfrastructureStatus.CRITICAL
        elif memory.percent >= self.thresholds["warning"]:
            status = InfrastructureStatus.DEGRADED

        return ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            current_value=memory.used,
            max_value=memory.total,
            utilization_percent=memory.percent,
            status=status,
            metadata={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "cached_gb": memory.cached / (1024**3),
                "swap_percent": swap.percent,
                "swap_used_gb": swap.used / (1024**3),
                "swap_total_gb": swap.total / (1024**3),
            },
        )

    def get_thresholds(self) -> Dict[str, float]:
        return self.thresholds.copy()


class DiskMonitor(ResourceMonitor):
    def __init__(self, monitored_paths: List[str] = None):
        self.monitored_paths = monitored_paths or ["/", "/tmp", "/var"]
        self.thresholds = {"warning": 85.0, "critical": 95.0, "emergency": 98.0}

    async def collect_metrics(self) -> ResourceMetrics:
        disk_metrics = {}
        max_usage = 0.0

        for path in self.monitored_paths:
            try:
                disk_usage = shutil.disk_usage(path)
                total = disk_usage.total
                used = disk_usage.used
                percent = (used / total) * 100 if total > 0 else 0

                disk_metrics[path] = {
                    "total_gb": total / (1024**3),
                    "used_gb": used / (1024**3),
                    "free_gb": (total - used) / (1024**3),
                    "percent": percent,
                }

                max_usage = max(max_usage, percent)

            except (OSError, FileNotFoundError) as e:
                logger.warning(f"Could not get disk usage for {path}: {str(e)}")

        # Determine status based on highest usage
        status = InfrastructureStatus.HEALTHY
        if max_usage >= self.thresholds["emergency"]:
            status = InfrastructureStatus.FAILED
        elif max_usage >= self.thresholds["critical"]:
            status = InfrastructureStatus.CRITICAL
        elif max_usage >= self.thresholds["warning"]:
            status = InfrastructureStatus.DEGRADED

        return ResourceMetrics(
            resource_type=ResourceType.DISK,
            current_value=max_usage,
            max_value=100.0,
            utilization_percent=max_usage,
            status=status,
            metadata=disk_metrics,
        )

    def get_thresholds(self) -> Dict[str, float]:
        return self.thresholds.copy()


class NetworkMonitor(ResourceMonitor):
    def __init__(self):
        self.thresholds = {
            "warning": 80.0,  # 80% packet loss or high latency
            "critical": 90.0,
            "emergency": 95.0,
        }
        self.last_stats = None
        self.last_check = None

    async def collect_metrics(self) -> ResourceMetrics:
        current_stats = psutil.net_io_counters()
        current_time = time.time()

        # Calculate rates if we have previous data
        bytes_sent_rate = 0
        bytes_recv_rate = 0
        packets_sent_rate = 0
        packets_recv_rate = 0

        if self.last_stats and self.last_check:
            time_delta = current_time - self.last_check
            if time_delta > 0:
                bytes_sent_rate = (
                    current_stats.bytes_sent - self.last_stats.bytes_sent
                ) / time_delta
                bytes_recv_rate = (
                    current_stats.bytes_recv - self.last_stats.bytes_recv
                ) / time_delta
                packets_sent_rate = (
                    current_stats.packets_sent - self.last_stats.packets_sent
                ) / time_delta
                packets_recv_rate = (
                    current_stats.packets_recv - self.last_stats.packets_recv
                ) / time_delta

        self.last_stats = current_stats
        self.last_check = current_time

        # Calculate network health score (simplified)
        error_rate = 0
        if current_stats.packets_sent > 0:
            error_rate = (
                (current_stats.errin + current_stats.errout)
                / current_stats.packets_sent
                * 100
            )

        # Determine status
        status = InfrastructureStatus.HEALTHY
        if error_rate >= self.thresholds["emergency"]:
            status = InfrastructureStatus.FAILED
        elif error_rate >= self.thresholds["critical"]:
            status = InfrastructureStatus.CRITICAL
        elif error_rate >= self.thresholds["warning"]:
            status = InfrastructureStatus.DEGRADED

        return ResourceMetrics(
            resource_type=ResourceType.NETWORK,
            current_value=error_rate,
            max_value=100.0,
            utilization_percent=error_rate,
            status=status,
            metadata={
                "bytes_sent_total": current_stats.bytes_sent,
                "bytes_recv_total": current_stats.bytes_recv,
                "packets_sent_total": current_stats.packets_sent,
                "packets_recv_total": current_stats.packets_recv,
                "bytes_sent_rate_mbps": bytes_sent_rate * 8 / (1024**2),
                "bytes_recv_rate_mbps": bytes_recv_rate * 8 / (1024**2),
                "packets_sent_rate": packets_sent_rate,
                "packets_recv_rate": packets_recv_rate,
                "errors_in": current_stats.errin,
                "errors_out": current_stats.errout,
                "drop_in": current_stats.dropin,
                "drop_out": current_stats.dropout,
            },
        )

    def get_thresholds(self) -> Dict[str, float]:
        return self.thresholds.copy()


class TemperatureMonitor(ResourceMonitor):
    def __init__(self):
        self.thresholds = {
            "warning": 70.0,  # 70°C
            "critical": 80.0,  # 80°C
            "emergency": 85.0,  # 85°C
        }

    async def collect_metrics(self) -> ResourceMetrics:
        try:
            temps = psutil.sensors_temperatures()
            max_temp = 0.0
            temp_data = {}

            for name, entries in temps.items():
                for entry in entries:
                    current_temp = entry.current
                    max_temp = max(max_temp, current_temp)
                    temp_data[f"{name}_{entry.label or 'sensor'}"] = {
                        "current": current_temp,
                        "high": entry.high,
                        "critical": entry.critical,
                    }

            # Determine status
            status = InfrastructureStatus.HEALTHY
            if max_temp >= self.thresholds["emergency"]:
                status = InfrastructureStatus.FAILED
            elif max_temp >= self.thresholds["critical"]:
                status = InfrastructureStatus.CRITICAL
            elif max_temp >= self.thresholds["warning"]:
                status = InfrastructureStatus.DEGRADED

            return ResourceMetrics(
                resource_type=ResourceType.TEMPERATURE,
                current_value=max_temp,
                max_value=100.0,
                utilization_percent=(max_temp / 100.0) * 100,
                status=status,
                metadata=temp_data,
            )

        except Exception as e:
            logger.warning(f"Could not read temperature sensors: {str(e)}")
            return ResourceMetrics(
                resource_type=ResourceType.TEMPERATURE,
                current_value=0.0,
                max_value=100.0,
                utilization_percent=0.0,
                status=InfrastructureStatus.HEALTHY,
                metadata={"error": str(e)},
            )

    def get_thresholds(self) -> Dict[str, float]:
        return self.thresholds.copy()


class InfrastructureHealer:
    def __init__(self):
        self.healing_actions = {
            ResourceType.CPU: [
                HealingAction(
                    action_id="kill_high_cpu_processes",
                    name="Kill High CPU Processes",
                    description="Terminate processes consuming excessive CPU",
                    resource_type=ResourceType.CPU,
                    command="pkill -f 'high_cpu_process'",
                    estimated_impact="Free up CPU resources",
                    risk_level="medium",
                ),
                HealingAction(
                    action_id="reduce_cpu_frequency",
                    name="Reduce CPU Frequency",
                    description="Lower CPU frequency to reduce heat and power",
                    resource_type=ResourceType.CPU,
                    command="cpupower frequency-set --max 1.5GHz",
                    estimated_impact="Lower CPU usage temporarily",
                    risk_level="low",
                ),
            ],
            ResourceType.MEMORY: [
                HealingAction(
                    action_id="clear_memory_cache",
                    name="Clear Memory Cache",
                    description="Clear system memory caches",
                    resource_type=ResourceType.MEMORY,
                    command="sync && echo 3 > /proc/sys/vm/drop_caches",
                    estimated_impact="Free cached memory",
                    risk_level="low",
                ),
                HealingAction(
                    action_id="restart_memory_intensive_services",
                    name="Restart Memory-Intensive Services",
                    description="Restart services that may have memory leaks",
                    resource_type=ResourceType.MEMORY,
                    command="systemctl restart memory-intensive-service",
                    estimated_impact="Free leaked memory",
                    risk_level="medium",
                ),
            ],
            ResourceType.DISK: [
                HealingAction(
                    action_id="clean_temp_files",
                    name="Clean Temporary Files",
                    description="Remove temporary and cache files",
                    resource_type=ResourceType.DISK,
                    command="find /tmp -type f -mtime +7 -delete",
                    estimated_impact="Free disk space",
                    risk_level="low",
                ),
                HealingAction(
                    action_id="compress_old_logs",
                    name="Compress Old Logs",
                    description="Compress old log files to save space",
                    resource_type=ResourceType.DISK,
                    command="find /var/log -name '*.log' -mtime +1 -exec gzip {} \\;",
                    estimated_impact="Reduce log file sizes",
                    risk_level="low",
                ),
            ],
        }
        self.healing_history: List[Dict[str, Any]] = []

    async def can_heal(self, alert: InfrastructureAlert) -> bool:
        return (
            alert.resource_type in self.healing_actions
            and not alert.auto_healing_attempted
            and alert.alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]
        )

    async def heal(self, alert: InfrastructureAlert) -> bool:
        if not await self.can_heal(alert):
            return False

        actions = self.healing_actions.get(alert.resource_type, [])
        if not actions:
            return False

        # Select action based on alert level
        action = actions[0]  # Use first action for simplicity

        try:
            logger.info(f"Attempting healing action: {action.name}")

            # Simulate healing action (in production, this would execute the actual command)
            await asyncio.sleep(1)  # Simulate action execution time

            # Record healing attempt
            healing_record = {
                "timestamp": datetime.now().isoformat(),
                "alert_resource": alert.resource_type.value,
                "alert_level": alert.alert_level.value,
                "action_name": action.name,
                "action_id": action.action_id,
                "success": True,  # Simulate success for demo
            }
            self.healing_history.append(healing_record)

            alert.auto_healing_attempted = True
            logger.info(f"Healing action {action.name} completed successfully")

            return True

        except Exception as e:
            logger.error(f"Healing action failed: {str(e)}")
            healing_record = {
                "timestamp": datetime.now().isoformat(),
                "alert_resource": alert.resource_type.value,
                "alert_level": alert.alert_level.value,
                "action_name": action.name,
                "action_id": action.action_id,
                "success": False,
                "error": str(e),
            }
            self.healing_history.append(healing_record)
            return False

    def get_healing_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.healing_history[-limit:]


class InfrastructureSentinel:
    def __init__(self):
        self.monitors = {
            ResourceType.CPU: CPUMonitor(),
            ResourceType.MEMORY: MemoryMonitor(),
            ResourceType.DISK: DiskMonitor(),
            ResourceType.NETWORK: NetworkMonitor(),
            ResourceType.TEMPERATURE: TemperatureMonitor(),
        }

        self.healer = InfrastructureHealer()
        self.active_alerts: List[InfrastructureAlert] = []
        self.alert_history: List[InfrastructureAlert] = []
        self.current_metrics: Dict[ResourceType, ResourceMetrics] = {}

        self.running = False
        self.monitoring_interval = 30  # seconds

        # Statistics
        self.total_alerts = 0
        self.total_healing_attempts = 0
        self.successful_healings = 0

    async def start_monitoring(self) -> None:
        if self.running:
            logger.warning("Infrastructure sentinel already running")
            return

        self.running = True
        logger.info("Starting infrastructure sentinel")

        try:
            while self.running:
                await self._monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
        except Exception as e:
            logger.error(f"Infrastructure monitoring error: {str(e)}")
        finally:
            self.running = False

    async def stop_monitoring(self) -> None:
        self.running = False
        logger.info("Stopping infrastructure sentinel")

    async def _monitoring_cycle(self) -> None:
        try:
            # Collect metrics from all monitors
            for resource_type, monitor in self.monitors.items():
                try:
                    metrics = await monitor.collect_metrics()
                    self.current_metrics[resource_type] = metrics

                    # Check for alerts
                    alerts = self._check_for_alerts(metrics, monitor)
                    for alert in alerts:
                        await self._handle_alert(alert)

                except Exception as e:
                    logger.error(f"Error monitoring {resource_type.value}: {str(e)}")

            # Check if any alerts have been resolved
            await self._check_resolved_alerts()

        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")

    def _check_for_alerts(
        self, metrics: ResourceMetrics, monitor: ResourceMonitor
    ) -> List[InfrastructureAlert]:
        alerts = []
        thresholds = monitor.get_thresholds()

        alert_level = None
        if metrics.utilization_percent >= thresholds.get("emergency", 100):
            alert_level = AlertLevel.EMERGENCY
        elif metrics.utilization_percent >= thresholds.get("critical", 100):
            alert_level = AlertLevel.CRITICAL
        elif metrics.utilization_percent >= thresholds.get("warning", 100):
            alert_level = AlertLevel.WARNING

        if alert_level:
            # Check if we already have an active alert for this resource
            existing_alert = next(
                (
                    alert
                    for alert in self.active_alerts
                    if alert.resource_type == metrics.resource_type
                    and not alert.resolved
                ),
                None,
            )

            if not existing_alert:
                alert = InfrastructureAlert(
                    resource_type=metrics.resource_type,
                    alert_level=alert_level,
                    message=f"{metrics.resource_type.value.upper()} usage at {metrics.utilization_percent:.1f}%",
                    current_value=metrics.utilization_percent,
                    threshold_value=thresholds.get(alert_level.value, 0),
                )
                alerts.append(alert)

        return alerts

    async def _handle_alert(self, alert: InfrastructureAlert) -> None:
        self.total_alerts += 1
        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        logger.warning(f"Infrastructure alert: {alert.message}")

        # Attempt auto-healing for critical alerts
        if alert.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            if await self.healer.can_heal(alert):
                self.total_healing_attempts += 1
                success = await self.healer.heal(alert)
                if success:
                    self.successful_healings += 1
                    # Give some time for the healing to take effect
                    await asyncio.sleep(5)

    async def _check_resolved_alerts(self) -> None:
        for alert in self.active_alerts.copy():
            if alert.resolved:
                continue

            # Check if the resource is back to healthy levels
            current_metrics = self.current_metrics.get(alert.resource_type)
            if (
                current_metrics
                and current_metrics.status == InfrastructureStatus.HEALTHY
            ):
                alert.resolved = True
                logger.info(f"Alert resolved: {alert.message}")

    def get_infrastructure_status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "overall_status": self._calculate_overall_status(),
            "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
            "total_alerts": self.total_alerts,
            "healing_success_rate": (
                self.successful_healings / max(1, self.total_healing_attempts)
            ),
            "resource_status": {
                resource_type.value: {
                    "status": metrics.status.value,
                    "utilization": metrics.utilization_percent,
                    "current_value": metrics.current_value,
                }
                for resource_type, metrics in self.current_metrics.items()
            },
        }

    def _calculate_overall_status(self) -> str:
        if not self.current_metrics:
            return InfrastructureStatus.HEALTHY.value

        statuses = [metrics.status for metrics in self.current_metrics.values()]

        if InfrastructureStatus.FAILED in statuses:
            return InfrastructureStatus.FAILED.value
        elif InfrastructureStatus.CRITICAL in statuses:
            return InfrastructureStatus.CRITICAL.value
        elif InfrastructureStatus.DEGRADED in statuses:
            return InfrastructureStatus.DEGRADED.value
        else:
            return InfrastructureStatus.HEALTHY.value

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        return [
            {
                "resource_type": alert.resource_type.value,
                "alert_level": alert.alert_level.value,
                "message": alert.message,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "timestamp": alert.timestamp.isoformat(),
                "auto_healing_attempted": alert.auto_healing_attempted,
                "resolved": alert.resolved,
            }
            for alert in self.active_alerts
            if not alert.resolved
        ]


# Global sentinel instance
_global_sentinel: Optional[InfrastructureSentinel] = None


def get_infrastructure_sentinel() -> InfrastructureSentinel:
    global _global_sentinel
    if _global_sentinel is None:
        _global_sentinel = InfrastructureSentinel()
    return _global_sentinel


async def start_infrastructure_monitoring() -> None:
    sentinel = get_infrastructure_sentinel()
    await sentinel.start_monitoring()


async def stop_infrastructure_monitoring() -> None:
    sentinel = get_infrastructure_sentinel()
    await sentinel.stop_monitoring()


def get_infrastructure_health() -> Dict[str, Any]:
    sentinel = get_infrastructure_sentinel()
    return sentinel.get_infrastructure_status()
