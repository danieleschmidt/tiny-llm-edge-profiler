"""
Unified Self-Healing Guard System
Coordinates pipeline monitoring, model drift detection, and infrastructure sentinel
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import json

from .pipeline_guard import (
    SelfHealingPipelineGuard,
    get_pipeline_guard,
    PipelineStage,
    HealthStatus,
)
from .model_drift_detector import (
    ModelDriftMonitor,
    get_drift_monitor,
    DriftType,
    DriftSeverity,
)
from .infrastructure_sentinel import (
    InfrastructureSentinel,
    get_infrastructure_sentinel,
    ResourceType,
    AlertLevel,
)

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class SystemAlert:
    source: str  # pipeline, drift, infrastructure
    alert_type: str
    priority: AlertPriority
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class SystemMetrics:
    pipeline_health: Dict[str, Any]
    drift_status: Dict[str, Any]
    infrastructure_status: Dict[str, Any]
    overall_status: SystemStatus
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedGuardSystem:
    def __init__(self):
        # Component instances
        self.pipeline_guard = get_pipeline_guard()
        self.drift_monitor = get_drift_monitor()
        self.infrastructure_sentinel = get_infrastructure_sentinel()

        # System state
        self.running = False
        self.system_alerts: List[SystemAlert] = []
        self.alert_callbacks: List[Callable[[SystemAlert], None]] = []

        # Configuration
        self.monitoring_interval = 30  # seconds
        self.alert_retention_hours = 24
        self.max_alerts = 1000

        # Statistics
        self.start_time: Optional[datetime] = None
        self.total_alerts = 0
        self.resolved_alerts = 0
        self.system_recoveries = 0

    async def start_unified_monitoring(self) -> None:
        if self.running:
            logger.warning("Unified guard system already running")
            return

        self.running = True
        self.start_time = datetime.now()
        logger.info("Starting unified self-healing guard system")

        try:
            # Start all component monitoring systems
            monitoring_tasks = [
                asyncio.create_task(self.pipeline_guard.start_monitoring()),
                asyncio.create_task(self.infrastructure_sentinel.start_monitoring()),
                asyncio.create_task(self._unified_monitoring_loop()),
            ]

            # Wait for all monitoring tasks
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in unified monitoring: {str(e)}")
        finally:
            self.running = False

    async def stop_unified_monitoring(self) -> None:
        self.running = False
        logger.info("Stopping unified guard system")

        # Stop component monitoring
        await self.pipeline_guard.stop_monitoring()
        await self.infrastructure_sentinel.stop_monitoring()

    async def _unified_monitoring_loop(self) -> None:
        while self.running:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()

                # Analyze system health
                alerts = await self._analyze_system_health(system_metrics)

                # Process new alerts
                for alert in alerts:
                    await self._handle_system_alert(alert)

                # Clean up old alerts
                self._cleanup_old_alerts()

                # Check for system recovery
                await self._check_system_recovery()

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in unified monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Short sleep on error

    async def _collect_system_metrics(self) -> SystemMetrics:
        # Collect metrics from all components
        pipeline_health = self.pipeline_guard.get_health_status()
        drift_status = self.drift_monitor.get_drift_summary()
        infrastructure_status = self.infrastructure_sentinel.get_infrastructure_status()

        # Calculate overall system status
        overall_status = self._calculate_overall_status(
            pipeline_health, drift_status, infrastructure_status
        )

        return SystemMetrics(
            pipeline_health=pipeline_health,
            drift_status=drift_status,
            infrastructure_status=infrastructure_status,
            overall_status=overall_status,
        )

    def _calculate_overall_status(
        self,
        pipeline_health: Dict[str, Any],
        drift_status: Dict[str, Any],
        infrastructure_status: Dict[str, Any],
    ) -> SystemStatus:

        # Check for failed states
        if (
            not pipeline_health.get("running", False)
            or infrastructure_status.get("overall_status") == "failed"
        ):
            return SystemStatus.FAILED

        # Check for critical states
        if (
            pipeline_health.get("active_failures", 0) > 0
            or drift_status.get("active_alerts_by_severity", {}).get("critical", 0) > 0
            or infrastructure_status.get("overall_status") == "critical"
        ):
            return SystemStatus.CRITICAL

        # Check for warning states
        if (
            drift_status.get("active_alerts_count", 0) > 0
            or infrastructure_status.get("overall_status") == "degraded"
        ):
            return SystemStatus.WARNING

        return SystemStatus.HEALTHY

    async def _analyze_system_health(self, metrics: SystemMetrics) -> List[SystemAlert]:
        alerts = []

        # Analyze pipeline health
        pipeline_alerts = self._analyze_pipeline_health(metrics.pipeline_health)
        alerts.extend(pipeline_alerts)

        # Analyze drift status
        drift_alerts = self._analyze_drift_status(metrics.drift_status)
        alerts.extend(drift_alerts)

        # Analyze infrastructure status
        infra_alerts = self._analyze_infrastructure_status(
            metrics.infrastructure_status
        )
        alerts.extend(infra_alerts)

        # Check for correlation patterns
        correlation_alerts = self._analyze_correlations(metrics)
        alerts.extend(correlation_alerts)

        return alerts

    def _analyze_pipeline_health(
        self, pipeline_health: Dict[str, Any]
    ) -> List[SystemAlert]:
        alerts = []

        # Check for active pipeline failures
        active_failures = pipeline_health.get("active_failures", 0)
        if active_failures > 0:
            priority = (
                AlertPriority.HIGH if active_failures > 2 else AlertPriority.MEDIUM
            )
            alerts.append(
                SystemAlert(
                    source="pipeline",
                    alert_type="pipeline_failures",
                    priority=priority,
                    message=f"{active_failures} active pipeline failures detected",
                    details={"active_failures": active_failures},
                )
            )

        # Check healing success rate
        healing_rate = pipeline_health.get("healing_success_rate", 1.0)
        if healing_rate < 0.5:
            alerts.append(
                SystemAlert(
                    source="pipeline",
                    alert_type="low_healing_rate",
                    priority=AlertPriority.MEDIUM,
                    message=f"Low pipeline healing success rate: {healing_rate:.2f}",
                    details={"healing_success_rate": healing_rate},
                )
            )

        return alerts

    def _analyze_drift_status(self, drift_status: Dict[str, Any]) -> List[SystemAlert]:
        alerts = []

        # Check for critical drift alerts
        critical_alerts = drift_status.get("active_alerts_by_severity", {}).get(
            "critical", 0
        )
        if critical_alerts > 0:
            alerts.append(
                SystemAlert(
                    source="drift",
                    alert_type="critical_model_drift",
                    priority=AlertPriority.CRITICAL,
                    message=f"{critical_alerts} critical model drift alerts",
                    details={"critical_drift_alerts": critical_alerts},
                )
            )

        # Check for high number of drift alerts
        total_alerts = drift_status.get("active_alerts_count", 0)
        if total_alerts > 5:
            alerts.append(
                SystemAlert(
                    source="drift",
                    alert_type="high_drift_alert_count",
                    priority=AlertPriority.MEDIUM,
                    message=f"High number of drift alerts: {total_alerts}",
                    details={"total_drift_alerts": total_alerts},
                )
            )

        return alerts

    def _analyze_infrastructure_status(
        self, infra_status: Dict[str, Any]
    ) -> List[SystemAlert]:
        alerts = []

        # Check overall infrastructure status
        overall_status = infra_status.get("overall_status", "healthy")
        if overall_status == "failed":
            alerts.append(
                SystemAlert(
                    source="infrastructure",
                    alert_type="infrastructure_failure",
                    priority=AlertPriority.EMERGENCY,
                    message="Infrastructure failure detected",
                    details={"overall_status": overall_status},
                )
            )
        elif overall_status == "critical":
            alerts.append(
                SystemAlert(
                    source="infrastructure",
                    alert_type="infrastructure_critical",
                    priority=AlertPriority.CRITICAL,
                    message="Critical infrastructure issues detected",
                    details={"overall_status": overall_status},
                )
            )

        # Check resource-specific issues
        resource_status = infra_status.get("resource_status", {})
        for resource, status_info in resource_status.items():
            if status_info.get("status") == "critical":
                alerts.append(
                    SystemAlert(
                        source="infrastructure",
                        alert_type="resource_critical",
                        priority=AlertPriority.HIGH,
                        message=f"Critical {resource} resource status",
                        details={
                            "resource": resource,
                            "status": status_info.get("status"),
                            "utilization": status_info.get("utilization"),
                        },
                    )
                )

        return alerts

    def _analyze_correlations(self, metrics: SystemMetrics) -> List[SystemAlert]:
        alerts = []

        # Check for cascading failures
        pipeline_failures = metrics.pipeline_health.get("active_failures", 0)
        infra_critical = (
            metrics.infrastructure_status.get("overall_status") == "critical"
        )
        drift_alerts = metrics.drift_status.get("active_alerts_count", 0)

        if pipeline_failures > 0 and infra_critical and drift_alerts > 0:
            alerts.append(
                SystemAlert(
                    source="correlation",
                    alert_type="cascading_failure",
                    priority=AlertPriority.EMERGENCY,
                    message="Cascading failure detected across multiple systems",
                    details={
                        "pipeline_failures": pipeline_failures,
                        "infrastructure_critical": infra_critical,
                        "drift_alerts": drift_alerts,
                    },
                )
            )

        # Check for resource exhaustion pattern
        resource_status = metrics.infrastructure_status.get("resource_status", {})
        high_utilization_resources = [
            resource
            for resource, status in resource_status.items()
            if status.get("utilization", 0) > 90
        ]

        if len(high_utilization_resources) >= 2 and pipeline_failures > 0:
            alerts.append(
                SystemAlert(
                    source="correlation",
                    alert_type="resource_exhaustion",
                    priority=AlertPriority.CRITICAL,
                    message="Resource exhaustion affecting pipeline performance",
                    details={
                        "high_utilization_resources": high_utilization_resources,
                        "pipeline_failures": pipeline_failures,
                    },
                )
            )

        return alerts

    async def _handle_system_alert(self, alert: SystemAlert) -> None:
        self.total_alerts += 1
        self.system_alerts.append(alert)

        logger.warning(f"System alert [{alert.priority.name}]: {alert.message}")

        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")

        # Automatic escalation for emergency alerts
        if alert.priority == AlertPriority.EMERGENCY:
            await self._handle_emergency_alert(alert)

    async def _handle_emergency_alert(self, alert: SystemAlert) -> None:
        logger.critical(f"EMERGENCY ALERT: {alert.message}")

        # Implement emergency response protocols
        if alert.alert_type == "cascading_failure":
            # Attempt system-wide recovery
            await self._attempt_system_recovery()
        elif alert.alert_type == "infrastructure_failure":
            # Focus on infrastructure healing
            logger.critical(
                "Infrastructure failure - check system resources immediately"
            )

    async def _attempt_system_recovery(self) -> None:
        logger.info("Attempting system-wide recovery")
        self.system_recoveries += 1

        try:
            # Stop and restart monitoring to clear transient issues
            await asyncio.sleep(5)  # Brief pause

            # Clear resolved alerts
            self._cleanup_old_alerts()

            logger.info("System recovery attempt completed")

        except Exception as e:
            logger.error(f"System recovery failed: {str(e)}")

    async def _check_system_recovery(self) -> None:
        # Check if critical alerts have been resolved
        critical_alerts = [
            alert
            for alert in self.system_alerts
            if not alert.resolved
            and alert.priority in [AlertPriority.CRITICAL, AlertPriority.EMERGENCY]
        ]

        if not critical_alerts:
            # Mark recent alerts as resolved if system is healthy
            recent_alerts = [
                alert
                for alert in self.system_alerts
                if not alert.resolved
                and (datetime.now() - alert.timestamp).seconds < 300
            ]

            for alert in recent_alerts:
                alert.resolved = True
                self.resolved_alerts += 1
                logger.info(f"Alert resolved: {alert.message}")

    def _cleanup_old_alerts(self) -> None:
        # Remove old alerts beyond retention period
        cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)

        old_alerts = [
            alert for alert in self.system_alerts if alert.timestamp < cutoff_time
        ]

        for alert in old_alerts:
            self.system_alerts.remove(alert)

        # Limit total alerts
        if len(self.system_alerts) > self.max_alerts:
            excess_count = len(self.system_alerts) - self.max_alerts
            self.system_alerts = self.system_alerts[excess_count:]

    def add_alert_callback(self, callback: Callable[[SystemAlert], None]) -> None:
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[SystemAlert], None]) -> None:
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def acknowledge_alert(self, alert_id: int) -> bool:
        if 0 <= alert_id < len(self.system_alerts):
            self.system_alerts[alert_id].acknowledged = True
            return True
        return False

    def get_system_status(self) -> Dict[str, Any]:
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        active_alerts = [alert for alert in self.system_alerts if not alert.resolved]

        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "overall_status": self._calculate_current_status().value,
            "active_alerts": len(active_alerts),
            "total_alerts": self.total_alerts,
            "resolved_alerts": self.resolved_alerts,
            "system_recoveries": self.system_recoveries,
            "alert_breakdown": {
                priority.name: len(
                    [alert for alert in active_alerts if alert.priority == priority]
                )
                for priority in AlertPriority
            },
            "component_status": {
                "pipeline_guard": self.pipeline_guard.get_health_status(),
                "drift_monitor": self.drift_monitor.get_drift_summary(),
                "infrastructure_sentinel": self.infrastructure_sentinel.get_infrastructure_status(),
            },
        }

    def _calculate_current_status(self) -> SystemStatus:
        active_alerts = [alert for alert in self.system_alerts if not alert.resolved]

        if any(alert.priority == AlertPriority.EMERGENCY for alert in active_alerts):
            return SystemStatus.FAILED
        elif any(alert.priority == AlertPriority.CRITICAL for alert in active_alerts):
            return SystemStatus.CRITICAL
        elif any(
            alert.priority in [AlertPriority.HIGH, AlertPriority.MEDIUM]
            for alert in active_alerts
        ):
            return SystemStatus.WARNING
        else:
            return SystemStatus.HEALTHY

    def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        active_alerts = [alert for alert in self.system_alerts if not alert.resolved][
            -limit:
        ]

        return [
            {
                "source": alert.source,
                "alert_type": alert.alert_type,
                "priority": alert.priority.name,
                "message": alert.message,
                "details": alert.details,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged,
            }
            for alert in active_alerts
        ]


# Global unified guard instance
_global_unified_guard: Optional[UnifiedGuardSystem] = None


def get_unified_guard() -> UnifiedGuardSystem:
    global _global_unified_guard
    if _global_unified_guard is None:
        _global_unified_guard = UnifiedGuardSystem()
    return _global_unified_guard


async def start_unified_guard() -> None:
    guard = get_unified_guard()
    await guard.start_unified_monitoring()


async def stop_unified_guard() -> None:
    guard = get_unified_guard()
    await guard.stop_unified_monitoring()


def get_unified_status() -> Dict[str, Any]:
    guard = get_unified_guard()
    return guard.get_system_status()


def add_alert_handler(callback: Callable[[SystemAlert], None]) -> None:
    guard = get_unified_guard()
    guard.add_alert_callback(callback)
