"""
Auto-scaling and load balancing for the Tiny LLM Edge Profiler.
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import queue

from .exceptions import TinyLLMProfilerError, ResourceError
from .logging_config import get_logger, PerformanceLogger
from .concurrent_utils import ProfilingTask, TaskResult
from .scalable_profiler import ScalableProfiler
from .resource_pool import ResourcePoolManager, ResourcePool
from .health import health_checker

logger = get_logger("scaling")
perf_logger = PerformanceLogger()


class ScalingPolicy(str, Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"  # Scale based on current load
    PREDICTIVE = "predictive"  # Scale based on predicted load
    SCHEDULED = "scheduled"  # Scale based on schedule
    HYBRID = "hybrid"  # Combination of policies


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    queue_length: int
    active_tasks: int
    completed_tasks_per_minute: float
    average_task_duration: float
    error_rate_percent: float
    response_time_p95: float


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    name: str
    metric_name: str
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    threshold: float
    action: str  # 'scale_up', 'scale_down'
    cooldown_seconds: int = 300  # 5 minutes
    last_triggered: datetime = field(default_factory=datetime.now)
    
    def should_trigger(self, metric_value: float) -> bool:
        """Check if rule should trigger based on metric value."""
        # Check cooldown
        if (datetime.now() - self.last_triggered).total_seconds() < self.cooldown_seconds:
            return False
        
        # Check condition
        if self.operator == 'gt':
            return metric_value > self.threshold
        elif self.operator == 'lt':
            return metric_value < self.threshold
        elif self.operator == 'gte':
            return metric_value >= self.threshold
        elif self.operator == 'lte':
            return metric_value <= self.threshold
        elif self.operator == 'eq':
            return abs(metric_value - self.threshold) < 0.01
        
        return False
    
    def trigger(self) -> None:
        """Mark rule as triggered."""
        self.last_triggered = datetime.now()


class LoadBalancer:
    """
    Load balancer for distributing tasks across multiple resources.
    """
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.resources: List[Dict[str, Any]] = []
        self.current_index = 0
        self.resource_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def add_resource(self, resource_id: str, resource: Any, capacity_weight: float = 1.0) -> None:
        """Add a resource to the load balancer."""
        with self._lock:
            resource_info = {
                "id": resource_id,
                "resource": resource,
                "capacity_weight": capacity_weight,
                "active_tasks": 0,
                "total_tasks": 0,
                "error_count": 0,
                "last_used": datetime.now(),
                "healthy": True
            }
            
            self.resources.append(resource_info)
            self.resource_stats[resource_id] = {
                "tasks_assigned": 0,
                "tasks_completed": 0,
                "tasks_failed": 0,
                "average_response_time": 0.0,
                "load_score": 0.0
            }
            
            logger.info(f"Added resource to load balancer: {resource_id}")
    
    def remove_resource(self, resource_id: str) -> bool:
        """Remove a resource from the load balancer."""
        with self._lock:
            for i, resource_info in enumerate(self.resources):
                if resource_info["id"] == resource_id:
                    del self.resources[i]
                    if resource_id in self.resource_stats:
                        del self.resource_stats[resource_id]
                    logger.info(f"Removed resource from load balancer: {resource_id}")
                    return True
            return False
    
    def select_resource(self, task: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Select the best resource for a task."""
        with self._lock:
            if not self.resources:
                return None
            
            healthy_resources = [r for r in self.resources if r["healthy"]]
            if not healthy_resources:
                return None
            
            if self.strategy == "round_robin":
                return self._round_robin_selection(healthy_resources)
            elif self.strategy == "least_connections":
                return self._least_connections_selection(healthy_resources)
            elif self.strategy == "weighted_round_robin":
                return self._weighted_round_robin_selection(healthy_resources)
            elif self.strategy == "least_response_time":
                return self._least_response_time_selection(healthy_resources)
            else:
                # Default to round robin
                return self._round_robin_selection(healthy_resources)
    
    def _round_robin_selection(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Round robin resource selection."""
        if self.current_index >= len(resources):
            self.current_index = 0
        
        selected = resources[self.current_index]
        self.current_index = (self.current_index + 1) % len(resources)
        return selected
    
    def _least_connections_selection(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select resource with least active connections."""
        return min(resources, key=lambda r: r["active_tasks"])
    
    def _weighted_round_robin_selection(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted round robin based on capacity."""
        # Simple implementation - could be more sophisticated
        weights = [r["capacity_weight"] for r in resources]
        total_weight = sum(weights)
        
        import random
        rand_val = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for resource in resources:
            cumulative_weight += resource["capacity_weight"]
            if rand_val <= cumulative_weight:
                return resource
        
        return resources[-1]  # Fallback
    
    def _least_response_time_selection(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select resource with lowest average response time."""
        return min(resources, key=lambda r: self.resource_stats[r["id"]]["average_response_time"])
    
    def report_task_start(self, resource_id: str, task_id: str) -> None:
        """Report that a task has started on a resource."""
        with self._lock:
            for resource_info in self.resources:
                if resource_info["id"] == resource_id:
                    resource_info["active_tasks"] += 1
                    resource_info["total_tasks"] += 1
                    resource_info["last_used"] = datetime.now()
                    break
            
            if resource_id in self.resource_stats:
                self.resource_stats[resource_id]["tasks_assigned"] += 1
    
    def report_task_completion(
        self,
        resource_id: str,
        task_id: str,
        success: bool,
        response_time: float
    ) -> None:
        """Report task completion."""
        with self._lock:
            for resource_info in self.resources:
                if resource_info["id"] == resource_id:
                    resource_info["active_tasks"] = max(0, resource_info["active_tasks"] - 1)
                    if not success:
                        resource_info["error_count"] += 1
                    break
            
            if resource_id in self.resource_stats:
                stats = self.resource_stats[resource_id]
                
                if success:
                    stats["tasks_completed"] += 1
                    # Update running average response time
                    count = stats["tasks_completed"]
                    current_avg = stats["average_response_time"]
                    stats["average_response_time"] = (current_avg * (count - 1) + response_time) / count
                else:
                    stats["tasks_failed"] += 1
    
    def update_resource_health(self, resource_id: str, healthy: bool) -> None:
        """Update resource health status."""
        with self._lock:
            for resource_info in self.resources:
                if resource_info["id"] == resource_id:
                    resource_info["healthy"] = healthy
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_resources = len(self.resources)
            healthy_resources = len([r for r in self.resources if r["healthy"]])
            total_active_tasks = sum(r["active_tasks"] for r in self.resources)
            
            return {
                "strategy": self.strategy,
                "total_resources": total_resources,
                "healthy_resources": healthy_resources,
                "total_active_tasks": total_active_tasks,
                "resource_stats": self.resource_stats.copy()
            }


class AutoScaler:
    """
    Auto-scaler for dynamic resource management.
    """
    
    def __init__(
        self,
        resource_pool_manager: ResourcePoolManager,
        concurrent_profiler: ConcurrentProfiler,
        min_instances: int = 1,
        max_instances: int = 10,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        check_interval_seconds: int = 60
    ):
        self.resource_pool_manager = resource_pool_manager
        self.concurrent_profiler = concurrent_profiler
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.check_interval_seconds = check_interval_seconds
        
        # Metrics collection
        self.metrics_history: deque = deque(maxlen=100)
        self.scaling_rules: List[ScalingRule] = []
        
        # Load balancer
        self.load_balancer = LoadBalancer(strategy="least_connections")
        
        # Control
        self.running = False
        self.scaling_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Statistics
        self.scaling_events = {
            "scale_up_events": 0,
            "scale_down_events": 0,
            "last_scale_event": None,
            "total_instances_created": 0,
            "total_instances_destroyed": 0
        }
        
        # Initialize default scaling rules
        self._create_default_scaling_rules()
    
    def _create_default_scaling_rules(self) -> None:
        """Create default scaling rules."""
        self.scaling_rules = [
            # Scale up rules
            ScalingRule(
                name="high_cpu_scale_up",
                metric_name="cpu_usage_percent",
                operator="gt",
                threshold=80.0,
                action="scale_up",
                cooldown_seconds=300
            ),
            ScalingRule(
                name="high_queue_scale_up",
                metric_name="queue_length",
                operator="gt",
                threshold=10,
                action="scale_up",
                cooldown_seconds=180
            ),
            ScalingRule(
                name="high_response_time_scale_up",
                metric_name="response_time_p95",
                operator="gt",
                threshold=5000,  # 5 seconds
                action="scale_up",
                cooldown_seconds=300
            ),
            
            # Scale down rules
            ScalingRule(
                name="low_cpu_scale_down",
                metric_name="cpu_usage_percent",
                operator="lt",
                threshold=20.0,
                action="scale_down",
                cooldown_seconds=600  # Longer cooldown for scale down
            ),
            ScalingRule(
                name="low_queue_scale_down",
                metric_name="queue_length",
                operator="lt",
                threshold=2,
                action="scale_down",
                cooldown_seconds=600
            )
        ]
    
    def start(self) -> None:
        """Start the auto-scaler."""
        with self._lock:
            if self.running:
                return
            
            self.running = True
            self.scaling_thread = threading.Thread(
                target=self._scaling_loop,
                name="AutoScaler",
                daemon=True
            )
            self.scaling_thread.start()
            
            logger.info("Auto-scaler started")
    
    def stop(self) -> None:
        """Stop the auto-scaler."""
        with self._lock:
            self.running = False
            
            if self.scaling_thread:
                self.scaling_thread.join(timeout=30)
            
            logger.info("Auto-scaler stopped")
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    
                    # Evaluate scaling rules
                    self._evaluate_scaling_rules(metrics)
                
                # Health check for load balancer resources
                self._health_check_resources()
                
                time.sleep(self.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Auto-scaler error: {e}")
                time.sleep(min(self.check_interval_seconds, 60))
    
    def _collect_metrics(self) -> Optional[ScalingMetrics]:
        """Collect current system metrics."""
        try:
            # Get system metrics
            system_metrics = health_checker.collect_system_metrics()
            
            # Get profiler stats
            profiler_stats = self.concurrent_profiler.get_stats()
            
            # Calculate task completion rate
            completed_tasks_per_minute = 0.0
            if len(self.metrics_history) > 0:
                recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
                if len(recent_metrics) >= 2:
                    time_span = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 60
                    if time_span > 0:
                        task_diff = recent_metrics[-1].completed_tasks_per_minute - recent_metrics[0].completed_tasks_per_minute
                        completed_tasks_per_minute = task_diff / time_span
            
            # Calculate average task duration and error rate
            average_duration = 0.0
            error_rate = 0.0
            
            if profiler_stats["completed_tasks"] > 0:
                average_duration = profiler_stats.get("average_duration_seconds", 0.0)
                error_rate = (profiler_stats.get("tasks_failed", 0) / profiler_stats["completed_tasks"]) * 100
            
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=system_metrics.cpu_percent,
                memory_usage_percent=system_metrics.memory_percent,
                queue_length=profiler_stats.get("queued_tasks", 0),
                active_tasks=profiler_stats.get("active_tasks", 0),
                completed_tasks_per_minute=completed_tasks_per_minute,
                average_task_duration=average_duration,
                error_rate_percent=error_rate,
                response_time_p95=0.0  # Would calculate from response time distribution
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return None
    
    def _evaluate_scaling_rules(self, metrics: ScalingMetrics) -> None:
        """Evaluate scaling rules against current metrics."""
        for rule in self.scaling_rules:
            try:
                # Get metric value
                metric_value = getattr(metrics, rule.metric_name, 0.0)
                
                if rule.should_trigger(metric_value):
                    logger.info(f"Scaling rule triggered: {rule.name} (value: {metric_value}, threshold: {rule.threshold})")
                    
                    if rule.action == "scale_up":
                        self._scale_up()
                    elif rule.action == "scale_down":
                        self._scale_down()
                    
                    rule.trigger()
                    
                    # Log scaling event
                    perf_logger.log_metric(
                        metric_name="scaling_event",
                        value=1.0,
                        unit="event",
                        rule_name=rule.name,
                        action=rule.action,
                        metric_value=metric_value
                    )
            
            except Exception as e:
                logger.error(f"Error evaluating scaling rule {rule.name}: {e}")
    
    def _scale_up(self) -> bool:
        """Scale up resources."""
        with self._lock:
            current_instances = len(self.load_balancer.resources)
            
            if current_instances >= self.max_instances:
                logger.warning(f"Cannot scale up: already at maximum instances ({self.max_instances})")
                return False
            
            try:
                # Create new resource instance
                instance_id = f"scaled_instance_{int(time.time() * 1000)}"
                
                # This would create actual resources - simplified for example
                new_resource = self._create_resource_instance(instance_id)
                
                if new_resource:
                    self.load_balancer.add_resource(instance_id, new_resource)
                    self.scaling_events["scale_up_events"] += 1
                    self.scaling_events["total_instances_created"] += 1
                    self.scaling_events["last_scale_event"] = datetime.now()
                    
                    logger.info(f"Scaled up: created instance {instance_id} (total: {current_instances + 1})")
                    return True
                
            except Exception as e:
                logger.error(f"Scale up failed: {e}")
            
            return False
    
    def _scale_down(self) -> bool:
        """Scale down resources."""
        with self._lock:
            current_instances = len(self.load_balancer.resources)
            
            if current_instances <= self.min_instances:
                logger.warning(f"Cannot scale down: already at minimum instances ({self.min_instances})")
                return False
            
            try:
                # Find least used resource
                least_used_resource = min(
                    self.load_balancer.resources,
                    key=lambda r: r["active_tasks"]
                )
                
                if least_used_resource["active_tasks"] == 0:
                    instance_id = least_used_resource["id"]
                    
                    # Remove from load balancer
                    self.load_balancer.remove_resource(instance_id)
                    
                    # Clean up resource
                    self._destroy_resource_instance(instance_id, least_used_resource["resource"])
                    
                    self.scaling_events["scale_down_events"] += 1
                    self.scaling_events["total_instances_destroyed"] += 1
                    self.scaling_events["last_scale_event"] = datetime.now()
                    
                    logger.info(f"Scaled down: removed instance {instance_id} (total: {current_instances - 1})")
                    return True
                else:
                    logger.debug("Cannot scale down: all resources have active tasks")
            
            except Exception as e:
                logger.error(f"Scale down failed: {e}")
            
            return False
    
    def _create_resource_instance(self, instance_id: str) -> Optional[Any]:
        """Create a new resource instance."""
        # This is a placeholder - in real implementation, this would:
        # 1. Create new worker threads/processes
        # 2. Initialize new device connections
        # 3. Set up new resource pools
        # For now, return a mock resource
        return {"id": instance_id, "type": "mock_resource", "created_at": datetime.now()}
    
    def _destroy_resource_instance(self, instance_id: str, resource: Any) -> None:
        """Destroy a resource instance."""
        # Placeholder for resource cleanup
        logger.debug(f"Destroying resource instance: {instance_id}")
    
    def _health_check_resources(self) -> None:
        """Perform health checks on load balancer resources."""
        unhealthy_resources = []
        
        for resource_info in self.load_balancer.resources:
            try:
                # Perform health check (simplified)
                is_healthy = self._check_resource_health(resource_info["resource"])
                
                if not is_healthy and resource_info["healthy"]:
                    logger.warning(f"Resource became unhealthy: {resource_info['id']}")
                    self.load_balancer.update_resource_health(resource_info["id"], False)
                    unhealthy_resources.append(resource_info["id"])
                elif is_healthy and not resource_info["healthy"]:
                    logger.info(f"Resource recovered: {resource_info['id']}")
                    self.load_balancer.update_resource_health(resource_info["id"], True)
                    
            except Exception as e:
                logger.error(f"Health check failed for resource {resource_info['id']}: {e}")
                self.load_balancer.update_resource_health(resource_info["id"], False)
        
        # Remove persistently unhealthy resources
        for resource_id in unhealthy_resources:
            # Add logic to remove resources that have been unhealthy for too long
            pass
    
    def _check_resource_health(self, resource: Any) -> bool:
        """Check if a resource is healthy."""
        # Placeholder health check
        return True
    
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a custom scaling rule."""
        with self._lock:
            self.scaling_rules.append(rule)
            logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str) -> bool:
        """Remove a scaling rule."""
        with self._lock:
            for i, rule in enumerate(self.scaling_rules):
                if rule.name == rule_name:
                    del self.scaling_rules[i]
                    logger.info(f"Removed scaling rule: {rule_name}")
                    return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        with self._lock:
            load_balancer_stats = self.load_balancer.get_stats()
            
            # Recent metrics summary
            recent_metrics = None
            if self.metrics_history:
                recent_metrics = {
                    "cpu_usage_percent": self.metrics_history[-1].cpu_usage_percent,
                    "memory_usage_percent": self.metrics_history[-1].memory_usage_percent,
                    "queue_length": self.metrics_history[-1].queue_length,
                    "active_tasks": self.metrics_history[-1].active_tasks,
                    "error_rate_percent": self.metrics_history[-1].error_rate_percent
                }
            
            return {
                "running": self.running,
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "current_instances": len(self.load_balancer.resources),
                "scaling_events": self.scaling_events.copy(),
                "scaling_rules_count": len(self.scaling_rules),
                "load_balancer": load_balancer_stats,
                "recent_metrics": recent_metrics,
                "metrics_history_size": len(self.metrics_history)
            }


# Global auto-scaler instance (would be configured in production)
_global_auto_scaler: Optional[AutoScaler] = None


def get_auto_scaler() -> Optional[AutoScaler]:
    """Get the global auto-scaler instance."""
    return _global_auto_scaler


def initialize_auto_scaler(
    resource_pool_manager: ResourcePoolManager,
    concurrent_profiler: ConcurrentProfiler,
    **kwargs
) -> AutoScaler:
    """Initialize the global auto-scaler."""
    global _global_auto_scaler
    
    _global_auto_scaler = AutoScaler(
        resource_pool_manager=resource_pool_manager,
        concurrent_profiler=concurrent_profiler,
        **kwargs
    )
    
    return _global_auto_scaler


def start_auto_scaling() -> bool:
    """Start auto-scaling if initialized."""
    if _global_auto_scaler:
        _global_auto_scaler.start()
        return True
    return False


def stop_auto_scaling() -> bool:
    """Stop auto-scaling if running."""
    if _global_auto_scaler:
        _global_auto_scaler.stop()
        return True
    return False