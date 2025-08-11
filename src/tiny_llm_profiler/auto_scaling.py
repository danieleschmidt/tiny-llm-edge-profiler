"""
Auto-scaling capabilities for dynamic resource allocation and load balancing.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import psutil

from .concurrent_utils import ProfilingTask, TaskResult, TaskStatus
from .monitoring import HealthMonitor, Metric, MetricType


class ScalingDecision(str, Enum):
    """Scaling decision types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


class LoadLevel(str, Enum):
    """System load levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime
    queue_length: int
    active_tasks: int
    cpu_utilization: float
    memory_utilization: float
    avg_response_time: float
    error_rate: float
    throughput: float  # tasks per second
    load_level: LoadLevel = LoadLevel.NORMAL


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    # Thresholds for scaling decisions
    scale_up_queue_threshold: int = 10
    scale_down_queue_threshold: int = 2
    cpu_scale_up_threshold: float = 80.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 85.0
    response_time_threshold: float = 5.0  # seconds
    error_rate_threshold: float = 0.1  # 10%
    
    # Worker limits
    min_workers: int = 2
    max_workers: int = 20
    
    # Scaling behavior
    scale_up_factor: int = 2
    scale_down_factor: float = 0.5
    cooldown_period_seconds: int = 60
    evaluation_window_seconds: int = 30
    
    # Emergency scaling
    emergency_queue_threshold: int = 50
    emergency_cpu_threshold: float = 95.0
    emergency_scale_factor: int = 4


class AutoScaler:
    """
    Intelligent auto-scaling system for profiling workloads.
    """
    
    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        enable_predictive_scaling: bool = True
    ):
        self.config = config or ScalingConfig()
        self.enable_predictive_scaling = enable_predictive_scaling
        
        # Scaling state
        self.current_workers = self.config.min_workers
        self.last_scaling_decision = datetime.now()
        self.scaling_history: List[Tuple[datetime, ScalingDecision, int]] = []
        
        # Metrics collection
        self.metrics_history: List[ScalingMetrics] = []
        self.metrics_lock = threading.Lock()
        
        # Monitoring
        self.health_monitor = HealthMonitor()
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.scaling_callbacks: List[Callable[[ScalingDecision, int, int], None]] = []
        
        # Predictive model (simple moving average for now)
        self.prediction_window = 5
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.health_monitor.start_monitoring()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.health_monitor.stop_monitoring()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Auto-scaler monitoring stopped")
    
    def add_scaling_callback(self, callback: Callable[[ScalingDecision, int, int], None]):
        """Add callback for scaling decisions."""
        self.scaling_callbacks.append(callback)
    
    def collect_metrics(
        self,
        queue_length: int,
        active_tasks: int,
        avg_response_time: float,
        error_rate: float,
        throughput: float
    ):
        """Collect current system metrics."""
        with self.metrics_lock:
            # Get system metrics
            cpu_util = psutil.cpu_percent()
            memory_util = psutil.virtual_memory().percent
            
            # Determine load level
            load_level = self._calculate_load_level(
                queue_length, cpu_util, memory_util, avg_response_time, error_rate
            )
            
            metrics = ScalingMetrics(
                timestamp=datetime.now(),
                queue_length=queue_length,
                active_tasks=active_tasks,
                cpu_utilization=cpu_util,
                memory_utilization=memory_util,
                avg_response_time=avg_response_time,
                error_rate=error_rate,
                throughput=throughput,
                load_level=load_level
            )
            
            self.metrics_history.append(metrics)
            
            # Keep only recent metrics
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp > cutoff_time
            ]
    
    def make_scaling_decision(self) -> Tuple[ScalingDecision, int]:
        """
        Make scaling decision based on current metrics.
        
        Returns:
            Tuple of (decision, target_workers)
        """
        if not self.metrics_history:
            return ScalingDecision.MAINTAIN, self.current_workers
        
        # Check cooldown period
        if self._is_in_cooldown():
            return ScalingDecision.MAINTAIN, self.current_workers
        
        # Get recent metrics for decision
        recent_metrics = self._get_recent_metrics()
        if not recent_metrics:
            return ScalingDecision.MAINTAIN, self.current_workers
        
        # Check for emergency scaling
        latest_metrics = recent_metrics[-1]
        if self._needs_emergency_scaling(latest_metrics):
            target_workers = min(
                self.current_workers * self.config.emergency_scale_factor,
                self.config.max_workers
            )
            return ScalingDecision.EMERGENCY_SCALE, target_workers
        
        # Analyze trends
        avg_queue_length = statistics.mean(m.queue_length for m in recent_metrics)
        avg_cpu_util = statistics.mean(m.cpu_utilization for m in recent_metrics)
        avg_memory_util = statistics.mean(m.memory_utilization for m in recent_metrics)
        avg_response_time = statistics.mean(m.avg_response_time for m in recent_metrics)
        avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
        
        # Scale up conditions
        scale_up_needed = (
            avg_queue_length > self.config.scale_up_queue_threshold or
            avg_cpu_util > self.config.cpu_scale_up_threshold or
            avg_memory_util > self.config.memory_scale_up_threshold or
            avg_response_time > self.config.response_time_threshold or
            avg_error_rate > self.config.error_rate_threshold
        )
        
        if scale_up_needed and self.current_workers < self.config.max_workers:
            target_workers = min(
                self.current_workers * self.config.scale_up_factor,
                self.config.max_workers
            )
            return ScalingDecision.SCALE_UP, target_workers
        
        # Scale down conditions
        scale_down_needed = (
            avg_queue_length < self.config.scale_down_queue_threshold and
            avg_cpu_util < self.config.cpu_scale_down_threshold and
            avg_memory_util < 50.0 and  # Conservative memory threshold for scale down
            avg_response_time < self.config.response_time_threshold * 0.5 and
            avg_error_rate < self.config.error_rate_threshold * 0.1
        )
        
        if scale_down_needed and self.current_workers > self.config.min_workers:
            target_workers = max(
                int(self.current_workers * self.config.scale_down_factor),
                self.config.min_workers
            )
            return ScalingDecision.SCALE_DOWN, target_workers
        
        # Use predictive scaling if enabled
        if self.enable_predictive_scaling:
            predicted_decision = self._predict_scaling_need()
            if predicted_decision != ScalingDecision.MAINTAIN:
                if predicted_decision == ScalingDecision.SCALE_UP:
                    target_workers = min(
                        self.current_workers + 1,  # Conservative predictive scaling
                        self.config.max_workers
                    )
                    return predicted_decision, target_workers
                else:  # SCALE_DOWN
                    target_workers = max(
                        self.current_workers - 1,
                        self.config.min_workers
                    )
                    return predicted_decision, target_workers
        
        return ScalingDecision.MAINTAIN, self.current_workers
    
    def execute_scaling_decision(self, decision: ScalingDecision, target_workers: int) -> bool:
        """
        Execute a scaling decision.
        
        Args:
            decision: Scaling decision to execute
            target_workers: Target number of workers
            
        Returns:
            True if scaling was successful
        """
        if decision == ScalingDecision.MAINTAIN:
            return True
        
        old_workers = self.current_workers
        self.current_workers = target_workers
        self.last_scaling_decision = datetime.now()
        
        # Record scaling history
        self.scaling_history.append((datetime.now(), decision, target_workers))
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.scaling_history = [
            (timestamp, dec, workers) for timestamp, dec, workers in self.scaling_history
            if timestamp > cutoff_time
        ]
        
        # Notify callbacks
        for callback in self.scaling_callbacks:
            try:
                callback(decision, old_workers, target_workers)
            except Exception as e:
                self.logger.error(f"Scaling callback failed: {e}")
        
        self.logger.info(
            f"Scaling decision executed: {decision.value} "
            f"({old_workers} -> {target_workers} workers)"
        )
        
        return True
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self.metrics_lock:
            recent_metrics = self._get_recent_metrics()
            
            if recent_metrics:
                latest = recent_metrics[-1]
                avg_queue = statistics.mean(m.queue_length for m in recent_metrics)
                avg_cpu = statistics.mean(m.cpu_utilization for m in recent_metrics)
            else:
                latest = None
                avg_queue = 0
                avg_cpu = 0
            
            # Scaling frequency
            recent_scaling = [
                timestamp for timestamp, _, _ in self.scaling_history
                if timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            return {
                "current_workers": self.current_workers,
                "target_range": {
                    "min": self.config.min_workers,
                    "max": self.config.max_workers
                },
                "current_metrics": {
                    "load_level": latest.load_level.value if latest else "unknown",
                    "queue_length": latest.queue_length if latest else 0,
                    "cpu_utilization": latest.cpu_utilization if latest else 0,
                    "memory_utilization": latest.memory_utilization if latest else 0
                },
                "recent_averages": {
                    "queue_length": avg_queue,
                    "cpu_utilization": avg_cpu
                },
                "scaling_activity": {
                    "total_scaling_events": len(self.scaling_history),
                    "recent_scaling_events": len(recent_scaling),
                    "last_scaling": self.last_scaling_decision.isoformat() if self.scaling_history else None
                },
                "configuration": {
                    "predictive_scaling": self.enable_predictive_scaling,
                    "cooldown_period": self.config.cooldown_period_seconds
                }
            }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Make scaling decision
                decision, target_workers = self.make_scaling_decision()
                
                # Execute if needed
                if decision != ScalingDecision.MAINTAIN:
                    self.execute_scaling_decision(decision, target_workers)
                
                # Wait before next evaluation
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(30)  # Longer wait on error
    
    def _calculate_load_level(
        self,
        queue_length: int,
        cpu_util: float,
        memory_util: float,
        response_time: float,
        error_rate: float
    ) -> LoadLevel:
        """Calculate current load level."""
        # Count critical conditions
        critical_count = 0
        high_count = 0
        
        if queue_length > self.config.emergency_queue_threshold:
            critical_count += 1
        elif queue_length > self.config.scale_up_queue_threshold:
            high_count += 1
        
        if cpu_util > self.config.emergency_cpu_threshold:
            critical_count += 1
        elif cpu_util > self.config.cpu_scale_up_threshold:
            high_count += 1
        
        if memory_util > 90.0:
            critical_count += 1
        elif memory_util > self.config.memory_scale_up_threshold:
            high_count += 1
        
        if response_time > self.config.response_time_threshold * 2:
            critical_count += 1
        elif response_time > self.config.response_time_threshold:
            high_count += 1
        
        if error_rate > self.config.error_rate_threshold * 2:
            critical_count += 1
        elif error_rate > self.config.error_rate_threshold:
            high_count += 1
        
        # Determine level
        if critical_count > 0:
            return LoadLevel.CRITICAL
        elif high_count >= 2:
            return LoadLevel.HIGH
        elif high_count == 1 or queue_length > 5:
            return LoadLevel.NORMAL
        else:
            return LoadLevel.LOW
    
    def _get_recent_metrics(self) -> List[ScalingMetrics]:
        """Get recent metrics within evaluation window."""
        cutoff_time = datetime.now() - timedelta(seconds=self.config.evaluation_window_seconds)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        cooldown_end = self.last_scaling_decision + timedelta(
            seconds=self.config.cooldown_period_seconds
        )
        return datetime.now() < cooldown_end
    
    def _needs_emergency_scaling(self, metrics: ScalingMetrics) -> bool:
        """Check if emergency scaling is needed."""
        return (
            metrics.queue_length > self.config.emergency_queue_threshold or
            metrics.cpu_utilization > self.config.emergency_cpu_threshold or
            metrics.memory_utilization > 95.0 or
            metrics.error_rate > 0.5  # 50% error rate
        )
    
    def _predict_scaling_need(self) -> ScalingDecision:
        """Predict future scaling need based on trends."""
        if len(self.metrics_history) < self.prediction_window:
            return ScalingDecision.MAINTAIN
        
        # Get recent metrics for trend analysis
        recent = self.metrics_history[-self.prediction_window:]
        
        # Calculate trends (simple linear trend)
        queue_trend = self._calculate_trend([m.queue_length for m in recent])
        cpu_trend = self._calculate_trend([m.cpu_utilization for m in recent])
        response_time_trend = self._calculate_trend([m.avg_response_time for m in recent])
        
        # Predict if trends will exceed thresholds
        current_queue = recent[-1].queue_length
        current_cpu = recent[-1].cpu_utilization
        current_response = recent[-1].avg_response_time
        
        # Predict values in next evaluation period
        predicted_queue = current_queue + queue_trend * 2
        predicted_cpu = current_cpu + cpu_trend * 2
        predicted_response = current_response + response_time_trend * 2
        
        # Check if predicted values exceed thresholds
        if (predicted_queue > self.config.scale_up_queue_threshold or
            predicted_cpu > self.config.cpu_scale_up_threshold or
            predicted_response > self.config.response_time_threshold):
            return ScalingDecision.SCALE_UP
        
        if (predicted_queue < self.config.scale_down_queue_threshold and
            predicted_cpu < self.config.cpu_scale_down_threshold and
            predicted_response < self.config.response_time_threshold * 0.5):
            return ScalingDecision.SCALE_DOWN
        
        return ScalingDecision.MAINTAIN
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # Linear regression slope
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0
        
        return (n * xy_sum - x_sum * y_sum) / denominator


class LoadBalancer:
    """
    Intelligent load balancer for distributing profiling tasks.
    """
    
    def __init__(self, auto_scaler: Optional[AutoScaler] = None):
        self.auto_scaler = auto_scaler
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.task_assignments: Dict[str, str] = {}
        self.lock = threading.Lock()
        
        # Load balancing strategies
        self.strategies = {
            "round_robin": self._round_robin_selection,
            "least_connections": self._least_connections_selection,
            "weighted_response_time": self._weighted_response_time_selection,
            "resource_aware": self._resource_aware_selection
        }
        
        self.current_strategy = "resource_aware"
        self.round_robin_counter = 0
        
        self.logger = logging.getLogger(__name__)
    
    def register_worker(self, worker_id: str, capabilities: Optional[Dict[str, Any]] = None):
        """Register a worker with the load balancer."""
        with self.lock:
            self.worker_stats[worker_id] = {
                "active_tasks": 0,
                "completed_tasks": 0,
                "total_response_time": 0.0,
                "avg_response_time": 0.0,
                "error_count": 0,
                "capabilities": capabilities or {},
                "last_assigned": datetime.now(),
                "load_score": 0.0
            }
            
            self.logger.debug(f"Registered worker: {worker_id}")
    
    def remove_worker(self, worker_id: str):
        """Remove a worker from the load balancer."""
        with self.lock:
            if worker_id in self.worker_stats:
                # Reassign active tasks if any
                active_tasks = [
                    task_id for task_id, assigned_worker in self.task_assignments.items()
                    if assigned_worker == worker_id
                ]
                
                for task_id in active_tasks:
                    del self.task_assignments[task_id]
                
                del self.worker_stats[worker_id]
                self.logger.info(f"Removed worker: {worker_id} ({len(active_tasks)} tasks reassigned)")
    
    def assign_task(self, task: ProfilingTask) -> Optional[str]:
        """
        Assign task to optimal worker.
        
        Args:
            task: ProfilingTask to assign
            
        Returns:
            Worker ID if assignment successful, None otherwise
        """
        with self.lock:
            if not self.worker_stats:
                return None
            
            # Select worker using current strategy
            strategy_func = self.strategies.get(self.current_strategy, self._round_robin_selection)
            selected_worker = strategy_func(task)
            
            if selected_worker:
                # Update assignments and stats
                self.task_assignments[task.task_id] = selected_worker
                self.worker_stats[selected_worker]["active_tasks"] += 1
                self.worker_stats[selected_worker]["last_assigned"] = datetime.now()
                
                self.logger.debug(f"Assigned task {task.task_id} to worker {selected_worker}")
                return selected_worker
            
            return None
    
    def complete_task(self, task_id: str, success: bool, response_time: float):
        """Record task completion."""
        with self.lock:
            if task_id in self.task_assignments:
                worker_id = self.task_assignments[task_id]
                
                if worker_id in self.worker_stats:
                    stats = self.worker_stats[worker_id]
                    stats["active_tasks"] = max(0, stats["active_tasks"] - 1)
                    stats["completed_tasks"] += 1
                    stats["total_response_time"] += response_time
                    
                    if stats["completed_tasks"] > 0:
                        stats["avg_response_time"] = stats["total_response_time"] / stats["completed_tasks"]
                    
                    if not success:
                        stats["error_count"] += 1
                    
                    # Update load score
                    stats["load_score"] = self._calculate_load_score(stats)
                
                del self.task_assignments[task_id]
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            total_active = sum(stats["active_tasks"] for stats in self.worker_stats.values())
            total_completed = sum(stats["completed_tasks"] for stats in self.worker_stats.values())
            total_errors = sum(stats["error_count"] for stats in self.worker_stats.values())
            
            return {
                "strategy": self.current_strategy,
                "active_workers": len(self.worker_stats),
                "total_active_tasks": total_active,
                "total_completed_tasks": total_completed,
                "total_errors": total_errors,
                "error_rate": total_errors / max(total_completed, 1),
                "worker_details": dict(self.worker_stats)
            }
    
    def set_strategy(self, strategy: str):
        """Set load balancing strategy."""
        if strategy in self.strategies:
            self.current_strategy = strategy
            self.logger.info(f"Load balancing strategy set to: {strategy}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _round_robin_selection(self, task: ProfilingTask) -> Optional[str]:
        """Round-robin worker selection."""
        workers = list(self.worker_stats.keys())
        if not workers:
            return None
        
        selected = workers[self.round_robin_counter % len(workers)]
        self.round_robin_counter += 1
        return selected
    
    def _least_connections_selection(self, task: ProfilingTask) -> Optional[str]:
        """Select worker with least active connections."""
        if not self.worker_stats:
            return None
        
        min_connections = min(stats["active_tasks"] for stats in self.worker_stats.values())
        candidates = [
            worker_id for worker_id, stats in self.worker_stats.items()
            if stats["active_tasks"] == min_connections
        ]
        
        # If tie, select by lowest error rate
        if len(candidates) > 1:
            candidates.sort(key=lambda w: self.worker_stats[w].get("error_count", 0))
        
        return candidates[0]
    
    def _weighted_response_time_selection(self, task: ProfilingTask) -> Optional[str]:
        """Select worker based on weighted response time."""
        if not self.worker_stats:
            return None
        
        # Calculate weights (inverse of response time)
        weights = {}
        for worker_id, stats in self.worker_stats.items():
            avg_time = stats.get("avg_response_time", 1.0)
            if avg_time > 0:
                weights[worker_id] = 1.0 / avg_time
            else:
                weights[worker_id] = 1.0
        
        # Select worker with highest weight (lowest response time)
        return max(weights.keys(), key=lambda w: weights[w])
    
    def _resource_aware_selection(self, task: ProfilingTask) -> Optional[str]:
        """Advanced resource-aware worker selection."""
        if not self.worker_stats:
            return None
        
        # Calculate composite score for each worker
        scores = {}
        for worker_id, stats in self.worker_stats.items():
            score = self._calculate_selection_score(worker_id, task, stats)
            scores[worker_id] = score
        
        # Select worker with best score
        return max(scores.keys(), key=lambda w: scores[w])
    
    def _calculate_selection_score(self, worker_id: str, task: ProfilingTask, stats: Dict[str, Any]) -> float:
        """Calculate selection score for resource-aware selection."""
        score = 100.0  # Base score
        
        # Penalize high active task count
        active_tasks = stats.get("active_tasks", 0)
        score -= active_tasks * 10
        
        # Penalize high response time
        avg_response_time = stats.get("avg_response_time", 0)
        if avg_response_time > 0:
            score -= min(avg_response_time * 5, 50)  # Cap penalty at 50
        
        # Penalize high error rate
        completed = stats.get("completed_tasks", 1)
        errors = stats.get("error_count", 0)
        error_rate = errors / completed
        score -= error_rate * 100
        
        # Bonus for platform specialization
        capabilities = stats.get("capabilities", {})
        specialized_platforms = capabilities.get("specialized_platforms", [])
        if task.platform in specialized_platforms:
            score += 20
        
        # Time since last assignment (load spreading)
        last_assigned = stats.get("last_assigned", datetime.now())
        time_since_last = (datetime.now() - last_assigned).total_seconds()
        score += min(time_since_last / 10, 10)  # Bonus up to 10 points
        
        return score
    
    def _calculate_load_score(self, stats: Dict[str, Any]) -> float:
        """Calculate load score for a worker."""
        active_tasks = stats.get("active_tasks", 0)
        avg_response_time = stats.get("avg_response_time", 0)
        error_count = stats.get("error_count", 0)
        completed_tasks = stats.get("completed_tasks", 1)
        
        # Higher score means higher load
        load_score = active_tasks * 10
        load_score += avg_response_time * 2
        load_score += (error_count / completed_tasks) * 50
        
        return load_score