"""
Unit tests for the auto_scaling module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import threading
import time

from tiny_llm_profiler.auto_scaling import (
    AutoScaler,
    LoadBalancer,
    ScalingConfig,
    ScalingMetrics,
    ScalingDecision,
    LoadLevel
)
from tiny_llm_profiler.concurrent import ProfilingTask


class TestScalingConfig:
    """Test the ScalingConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = ScalingConfig()
        
        assert config.scale_up_queue_threshold == 10
        assert config.scale_down_queue_threshold == 2
        assert config.cpu_scale_up_threshold == 80.0
        assert config.cpu_scale_down_threshold == 30.0
        assert config.min_workers == 2
        assert config.max_workers == 20
        assert config.cooldown_period_seconds == 60
    
    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = ScalingConfig(
            min_workers=4,
            max_workers=50,
            scale_up_queue_threshold=20,
            cooldown_period_seconds=120
        )
        
        assert config.min_workers == 4
        assert config.max_workers == 50
        assert config.scale_up_queue_threshold == 20
        assert config.cooldown_period_seconds == 120


class TestScalingMetrics:
    """Test the ScalingMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test metrics creation with all fields."""
        timestamp = datetime.now()
        metrics = ScalingMetrics(
            timestamp=timestamp,
            queue_length=5,
            active_tasks=3,
            cpu_utilization=65.0,
            memory_utilization=70.0,
            avg_response_time=2.5,
            error_rate=0.05,
            throughput=10.0,
            load_level=LoadLevel.NORMAL
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.queue_length == 5
        assert metrics.active_tasks == 3
        assert metrics.cpu_utilization == 65.0
        assert metrics.load_level == LoadLevel.NORMAL


class TestAutoScaler:
    """Test the AutoScaler class."""
    
    def test_initialization_default_config(self):
        """Test auto scaler initialization with default config."""
        scaler = AutoScaler()
        
        assert scaler.config.min_workers == 2
        assert scaler.config.max_workers == 20
        assert scaler.current_workers == 2
        assert scaler.enable_predictive_scaling is True
        assert scaler.is_monitoring is False
    
    def test_initialization_custom_config(self):
        """Test auto scaler initialization with custom config."""
        config = ScalingConfig(min_workers=5, max_workers=15)
        scaler = AutoScaler(config=config, enable_predictive_scaling=False)
        
        assert scaler.config.min_workers == 5
        assert scaler.config.max_workers == 15
        assert scaler.current_workers == 5
        assert scaler.enable_predictive_scaling is False
    
    @patch('tiny_llm_profiler.auto_scaling.psutil.cpu_percent')
    @patch('tiny_llm_profiler.auto_scaling.psutil.virtual_memory')
    def test_collect_metrics(self, mock_memory, mock_cpu):
        """Test metrics collection."""
        mock_cpu.return_value = 75.0
        mock_memory.return_value = Mock(percent=65.0)
        
        scaler = AutoScaler()
        scaler.collect_metrics(
            queue_length=8,
            active_tasks=5,
            avg_response_time=3.0,
            error_rate=0.1,
            throughput=12.0
        )
        
        assert len(scaler.metrics_history) == 1
        metrics = scaler.metrics_history[0]
        
        assert metrics.queue_length == 8
        assert metrics.active_tasks == 5
        assert metrics.cpu_utilization == 75.0
        assert metrics.memory_utilization == 65.0
        assert metrics.avg_response_time == 3.0
        assert metrics.error_rate == 0.1
        assert metrics.throughput == 12.0
    
    def test_calculate_load_level_low(self):
        """Test load level calculation for low load."""
        scaler = AutoScaler()
        
        load_level = scaler._calculate_load_level(
            queue_length=1,
            cpu_util=20.0,
            memory_util=30.0,
            response_time=0.5,
            error_rate=0.01
        )
        
        assert load_level == LoadLevel.LOW
    
    def test_calculate_load_level_normal(self):
        """Test load level calculation for normal load."""
        scaler = AutoScaler()
        
        load_level = scaler._calculate_load_level(
            queue_length=6,
            cpu_util=50.0,
            memory_util=60.0,
            response_time=2.0,
            error_rate=0.05
        )
        
        assert load_level == LoadLevel.NORMAL
    
    def test_calculate_load_level_high(self):
        """Test load level calculation for high load."""
        scaler = AutoScaler()
        
        load_level = scaler._calculate_load_level(
            queue_length=15,
            cpu_util=85.0,
            memory_util=80.0,
            response_time=4.0,
            error_rate=0.08
        )
        
        assert load_level == LoadLevel.HIGH
    
    def test_calculate_load_level_critical(self):
        """Test load level calculation for critical load."""
        scaler = AutoScaler()
        
        load_level = scaler._calculate_load_level(
            queue_length=60,  # Above emergency threshold
            cpu_util=98.0,    # Above emergency threshold
            memory_util=95.0, # Above critical memory threshold
            response_time=12.0,
            error_rate=0.3
        )
        
        assert load_level == LoadLevel.CRITICAL
    
    def test_make_scaling_decision_no_metrics(self):
        """Test scaling decision with no metrics."""
        scaler = AutoScaler()
        
        decision, target_workers = scaler.make_scaling_decision()
        
        assert decision == ScalingDecision.MAINTAIN
        assert target_workers == scaler.current_workers
    
    def test_make_scaling_decision_in_cooldown(self):
        """Test scaling decision during cooldown period."""
        scaler = AutoScaler()
        scaler.last_scaling_decision = datetime.now()  # Recent scaling
        
        # Add some metrics
        scaler.metrics_history.append(ScalingMetrics(
            timestamp=datetime.now(),
            queue_length=20,  # Should trigger scale up
            active_tasks=10,
            cpu_utilization=90.0,
            memory_utilization=85.0,
            avg_response_time=5.0,
            error_rate=0.1,
            throughput=5.0
        ))
        
        decision, target_workers = scaler.make_scaling_decision()
        
        assert decision == ScalingDecision.MAINTAIN
        assert target_workers == scaler.current_workers
    
    def test_make_scaling_decision_emergency_scaling(self):
        """Test emergency scaling decision."""
        scaler = AutoScaler()
        scaler.last_scaling_decision = datetime.now() - timedelta(minutes=5)  # Out of cooldown
        
        # Add emergency metrics
        scaler.metrics_history.append(ScalingMetrics(
            timestamp=datetime.now(),
            queue_length=60,  # Above emergency threshold
            active_tasks=50,
            cpu_utilization=98.0,  # Above emergency threshold
            memory_utilization=95.0,
            avg_response_time=10.0,
            error_rate=0.2,
            throughput=2.0
        ))
        
        decision, target_workers = scaler.make_scaling_decision()
        
        assert decision == ScalingDecision.EMERGENCY_SCALE
        assert target_workers > scaler.current_workers
    
    def test_make_scaling_decision_scale_up(self):
        """Test scale up decision."""
        scaler = AutoScaler()
        scaler.current_workers = 5
        scaler.last_scaling_decision = datetime.now() - timedelta(minutes=5)  # Out of cooldown
        
        # Add multiple metrics indicating high load
        for i in range(3):
            scaler.metrics_history.append(ScalingMetrics(
                timestamp=datetime.now() - timedelta(seconds=i*10),
                queue_length=15,  # Above scale up threshold
                active_tasks=12,
                cpu_utilization=85.0,  # Above scale up threshold
                memory_utilization=80.0,
                avg_response_time=6.0,  # Above threshold
                error_rate=0.08,
                throughput=8.0
            ))
        
        decision, target_workers = scaler.make_scaling_decision()
        
        assert decision == ScalingDecision.SCALE_UP
        assert target_workers > scaler.current_workers
    
    def test_make_scaling_decision_scale_down(self):
        """Test scale down decision."""
        scaler = AutoScaler()
        scaler.current_workers = 10
        scaler.last_scaling_decision = datetime.now() - timedelta(minutes=5)  # Out of cooldown
        
        # Add multiple metrics indicating low load
        for i in range(3):
            scaler.metrics_history.append(ScalingMetrics(
                timestamp=datetime.now() - timedelta(seconds=i*10),
                queue_length=1,   # Below scale down threshold
                active_tasks=2,
                cpu_utilization=25.0,  # Below scale down threshold
                memory_utilization=40.0,
                avg_response_time=1.0,
                error_rate=0.01,
                throughput=15.0
            ))
        
        decision, target_workers = scaler.make_scaling_decision()
        
        assert decision == ScalingDecision.SCALE_DOWN
        assert target_workers < scaler.current_workers
        assert target_workers >= scaler.config.min_workers
    
    def test_execute_scaling_decision_maintain(self):
        """Test executing maintain decision."""
        scaler = AutoScaler()
        original_workers = scaler.current_workers
        
        result = scaler.execute_scaling_decision(ScalingDecision.MAINTAIN, original_workers)
        
        assert result is True
        assert scaler.current_workers == original_workers
        assert len(scaler.scaling_history) == 0
    
    def test_execute_scaling_decision_scale_up(self):
        """Test executing scale up decision."""
        scaler = AutoScaler()
        original_workers = scaler.current_workers
        target_workers = original_workers + 2
        
        result = scaler.execute_scaling_decision(ScalingDecision.SCALE_UP, target_workers)
        
        assert result is True
        assert scaler.current_workers == target_workers
        assert len(scaler.scaling_history) == 1
        
        # Check history entry
        timestamp, decision, workers = scaler.scaling_history[0]
        assert decision == ScalingDecision.SCALE_UP
        assert workers == target_workers
    
    def test_add_scaling_callback(self):
        """Test adding scaling callback."""
        scaler = AutoScaler()
        callback_called = []
        
        def test_callback(decision, old_workers, new_workers):
            callback_called.append((decision, old_workers, new_workers))
        
        scaler.add_scaling_callback(test_callback)
        scaler.execute_scaling_decision(ScalingDecision.SCALE_UP, 5)
        
        assert len(callback_called) == 1
        decision, old_workers, new_workers = callback_called[0]
        assert decision == ScalingDecision.SCALE_UP
        assert old_workers == 2  # Default min_workers
        assert new_workers == 5
    
    def test_get_scaling_stats_no_metrics(self):
        """Test getting scaling stats with no metrics."""
        scaler = AutoScaler()
        
        stats = scaler.get_scaling_stats()
        
        assert stats["current_workers"] == 2
        assert stats["target_range"]["min"] == 2
        assert stats["target_range"]["max"] == 20
        assert stats["current_metrics"]["load_level"] == "unknown"
        assert stats["scaling_activity"]["total_scaling_events"] == 0
    
    def test_get_scaling_stats_with_metrics(self):
        """Test getting scaling stats with metrics."""
        scaler = AutoScaler()
        
        # Add metrics
        scaler.metrics_history.append(ScalingMetrics(
            timestamp=datetime.now(),
            queue_length=8,
            active_tasks=5,
            cpu_utilization=70.0,
            memory_utilization=65.0,
            avg_response_time=3.0,
            error_rate=0.05,
            throughput=12.0,
            load_level=LoadLevel.NORMAL
        ))
        
        # Add scaling history
        scaler.scaling_history.append((datetime.now(), ScalingDecision.SCALE_UP, 4))
        
        stats = scaler.get_scaling_stats()
        
        assert stats["current_workers"] == 2
        assert stats["current_metrics"]["load_level"] == "normal"
        assert stats["current_metrics"]["queue_length"] == 8
        assert stats["scaling_activity"]["total_scaling_events"] == 1
    
    def test_calculate_trend_simple(self):
        """Test trend calculation with simple data."""
        scaler = AutoScaler()
        
        # Upward trend
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = scaler._calculate_trend(values)
        assert trend > 0  # Should be positive trend
        
        # Downward trend
        values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = scaler._calculate_trend(values)
        assert trend < 0  # Should be negative trend
        
        # Flat trend
        values = [3.0, 3.0, 3.0, 3.0, 3.0]
        trend = scaler._calculate_trend(values)
        assert abs(trend) < 0.1  # Should be near zero
    
    def test_calculate_trend_edge_cases(self):
        """Test trend calculation edge cases."""
        scaler = AutoScaler()
        
        # Empty list
        trend = scaler._calculate_trend([])
        assert trend == 0.0
        
        # Single value
        trend = scaler._calculate_trend([5.0])
        assert trend == 0.0
    
    @patch('tiny_llm_profiler.auto_scaling.psutil.cpu_percent')
    @patch('tiny_llm_profiler.auto_scaling.psutil.virtual_memory')
    def test_start_stop_monitoring(self, mock_memory, mock_cpu):
        """Test starting and stopping monitoring."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        
        scaler = AutoScaler()
        
        # Start monitoring
        scaler.start_monitoring()
        assert scaler.is_monitoring is True
        assert scaler.monitor_thread is not None
        
        # Stop monitoring
        scaler.stop_monitoring()
        assert scaler.is_monitoring is False


class TestLoadBalancer:
    """Test the LoadBalancer class."""
    
    def test_initialization(self):
        """Test load balancer initialization."""
        balancer = LoadBalancer()
        
        assert balancer.auto_scaler is None
        assert len(balancer.worker_stats) == 0
        assert len(balancer.task_assignments) == 0
        assert balancer.current_strategy == "resource_aware"
    
    def test_initialization_with_auto_scaler(self):
        """Test load balancer initialization with auto scaler."""
        auto_scaler = AutoScaler()
        balancer = LoadBalancer(auto_scaler=auto_scaler)
        
        assert balancer.auto_scaler is auto_scaler
    
    def test_register_worker(self):
        """Test registering a worker."""
        balancer = LoadBalancer()
        
        capabilities = {"specialized_platforms": ["esp32", "stm32f4"]}
        balancer.register_worker("worker1", capabilities)
        
        assert "worker1" in balancer.worker_stats
        stats = balancer.worker_stats["worker1"]
        assert stats["active_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["capabilities"] == capabilities
    
    def test_remove_worker(self):
        """Test removing a worker."""
        balancer = LoadBalancer()
        
        # Register worker
        balancer.register_worker("worker1")
        
        # Assign a task
        task = ProfilingTask(
            task_id="task1",
            model_path="test_model",
            platform="esp32"
        )
        balancer.task_assignments["task1"] = "worker1"
        
        # Remove worker
        balancer.remove_worker("worker1")
        
        assert "worker1" not in balancer.worker_stats
        assert "task1" not in balancer.task_assignments
    
    def test_assign_task_no_workers(self):
        """Test task assignment with no workers."""
        balancer = LoadBalancer()
        
        task = ProfilingTask(
            task_id="task1",
            model_path="test_model",
            platform="esp32"
        )
        
        assigned_worker = balancer.assign_task(task)
        assert assigned_worker is None
    
    def test_assign_task_round_robin(self):
        """Test round-robin task assignment."""
        balancer = LoadBalancer()
        balancer.set_strategy("round_robin")
        
        # Register workers
        balancer.register_worker("worker1")
        balancer.register_worker("worker2")
        balancer.register_worker("worker3")
        
        # Assign tasks
        task1 = ProfilingTask(task_id="task1", model_path="model", platform="esp32")
        task2 = ProfilingTask(task_id="task2", model_path="model", platform="esp32")
        task3 = ProfilingTask(task_id="task3", model_path="model", platform="esp32")
        task4 = ProfilingTask(task_id="task4", model_path="model", platform="esp32")
        
        worker1 = balancer.assign_task(task1)
        worker2 = balancer.assign_task(task2)
        worker3 = balancer.assign_task(task3)
        worker4 = balancer.assign_task(task4)
        
        # Should cycle through workers
        assert worker1 != worker2
        assert worker2 != worker3
        assert worker1 == worker4  # Should cycle back
    
    def test_assign_task_least_connections(self):
        """Test least connections task assignment."""
        balancer = LoadBalancer()
        balancer.set_strategy("least_connections")
        
        # Register workers with different loads
        balancer.register_worker("worker1")
        balancer.register_worker("worker2")
        balancer.worker_stats["worker1"]["active_tasks"] = 5
        balancer.worker_stats["worker2"]["active_tasks"] = 2
        
        task = ProfilingTask(task_id="task1", model_path="model", platform="esp32")
        assigned_worker = balancer.assign_task(task)
        
        # Should assign to worker with fewer connections
        assert assigned_worker == "worker2"
    
    def test_complete_task_success(self):
        """Test successful task completion."""
        balancer = LoadBalancer()
        
        # Register worker and assign task
        balancer.register_worker("worker1")
        task = ProfilingTask(task_id="task1", model_path="model", platform="esp32")
        balancer.assign_task(task)
        
        # Complete task
        balancer.complete_task("task1", success=True, response_time=2.5)
        
        stats = balancer.worker_stats["worker1"]
        assert stats["active_tasks"] == 0
        assert stats["completed_tasks"] == 1
        assert stats["total_response_time"] == 2.5
        assert stats["avg_response_time"] == 2.5
        assert stats["error_count"] == 0
        assert "task1" not in balancer.task_assignments
    
    def test_complete_task_failure(self):
        """Test failed task completion."""
        balancer = LoadBalancer()
        
        # Register worker and assign task
        balancer.register_worker("worker1")
        task = ProfilingTask(task_id="task1", model_path="model", platform="esp32")
        balancer.assign_task(task)
        
        # Complete task with failure
        balancer.complete_task("task1", success=False, response_time=1.0)
        
        stats = balancer.worker_stats["worker1"]
        assert stats["active_tasks"] == 0
        assert stats["completed_tasks"] == 1
        assert stats["error_count"] == 1
    
    def test_get_load_balancing_stats(self):
        """Test getting load balancing statistics."""
        balancer = LoadBalancer()
        
        # Register workers with different stats
        balancer.register_worker("worker1")
        balancer.register_worker("worker2")
        
        balancer.worker_stats["worker1"]["active_tasks"] = 3
        balancer.worker_stats["worker1"]["completed_tasks"] = 10
        balancer.worker_stats["worker1"]["error_count"] = 1
        
        balancer.worker_stats["worker2"]["active_tasks"] = 2
        balancer.worker_stats["worker2"]["completed_tasks"] = 15
        balancer.worker_stats["worker2"]["error_count"] = 0
        
        stats = balancer.get_load_balancing_stats()
        
        assert stats["active_workers"] == 2
        assert stats["total_active_tasks"] == 5
        assert stats["total_completed_tasks"] == 25
        assert stats["total_errors"] == 1
        assert stats["error_rate"] == 1/25
    
    def test_set_strategy_valid(self):
        """Test setting valid load balancing strategy."""
        balancer = LoadBalancer()
        
        balancer.set_strategy("least_connections")
        assert balancer.current_strategy == "least_connections"
    
    def test_set_strategy_invalid(self):
        """Test setting invalid load balancing strategy."""
        balancer = LoadBalancer()
        
        with pytest.raises(ValueError):
            balancer.set_strategy("invalid_strategy")
    
    def test_resource_aware_selection(self):
        """Test resource-aware worker selection."""
        balancer = LoadBalancer()
        balancer.set_strategy("resource_aware")
        
        # Register workers with different characteristics
        balancer.register_worker("worker1", {"specialized_platforms": ["esp32"]})
        balancer.register_worker("worker2", {"specialized_platforms": ["stm32f4"]})
        
        # Set different performance stats
        balancer.worker_stats["worker1"]["avg_response_time"] = 1.0
        balancer.worker_stats["worker1"]["active_tasks"] = 2
        balancer.worker_stats["worker1"]["error_count"] = 0
        balancer.worker_stats["worker1"]["completed_tasks"] = 10
        
        balancer.worker_stats["worker2"]["avg_response_time"] = 3.0
        balancer.worker_stats["worker2"]["active_tasks"] = 5
        balancer.worker_stats["worker2"]["error_count"] = 2
        balancer.worker_stats["worker2"]["completed_tasks"] = 8
        
        # Task for ESP32 should prefer worker1 (specialized + better performance)
        task = ProfilingTask(task_id="task1", model_path="model", platform="esp32")
        assigned_worker = balancer.assign_task(task)
        
        assert assigned_worker == "worker1"
    
    def test_calculate_selection_score(self):
        """Test selection score calculation."""
        balancer = LoadBalancer()
        
        task = ProfilingTask(task_id="task1", model_path="model", platform="esp32")
        
        # Good worker stats
        good_stats = {
            "active_tasks": 1,
            "avg_response_time": 1.0,
            "error_count": 0,
            "completed_tasks": 10,
            "capabilities": {"specialized_platforms": ["esp32"]},
            "last_assigned": datetime.now() - timedelta(minutes=5)
        }
        
        # Poor worker stats
        poor_stats = {
            "active_tasks": 10,
            "avg_response_time": 5.0,
            "error_count": 5,
            "completed_tasks": 10,
            "capabilities": {"specialized_platforms": ["stm32f4"]},
            "last_assigned": datetime.now()
        }
        
        good_score = balancer._calculate_selection_score("worker1", task, good_stats)
        poor_score = balancer._calculate_selection_score("worker2", task, poor_stats)
        
        # Good worker should have higher score
        assert good_score > poor_score
    
    def test_calculate_load_score(self):
        """Test load score calculation."""
        balancer = LoadBalancer()
        
        # High load stats
        high_load_stats = {
            "active_tasks": 10,
            "avg_response_time": 5.0,
            "error_count": 5,
            "completed_tasks": 20
        }
        
        # Low load stats
        low_load_stats = {
            "active_tasks": 2,
            "avg_response_time": 1.0,
            "error_count": 0,
            "completed_tasks": 20
        }
        
        high_load_score = balancer._calculate_load_score(high_load_stats)
        low_load_score = balancer._calculate_load_score(low_load_stats)
        
        # High load should have higher score
        assert high_load_score > low_load_score


if __name__ == "__main__":
    pytest.main([__file__])