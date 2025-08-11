"""
Unit tests for advanced monitoring and alerting system.

Tests the metric collection, alert management, and monitoring
capabilities implemented in Generation 2.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from tiny_llm_profiler.advanced_monitoring import (
    Metric, MetricType, MetricCollector, Alert, AlertRule, AlertSeverity,
    AlertState, AlertManager, MonitoringSystem, get_monitoring_system,
    start_monitoring, stop_monitoring, record_metric, get_health_summary
)


class TestMetric:
    """Test metric data structure."""
    
    def test_metric_creation(self):
        """Test creating metric instances."""
        timestamp = datetime.now()
        metric = Metric(
            name="test_metric",
            value=42.0,
            timestamp=timestamp,
            labels={"platform": "esp32"},
            metric_type=MetricType.GAUGE
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.timestamp == timestamp
        assert metric.labels == {"platform": "esp32"}
        assert metric.metric_type == MetricType.GAUGE


class TestMetricCollector:
    """Test metric collection functionality."""
    
    @pytest.fixture
    def collector(self):
        """Fixture providing a metric collector."""
        return MetricCollector(max_retention_hours=1)
    
    def test_record_metric(self, collector):
        """Test recording metrics."""
        metric = Metric(
            name="cpu_usage",
            value=75.5,
            timestamp=datetime.now()
        )
        
        collector.record(metric)
        
        # Should be able to retrieve the metric
        metrics = collector.get_metrics("cpu_usage")
        assert len(metrics) == 1
        assert metrics[0].value == 75.5
    
    def test_record_metrics_with_labels(self, collector):
        """Test recording metrics with different labels."""
        timestamp = datetime.now()
        
        metrics_to_record = [
            Metric("memory_usage", 60.0, timestamp, {"host": "server1"}),
            Metric("memory_usage", 70.0, timestamp, {"host": "server2"}),
            Metric("memory_usage", 50.0, timestamp, {"host": "server1"})
        ]
        
        for metric in metrics_to_record:
            collector.record(metric)
        
        # Should be able to retrieve metrics by labels
        server1_metrics = collector.get_metrics("memory_usage", {"host": "server1"})
        server2_metrics = collector.get_metrics("memory_usage", {"host": "server2"})
        
        assert len(server1_metrics) == 2
        assert len(server2_metrics) == 1
        assert server1_metrics[0].labels["host"] == "server1"
        assert server2_metrics[0].labels["host"] == "server2"
    
    def test_get_latest_metric(self, collector):
        """Test retrieving the latest metric value."""
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(seconds=1)
        
        collector.record(Metric("temperature", 20.0, timestamp1))
        collector.record(Metric("temperature", 25.0, timestamp2))
        
        latest = collector.get_latest("temperature")
        assert latest is not None
        assert latest.value == 25.0
        assert latest.timestamp == timestamp2
    
    def test_get_aggregated_metrics(self, collector):
        """Test metric aggregation functions."""
        timestamp = datetime.now()
        
        # Record several data points
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for i, value in enumerate(values):
            metric_time = timestamp - timedelta(minutes=i)
            collector.record(Metric("test_metric", value, metric_time))
        
        # Test different aggregations
        avg = collector.get_aggregated("test_metric", "avg", duration_minutes=10)
        sum_val = collector.get_aggregated("test_metric", "sum", duration_minutes=10)
        min_val = collector.get_aggregated("test_metric", "min", duration_minutes=10)
        max_val = collector.get_aggregated("test_metric", "max", duration_minutes=10)
        count = collector.get_aggregated("test_metric", "count", duration_minutes=10)
        
        assert avg == 30.0  # (10+20+30+40+50) / 5
        assert sum_val == 150.0
        assert min_val == 10.0
        assert max_val == 50.0
        assert count == 5
    
    def test_time_range_filtering(self, collector):
        """Test filtering metrics by time range."""
        base_time = datetime.now()
        
        # Record metrics at different times
        collector.record(Metric("test", 1.0, base_time - timedelta(hours=2)))
        collector.record(Metric("test", 2.0, base_time - timedelta(minutes=30)))
        collector.record(Metric("test", 3.0, base_time))
        
        # Get metrics from last hour
        start_time = base_time - timedelta(hours=1)
        recent_metrics = collector.get_metrics("test", start_time=start_time)
        
        assert len(recent_metrics) == 2  # Only the last 2 metrics
        assert recent_metrics[0].value == 2.0
        assert recent_metrics[1].value == 3.0
    
    def test_metric_key_generation(self, collector):
        """Test unique key generation for metrics with labels."""
        # Test with no labels
        key1 = collector._get_metric_key("cpu_usage", {})
        assert key1 == "cpu_usage"
        
        # Test with labels
        labels = {"host": "server1", "cpu": "0"}
        key2 = collector._get_metric_key("cpu_usage", labels)
        assert key2 == "cpu_usage{cpu=0,host=server1}"  # Sorted labels
    
    def test_metric_cleanup(self):
        """Test automatic cleanup of old metrics."""
        # Create collector with very short retention
        collector = MetricCollector(max_retention_hours=0.001)  # ~3.6 seconds
        
        # Record a metric
        old_time = datetime.now() - timedelta(hours=1)  # Very old
        collector.record(Metric("old_metric", 1.0, old_time))
        
        # Trigger cleanup
        collector._cleanup_old_metrics()
        
        # Old metric should be cleaned up
        metrics = collector.get_metrics("old_metric")
        assert len(metrics) == 0
    
    def test_collector_stop(self, collector):
        """Test stopping the collector."""
        collector.stop()
        assert not collector.cleanup_running


class TestAlert:
    """Test alert data structures."""
    
    def test_alert_creation(self):
        """Test creating alert instances."""
        alert = Alert(
            id="test_alert_1",
            rule_name="high_cpu",
            metric_name="cpu_percent",
            current_value=95.0,
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            state=AlertState.ACTIVE,
            started_at=datetime.now(),
            description="CPU usage is high"
        )
        
        assert alert.id == "test_alert_1"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.state == AlertState.ACTIVE
    
    def test_alert_to_dict(self):
        """Test alert serialization to dictionary."""
        started_at = datetime.now()
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            metric_name="test_metric", 
            current_value=100.0,
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            state=AlertState.ACTIVE,
            started_at=started_at
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["id"] == "test_alert"
        assert alert_dict["severity"] == "critical"
        assert alert_dict["started_at"] == started_at.isoformat()
        assert alert_dict["resolved_at"] is None


class TestAlertRule:
    """Test alert rule definitions."""
    
    def test_alert_rule_creation(self):
        """Test creating alert rules."""
        rule = AlertRule(
            name="high_memory",
            metric_name="memory_percent",
            condition="gt",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=120.0,
            description="Memory usage is high"
        )
        
        assert rule.name == "high_memory"
        assert rule.condition == "gt"
        assert rule.threshold == 85.0
        assert rule.enabled is True


class TestAlertManager:
    """Test alert management functionality."""
    
    @pytest.fixture
    def alert_manager(self):
        """Fixture providing an alert manager with mock metric collector."""
        mock_collector = Mock()
        manager = AlertManager(mock_collector)
        # Stop the evaluation loop for testing
        manager.evaluation_running = False
        return manager
    
    def test_add_remove_rules(self, alert_manager):
        """Test adding and removing alert rules."""
        rule = AlertRule(
            name="test_rule",
            metric_name="test_metric",
            condition="gt", 
            threshold=100.0,
            severity=AlertSeverity.ERROR
        )
        
        alert_manager.add_rule(rule)
        assert "test_rule" in alert_manager.alert_rules
        
        alert_manager.remove_rule("test_rule")
        assert "test_rule" not in alert_manager.alert_rules
    
    def test_alert_handlers(self, alert_manager):
        """Test alert handler registration and notification."""
        handler_calls = []
        
        def test_handler(alert):
            handler_calls.append(alert)
        
        alert_manager.add_alert_handler(test_handler)
        
        # Simulate an alert
        alert = Alert(
            id="test",
            rule_name="test_rule",
            metric_name="test_metric",
            current_value=100.0,
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            state=AlertState.ACTIVE,
            started_at=datetime.now()
        )
        
        alert_manager._notify_handlers(alert)
        
        assert len(handler_calls) == 1
        assert handler_calls[0] == alert
    
    def test_condition_checking(self, alert_manager):
        """Test alert condition evaluation."""
        rule = AlertRule(
            name="test_rule",
            metric_name="test_metric",
            condition="gt",
            threshold=50.0,
            severity=AlertSeverity.WARNING
        )
        
        # Test different conditions
        assert alert_manager._check_condition(rule, 60.0) is True   # 60 > 50
        assert alert_manager._check_condition(rule, 40.0) is False  # 40 > 50
        
        # Test other conditions
        rule.condition = "lt"
        assert alert_manager._check_condition(rule, 40.0) is True   # 40 < 50
        assert alert_manager._check_condition(rule, 60.0) is False  # 60 < 50
        
        rule.condition = "eq"
        assert alert_manager._check_condition(rule, 50.0) is True   # 50 == 50
        assert alert_manager._check_condition(rule, 40.0) is False  # 40 == 50
    
    def test_alert_evaluation_trigger(self, alert_manager):
        """Test alert rule evaluation and triggering."""
        # Setup mock metric collector
        current_metric = Metric("cpu_percent", 95.0, datetime.now())
        alert_manager.metric_collector.get_latest.return_value = current_metric
        
        # Add rule
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_percent",
            condition="gt",
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=0.0  # No duration requirement
        )
        alert_manager.add_rule(rule)
        
        # Evaluate rules
        alert_manager._evaluate_rule(rule)
        
        # Should have created an alert
        alert_id = "high_cpu_cpu_percent"
        assert alert_id in alert_manager.active_alerts
        
        active_alert = alert_manager.active_alerts[alert_id]
        assert active_alert.current_value == 95.0
        assert active_alert.severity == AlertSeverity.WARNING
    
    def test_alert_resolution(self, alert_manager):
        """Test alert resolution when conditions are no longer met."""
        # Create an active alert
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            metric_name="cpu_percent",
            current_value=95.0,
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            state=AlertState.ACTIVE,
            started_at=datetime.now()
        )
        alert_manager.active_alerts["test_alert"] = alert
        
        # Setup mock to return value below threshold
        current_metric = Metric("cpu_percent", 80.0, datetime.now())
        alert_manager.metric_collector.get_latest.return_value = current_metric
        
        rule = AlertRule(
            name="test_rule",
            metric_name="cpu_percent",
            condition="gt",
            threshold=90.0,
            severity=AlertSeverity.WARNING
        )
        
        # Evaluate rule
        alert_manager._evaluate_rule(rule)
        
        # Alert should be resolved and removed from active alerts
        assert "test_alert" not in alert_manager.active_alerts
        
        # Should be in history with resolved state
        assert len(alert_manager.alert_history) > 0
        resolved_alert = next(a for a in alert_manager.alert_history if a.id == "test_alert")
        assert resolved_alert.state == AlertState.RESOLVED
        assert resolved_alert.resolved_at is not None
    
    def test_alert_silencing(self, alert_manager):
        """Test alert silencing functionality."""
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            metric_name="test_metric",
            current_value=100.0,
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            state=AlertState.ACTIVE,
            started_at=datetime.now()
        )
        alert_manager.active_alerts["test_alert"] = alert
        
        # Silence the alert
        alert_manager.silence_alert("test_alert", duration_minutes=1)
        
        # Alert should be silenced
        assert alert_manager.active_alerts["test_alert"].state == AlertState.SILENCED
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts with filtering."""
        # Add alerts with different severities
        alerts = [
            Alert("alert1", "rule1", "metric1", 100, 90, AlertSeverity.WARNING, AlertState.ACTIVE, datetime.now()),
            Alert("alert2", "rule2", "metric2", 200, 180, AlertSeverity.CRITICAL, AlertState.ACTIVE, datetime.now()),
            Alert("alert3", "rule3", "metric3", 50, 40, AlertSeverity.INFO, AlertState.ACTIVE, datetime.now()),
        ]
        
        for alert in alerts:
            alert_manager.active_alerts[alert.id] = alert
        
        # Test getting all active alerts
        all_alerts = alert_manager.get_active_alerts()
        assert len(all_alerts) == 3
        
        # Test filtering by severity
        critical_alerts = alert_manager.get_active_alerts(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_alert_history(self, alert_manager):
        """Test alert history functionality."""
        # Add some historical alerts
        now = datetime.now()
        old_alert = Alert("old", "rule", "metric", 100, 90, AlertSeverity.WARNING, AlertState.RESOLVED, now - timedelta(hours=25))
        recent_alert = Alert("recent", "rule", "metric", 100, 90, AlertSeverity.ERROR, AlertState.RESOLVED, now - timedelta(hours=1))
        
        alert_manager.alert_history = [old_alert, recent_alert]
        
        # Get last 24 hours
        recent_history = alert_manager.get_alert_history(hours=24)
        
        assert len(recent_history) == 1
        assert recent_history[0].id == "recent"


class TestMonitoringSystem:
    """Test the comprehensive monitoring system."""
    
    @pytest.fixture
    def monitoring_system(self):
        """Fixture providing a monitoring system."""
        return MonitoringSystem()
    
    def test_monitoring_system_initialization(self, monitoring_system):
        """Test monitoring system initialization."""
        assert monitoring_system.metric_collector is not None
        assert monitoring_system.alert_manager is not None
        assert monitoring_system.monitoring_active is False
    
    def test_record_metric(self, monitoring_system):
        """Test recording metrics through the monitoring system."""
        monitoring_system.record_metric("test_metric", 42.0, {"source": "test"})
        
        metrics = monitoring_system.get_metrics("test_metric")
        assert len(metrics) == 1
        assert metrics[0].value == 42.0
        assert metrics[0].labels == {"source": "test"}
    
    def test_health_summary(self, monitoring_system):
        """Test getting health summary."""
        # Add some test metrics
        monitoring_system.record_metric("system_memory_percent", 75.0)
        monitoring_system.record_metric("system_cpu_percent", 60.0)
        
        health_summary = monitoring_system.get_health_summary()
        
        assert "timestamp" in health_summary
        assert "overall_health" in health_summary
        assert "active_alerts" in health_summary
        assert "system_metrics" in health_summary
        assert health_summary["monitoring_active"] is False
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_system_monitoring_loop(self, mock_cpu, mock_memory, monitoring_system):
        """Test the system monitoring loop."""
        # Mock system metrics
        mock_memory.return_value = Mock(percent=80.0, available=1024*1024*1024)
        mock_cpu.return_value = 60.0
        
        # Start monitoring briefly
        monitoring_system.start()
        time.sleep(0.1)  # Let it collect some data
        monitoring_system.stop()
        
        # Check that system metrics were collected
        memory_metrics = monitoring_system.get_metrics("system_memory_percent")
        cpu_metrics = monitoring_system.get_metrics("system_cpu_percent")
        
        assert len(memory_metrics) > 0
        assert len(cpu_metrics) > 0
    
    def test_export_metrics(self, monitoring_system):
        """Test exporting metrics to file."""
        # Record some test metrics
        monitoring_system.record_metric("export_test", 123.45)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            monitoring_system.export_metrics(output_path)
            
            # Verify exported data
            with open(output_path, 'r') as f:
                exported_data = json.load(f)
            
            assert "export_test" in exported_data
            assert len(exported_data["export_test"]) == 1
            assert exported_data["export_test"][0]["value"] == 123.45
        
        finally:
            output_path.unlink()  # Cleanup


class TestGlobalMonitoringFunctions:
    """Test global monitoring convenience functions."""
    
    def test_get_monitoring_system(self):
        """Test getting the global monitoring system instance."""
        system1 = get_monitoring_system()
        system2 = get_monitoring_system()
        
        # Should return the same instance
        assert system1 is system2
    
    def test_record_metric_global(self):
        """Test recording metrics through global function."""
        record_metric("global_test_metric", 99.9, {"test": "global"})
        
        # Should be recorded in the global system
        system = get_monitoring_system()
        metrics = system.get_metrics("global_test_metric")
        
        assert len(metrics) >= 1
        assert any(m.value == 99.9 for m in metrics)
    
    def test_health_summary_global(self):
        """Test getting health summary through global function."""
        summary = get_health_summary()
        
        assert "timestamp" in summary
        assert "overall_health" in summary


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for the monitoring system."""
    
    def test_end_to_end_monitoring(self):
        """Test complete monitoring workflow."""
        system = MonitoringSystem()
        
        # Start monitoring
        system.start()
        
        try:
            # Record some metrics that should trigger alerts
            system.record_metric("system_memory_percent", 90.0)  # Should trigger warning
            system.record_metric("system_memory_percent", 97.0)  # Should trigger critical
            
            # Give some time for alert evaluation
            time.sleep(1.0)
            
            # Check that alerts were triggered
            active_alerts = system.alert_manager.get_active_alerts()
            
            # Should have alerts for high memory usage
            memory_alerts = [a for a in active_alerts if "memory" in a.metric_name]
            assert len(memory_alerts) > 0
            
            # Record lower memory to resolve alerts
            system.record_metric("system_memory_percent", 70.0)
            time.sleep(1.0)
            
            # Check health summary
            health = system.get_health_summary()
            assert health["overall_health"] in ["healthy", "warning", "critical"]
        
        finally:
            system.stop()
    
    def test_alert_handler_integration(self):
        """Test custom alert handlers."""
        system = MonitoringSystem()
        
        alerts_received = []
        
        def custom_handler(alert):
            alerts_received.append(alert)
        
        system.alert_manager.add_alert_handler(custom_handler)
        
        # Add a rule that will trigger immediately
        rule = AlertRule(
            name="test_integration",
            metric_name="integration_test",
            condition="gt",
            threshold=50.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=0.0
        )
        system.alert_manager.add_rule(rule)
        
        system.start()
        
        try:
            # Record metric that should trigger alert
            system.record_metric("integration_test", 75.0)
            
            # Give time for processing
            time.sleep(1.0)
            
            # Check that handler was called
            assert len(alerts_received) > 0
            assert any(alert.rule_name == "test_integration" for alert in alerts_received)
        
        finally:
            system.stop()


if __name__ == "__main__":
    pytest.main([__file__])