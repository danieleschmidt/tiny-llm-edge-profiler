"""
Comprehensive tests for Self-Healing Pipeline Guard System
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.tiny_llm_profiler.pipeline_guard import (
    SelfHealingPipelineGuard, PipelineStage, HealthStatus, FailureType,
    PipelineMetrics, FailureEvent, HealingAction,
    CIPipelineMonitor, BasicAutoHealer, FailurePredictor
)
from src.tiny_llm_profiler.model_drift_detector import (
    ModelDriftMonitor, ModelPerformanceMetrics, DriftType, DriftSeverity
)
from src.tiny_llm_profiler.infrastructure_sentinel import (
    InfrastructureSentinel, ResourceType, ResourceMetrics, InfrastructureStatus
)
from src.tiny_llm_profiler.unified_guard_system import (
    UnifiedGuardSystem, SystemStatus, AlertPriority
)


class TestPipelineGuard:
    
    @pytest.fixture
    def pipeline_guard(self):
        return SelfHealingPipelineGuard()
    
    @pytest.fixture
    def mock_monitor(self):
        monitor = Mock(spec=CIPipelineMonitor)
        monitor.collect_metrics = AsyncMock()
        monitor.detect_failures = AsyncMock()
        return monitor
    
    @pytest.fixture
    def mock_healer(self):
        healer = Mock(spec=BasicAutoHealer)
        healer.can_heal = AsyncMock(return_value=True)
        healer.heal = AsyncMock(return_value=True)
        return healer
    
    def test_pipeline_guard_initialization(self, pipeline_guard):
        assert pipeline_guard.running is False
        assert len(pipeline_guard.active_failures) == 0
        assert pipeline_guard.total_failures_detected == 0
        assert pipeline_guard.total_failures_healed == 0
    
    @pytest.mark.asyncio
    async def test_monitoring_cycle_success(self, pipeline_guard, mock_monitor, mock_healer):
        # Setup
        pipeline_guard.monitor = mock_monitor
        pipeline_guard.healer = mock_healer
        
        # Mock healthy metrics
        healthy_metrics = [
            PipelineMetrics(
                stage=PipelineStage.BUILD,
                status=HealthStatus.HEALTHY,
                duration_seconds=120.0,
                memory_usage_mb=500.0,
                cpu_usage_percent=60.0,
                success_rate=0.95,
                error_count=1
            )
        ]
        
        mock_monitor.collect_metrics.return_value = healthy_metrics
        mock_monitor.detect_failures.return_value = []
        
        # Execute monitoring cycle
        await pipeline_guard._monitoring_cycle()
        
        # Assertions
        mock_monitor.collect_metrics.assert_called_once()
        mock_monitor.detect_failures.assert_called_once_with(healthy_metrics)
        assert pipeline_guard.total_failures_detected == 0
    
    @pytest.mark.asyncio
    async def test_failure_detection_and_healing(self, pipeline_guard, mock_monitor, mock_healer):
        # Setup
        pipeline_guard.monitor = mock_monitor
        pipeline_guard.healer = mock_healer
        
        # Mock failed metrics
        failed_metrics = [
            PipelineMetrics(
                stage=PipelineStage.BUILD,
                status=HealthStatus.CRITICAL,
                duration_seconds=300.0,
                memory_usage_mb=1500.0,
                cpu_usage_percent=95.0,
                success_rate=0.6,
                error_count=8
            )
        ]
        
        failure_event = FailureEvent(
            stage=PipelineStage.BUILD,
            failure_type=FailureType.PERSISTENT,
            error_message="Build stage failed with 8 errors",
            stack_trace=None,
            metrics=failed_metrics[0]
        )
        
        mock_monitor.collect_metrics.return_value = failed_metrics
        mock_monitor.detect_failures.return_value = [failure_event]
        mock_healer.can_heal.return_value = True
        mock_healer.heal.return_value = True
        
        # Execute monitoring cycle
        await pipeline_guard._monitoring_cycle()
        
        # Assertions
        assert pipeline_guard.total_failures_detected == 1
        assert pipeline_guard.total_healing_attempts == 1
        assert pipeline_guard.total_failures_healed == 1
        mock_healer.heal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_healing_failure(self, pipeline_guard, mock_monitor, mock_healer):
        # Setup
        pipeline_guard.monitor = mock_monitor
        pipeline_guard.healer = mock_healer
        
        failure_event = FailureEvent(
            stage=PipelineStage.TEST,
            failure_type=FailureType.TRANSIENT,
            error_message="Test stage timeout",
            stack_trace=None,
            metrics=Mock()
        )
        
        mock_monitor.collect_metrics.return_value = []
        mock_monitor.detect_failures.return_value = [failure_event]
        mock_healer.can_heal.return_value = True
        mock_healer.heal.return_value = False  # Healing fails
        
        # Execute
        await pipeline_guard._monitoring_cycle()
        
        # Assertions
        assert pipeline_guard.total_failures_detected == 1
        assert pipeline_guard.total_healing_attempts == 1
        assert pipeline_guard.total_failures_healed == 0
        assert len(pipeline_guard.active_failures) == 1
    
    def test_get_health_status(self, pipeline_guard):
        pipeline_guard.total_failures_detected = 10
        pipeline_guard.total_failures_healed = 8
        pipeline_guard.total_healing_attempts = 10
        
        status = pipeline_guard.get_health_status()
        
        assert status["total_failures_detected"] == 10
        assert status["total_failures_healed"] == 8
        assert status["healing_success_rate"] == 0.8
        assert "active_failure_stages" in status


class TestFailurePredictor:
    
    @pytest.fixture
    def predictor(self):
        return FailurePredictor(window_size=10)
    
    def test_initialization(self, predictor):
        assert predictor.window_size == 10
        assert len(predictor.metrics_history) == 0
        assert predictor.is_trained is False
    
    def test_add_metrics(self, predictor):
        metric = PipelineMetrics(
            stage=PipelineStage.BUILD,
            status=HealthStatus.HEALTHY,
            duration_seconds=120.0,
            memory_usage_mb=500.0,
            cpu_usage_percent=60.0,
            success_rate=0.95,
            error_count=1
        )
        
        predictor.add_metrics(metric)
        
        assert len(predictor.metrics_history) == 1
        assert predictor.metrics_history[0] == metric
    
    def test_window_size_maintenance(self, predictor):
        # Add more metrics than window size
        for i in range(15):
            metric = PipelineMetrics(
                stage=PipelineStage.BUILD,
                status=HealthStatus.HEALTHY,
                duration_seconds=120.0 + i,
                memory_usage_mb=500.0,
                cpu_usage_percent=60.0,
                success_rate=0.95,
                error_count=1
            )
            predictor.add_metrics(metric)
        
        # Should maintain window size
        assert len(predictor.metrics_history) == predictor.window_size
        assert predictor.metrics_history[-1].duration_seconds == 134.0  # Last added
    
    def test_prediction_with_insufficient_data(self, predictor):
        metric = PipelineMetrics(
            stage=PipelineStage.BUILD,
            status=HealthStatus.HEALTHY,
            duration_seconds=120.0,
            memory_usage_mb=500.0,
            cpu_usage_percent=60.0,
            success_rate=0.95,
            error_count=1
        )
        
        probability = predictor.predict_failure_probability(metric)
        assert probability == 0.0  # Not trained yet
    
    def test_model_training(self, predictor):
        # Add enough data for training
        for i in range(25):
            metric = PipelineMetrics(
                stage=PipelineStage.BUILD,
                status=HealthStatus.HEALTHY,
                duration_seconds=120.0 + i,
                memory_usage_mb=500.0 + i * 10,
                cpu_usage_percent=60.0 + i,
                success_rate=0.95 - i * 0.01,
                error_count=i // 5
            )
            predictor.add_metrics(metric)
        
        assert predictor.is_trained is True


class TestModelDriftDetector:
    
    @pytest.fixture
    def drift_monitor(self):
        return ModelDriftMonitor(window_size=10)
    
    def test_initialization(self, drift_monitor):
        assert drift_monitor.window_size == 10
        assert len(drift_monitor.metrics_history) == 0
        assert drift_monitor.baseline_metrics is None
    
    def test_add_metrics_creates_baseline(self, drift_monitor):
        metrics = ModelPerformanceMetrics(
            accuracy=0.85,
            f1_score=0.80,
            latency_ms=150.0,
            throughput_tokens_per_sec=25.0,
            memory_usage_mb=800.0,
            error_rate=0.05
        )
        
        alerts = drift_monitor.add_metrics(metrics)
        
        assert drift_monitor.baseline_metrics is not None
        assert len(alerts) == 0  # No drift alerts for first metric
    
    def test_drift_detection_accuracy_drop(self, drift_monitor):
        # Set baseline
        baseline_metrics = ModelPerformanceMetrics(
            accuracy=0.90,
            f1_score=0.85,
            latency_ms=100.0,
            throughput_tokens_per_sec=30.0,
            memory_usage_mb=600.0,
            error_rate=0.02
        )
        drift_monitor.add_metrics(baseline_metrics)
        
        # Add metrics with significant accuracy drop
        current_metrics = ModelPerformanceMetrics(
            accuracy=0.75,  # 15% drop
            f1_score=0.80,
            latency_ms=105.0,
            throughput_tokens_per_sec=28.0,
            memory_usage_mb=650.0,
            error_rate=0.03
        )
        
        alerts = drift_monitor.add_metrics(current_metrics)
        
        # Should detect accuracy drift
        accuracy_alerts = [alert for alert in alerts if alert.metric_name == "accuracy"]
        assert len(accuracy_alerts) > 0
        assert accuracy_alerts[0].severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
    
    def test_drift_summary(self, drift_monitor):
        # Add some metrics
        for i in range(5):
            metrics = ModelPerformanceMetrics(
                accuracy=0.85 - i * 0.02,
                f1_score=0.80,
                latency_ms=150.0,
                throughput_tokens_per_sec=25.0,
                memory_usage_mb=800.0,
                error_rate=0.05
            )
            drift_monitor.add_metrics(metrics)
        
        summary = drift_monitor.get_drift_summary()
        
        assert "baseline_timestamp" in summary
        assert "active_alerts_count" in summary
        assert "metrics_history_length" in summary
        assert summary["metrics_history_length"] == 5


class TestInfrastructureSentinel:
    
    @pytest.fixture
    def sentinel(self):
        return InfrastructureSentinel()
    
    def test_initialization(self, sentinel):
        assert sentinel.running is False
        assert len(sentinel.active_alerts) == 0
        assert ResourceType.CPU in sentinel.monitors
        assert ResourceType.MEMORY in sentinel.monitors
    
    @pytest.mark.asyncio
    async def test_monitoring_cycle(self, sentinel):
        # Mock one monitor to avoid actual system calls
        mock_cpu_monitor = AsyncMock()
        mock_cpu_monitor.collect_metrics.return_value = ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_usage=45.0,
            capacity=100.0,
            utilization_percent=45.0,
            status=InfrastructureStatus.HEALTHY
        )
        
        sentinel.monitors[ResourceType.CPU] = mock_cpu_monitor
        
        # Remove other monitors to avoid system dependencies
        sentinel.monitors = {ResourceType.CPU: mock_cpu_monitor}
        
        await sentinel._monitoring_cycle()
        
        mock_cpu_monitor.collect_metrics.assert_called_once()
        assert ResourceType.CPU in sentinel.current_metrics
    
    def test_infrastructure_status(self, sentinel):
        # Add some mock metrics
        sentinel.current_metrics[ResourceType.CPU] = ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_usage=85.0,
            capacity=100.0,
            utilization_percent=85.0,
            status=InfrastructureStatus.DEGRADED
        )
        
        status = sentinel.get_infrastructure_status()
        
        assert "running" in status
        assert "overall_status" in status
        assert "resource_status" in status
        assert ResourceType.CPU.value in status["resource_status"]


class TestUnifiedGuardSystem:
    
    @pytest.fixture
    def unified_guard(self):
        return UnifiedGuardSystem()
    
    def test_initialization(self, unified_guard):
        assert unified_guard.running is False
        assert len(unified_guard.system_alerts) == 0
        assert unified_guard.pipeline_guard is not None
        assert unified_guard.drift_monitor is not None
        assert unified_guard.infrastructure_sentinel is not None
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, unified_guard):
        # Mock component health status
        with patch.object(unified_guard.pipeline_guard, 'get_health_status') as mock_pipeline, \
             patch.object(unified_guard.drift_monitor, 'get_drift_summary') as mock_drift, \
             patch.object(unified_guard.infrastructure_sentinel, 'get_infrastructure_status') as mock_infra:
            
            mock_pipeline.return_value = {"running": True, "active_failures": 0}
            mock_drift.return_value = {"active_alerts_count": 0}
            mock_infra.return_value = {"overall_status": "healthy"}
            
            metrics = await unified_guard._collect_system_metrics()
            
            assert metrics.overall_status == SystemStatus.HEALTHY
            assert metrics.pipeline_health["running"] is True
    
    def test_overall_status_calculation(self, unified_guard):
        # Test critical state calculation
        pipeline_health = {"running": True, "active_failures": 2}
        drift_status = {"active_alerts_count": 1, "active_alerts_by_severity": {"critical": 1}}
        infrastructure_status = {"overall_status": "healthy"}
        
        status = unified_guard._calculate_overall_status(
            pipeline_health, drift_status, infrastructure_status
        )
        
        assert status == SystemStatus.CRITICAL
    
    def test_alert_handling(self, unified_guard):
        initial_count = len(unified_guard.system_alerts)
        
        # Create a test alert
        from src.tiny_llm_profiler.unified_guard_system import SystemAlert
        alert = SystemAlert(
            source="test",
            alert_type="test_alert",
            priority=AlertPriority.HIGH,
            message="Test alert message",
            details={"test": "data"}
        )
        
        # This would normally be called by _handle_system_alert
        unified_guard.system_alerts.append(alert)
        unified_guard.total_alerts += 1
        
        assert len(unified_guard.system_alerts) == initial_count + 1
        assert unified_guard.total_alerts > 0


class TestIntegration:
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_healing(self):
        """Test complete pipeline from failure detection to healing"""
        
        # Create pipeline guard with mocked components
        guard = SelfHealingPipelineGuard()
        
        # Mock a failure scenario
        with patch.object(guard.monitor, 'collect_metrics') as mock_collect, \
             patch.object(guard.monitor, 'detect_failures') as mock_detect, \
             patch.object(guard.healer, 'can_heal') as mock_can_heal, \
             patch.object(guard.healer, 'heal') as mock_heal:
            
            # Setup failure scenario
            failed_metric = PipelineMetrics(
                stage=PipelineStage.BUILD,
                status=HealthStatus.CRITICAL,
                duration_seconds=300.0,
                memory_usage_mb=1500.0,
                cpu_usage_percent=95.0,
                success_rate=0.5,
                error_count=10
            )
            
            failure = FailureEvent(
                stage=PipelineStage.BUILD,
                failure_type=FailureType.PERSISTENT,
                error_message="Critical build failure",
                stack_trace=None,
                metrics=failed_metric
            )
            
            mock_collect.return_value = [failed_metric]
            mock_detect.return_value = [failure]
            mock_can_heal.return_value = True
            mock_heal.return_value = True
            
            # Execute one monitoring cycle
            await guard._monitoring_cycle()
            
            # Verify end-to-end flow
            assert guard.total_failures_detected == 1
            assert guard.total_healing_attempts == 1
            assert guard.total_failures_healed == 1
            mock_heal.assert_called_once_with(failure)
    
    @pytest.mark.asyncio 
    async def test_unified_system_coordination(self):
        """Test coordination between all guard components"""
        
        unified = UnifiedGuardSystem()
        
        # Mock all component statuses for a coordinated test
        with patch.object(unified.pipeline_guard, 'get_health_status') as mock_pipeline, \
             patch.object(unified.drift_monitor, 'get_drift_summary') as mock_drift, \
             patch.object(unified.infrastructure_sentinel, 'get_infrastructure_status') as mock_infra:
            
            # Simulate a cascading failure scenario
            mock_pipeline.return_value = {
                "running": True,
                "active_failures": 3,
                "healing_success_rate": 0.6
            }
            
            mock_drift.return_value = {
                "active_alerts_count": 2,
                "active_alerts_by_severity": {"critical": 1, "high": 1}
            }
            
            mock_infra.return_value = {
                "overall_status": "critical",
                "resource_status": {
                    "cpu": {"status": "critical", "utilization": 95},
                    "memory": {"status": "degraded", "utilization": 88}
                }
            }
            
            # Collect metrics and analyze
            metrics = await unified._collect_system_metrics()
            alerts = await unified._analyze_system_health(metrics)
            
            # Should detect cascading failure
            assert metrics.overall_status in [SystemStatus.CRITICAL, SystemStatus.FAILED]
            assert len(alerts) > 0
            
            # Check for correlation alerts
            correlation_alerts = [a for a in alerts if a.source == "correlation"]
            assert len(correlation_alerts) > 0


# Performance and benchmark tests
class TestPerformanceBenchmarks:
    
    @pytest.mark.asyncio
    async def test_monitoring_cycle_performance(self):
        """Benchmark monitoring cycle performance"""
        guard = SelfHealingPipelineGuard()
        
        # Mock fast responses
        with patch.object(guard.monitor, 'collect_metrics') as mock_collect, \
             patch.object(guard.monitor, 'detect_failures') as mock_detect:
            
            mock_collect.return_value = []
            mock_detect.return_value = []
            
            # Measure performance
            start_time = time.time()
            for _ in range(10):
                await guard._monitoring_cycle()
            execution_time = time.time() - start_time
            
            # Should complete 10 cycles in under 1 second
            assert execution_time < 1.0
            assert mock_collect.call_count == 10
    
    @pytest.mark.asyncio
    async def test_prediction_performance(self):
        """Benchmark failure prediction performance"""
        predictor = FailurePredictor(window_size=50)
        
        # Add training data
        for i in range(100):
            metric = PipelineMetrics(
                stage=PipelineStage.BUILD,
                status=HealthStatus.HEALTHY,
                duration_seconds=120.0 + i,
                memory_usage_mb=500.0,
                cpu_usage_percent=60.0,
                success_rate=0.95,
                error_count=1
            )
            predictor.add_metrics(metric)
        
        # Benchmark prediction
        test_metric = PipelineMetrics(
            stage=PipelineStage.BUILD,
            status=HealthStatus.HEALTHY,
            duration_seconds=150.0,
            memory_usage_mb=600.0,
            cpu_usage_percent=70.0,
            success_rate=0.90,
            error_count=2
        )
        
        start_time = time.time()
        for _ in range(100):
            predictor.predict_failure_probability(test_metric)
        prediction_time = time.time() - start_time
        
        # Should complete 100 predictions in under 1 second
        assert prediction_time < 1.0
    
    def test_memory_usage_stability(self):
        """Test that components don't leak memory"""
        import gc
        import sys
        
        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and destroy many objects
        for _ in range(100):
            guard = SelfHealingPipelineGuard()
            predictor = FailurePredictor()
            monitor = ModelDriftMonitor()
            
            # Add some data
            for j in range(10):
                metric = PipelineMetrics(
                    stage=PipelineStage.BUILD,
                    status=HealthStatus.HEALTHY,
                    duration_seconds=120.0,
                    memory_usage_mb=500.0,
                    cpu_usage_percent=60.0,
                    success_rate=0.95,
                    error_count=1
                )
                predictor.add_metrics(metric)
            
            # Clean up references
            del guard, predictor, monitor
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should not grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Allow some growth but not excessive


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])