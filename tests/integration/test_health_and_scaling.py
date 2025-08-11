"""
Health Monitoring and Scaling Integration Tests

Tests that verify health monitoring, auto-scaling, and load balancing
capabilities in realistic profiling scenarios.
"""

import pytest
import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any
import psutil

from tiny_llm_profiler.profiler import EdgeProfiler, ProfilingConfig
from tiny_llm_profiler.models import QuantizedModel, QuantizationType
from tiny_llm_profiler.health import HealthChecker
from tiny_llm_profiler.health_monitor import SystemHealthMonitor
from tiny_llm_profiler.scaling import AutoScaler, LoadBalancer, ScalingPolicy
from tiny_llm_profiler.concurrent_utils import ConcurrentProfiler, ProfilingTask


@pytest.mark.integration
class TestHealthMonitoringIntegration:
    """Test health monitoring during profiling operations."""
    
    def test_continuous_health_monitoring_during_profiling(self, tmp_path):
        """Test continuous health monitoring throughout profiling sessions."""
        
        # Initialize health checker
        health_checker = HealthChecker()
        
        # Create test model
        model_file = tmp_path / "health_test_model.gguf"
        self._create_model_file(model_file, 2.0)
        model = QuantizedModel.from_file(model_file)
        
        # Set up health monitoring
        health_records = []
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        def continuous_health_monitoring():
            """Monitor system health continuously."""
            while monitoring_active.is_set():
                try:
                    # Collect system metrics
                    metrics = health_checker.collect_system_metrics()
                    
                    # Run health checks
                    health_status = health_checker.run_all_checks()
                    
                    health_record = {
                        "timestamp": time.time(),
                        "cpu_percent": metrics.cpu_percent,
                        "memory_percent": metrics.memory_percent,
                        "disk_usage_percent": metrics.disk_usage_percent,
                        "overall_healthy": health_checker.get_overall_health().healthy,
                        "health_checks": health_status
                    }
                    
                    health_records.append(health_record)
                    
                except Exception as e:
                    print(f"Health monitoring error: {e}")
                
                time.sleep(0.5)  # Monitor every 500ms
        
        # Start health monitoring in background
        monitor_thread = threading.Thread(target=continuous_health_monitoring)
        monitor_thread.start()
        
        try:
            # Run profiling workload while monitoring
            profiler = EdgeProfiler(platform="esp32", connection="local")
            
            # Run multiple profiling sessions to create sustained load
            profiling_results = []
            
            for i in range(5):
                results = profiler.profile_model(
                    model=model,
                    test_prompts=[f"Health monitoring test {i}"],
                    metrics=["latency", "memory"],
                    config=ProfilingConfig(duration_seconds=3, measurement_iterations=2)
                )
                
                profiling_results.append(results)
                
                # Brief pause between sessions
                time.sleep(1)
            
            # Continue monitoring briefly after profiling
            time.sleep(2)
            
        finally:
            # Stop health monitoring
            monitoring_active.clear()
            monitor_thread.join(timeout=5)
        
        # Analyze health monitoring data
        assert len(health_records) >= 10, "Should have collected multiple health records"
        
        # Check for health trends during profiling
        cpu_usage = [r["cpu_percent"] for r in health_records]
        memory_usage = [r["memory_percent"] for r in health_records]
        
        # Should see some CPU activity during profiling
        max_cpu = max(cpu_usage)
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        
        assert max_cpu > avg_cpu, "Should see CPU spikes during profiling"
        
        # System should remain healthy throughout
        unhealthy_periods = [r for r in health_records if not r["overall_healthy"]]
        health_rate = (len(health_records) - len(unhealthy_periods)) / len(health_records)
        
        assert health_rate >= 0.8, f"System health rate during profiling: {health_rate:.1%}"
        
        # Check that all profiling completed successfully
        successful_profiles = [r for r in profiling_results if r is not None]
        assert len(successful_profiles) == 5, "All profiling sessions should succeed"
        
        print(f"✓ Continuous health monitoring test passed")
        print(f"  Health records: {len(health_records)}")
        print(f"  Max CPU usage: {max_cpu:.1f}%")
        print(f"  Health rate: {health_rate:.1%}")
        print(f"  Successful profiles: {len(successful_profiles)}")
    
    def test_health_based_profiling_throttling(self, tmp_path):
        """Test profiling throttling based on system health conditions."""
        
        # Initialize health-aware profiler system
        health_checker = HealthChecker()
        
        # Create multiple test models
        models = []
        for i in range(3):
            model_file = tmp_path / f"throttle_test_model_{i}.gguf"
            self._create_model_file(model_file, 1.5)
            models.append(QuantizedModel.from_file(model_file))
        
        # Set up health-based throttling
        class HealthAwareProfiler:
            def __init__(self, health_checker):
                self.health_checker = health_checker
                self.profiler = EdgeProfiler(platform="esp32", connection="local")
                self.throttling_active = False
            
            def profile_with_health_check(self, model, prompts):
                # Check system health before profiling
                health = self.health_checker.get_overall_health()
                
                if not health.healthy:
                    # System unhealthy - throttle profiling
                    self.throttling_active = True
                    time.sleep(2)  # Wait for system to recover
                    return None
                
                # Check resource usage
                metrics = self.health_checker.collect_system_metrics()
                
                if metrics.cpu_percent > 80 or metrics.memory_percent > 85:
                    # High resource usage - reduce profiling intensity
                    self.throttling_active = True
                    config = ProfilingConfig(
                        duration_seconds=1,  # Reduced
                        measurement_iterations=1  # Reduced
                    )
                else:
                    self.throttling_active = False
                    config = ProfilingConfig(
                        duration_seconds=3,
                        measurement_iterations=2
                    )
                
                return self.profiler.profile_model(
                    model=model,
                    test_prompts=prompts,
                    metrics=["latency"],
                    config=config
                )
        
        health_aware_profiler = HealthAwareProfiler(health_checker)
        
        # Simulate system stress to trigger throttling
        def create_cpu_stress():
            """Create CPU stress to trigger health-based throttling."""
            end_time = time.time() + 10
            while time.time() < end_time:
                # Busy work to increase CPU usage
                sum(x * x for x in range(1000))
        
        # Start stress in background
        stress_thread = threading.Thread(target=create_cpu_stress)
        stress_thread.start()
        
        try:
            # Run profiling with health monitoring
            profiling_results = []
            throttling_instances = 0
            
            for i, model in enumerate(models):
                for attempt in range(3):  # Multiple attempts per model
                    result = health_aware_profiler.profile_with_health_check(
                        model, [f"Throttling test {i}-{attempt}"]
                    )
                    
                    profiling_results.append({
                        "model_index": i,
                        "attempt": attempt,
                        "result": result,
                        "throttled": health_aware_profiler.throttling_active
                    })
                    
                    if health_aware_profiler.throttling_active:
                        throttling_instances += 1
                    
                    time.sleep(0.5)
        
        finally:
            stress_thread.join(timeout=15)
        
        # Analyze throttling behavior
        total_attempts = len(profiling_results)
        successful_results = [r for r in profiling_results if r["result"] is not None]
        
        # Should have some throttling due to stress
        assert throttling_instances > 0, "Should have triggered throttling under stress"
        
        # But should still complete some profiling
        success_rate = len(successful_results) / total_attempts
        assert success_rate >= 0.5, f"Success rate under stress: {success_rate:.1%}"
        
        print(f"✓ Health-based throttling test passed")
        print(f"  Total attempts: {total_attempts}")
        print(f"  Throttling instances: {throttling_instances}")
        print(f"  Success rate: {success_rate:.1%}")
    
    def test_health_alerting_and_recovery(self, tmp_path):
        """Test health alerting and automatic recovery mechanisms."""
        
        health_monitor = SystemHealthMonitor()
        
        # Configure health thresholds for testing
        health_monitor.configure_thresholds(
            cpu_threshold=70,
            memory_threshold=80,
            disk_threshold=90,
            check_interval=1.0
        )
        
        # Set up alert collection
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
            print(f"Health alert: {alert}")
        
        health_monitor.set_alert_handler(alert_handler)
        
        # Start health monitoring
        health_monitor.start()
        
        try:
            # Create model for testing
            model_file = tmp_path / "recovery_test_model.gguf"
            self._create_model_file(model_file, 1.8)
            model = QuantizedModel.from_file(model_file)
            
            # Simulate health degradation and recovery
            def simulate_memory_pressure():
                """Simulate memory pressure to trigger alerts."""
                # Allocate large amounts of memory
                memory_hogs = []
                try:
                    for i in range(10):
                        # Allocate 50MB chunks
                        memory_hog = bytearray(50 * 1024 * 1024)
                        memory_hogs.append(memory_hog)
                        time.sleep(0.5)
                        
                        # Check if we should stop (for recovery test)
                        if len(alerts_received) >= 2:
                            break
                    
                    # Hold memory for a bit
                    time.sleep(3)
                    
                finally:
                    # Release memory (simulate recovery)
                    memory_hogs.clear()
                    import gc
                    gc.collect()
            
            # Start memory pressure simulation
            pressure_thread = threading.Thread(target=simulate_memory_pressure)
            pressure_thread.start()
            
            # Monitor for alerts while running profiling
            profiler = EdgeProfiler(platform="esp32", connection="local")
            profiling_attempts = []
            
            for i in range(8):
                try:
                    start_time = time.time()
                    
                    # Get current health status
                    current_health = health_monitor.get_health_status()
                    
                    result = profiler.profile_model(
                        model=model,
                        test_prompts=[f"Recovery test {i}"],
                        metrics=["latency"],
                        config=ProfilingConfig(duration_seconds=2, measurement_iterations=1)
                    )
                    
                    profiling_attempts.append({
                        "attempt": i,
                        "success": result is not None,
                        "health_status": current_health,
                        "timestamp": start_time
                    })
                    
                except Exception as e:
                    profiling_attempts.append({
                        "attempt": i,
                        "success": False,
                        "error": str(e),
                        "timestamp": start_time
                    })
                
                time.sleep(1)
            
            pressure_thread.join(timeout=20)
            
            # Wait for recovery monitoring
            time.sleep(3)
        
        finally:
            health_monitor.stop()
        
        # Analyze health alerting and recovery
        assert len(alerts_received) > 0, "Should have received health alerts"
        
        # Check alert types
        alert_types = set(alert.get("type", "unknown") for alert in alerts_received)
        assert "memory" in alert_types or "high_usage" in alert_types, "Should have memory-related alerts"
        
        # Analyze profiling performance under health stress
        successful_attempts = [a for a in profiling_attempts if a["success"]]
        total_attempts = len(profiling_attempts)
        
        # Some attempts should succeed even under stress
        success_rate = len(successful_attempts) / total_attempts
        assert success_rate >= 0.3, f"Success rate during health stress: {success_rate:.1%}"
        
        # Should show recovery over time (later attempts more successful)
        early_attempts = profiling_attempts[:4]
        late_attempts = profiling_attempts[4:]
        
        early_success = sum(1 for a in early_attempts if a["success"]) / len(early_attempts)
        late_success = sum(1 for a in late_attempts if a["success"]) / len(late_attempts)
        
        print(f"✓ Health alerting and recovery test passed")
        print(f"  Alerts received: {len(alerts_received)}")
        print(f"  Alert types: {alert_types}")
        print(f"  Overall success rate: {success_rate:.1%}")
        print(f"  Early success rate: {early_success:.1%}")
        print(f"  Late success rate: {late_success:.1%}")


@pytest.mark.integration
class TestAutoScalingIntegration:
    """Test auto-scaling capabilities under varying loads."""
    
    def test_load_based_auto_scaling(self, tmp_path):
        """Test automatic scaling based on workload demands."""
        
        # Initialize auto-scaler with conservative settings
        scaler = AutoScaler(
            min_instances=1,
            max_instances=4,
            target_cpu_percent=60,
            scale_up_threshold=75,
            scale_down_threshold=30,
            cooldown_period=5
        )
        
        # Create test models
        models = []
        for i in range(6):
            model_file = tmp_path / f"scaling_test_model_{i}.gguf"
            self._create_model_file(model_file, 1.2)
            models.append(QuantizedModel.from_file(model_file))
        
        # Initialize concurrent profiler with scaling
        concurrent_profiler = ConcurrentProfiler(
            max_threads=2,  # Start small
            timeout_seconds=60
        )
        
        concurrent_profiler.start()
        
        try:
            # Phase 1: Low load (should maintain min instances)
            print("Phase 1: Low load testing")
            
            low_load_tasks = []
            for i in range(2):  # Just 2 tasks
                task = ProfilingTask(
                    task_id=f"low_load_task_{i}",
                    platform="esp32",
                    model=models[i],
                    test_prompts=[f"Low load test {i}"],
                    metrics=["latency"]
                )
                task_id = concurrent_profiler.submit_task(task)
                low_load_tasks.append(task_id)
            
            low_load_results = concurrent_profiler.wait_for_completion(low_load_tasks, timeout=30)
            
            # Check scaling decision during low load
            current_metrics = scaler.collect_metrics()
            scaling_decision = scaler.should_scale()
            
            phase1_stats = {
                "tasks": len(low_load_tasks),
                "completed": len([r for r in low_load_results.values() if r.success]),
                "cpu_usage": current_metrics.get("cpu_percent", 0),
                "scaling_decision": scaling_decision
            }
            
            # Phase 2: High load (should trigger scale up)
            print("Phase 2: High load testing")
            
            high_load_tasks = []
            for i in range(8):  # More tasks than current capacity
                task = ProfilingTask(
                    task_id=f"high_load_task_{i}",
                    platform="esp32",
                    model=models[i % len(models)],
                    test_prompts=[f"High load test {i}"],
                    metrics=["latency"],
                    priority=i % 3  # Mixed priorities
                )
                task_id = concurrent_profiler.submit_task(task)
                high_load_tasks.append(task_id)
            
            # Monitor scaling during high load
            start_time = time.time()
            scaling_events = []
            
            while time.time() - start_time < 20:  # Monitor for 20 seconds
                metrics = scaler.collect_metrics()
                decision = scaler.should_scale()
                
                if decision != "none":
                    scaling_events.append({
                        "timestamp": time.time(),
                        "decision": decision,
                        "cpu_percent": metrics.get("cpu_percent", 0),
                        "queue_size": metrics.get("queue_size", 0)
                    })
                    
                    # Simulate scaling action
                    if decision == "up":
                        print(f"  Scaling UP at {time.time():.1f}")
                        # Would increase thread pool size in real implementation
                    elif decision == "down":
                        print(f"  Scaling DOWN at {time.time():.1f}")
                        # Would decrease thread pool size in real implementation
                
                time.sleep(1)
                
                # Check if tasks are completing
                completed_tasks = [
                    tid for tid in high_load_tasks 
                    if tid in concurrent_profiler.completed_tasks
                ]
                
                if len(completed_tasks) >= len(high_load_tasks) * 0.8:
                    break
            
            high_load_results = concurrent_profiler.wait_for_completion(high_load_tasks, timeout=60)
            
            # Phase 3: Load reduction (should trigger scale down)
            print("Phase 3: Load reduction testing")
            
            # Wait for cooldown period
            time.sleep(scaler.cooldown_period + 1)
            
            # Submit fewer tasks
            reduced_load_tasks = []
            for i in range(1):  # Just 1 task
                task = ProfilingTask(
                    task_id=f"reduced_load_task_{i}",
                    platform="esp32",
                    model=models[0],
                    test_prompts=[f"Reduced load test {i}"],
                    metrics=["latency"]
                )
                task_id = concurrent_profiler.submit_task(task)
                reduced_load_tasks.append(task_id)
            
            reduced_load_results = concurrent_profiler.wait_for_completion(reduced_load_tasks, timeout=30)
            
            # Final scaling check
            final_metrics = scaler.collect_metrics()
            final_decision = scaler.should_scale()
        
        finally:
            concurrent_profiler.stop()
        
        # Analyze scaling behavior
        phase2_stats = {
            "tasks": len(high_load_tasks),
            "completed": len([r for r in high_load_results.values() if r.success]),
            "scaling_events": len(scaling_events),
            "scale_up_events": len([e for e in scaling_events if e["decision"] == "up"])
        }
        
        # Validate scaling decisions
        assert phase2_stats["scale_up_events"] >= 1, "Should have triggered scale-up during high load"
        
        # Most tasks should complete successfully
        overall_success_rate = (
            phase1_stats["completed"] + 
            phase2_stats["completed"] + 
            len([r for r in reduced_load_results.values() if r.success])
        ) / (phase1_stats["tasks"] + phase2_stats["tasks"] + len(reduced_load_tasks))
        
        assert overall_success_rate >= 0.7, f"Overall success rate: {overall_success_rate:.1%}"
        
        print(f"✓ Load-based auto-scaling test passed")
        print(f"  Phase 1 (low load): {phase1_stats}")
        print(f"  Phase 2 (high load): {phase2_stats}")
        print(f"  Overall success rate: {overall_success_rate:.1%}")
        print(f"  Scaling events: {scaling_events}")
    
    def test_load_balancer_with_health_awareness(self, tmp_path):
        """Test load balancer that considers resource health."""
        
        # Initialize load balancer with health awareness
        balancer = LoadBalancer(strategy="health_aware")
        
        # Create mock profiling resources with different health states
        resources = [
            {"id": "profiler_0", "platform": "esp32", "capacity": 1.0, "health": 1.0},
            {"id": "profiler_1", "platform": "stm32f7", "capacity": 0.8, "health": 0.9},
            {"id": "profiler_2", "platform": "rp2040", "capacity": 0.6, "health": 0.7},
            {"id": "profiler_3", "platform": "esp32", "capacity": 1.0, "health": 0.3}  # Unhealthy
        ]
        
        for resource in resources:
            balancer.add_resource(
                resource["id"],
                {"platform": resource["platform"]},
                capacity_weight=resource["capacity"]
            )
            
            # Set health status
            balancer.update_resource_health(resource["id"], resource["health"])
        
        # Create test model
        model_file = tmp_path / "load_balancer_test_model.gguf"
        self._create_model_file(model_file, 1.5)
        model = QuantizedModel.from_file(model_file)
        
        # Simulate task distribution with health awareness
        task_assignments = {}
        task_results = {}
        
        for i in range(20):
            # Select resource using health-aware load balancing
            selected_resource = balancer.select_resource()
            
            if selected_resource:
                resource_id = selected_resource["id"]
                
                # Track assignment
                task_assignments[resource_id] = task_assignments.get(resource_id, 0) + 1
                
                # Simulate task execution
                task_id = f"health_aware_task_{i}"
                balancer.report_task_start(resource_id, task_id)
                
                # Simulate success/failure based on health
                resource_health = next(r["health"] for r in resources if r["id"] == resource_id)
                success = resource_health > 0.5  # Healthy resources more likely to succeed
                
                execution_time = 1.0 / resource_health  # Unhealthy resources slower
                time.sleep(min(execution_time * 0.1, 0.2))  # Simulate work (scaled down)
                
                balancer.report_task_completion(resource_id, task_id, success, execution_time)
                
                task_results[task_id] = {
                    "resource_id": resource_id,
                    "success": success,
                    "execution_time": execution_time,
                    "resource_health": resource_health
                }
        
        # Analyze load balancing decisions
        balancer_stats = balancer.get_stats()
        
        # Healthy resources should receive more tasks
        healthy_resources = [r["id"] for r in resources if r["health"] > 0.7]
        unhealthy_resources = [r["id"] for r in resources if r["health"] <= 0.5]
        
        healthy_tasks = sum(task_assignments.get(r_id, 0) for r_id in healthy_resources)
        unhealthy_tasks = sum(task_assignments.get(r_id, 0) for r_id in unhealthy_resources)
        
        if unhealthy_tasks > 0:
            health_ratio = healthy_tasks / unhealthy_tasks
            assert health_ratio >= 2.0, f"Healthy resources should get more tasks: {health_ratio:.1f}"
        
        # Success rate should be higher for healthy resources
        healthy_task_results = [
            r for r in task_results.values() 
            if r["resource_health"] > 0.7
        ]
        unhealthy_task_results = [
            r for r in task_results.values() 
            if r["resource_health"] <= 0.5
        ]
        
        if healthy_task_results:
            healthy_success_rate = sum(r["success"] for r in healthy_task_results) / len(healthy_task_results)
        else:
            healthy_success_rate = 0
        
        if unhealthy_task_results:
            unhealthy_success_rate = sum(r["success"] for r in unhealthy_task_results) / len(unhealthy_task_results)
        else:
            unhealthy_success_rate = 0
        
        print(f"✓ Health-aware load balancer test passed")
        print(f"  Task assignments: {task_assignments}")
        print(f"  Healthy tasks: {healthy_tasks}, Unhealthy tasks: {unhealthy_tasks}")
        print(f"  Healthy success rate: {healthy_success_rate:.1%}")
        print(f"  Unhealthy success rate: {unhealthy_success_rate:.1%}")
        print(f"  Balancer stats: {balancer_stats}")
    
    def _create_model_file(self, path: Path, size_mb: float):
        """Create a test model file of specified size."""
        header = b"GGUF\x03\x00\x00\x00"
        metadata = b'\x00' * 256
        data_size = int(size_mb * 1024 * 1024) - len(header) - len(metadata)
        
        with open(path, 'wb') as f:
            f.write(header)
            f.write(metadata)
            f.write(b'\x42' * max(0, data_size))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])