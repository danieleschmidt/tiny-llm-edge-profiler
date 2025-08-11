"""
Concurrent Profiling Integration Tests

Tests that verify concurrent profiling capabilities including multi-device profiling,
task scheduling, and resource management in concurrent scenarios.
"""

import pytest
import asyncio
import threading
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import concurrent.futures

from tiny_llm_profiler.profiler import EdgeProfiler, ProfilingConfig
from tiny_llm_profiler.models import QuantizedModel, QuantizationType
from tiny_llm_profiler.concurrent_utils import ConcurrentProfiler, ProfilingTask, TaskResult, TaskStatus
from tiny_llm_profiler.scalable_profiler import ProfileTask, ProfileTaskQueue, ProfileTaskResult
from tiny_llm_profiler.results import ProfileResults


@pytest.mark.integration
class TestConcurrentProfilingWorkflows:
    """Test concurrent profiling across multiple devices and models."""
    
    def test_multi_device_concurrent_profiling(self, tmp_path):
        """Test profiling multiple models on different devices concurrently."""
        
        # Create multiple test models
        models = {}
        for i, (size_mb, quant) in enumerate([(1.5, QuantizationType.INT2), (2.5, QuantizationType.INT3), (3.0, QuantizationType.INT4)]):
            model_file = tmp_path / f"concurrent_model_{i}.gguf"
            self._create_model_file(model_file, size_mb)
            models[f"model_{i}"] = QuantizedModel.from_file(model_file, quantization=quant)
        
        # Define multiple target devices/platforms
        device_configs = [
            {"platform": "esp32", "device": "/dev/mock_esp32_0"},
            {"platform": "stm32f7", "device": "/dev/mock_stm32_0"}, 
            {"platform": "rp2040", "device": "/dev/mock_rp2040_0"},
        ]
        
        # Initialize concurrent profiler
        concurrent_profiler = ConcurrentProfiler(
            max_threads=4,
            timeout_seconds=60
        )
        
        # Start the concurrent profiler
        concurrent_profiler.start()
        
        try:
            # Submit profiling tasks for different model-device combinations
            submitted_tasks = []
            
            for i, (model_name, model) in enumerate(models.items()):
                for j, device_config in enumerate(device_configs):
                    task = ProfilingTask(
                        task_id=f"task_{model_name}_{device_config['platform']}",
                        platform=device_config["platform"],
                        model=model,
                        device_path=device_config["device"],
                        test_prompts=[f"Concurrent test {i}-{j}"],
                        metrics=["latency", "memory"],
                        priority=i  # Different priorities
                    )
                    
                    task_id = concurrent_profiler.submit_task(task)
                    submitted_tasks.append(task_id)
            
            # Wait for all tasks to complete
            results = concurrent_profiler.wait_for_completion(
                submitted_tasks, 
                timeout=120
            )
            
            # Validate results
            assert len(results) == len(submitted_tasks), "All tasks should complete"
            
            successful_tasks = [
                task_id for task_id, result in results.items() 
                if result.success and result.result is not None
            ]
            
            # At least 70% should succeed (allowing for some platform incompatibilities)
            success_rate = len(successful_tasks) / len(submitted_tasks)
            assert success_rate >= 0.7, f"Success rate too low: {success_rate:.1%}"
            
            # Verify concurrent execution (should be faster than sequential)
            total_execution_time = max(
                result.duration_seconds for result in results.values() 
                if result.success
            )
            
            # Should complete much faster than sequential execution
            estimated_sequential_time = len(submitted_tasks) * 5  # 5 seconds per task
            assert total_execution_time < estimated_sequential_time * 0.5, "Concurrent execution should be faster"
            
            # Analyze performance across different combinations
            performance_matrix = {}
            for task_id, result in results.items():
                if result.success:
                    parts = task_id.split("_")
                    model_name = f"{parts[1]}_{parts[2]}"  # model_X
                    platform = parts[3]  # platform name
                    
                    if model_name not in performance_matrix:
                        performance_matrix[model_name] = {}
                    
                    performance_matrix[model_name][platform] = {
                        "tokens_per_second": result.result.latency_profile.tokens_per_second,
                        "memory_usage_kb": result.result.memory_profile.peak_memory_kb,
                        "efficiency_score": result.result.calculate_efficiency_score()
                    }
            
            print(f"✓ Multi-device concurrent profiling test passed")
            print(f"  Tasks submitted: {len(submitted_tasks)}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Execution time: {total_execution_time:.1f}s")
            print(f"  Performance matrix:")
            for model, platforms in performance_matrix.items():
                print(f"    {model}: {len(platforms)} platforms tested")
        
        finally:
            concurrent_profiler.stop()
    
    def test_task_priority_scheduling(self, tmp_path):
        """Test that higher priority tasks are executed first in concurrent profiling."""
        
        # Create test models
        model_file = tmp_path / "priority_test_model.gguf"
        self._create_model_file(model_file, 2.0)
        model = QuantizedModel.from_file(model_file)
        
        # Create task queue
        task_queue = ProfileTaskQueue(maxsize=100)
        
        # Submit tasks with different priorities (1=highest, 10=lowest)
        tasks = [
            ProfileTask(
                task_id=f"low_priority_{i}",
                platform="esp32",
                test_prompts=[f"Low priority test {i}"],
                config={"duration": 1},
                priority=8  # Low priority
            ) for i in range(3)
        ] + [
            ProfileTask(
                task_id=f"high_priority_{i}",
                platform="esp32", 
                test_prompts=[f"High priority test {i}"],
                config={"duration": 1},
                priority=2  # High priority
            ) for i in range(2)
        ] + [
            ProfileTask(
                task_id=f"medium_priority_{i}",
                platform="esp32",
                test_prompts=[f"Medium priority test {i}"],
                config={"duration": 1},
                priority=5  # Medium priority
            ) for i in range(2)
        ]
        
        # Add tasks to queue (in random order)
        import random
        random.shuffle(tasks)
        for task in tasks:
            task_queue.add_task(task)
        
        # Execute tasks and track execution order
        execution_order = []
        
        while True:
            task = task_queue.get_task(timeout=0.1)
            if task is None:
                break
                
            execution_order.append((task.task_id, task.priority))
            
            # Simulate task execution
            result = ProfileTaskResult(
                task_id=task.task_id,
                platform=task.platform,
                result=self._create_mock_profile(),
                execution_time_s=0.1
            )
            task_queue.add_result(result)
        
        # Verify priority ordering
        priorities = [priority for _, priority in execution_order]
        
        # Should generally execute higher priority (lower number) first
        # Allow some flexibility as equal priorities can be in any order
        high_priority_indices = [i for i, p in enumerate(priorities) if p <= 3]
        low_priority_indices = [i for i, p in enumerate(priorities) if p >= 7]
        
        if high_priority_indices and low_priority_indices:
            avg_high_pos = sum(high_priority_indices) / len(high_priority_indices)
            avg_low_pos = sum(low_priority_indices) / len(low_priority_indices)
            
            assert avg_high_pos < avg_low_pos, "High priority tasks should execute before low priority"
        
        stats = task_queue.get_stats()
        assert stats["completed_tasks"] == len(tasks)
        assert stats["failed_tasks"] == 0
        
        print(f"✓ Task priority scheduling test passed")
        print(f"  Tasks executed: {len(execution_order)}")
        print(f"  Execution order by priority: {priorities}")
    
    def test_concurrent_resource_contention(self, tmp_path):
        """Test handling of resource contention in concurrent profiling."""
        
        # Create multiple models that will compete for resources
        models = []
        for i in range(5):
            model_file = tmp_path / f"contention_model_{i}.gguf"
            self._create_model_file(model_file, 1.8)
            models.append(QuantizedModel.from_file(model_file))
        
        # Use a limited thread pool to force contention
        concurrent_profiler = ConcurrentProfiler(
            max_threads=2,  # Limited threads
            timeout_seconds=30
        )
        
        concurrent_profiler.start()
        
        try:
            # Submit many tasks simultaneously
            task_ids = []
            start_time = time.time()
            
            for i, model in enumerate(models):
                for platform in ["esp32", "stm32f7"]:
                    task = ProfilingTask(
                        task_id=f"contention_task_{i}_{platform}",
                        platform=platform,
                        model=model,
                        test_prompts=[f"Contention test {i}"],
                        metrics=["latency"],
                        priority=i % 3  # Mix of priorities
                    )
                    
                    task_id = concurrent_profiler.submit_task(task)
                    task_ids.append(task_id)
            
            submission_time = time.time() - start_time
            
            # Wait for completion
            results = concurrent_profiler.wait_for_completion(task_ids, timeout=60)
            
            completion_time = time.time() - start_time
            
            # Validate contention handling
            assert len(results) == len(task_ids), "All tasks should complete despite contention"
            
            successful_results = [r for r in results.values() if r.success]
            success_rate = len(successful_results) / len(results)
            
            assert success_rate >= 0.8, f"Success rate under contention too low: {success_rate:.1%}"
            
            # Check that tasks were queued and executed over time (not all at once)
            execution_times = [r.duration_seconds for r in successful_results]
            total_work_time = sum(execution_times)
            
            # With resource contention, total time should be longer than fastest possible
            min_possible_time = max(execution_times)  # Longest single task
            
            assert completion_time >= min_possible_time, "Should respect resource limitations"
            
            print(f"✓ Concurrent resource contention test passed")
            print(f"  Tasks: {len(task_ids)}, Success: {success_rate:.1%}")
            print(f"  Submission time: {submission_time:.2f}s, Total time: {completion_time:.1f}s")
            
        finally:
            concurrent_profiler.stop()
    
    @pytest.mark.asyncio
    async def test_async_concurrent_profiling(self, tmp_path):
        """Test asynchronous concurrent profiling capabilities."""
        
        # Create test model
        model_file = tmp_path / "async_test_model.gguf"
        self._create_model_file(model_file, 1.5)
        model = QuantizedModel.from_file(model_file)
        
        # Define async profiling tasks
        async def profile_model_async(platform: str, model: QuantizedModel, test_id: int):
            """Async profiling function."""
            profiler = EdgeProfiler(platform=platform, connection="local")
            
            # Simulate async I/O delay
            await asyncio.sleep(0.1 * test_id)
            
            config = ProfilingConfig(
                duration_seconds=1,
                measurement_iterations=1
            )
            
            results = profiler.profile_model(
                model=model,
                test_prompts=[f"Async test {test_id}"],
                metrics=["latency"],
                config=config
            )
            
            return {
                "test_id": test_id,
                "platform": platform,
                "results": results,
                "timestamp": time.time()
            }
        
        # Launch multiple async tasks
        platforms = ["esp32", "stm32f7", "rp2040"]
        tasks = []
        
        start_time = time.time()
        
        for i, platform in enumerate(platforms * 3):  # 9 total tasks
            task = profile_model_async(platform, model, i)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Filter successful results
        successful_results = [
            r for r in results 
            if isinstance(r, dict) and "results" in r
        ]
        
        assert len(successful_results) >= 6, "Most async tasks should succeed"
        
        # Verify concurrent execution
        timestamps = [r["timestamp"] for r in successful_results]
        time_span = max(timestamps) - min(timestamps)
        
        # Should complete much faster than sequential (would be ~4.5s sequential)
        assert execution_time < 3.0, f"Async execution too slow: {execution_time:.1f}s"
        
        # Verify results quality
        for result in successful_results:
            profile_results = result["results"]
            assert profile_results.latency_profile is not None
            assert profile_results.latency_profile.tokens_per_second > 0
        
        print(f"✓ Async concurrent profiling test passed")
        print(f"  Tasks: {len(tasks)}, Successful: {len(successful_results)}")
        print(f"  Execution time: {execution_time:.1f}s")
        print(f"  Time span: {time_span:.1f}s")
    
    def test_concurrent_profiling_error_handling(self, tmp_path):
        """Test error handling and recovery in concurrent profiling scenarios."""
        
        # Create mix of valid and problematic models
        models = {}
        
        # Valid model
        valid_model_file = tmp_path / "valid_model.gguf"
        self._create_model_file(valid_model_file, 1.5)
        models["valid"] = QuantizedModel.from_file(valid_model_file)
        
        # Oversized model (should cause memory issues)
        large_model_file = tmp_path / "oversized_model.gguf"
        self._create_model_file(large_model_file, 10.0)  # 10MB - too large
        models["oversized"] = QuantizedModel.from_file(large_model_file)
        
        # Create concurrent profiler with error handling
        concurrent_profiler = ConcurrentProfiler(
            max_threads=3,
            timeout_seconds=15  # Short timeout to test timeout handling
        )
        
        concurrent_profiler.start()
        
        try:
            task_ids = []
            
            # Submit mix of valid and problematic tasks
            for model_name, model in models.items():
                for platform in ["stm32f4", "esp32"]:  # stm32f4 has tight memory constraints
                    task = ProfilingTask(
                        task_id=f"error_test_{model_name}_{platform}",
                        platform=platform,
                        model=model,
                        test_prompts=["Error handling test"],
                        metrics=["latency", "memory"]
                    )
                    
                    task_id = concurrent_profiler.submit_task(task)
                    task_ids.append(task_id)
                    
                    # Also add a task that will timeout
                    timeout_task = ProfilingTask(
                        task_id=f"timeout_test_{model_name}_{platform}",
                        platform=platform,
                        model=model,
                        test_prompts=["Timeout test"],
                        metrics=["latency"]
                    )
                    
                    # Mock this task to take longer than timeout
                    with patch.object(EdgeProfiler, 'profile_model') as mock_profile:
                        def slow_profile(*args, **kwargs):
                            time.sleep(20)  # Longer than timeout
                            return ProfileResults("test", "test", 1.0, "4bit")
                        
                        mock_profile.side_effect = slow_profile
                        
                        timeout_task_id = concurrent_profiler.submit_task(timeout_task)
                        task_ids.append(timeout_task_id)
            
            # Wait for completion (with errors expected)
            results = concurrent_profiler.wait_for_completion(task_ids, timeout=30)
            
            # Analyze results
            successful_tasks = [tid for tid, r in results.items() if r.success]
            failed_tasks = [tid for tid, r in results.items() if not r.success]
            
            # Should have both successes and failures
            assert len(successful_tasks) > 0, "Some tasks should succeed"
            assert len(failed_tasks) > 0, "Some tasks should fail (demonstrating error handling)"
            
            # Validate error types
            error_types = {}
            for task_id, result in results.items():
                if not result.success and result.error:
                    error_type = type(result.error).__name__
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            print(f"✓ Concurrent error handling test passed")
            print(f"  Successful: {len(successful_tasks)}, Failed: {len(failed_tasks)}")
            print(f"  Error types: {error_types}")
            
            # Verify that valid tasks succeeded despite other failures
            valid_task_results = [
                r for tid, r in results.items() 
                if "valid" in tid and r.success
            ]
            assert len(valid_task_results) > 0, "Valid tasks should succeed despite other failures"
        
        finally:
            concurrent_profiler.stop()
    
    def test_concurrent_profiling_resource_limits(self, tmp_path):
        """Test concurrent profiling behavior under resource limits."""
        
        # Create test model
        model_file = tmp_path / "resource_limit_model.gguf"
        self._create_model_file(model_file, 2.0)
        model = QuantizedModel.from_file(model_file)
        
        # Test with very limited resources
        concurrent_profiler = ConcurrentProfiler(
            max_threads=1,  # Single thread
            max_processes=1,  # Single process
            timeout_seconds=20,
            queue_size=50  # Limited queue
        )
        
        concurrent_profiler.start()
        
        try:
            # Submit more tasks than can run simultaneously
            num_tasks = 20
            task_ids = []
            
            submission_start = time.time()
            
            for i in range(num_tasks):
                task = ProfilingTask(
                    task_id=f"resource_limit_task_{i}",
                    platform="esp32",
                    model=model,
                    test_prompts=[f"Resource limit test {i}"],
                    metrics=["latency"],
                    priority=i % 5  # Mix of priorities
                )
                
                task_id = concurrent_profiler.submit_task(task)
                task_ids.append(task_id)
            
            submission_time = time.time() - submission_start
            
            # Wait for completion
            results = concurrent_profiler.wait_for_completion(task_ids, timeout=60)
            
            total_time = time.time() - submission_start
            
            # All tasks should complete despite resource limits
            assert len(results) == num_tasks, "All tasks should complete"
            
            successful_results = [r for r in results.values() if r.success]
            success_rate = len(successful_results) / len(results)
            
            assert success_rate >= 0.9, f"High success rate expected: {success_rate:.1%}"
            
            # Verify sequential execution due to resource limits
            # With single thread, tasks should execute roughly sequentially
            execution_times = [r.duration_seconds for r in successful_results if r.success]
            
            # Total execution time should be roughly sum of individual times
            expected_min_time = sum(execution_times) * 0.8  # Allow some overhead
            
            assert total_time >= expected_min_time, "Should respect resource limits"
            
            print(f"✓ Resource limits test passed")
            print(f"  Tasks: {num_tasks}, Success rate: {success_rate:.1%}")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Average task time: {sum(execution_times)/len(execution_times):.2f}s")
        
        finally:
            concurrent_profiler.stop()
    
    def _create_model_file(self, path: Path, size_mb: float):
        """Create a test model file of specified size."""
        header = b"GGUF\x03\x00\x00\x00"
        metadata = b'\x00' * 256
        data_size = int(size_mb * 1024 * 1024) - len(header) - len(metadata)
        
        with open(path, 'wb') as f:
            f.write(header)
            f.write(metadata)
            f.write(b'\x42' * max(0, data_size))
    
    def _create_mock_profile(self) -> ProfileResults:
        """Create a mock profile result for testing."""
        from tiny_llm_profiler.results import LatencyProfile
        
        results = ProfileResults("esp32", "mock_model", 2.0, "4bit")
        
        latency_profile = LatencyProfile(
            first_token_latency_ms=50.0,
            inter_token_latency_ms=25.0,
            total_latency_ms=100.0,
            tokens_per_second=10.0,
            latency_std_ms=5.0
        )
        
        results.add_latency_profile(latency_profile)
        return results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])