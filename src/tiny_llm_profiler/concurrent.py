"""
Concurrent processing capabilities for the Tiny LLM Edge Profiler.
"""

import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Union, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import queue
from pathlib import Path

from .exceptions import TinyLLMProfilerError, ProfilingTimeoutError
from .logging_config import get_logger, PerformanceLogger
from .results import ProfileResults
from .models import QuantizedModel
from .platforms import PlatformManager

logger = get_logger("concurrent")
perf_logger = PerformanceLogger()

T = TypeVar('T')


@dataclass
class TaskResult(Generic[T]):
    """Result of a concurrent task."""
    task_id: str
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    def mark_completed(self, result: Optional[T] = None, error: Optional[Exception] = None):
        """Mark task as completed."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = error is None
        self.result = result
        self.error = error


@dataclass
class ProfilingTask:
    """Task for concurrent profiling."""
    task_id: str
    platform: str
    model: QuantizedModel
    device_path: Optional[str] = None
    test_prompts: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=lambda: ["latency", "memory"])
    config: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher number = higher priority


class ConcurrentProfiler:
    """
    Concurrent profiler for handling multiple devices and models simultaneously.
    """
    
    def __init__(
        self,
        max_threads: int = 4,
        max_processes: int = None,
        timeout_seconds: int = 300,
        queue_size: int = 100
    ):
        self.max_threads = max_threads
        self.max_processes = max_processes or mp.cpu_count()
        self.timeout_seconds = timeout_seconds
        self.queue_size = queue_size
        
        # Thread pool for I/O bound tasks (device communication)
        self.thread_executor = ThreadPoolExecutor(max_workers=max_threads)
        
        # Process pool for CPU bound tasks (analysis)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Task management
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self.active_tasks: Dict[str, Future] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_duration": 0.0
        }
        
        self._lock = threading.RLock()
    
    def start(self) -> None:
        """Start the concurrent profiler workers."""
        with self._lock:
            if self.running:
                return
            
            self.running = True
            
            # Start worker threads
            for i in range(self.max_threads):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"ProfilerWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
            
            logger.info(f"Started concurrent profiler with {self.max_threads} workers")
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the concurrent profiler."""
        with self._lock:
            if not self.running:
                return
            
            self.running = False
            
            # Cancel active tasks
            for task_id, future in self.active_tasks.items():
                future.cancel()
                logger.debug(f"Cancelled task: {task_id}")
            
            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=timeout / len(self.workers))
            
            # Shutdown executors
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            logger.info("Concurrent profiler stopped")
    
    def submit_task(self, task: ProfilingTask) -> str:
        """
        Submit a profiling task for concurrent execution.
        
        Args:
            task: ProfilingTask to execute
            
        Returns:
            Task ID for tracking
        """
        with self._lock:
            if not self.running:
                self.start()
            
            try:
                # Priority queue uses tuple (priority, item)
                # Negate priority for descending order
                self.task_queue.put((-task.priority, task), timeout=1.0)
                self.stats["tasks_submitted"] += 1
                
                logger.debug(f"Submitted task: {task.task_id} (priority: {task.priority})")
                return task.task_id
                
            except queue.Full:
                raise TinyLLMProfilerError(
                    f"Task queue full (max size: {self.queue_size})",
                    "QUEUE_FULL"
                )
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Get result of a completed task.
        
        Args:
            task_id: ID of the task
            timeout: Maximum time to wait for completion
            
        Returns:
            TaskResult if completed, None if not found or not completed
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                # Check if task is completed
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
                
                # Check if task is still active
                if task_id in self.active_tasks:
                    future = self.active_tasks[task_id]
                    
                    # Check if future is done
                    if future.done():
                        try:
                            result = future.result()
                            task_result = TaskResult(
                                task_id=task_id,
                                success=True,
                                result=result
                            )
                        except Exception as e:
                            task_result = TaskResult(
                                task_id=task_id,
                                success=False,
                                error=e
                            )
                        
                        # Move to completed tasks
                        self.completed_tasks[task_id] = task_result
                        del self.active_tasks[task_id]
                        
                        return task_result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Wait a bit before checking again
            time.sleep(0.1)
        
        return None
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Maximum time to wait
            
        Returns:
            Dictionary mapping task IDs to their results
        """
        results = {}
        start_time = time.time()
        
        while len(results) < len(task_ids):
            for task_id in task_ids:
                if task_id not in results:
                    result = self.get_task_result(task_id, timeout=0.1)
                    if result:
                        results[task_id] = result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            time.sleep(0.1)
        
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        with self._lock:
            if task_id in self.active_tasks:
                future = self.active_tasks[task_id]
                cancelled = future.cancel()
                
                if cancelled:
                    del self.active_tasks[task_id]
                    logger.debug(f"Cancelled task: {task_id}")
                
                return cancelled
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        with self._lock:
            active_count = len(self.active_tasks)
            queue_size = self.task_queue.qsize()
            completed_count = len(self.completed_tasks)
            
            # Calculate average duration
            durations = [
                result.duration_seconds 
                for result in self.completed_tasks.values() 
                if result.duration_seconds > 0
            ]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            
            return {
                "running": self.running,
                "active_tasks": active_count,
                "queued_tasks": queue_size,
                "completed_tasks": completed_count,
                "tasks_submitted": self.stats["tasks_submitted"],
                "tasks_failed": self.stats["tasks_failed"],
                "average_duration_seconds": avg_duration,
                "worker_threads": len(self.workers),
                "max_threads": self.max_threads,
                "max_processes": self.max_processes
            }
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        while self.running:
            try:
                # Get task from queue (with timeout to allow clean shutdown)
                try:
                    priority, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Execute task
                self._execute_task(task)
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _execute_task(self, task: ProfilingTask) -> None:
        """Execute a profiling task."""
        start_time = datetime.now()
        
        try:
            # Submit to appropriate executor
            if self._is_io_bound_task(task):
                future = self.thread_executor.submit(self._run_profiling_task, task)
            else:
                future = self.process_executor.submit(self._run_profiling_task, task)
            
            # Track active task
            with self._lock:
                self.active_tasks[task.task_id] = future
            
            # Log task start
            perf_logger.log_profiling_start(
                platform=task.platform,
                model_name=task.model.name,
                session_id=task.task_id,
                concurrent=True
            )
            
        except Exception as e:
            # Handle execution error
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=e,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration
            )
            
            with self._lock:
                self.completed_tasks[task.task_id] = task_result
                self.stats["tasks_failed"] += 1
            
            logger.error(f"Task execution failed: {task.task_id} - {e}")
    
    def _is_io_bound_task(self, task: ProfilingTask) -> bool:
        """Determine if task is I/O bound (device communication) or CPU bound."""
        # Tasks with device communication are I/O bound
        return task.device_path is not None
    
    def _run_profiling_task(self, task: ProfilingTask) -> ProfileResults:
        """Run the actual profiling task."""
        from .profiler import EdgeProfiler, ProfilingConfig
        
        # Create profiler
        profiler = EdgeProfiler(
            platform=task.platform,
            device=task.device_path
        )
        
        # Create config from task
        config = ProfilingConfig()
        if task.config:
            for key, value in task.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Run profiling
        results = profiler.profile_model(
            model=task.model,
            test_prompts=task.test_prompts,
            metrics=task.metrics,
            config=config
        )
        
        return results


class BatchProfiler:
    """
    Batch profiler for running multiple profiling sessions efficiently.
    """
    
    def __init__(self, concurrent_profiler: Optional[ConcurrentProfiler] = None):
        self.concurrent_profiler = concurrent_profiler or ConcurrentProfiler()
        self.logger = get_logger("batch")
    
    def profile_multiple_models(
        self,
        models: List[QuantizedModel],
        platforms: List[str],
        test_prompts: List[str],
        device_mapping: Optional[Dict[str, str]] = None,
        max_concurrent: int = 4
    ) -> Dict[str, Dict[str, ProfileResults]]:
        """
        Profile multiple models across multiple platforms.
        
        Args:
            models: List of models to profile
            platforms: List of platforms to test on
            test_prompts: Test prompts to use
            device_mapping: Optional mapping of platform to device path
            max_concurrent: Maximum concurrent tasks
            
        Returns:
            Nested dictionary: {model_name: {platform: ProfileResults}}
        """
        device_mapping = device_mapping or {}
        
        # Generate all tasks
        tasks = []
        task_mapping = {}  # task_id -> (model_name, platform)
        
        for model in models:
            for platform in platforms:
                task_id = f"{model.name}_{platform}_{int(time.time() * 1000)}"
                
                task = ProfilingTask(
                    task_id=task_id,
                    platform=platform,
                    model=model,
                    device_path=device_mapping.get(platform),
                    test_prompts=test_prompts,
                    metrics=["latency", "memory"],
                    priority=1
                )
                
                tasks.append(task)
                task_mapping[task_id] = (model.name, platform)
        
        self.logger.info(f"Starting batch profiling: {len(tasks)} tasks")
        
        # Ensure profiler is running
        if not self.concurrent_profiler.running:
            self.concurrent_profiler.start()
        
        # Submit tasks in batches
        submitted_tasks = []
        batch_size = max_concurrent
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_task_ids = []
            
            for task in batch:
                task_id = self.concurrent_profiler.submit_task(task)
                batch_task_ids.append(task_id)
                submitted_tasks.append(task_id)
            
            # Wait for current batch to complete before submitting next
            if i + batch_size < len(tasks):  # Not the last batch
                self.concurrent_profiler.wait_for_completion(
                    batch_task_ids,
                    timeout=300  # 5 minutes per batch
                )
        
        # Wait for all tasks to complete
        results = self.concurrent_profiler.wait_for_completion(
            submitted_tasks,
            timeout=600  # 10 minutes total
        )
        
        # Organize results
        organized_results = {}
        
        for task_id, result in results.items():
            if result.success and result.result:
                model_name, platform = task_mapping[task_id]
                
                if model_name not in organized_results:
                    organized_results[model_name] = {}
                
                organized_results[model_name][platform] = result.result
            else:
                self.logger.error(f"Task failed: {task_id} - {result.error}")
        
        self.logger.info(f"Batch profiling completed: {len(organized_results)} successful results")
        
        return organized_results
    
    def profile_platform_comparison(
        self,
        model: QuantizedModel,
        platforms: List[str],
        test_prompts: List[str],
        device_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, ProfileResults]:
        """
        Profile a single model across multiple platforms for comparison.
        
        Args:
            model: Model to profile
            platforms: List of platforms to compare
            test_prompts: Test prompts
            device_mapping: Platform to device mapping
            
        Returns:
            Dictionary mapping platform to ProfileResults
        """
        result = self.profile_multiple_models(
            models=[model],
            platforms=platforms,
            test_prompts=test_prompts,
            device_mapping=device_mapping,
            max_concurrent=len(platforms)
        )
        
        return result.get(model.name, {})


class AsyncProfiler:
    """
    Asynchronous profiler using asyncio for concurrent device communication.
    """
    
    def __init__(self):
        self.logger = get_logger("async")
    
    async def profile_async(
        self,
        tasks: List[ProfilingTask],
        max_concurrent: int = 10
    ) -> Dict[str, TaskResult]:
        """
        Run profiling tasks asynchronously.
        
        Args:
            tasks: List of profiling tasks
            max_concurrent: Maximum concurrent tasks
            
        Returns:
            Dictionary mapping task IDs to results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_task(task: ProfilingTask) -> TaskResult:
            async with semaphore:
                task_result = TaskResult(task_id=task.task_id, success=False)
                
                try:
                    # Run profiling task in thread pool (since it's not truly async)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        self._run_sync_profiling,
                        task
                    )
                    
                    task_result.mark_completed(result=result)
                    
                except Exception as e:
                    task_result.mark_completed(error=e)
                    self.logger.error(f"Async task failed: {task.task_id} - {e}")
                
                return task_result
        
        # Create tasks
        async_tasks = [run_single_task(task) for task in tasks]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Organize results
        result_dict = {}
        for i, result in enumerate(results):
            if isinstance(result, TaskResult):
                result_dict[result.task_id] = result
            else:
                # Handle exceptions from gather
                task_id = tasks[i].task_id
                task_result = TaskResult(task_id=task_id, success=False)
                task_result.mark_completed(error=result)
                result_dict[task_id] = task_result
        
        return result_dict
    
    def _run_sync_profiling(self, task: ProfilingTask) -> ProfileResults:
        """Run synchronous profiling task."""
        from .profiler import EdgeProfiler, ProfilingConfig
        
        profiler = EdgeProfiler(
            platform=task.platform,
            device=task.device_path
        )
        
        config = ProfilingConfig()
        if task.config:
            for key, value in task.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return profiler.profile_model(
            model=task.model,
            test_prompts=task.test_prompts,
            metrics=task.metrics,
            config=config
        )


# Global concurrent profiler instance
_global_concurrent_profiler = ConcurrentProfiler()


def get_concurrent_profiler() -> ConcurrentProfiler:
    """Get the global concurrent profiler instance."""
    return _global_concurrent_profiler


def submit_profiling_task(
    platform: str,
    model: QuantizedModel,
    test_prompts: List[str],
    device_path: Optional[str] = None,
    metrics: List[str] = None,
    priority: int = 0
) -> str:
    """
    Submit a profiling task for concurrent execution.
    
    Args:
        platform: Target platform
        model: Model to profile
        test_prompts: Test prompts
        device_path: Optional device path
        metrics: Metrics to collect
        priority: Task priority
        
    Returns:
        Task ID for tracking
    """
    task_id = f"profile_{platform}_{model.name}_{int(time.time() * 1000)}"
    
    task = ProfilingTask(
        task_id=task_id,
        platform=platform,
        model=model,
        device_path=device_path,
        test_prompts=test_prompts or ["Hello world"],
        metrics=metrics or ["latency", "memory"],
        priority=priority
    )
    
    return _global_concurrent_profiler.submit_task(task)


def wait_for_task(task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
    """Wait for a task to complete and return its result."""
    return _global_concurrent_profiler.get_task_result(task_id, timeout)