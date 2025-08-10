"""
Concurrent and scalable profiling implementation for high-throughput operations.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import logging
import queue
from pathlib import Path

try:
    from .core_lite import SimpleProfiler, SimplifiedProfile, SimplePlatformManager
except ImportError:
    from core_lite import SimpleProfiler, SimplifiedProfile, SimplePlatformManager

logger = logging.getLogger(__name__)


@dataclass
class ProfileTask:
    """Represents a profiling task."""
    task_id: str
    platform: str
    test_prompts: List[str]
    config: Dict[str, Any]
    priority: int = 5  # 1=highest, 10=lowest
    
    def __lt__(self, other):
        """Enable comparison for priority queue."""
        return self.priority < other.priority


@dataclass 
class ProfileTaskResult:
    """Result of a profiling task."""
    task_id: str
    platform: str
    result: SimplifiedProfile
    execution_time_s: float
    worker_id: Optional[str] = None


class ProfileTaskQueue:
    """Thread-safe task queue for profiling operations."""
    
    def __init__(self, maxsize: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=maxsize)
        self.results: Dict[str, ProfileTaskResult] = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
    
    def add_task(self, task: ProfileTask):
        """Add a task to the queue."""
        # Priority queue uses (priority, item) tuples
        self.queue.put((task.priority, task))
        logger.debug(f"Added task {task.task_id} to queue with priority {task.priority}")
    
    def get_task(self, timeout: float = 1.0) -> Optional[ProfileTask]:
        """Get next task from queue."""
        try:
            priority, task = self.queue.get(timeout=timeout)
            return task
        except queue.Empty:
            return None
    
    def add_result(self, result: ProfileTaskResult):
        """Add a task result."""
        with self.lock:
            self.results[result.task_id] = result
            if result.result.success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1
    
    def get_result(self, task_id: str) -> Optional[ProfileTaskResult]:
        """Get result for a specific task."""
        with self.lock:
            return self.results.get(task_id)
    
    def get_all_results(self) -> List[ProfileTaskResult]:
        """Get all completed results."""
        with self.lock:
            return list(self.results.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                "pending_tasks": self.queue.qsize(),
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_results": len(self.results)
            }


class ProfileWorker:
    """Worker that processes profiling tasks."""
    
    def __init__(self, worker_id: str, task_queue: ProfileTaskQueue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.tasks_processed = 0
    
    def start(self):
        """Start the worker thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info(f"Worker {self.worker_id} started")
    
    def stop(self):
        """Stop the worker thread."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _worker_loop(self):
        """Main worker loop."""
        logger.debug(f"Worker {self.worker_id} started processing")
        
        while self.is_running:
            try:
                # Get next task
                task = self.task_queue.get_task(timeout=1.0)
                if task is None:
                    continue
                
                logger.debug(f"Worker {self.worker_id} processing task {task.task_id}")
                
                # Process task
                start_time = time.time()
                try:
                    profiler = SimpleProfiler(task.platform)
                    profile_result = profiler.simulate_profiling(task.test_prompts)
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} failed on task {task.task_id}: {e}")
                    profile_result = SimplifiedProfile(
                        success=False,
                        error_message=str(e)
                    )
                
                execution_time = time.time() - start_time
                
                # Store result
                result = ProfileTaskResult(
                    task_id=task.task_id,
                    platform=task.platform,
                    result=profile_result,
                    execution_time_s=execution_time,
                    worker_id=self.worker_id
                )
                
                self.task_queue.add_result(result)
                self.tasks_processed += 1
                
                logger.debug(f"Worker {self.worker_id} completed task {task.task_id} in {execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                time.sleep(0.1)


class ConcurrentProfiler:
    """High-performance concurrent profiler."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue = ProfileTaskQueue()
        self.workers: List[ProfileWorker] = []
        self.is_running = False
        
        # Performance monitoring
        self.start_time = time.time()
        self.tasks_submitted = 0
    
    def start(self):
        """Start the concurrent profiler."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = ProfileWorker(f"worker_{i}", self.task_queue)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Concurrent profiler started with {self.max_workers} workers")
    
    def stop(self):
        """Stop the concurrent profiler."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all workers
        for worker in self.workers:
            worker.stop()
        
        self.workers.clear()
        logger.info("Concurrent profiler stopped")
    
    def submit_task(self, platform: str, test_prompts: List[str], 
                   config: Dict[str, Any] = None, priority: int = 5) -> str:
        """Submit a profiling task."""
        if not self.is_running:
            raise RuntimeError("Concurrent profiler not started")
        
        task_id = f"task_{self.tasks_submitted}_{int(time.time())}"
        task = ProfileTask(
            task_id=task_id,
            platform=platform,
            test_prompts=test_prompts,
            config=config or {},
            priority=priority
        )
        
        self.task_queue.add_task(task)
        self.tasks_submitted += 1
        
        logger.debug(f"Submitted task {task_id} for platform {platform}")
        return task_id
    
    def get_result(self, task_id: str, timeout: float = 10.0) -> Optional[ProfileTaskResult]:
        """Get result for a specific task, waiting if necessary."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.task_queue.get_result(task_id)
            if result is not None:
                return result
            time.sleep(0.1)
        
        return None
    
    def get_all_results(self) -> List[ProfileTaskResult]:
        """Get all completed results."""
        return self.task_queue.get_all_results()
    
    def wait_for_completion(self, timeout: float = 300.0) -> bool:
        """Wait for all submitted tasks to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            stats = self.get_stats()
            if stats["completed_tasks"] + stats["failed_tasks"] >= self.tasks_submitted:
                return True
            time.sleep(0.5)
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        queue_stats = self.task_queue.get_stats()
        
        # Worker statistics
        active_workers = sum(1 for w in self.workers if w.is_running)
        total_processed = sum(w.tasks_processed for w in self.workers)
        
        # Performance metrics
        runtime = time.time() - self.start_time
        throughput = total_processed / runtime if runtime > 0 else 0
        
        return {
            "runtime_seconds": runtime,
            "tasks_submitted": self.tasks_submitted,
            "active_workers": active_workers,
            "total_workers": len(self.workers),
            "tasks_processed": total_processed,
            "throughput_tasks_per_second": throughput,
            **queue_stats
        }


class BatchProfiler:
    """Batch profiling operations for large-scale benchmarking."""
    
    def __init__(self, concurrent_profiler: ConcurrentProfiler):
        self.profiler = concurrent_profiler
    
    def profile_platforms_batch(
        self,
        platforms: List[str],
        test_prompts: List[str],
        config: Dict[str, Any] = None,
        priority: int = 5
    ) -> Dict[str, str]:
        """Submit profiling tasks for multiple platforms."""
        task_ids = {}
        
        for platform in platforms:
            try:
                task_id = self.profiler.submit_task(
                    platform=platform,
                    test_prompts=test_prompts,
                    config=config,
                    priority=priority
                )
                task_ids[platform] = task_id
            except Exception as e:
                logger.error(f"Failed to submit task for platform {platform}: {e}")
        
        return task_ids
    
    def profile_prompt_variations_batch(
        self,
        platform: str,
        prompt_variations: List[List[str]],
        config: Dict[str, Any] = None,
        priority: int = 5
    ) -> List[str]:
        """Submit profiling tasks for multiple prompt variations."""
        task_ids = []
        
        for i, prompts in enumerate(prompt_variations):
            try:
                task_id = self.profiler.submit_task(
                    platform=platform,
                    test_prompts=prompts,
                    config=config,
                    priority=priority
                )
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Failed to submit prompt variation {i}: {e}")
        
        return task_ids
    
    def comprehensive_benchmark(
        self,
        platforms: List[str],
        prompt_sets: Dict[str, List[str]],
        timeout: float = 600.0
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across platforms and prompt sets."""
        logger.info(f"Starting comprehensive benchmark: {len(platforms)} platforms, {len(prompt_sets)} prompt sets")
        
        all_task_ids = []
        task_metadata = {}
        
        # Submit all tasks
        for platform in platforms:
            for prompt_set_name, prompts in prompt_sets.items():
                try:
                    task_id = self.profiler.submit_task(
                        platform=platform,
                        test_prompts=prompts,
                        priority=3  # Higher priority for comprehensive benchmark
                    )
                    all_task_ids.append(task_id)
                    task_metadata[task_id] = {
                        "platform": platform,
                        "prompt_set": prompt_set_name
                    }
                except Exception as e:
                    logger.error(f"Failed to submit task for {platform}/{prompt_set_name}: {e}")
        
        logger.info(f"Submitted {len(all_task_ids)} tasks")
        
        # Wait for completion
        if not self.profiler.wait_for_completion(timeout):
            logger.warning("Benchmark timed out before all tasks completed")
        
        # Collect results
        results = {}
        successful_tasks = 0
        failed_tasks = 0
        
        for task_id in all_task_ids:
            result = self.profiler.get_result(task_id)
            if result:
                metadata = task_metadata[task_id]
                platform = metadata["platform"]
                prompt_set = metadata["prompt_set"]
                
                if platform not in results:
                    results[platform] = {}
                
                results[platform][prompt_set] = result
                
                if result.result.success:
                    successful_tasks += 1
                else:
                    failed_tasks += 1
        
        # Generate summary
        summary = {
            "total_tasks": len(all_task_ids),
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (successful_tasks / len(all_task_ids)) * 100 if all_task_ids else 0,
            "profiler_stats": self.profiler.get_stats(),
            "results": results
        }
        
        logger.info(f"Benchmark completed: {successful_tasks}/{len(all_task_ids)} tasks successful")
        return summary


async def async_profile_single(platform: str, test_prompts: List[str]) -> SimplifiedProfile:
    """Asynchronous single platform profiling."""
    loop = asyncio.get_event_loop()
    
    # Run profiling in thread pool
    with ThreadPoolExecutor(max_workers=1) as executor:
        profiler = SimpleProfiler(platform)
        future = executor.submit(profiler.simulate_profiling, test_prompts)
        result = await loop.run_in_executor(None, lambda: future.result())
    
    return result


async def async_profile_batch(
    platform_prompts: List[Tuple[str, List[str]]]
) -> List[Tuple[str, SimplifiedProfile]]:
    """Asynchronous batch profiling."""
    tasks = [
        async_profile_single(platform, prompts)
        for platform, prompts in platform_prompts
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Pair results with platforms
    paired_results = []
    for i, (platform, prompts) in enumerate(platform_prompts):
        result = results[i]
        if isinstance(result, Exception):
            result = SimplifiedProfile(
                success=False,
                error_message=str(result)
            )
        paired_results.append((platform, result))
    
    return paired_results


def run_concurrent_benchmark_demo():
    """Demo of concurrent profiling capabilities."""
    logger.info("Starting concurrent profiling demo")
    
    # Initialize concurrent profiler
    profiler = ConcurrentProfiler(max_workers=4)
    profiler.start()
    
    try:
        # Submit various tasks
        platforms = ["esp32", "stm32f4", "stm32f7", "rp2040"]
        test_prompts = ["Hello world", "Generate code", "Explain AI"]
        
        task_ids = []
        for platform in platforms:
            for i in range(3):  # 3 tasks per platform
                task_id = profiler.submit_task(
                    platform=platform,
                    test_prompts=test_prompts,
                    priority=i + 1
                )
                task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} tasks")
        
        # Wait for completion
        if profiler.wait_for_completion(timeout=30):
            logger.info("All tasks completed")
        else:
            logger.warning("Some tasks may not have completed")
        
        # Get results
        results = profiler.get_all_results()
        logger.info(f"Collected {len(results)} results")
        
        # Print statistics
        stats = profiler.get_stats()
        logger.info(f"Profiler stats: {stats}")
        
        # Show sample results
        successful_results = [r for r in results if r.result.success]
        if successful_results:
            avg_tokens_per_sec = sum(r.result.tokens_per_second for r in successful_results) / len(successful_results)
            avg_execution_time = sum(r.execution_time_s for r in successful_results) / len(successful_results)
            
            logger.info(f"Average performance: {avg_tokens_per_sec:.1f} tok/s")
            logger.info(f"Average execution time: {avg_execution_time:.2f}s")
        
    finally:
        profiler.stop()


if __name__ == "__main__":
    run_concurrent_benchmark_demo()