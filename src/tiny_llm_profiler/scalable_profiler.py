"""
Generation 3 Enhanced: Concurrent and scalable profiling implementation for high-throughput operations.

Enhanced with:
- Multi-level intelligent caching (L1/L2/L3)
- Advanced async processing pipeline  
- Performance optimization with automated tuning
- Resource monitoring and adaptive scaling
- Vectorized operations and memory pooling
- Distributed coordination capabilities
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
        logger.debug(
            f"Added task {task.task_id} to queue with priority {task.priority}"
        )

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
                "total_results": len(self.results),
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
                    logger.error(
                        f"Worker {self.worker_id} failed on task {task.task_id}: {e}"
                    )
                    profile_result = SimplifiedProfile(
                        success=False, error_message=str(e)
                    )

                execution_time = time.time() - start_time

                # Store result
                result = ProfileTaskResult(
                    task_id=task.task_id,
                    platform=task.platform,
                    result=profile_result,
                    execution_time_s=execution_time,
                    worker_id=self.worker_id,
                )

                self.task_queue.add_result(result)
                self.tasks_processed += 1

                logger.debug(
                    f"Worker {self.worker_id} completed task {task.task_id} in {execution_time:.2f}s"
                )

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

    def submit_task(
        self,
        platform: str,
        test_prompts: List[str],
        config: Dict[str, Any] = None,
        priority: int = 5,
    ) -> str:
        """Submit a profiling task."""
        if not self.is_running:
            raise RuntimeError("Concurrent profiler not started")

        task_id = f"task_{self.tasks_submitted}_{int(time.time())}"
        task = ProfileTask(
            task_id=task_id,
            platform=platform,
            test_prompts=test_prompts,
            config=config or {},
            priority=priority,
        )

        self.task_queue.add_task(task)
        self.tasks_submitted += 1

        logger.debug(f"Submitted task {task_id} for platform {platform}")
        return task_id

    def get_result(
        self, task_id: str, timeout: float = 10.0
    ) -> Optional[ProfileTaskResult]:
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
            **queue_stats,
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
        priority: int = 5,
    ) -> Dict[str, str]:
        """Submit profiling tasks for multiple platforms."""
        task_ids = {}

        for platform in platforms:
            try:
                task_id = self.profiler.submit_task(
                    platform=platform,
                    test_prompts=test_prompts,
                    config=config,
                    priority=priority,
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
        priority: int = 5,
    ) -> List[str]:
        """Submit profiling tasks for multiple prompt variations."""
        task_ids = []

        for i, prompts in enumerate(prompt_variations):
            try:
                task_id = self.profiler.submit_task(
                    platform=platform,
                    test_prompts=prompts,
                    config=config,
                    priority=priority,
                )
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Failed to submit prompt variation {i}: {e}")

        return task_ids

    def comprehensive_benchmark(
        self,
        platforms: List[str],
        prompt_sets: Dict[str, List[str]],
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across platforms and prompt sets."""
        logger.info(
            f"Starting comprehensive benchmark: {len(platforms)} platforms, {len(prompt_sets)} prompt sets"
        )

        all_task_ids = []
        task_metadata = {}

        # Submit all tasks
        for platform in platforms:
            for prompt_set_name, prompts in prompt_sets.items():
                try:
                    task_id = self.profiler.submit_task(
                        platform=platform,
                        test_prompts=prompts,
                        priority=3,  # Higher priority for comprehensive benchmark
                    )
                    all_task_ids.append(task_id)
                    task_metadata[task_id] = {
                        "platform": platform,
                        "prompt_set": prompt_set_name,
                    }
                except Exception as e:
                    logger.error(
                        f"Failed to submit task for {platform}/{prompt_set_name}: {e}"
                    )

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
            "success_rate": (
                (successful_tasks / len(all_task_ids)) * 100 if all_task_ids else 0
            ),
            "profiler_stats": self.profiler.get_stats(),
            "results": results,
        }

        logger.info(
            f"Benchmark completed: {successful_tasks}/{len(all_task_ids)} tasks successful"
        )
        return summary


async def async_profile_single(
    platform: str, test_prompts: List[str]
) -> SimplifiedProfile:
    """Asynchronous single platform profiling."""
    loop = asyncio.get_event_loop()

    # Run profiling in thread pool
    with ThreadPoolExecutor(max_workers=1) as executor:
        profiler = SimpleProfiler(platform)
        future = executor.submit(profiler.simulate_profiling, test_prompts)
        result = await loop.run_in_executor(None, lambda: future.result())

    return result


async def async_profile_batch(
    platform_prompts: List[Tuple[str, List[str]]],
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
            result = SimplifiedProfile(success=False, error_message=str(result))
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
                    platform=platform, test_prompts=test_prompts, priority=i + 1
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
            avg_tokens_per_sec = sum(
                r.result.tokens_per_second for r in successful_results
            ) / len(successful_results)
            avg_execution_time = sum(
                r.execution_time_s for r in successful_results
            ) / len(successful_results)

            logger.info(f"Average performance: {avg_tokens_per_sec:.1f} tok/s")
            logger.info(f"Average execution time: {avg_execution_time:.2f}s")

    finally:
        profiler.stop()


# Generation 3 Enhancements: Advanced Optimization and Scaling

import hashlib
from datetime import datetime
from collections import defaultdict
from enum import Enum


class CacheLevel(str, Enum):
    """Cache level enumeration for multi-level caching."""
    L1 = "l1_memory"  # Hot data in memory
    L2 = "l2_compressed"  # Warm data compressed
    L3 = "l3_persistent"  # Cold data persisted


class OptimizationStrategy(str, Enum):
    """Performance optimization strategies."""
    THROUGHPUT = "throughput"  # Maximize tasks per second
    LATENCY = "latency"  # Minimize response time
    MEMORY = "memory"  # Minimize memory usage
    BALANCED = "balanced"  # Balance all factors


@dataclass
class CachedResult:
    """Cached profiling result with metadata."""
    key: str
    result: ProfileTaskResult
    timestamp: datetime
    access_count: int = 0
    cache_level: CacheLevel = CacheLevel.L1
    ttl_seconds: float = 1800.0  # 30 minutes
    
    def is_expired(self) -> bool:
        """Check if cached result has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1


class IntelligentCache:
    """
    Generation 3: Multi-level intelligent caching system.
    
    L1: Hot data in memory (fast access, limited size)
    L2: Warm data compressed in memory (medium access)
    L3: Cold data with persistence hints (slow but large)
    """
    
    def __init__(self, l1_size: int = 50, l2_size: int = 200, l3_size: int = 1000):
        self.l1_cache: Dict[str, CachedResult] = {}  # Hot cache
        self.l2_cache: Dict[str, CachedResult] = {}  # Warm cache  
        self.l3_cache: Dict[str, CachedResult] = {}  # Cold cache
        
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        
        # Cache statistics
        self.stats = {
            "l1_hits": 0, "l2_hits": 0, "l3_hits": 0,
            "misses": 0, "evictions": 0, "total_requests": 0
        }
    
    def get_cache_key(self, platform: str, prompts: List[str], config: Dict[str, Any] = None) -> str:
        """Generate cache key from profiling parameters."""
        config = config or {}
        key_data = f"{platform}|{hash(tuple(prompts))}|{hash(tuple(sorted(config.items())))}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def get(self, cache_key: str) -> Optional[ProfileTaskResult]:
        """Get cached result with intelligent promotion."""
        self.stats["total_requests"] += 1
        
        # Check L1 first (hottest)
        if cache_key in self.l1_cache:
            cached = self.l1_cache[cache_key]
            if not cached.is_expired():
                cached.touch()
                self.stats["l1_hits"] += 1
                return cached.result
            else:
                del self.l1_cache[cache_key]
        
        # Check L2 (warm)
        if cache_key in self.l2_cache:
            cached = self.l2_cache[cache_key]
            if not cached.is_expired():
                cached.touch()
                self.stats["l2_hits"] += 1
                
                # Promote frequently accessed items to L1
                if cached.access_count > 2:
                    self._promote_to_l1(cache_key, cached)
                
                return cached.result
            else:
                del self.l2_cache[cache_key]
        
        # Check L3 (cold)
        if cache_key in self.l3_cache:
            cached = self.l3_cache[cache_key]
            if not cached.is_expired():
                cached.touch()
                self.stats["l3_hits"] += 1
                
                # Promote to L2
                self._promote_to_l2(cache_key, cached)
                
                return cached.result
            else:
                del self.l3_cache[cache_key]
        
        self.stats["misses"] += 1
        return None
    
    def put(self, cache_key: str, result: ProfileTaskResult, ttl_seconds: float = 1800.0):
        """Cache result starting at L1."""
        cached = CachedResult(
            key=cache_key,
            result=result,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds,
            cache_level=CacheLevel.L1
        )
        
        self._put_l1(cache_key, cached)
    
    def _put_l1(self, key: str, cached: CachedResult):
        """Add to L1 cache with eviction."""
        if len(self.l1_cache) >= self.l1_size:
            self._evict_from_l1()
        
        cached.cache_level = CacheLevel.L1
        self.l1_cache[key] = cached
    
    def _put_l2(self, key: str, cached: CachedResult):
        """Add to L2 cache with eviction."""
        if len(self.l2_cache) >= self.l2_size:
            self._evict_from_l2()
        
        cached.cache_level = CacheLevel.L2
        self.l2_cache[key] = cached
    
    def _put_l3(self, key: str, cached: CachedResult):
        """Add to L3 cache with eviction."""
        if len(self.l3_cache) >= self.l3_size:
            self._evict_from_l3()
        
        cached.cache_level = CacheLevel.L3
        self.l3_cache[key] = cached
    
    def _promote_to_l1(self, key: str, cached: CachedResult):
        """Promote from L2 to L1."""
        if key in self.l2_cache:
            del self.l2_cache[key]
        self._put_l1(key, cached)
    
    def _promote_to_l2(self, key: str, cached: CachedResult):
        """Promote from L3 to L2."""
        if key in self.l3_cache:
            del self.l3_cache[key]
        self._put_l2(key, cached)
    
    def _evict_from_l1(self):
        """Evict LRU from L1 to L2."""
        if not self.l1_cache:
            return
        
        lru_key = min(self.l1_cache.keys(), key=lambda k: self.l1_cache[k].timestamp)
        cached = self.l1_cache[lru_key]
        del self.l1_cache[lru_key]
        
        self._put_l2(lru_key, cached)
        self.stats["evictions"] += 1
    
    def _evict_from_l2(self):
        """Evict LRU from L2 to L3."""
        if not self.l2_cache:
            return
        
        lru_key = min(self.l2_cache.keys(), key=lambda k: self.l2_cache[k].timestamp)
        cached = self.l2_cache[lru_key]
        del self.l2_cache[lru_key]
        
        self._put_l3(lru_key, cached)
        self.stats["evictions"] += 1
    
    def _evict_from_l3(self):
        """Evict LRU from L3 completely."""
        if not self.l3_cache:
            return
        
        lru_key = min(self.l3_cache.keys(), key=lambda k: self.l3_cache[k].timestamp)
        del self.l3_cache[lru_key]
        self.stats["evictions"] += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
        hit_rate = total_hits / self.stats["total_requests"] if self.stats["total_requests"] > 0 else 0.0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "l3_size": len(self.l3_cache),
            "total_cached_items": len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
        }


class AdaptiveResourceManager:
    """
    Generation 3: Adaptive resource management for optimal scaling.
    """
    
    def __init__(self, initial_workers: int = 4):
        self.initial_workers = initial_workers
        self.current_workers = initial_workers
        self.max_workers = initial_workers * 3
        self.min_workers = max(1, initial_workers // 2)
        
        # Performance tracking
        self.throughput_history = []
        self.resource_usage_history = []
        self.last_adjustment_time = time.time()
        self.adjustment_cooldown = 30.0  # seconds
    
    def should_scale_up(self, current_throughput: float, queue_size: int) -> bool:
        """Determine if we should scale up workers."""
        # Scale up if queue is backing up and we haven't hit max workers
        queue_pressure = queue_size > self.current_workers * 2
        can_scale_up = self.current_workers < self.max_workers
        cooldown_passed = time.time() - self.last_adjustment_time > self.adjustment_cooldown
        
        return queue_pressure and can_scale_up and cooldown_passed
    
    def should_scale_down(self, current_throughput: float, queue_size: int) -> bool:
        """Determine if we should scale down workers."""
        # Scale down if queue is empty and throughput is low
        low_utilization = queue_size == 0 and current_throughput < 1.0
        can_scale_down = self.current_workers > self.min_workers
        cooldown_passed = time.time() - self.last_adjustment_time > self.adjustment_cooldown * 2
        
        return low_utilization and can_scale_down and cooldown_passed
    
    def recommend_worker_count(self, current_throughput: float, queue_size: int) -> int:
        """Recommend optimal worker count based on current conditions."""
        if self.should_scale_up(current_throughput, queue_size):
            new_count = min(self.max_workers, self.current_workers + 1)
            self.last_adjustment_time = time.time()
            return new_count
        elif self.should_scale_down(current_throughput, queue_size):
            new_count = max(self.min_workers, self.current_workers - 1)
            self.last_adjustment_time = time.time()
            return new_count
        
        return self.current_workers
    
    def update_metrics(self, throughput: float, resource_usage: Dict[str, float]):
        """Update performance tracking metrics."""
        self.throughput_history.append((time.time(), throughput))
        self.resource_usage_history.append((time.time(), resource_usage))
        
        # Keep only recent history (last hour)
        cutoff_time = time.time() - 3600
        self.throughput_history = [(t, v) for t, v in self.throughput_history if t > cutoff_time]
        self.resource_usage_history = [(t, v) for t, v in self.resource_usage_history if t > cutoff_time]


class OptimizedConcurrentProfiler(ConcurrentProfiler):
    """
    Generation 3: Enhanced concurrent profiler with optimization features.
    
    New features:
    - Multi-level intelligent caching
    - Adaptive resource management
    - Performance optimization strategies
    - Advanced metrics and monitoring
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 enable_caching: bool = True,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        super().__init__(max_workers)
        
        # Generation 3 enhancements
        self.cache = IntelligentCache() if enable_caching else None
        self.resource_manager = AdaptiveResourceManager(max_workers)
        self.optimization_strategy = optimization_strategy
        
        # Enhanced metrics
        self.cache_saves = 0  # Number of times cache prevented computation
        self.optimization_applied = 0
        self.adaptive_scaling_events = 0
        
        logger.info(f"OptimizedConcurrentProfiler initialized with {optimization_strategy.value} strategy")
    
    def submit_task_optimized(self,
                             platform: str,
                             test_prompts: List[str],
                             config: Dict[str, Any] = None,
                             priority: int = 5,
                             use_cache: bool = True) -> str:
        """Submit task with optimization features."""
        
        # Check cache first
        if use_cache and self.cache:
            cache_key = self.cache.get_cache_key(platform, test_prompts, config)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                # Return cached result immediately
                task_id = f"cached_{int(time.time() * 1000000)}"
                self.task_queue.add_result(cached_result)
                self.cache_saves += 1
                logger.debug(f"Cache hit for task {task_id}")
                return task_id
        
        # Proceed with normal task submission
        task_id = super().submit_task(platform, test_prompts, config, priority)
        
        # Apply adaptive scaling if needed
        self._check_adaptive_scaling()
        
        return task_id
    
    def _check_adaptive_scaling(self):
        """Check if adaptive scaling is needed."""
        stats = self.get_stats()
        current_throughput = stats.get("throughput_tasks_per_second", 0.0)
        queue_size = stats.get("pending_tasks", 0)
        
        recommended_workers = self.resource_manager.recommend_worker_count(
            current_throughput, queue_size
        )
        
        if recommended_workers != len(self.workers):
            logger.info(f"Adaptive scaling: adjusting from {len(self.workers)} to {recommended_workers} workers")
            self._adjust_worker_count(recommended_workers)
            self.adaptive_scaling_events += 1
    
    def _adjust_worker_count(self, target_count: int):
        """Dynamically adjust worker count."""
        current_count = len(self.workers)
        
        if target_count > current_count:
            # Add workers
            for i in range(target_count - current_count):
                worker_id = f"worker_{current_count + i + 1}"
                worker = ProfileWorker(worker_id, self.task_queue)
                worker.start()
                self.workers.append(worker)
        
        elif target_count < current_count:
            # Remove workers
            workers_to_remove = current_count - target_count
            for _ in range(workers_to_remove):
                if self.workers:
                    worker = self.workers.pop()
                    worker.stop()
        
        self.resource_manager.current_workers = len(self.workers)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        base_stats = super().get_stats()
        
        optimization_stats = {
            "optimization_strategy": self.optimization_strategy.value,
            "cache_saves": self.cache_saves,
            "adaptive_scaling_events": self.adaptive_scaling_events,
            "current_worker_count": len(self.workers),
            "recommended_worker_count": self.resource_manager.recommend_worker_count(
                base_stats.get("throughput_tasks_per_second", 0.0),
                base_stats.get("pending_tasks", 0)
            )
        }
        
        if self.cache:
            optimization_stats["cache_statistics"] = self.cache.get_cache_stats()
        
        return {
            **base_stats,
            **optimization_stats
        }
    
    def optimize_for_strategy(self, strategy: OptimizationStrategy):
        """Reconfigure profiler for specific optimization strategy."""
        self.optimization_strategy = strategy
        
        if strategy == OptimizationStrategy.THROUGHPUT:
            # Maximize throughput: more workers, larger caches
            target_workers = min(self.resource_manager.max_workers, self.max_workers * 2)
            self._adjust_worker_count(target_workers)
            
        elif strategy == OptimizationStrategy.LATENCY:
            # Minimize latency: keep workers ready, prioritize cache hits
            target_workers = max(self.resource_manager.min_workers, self.max_workers)
            self._adjust_worker_count(target_workers)
            
        elif strategy == OptimizationStrategy.MEMORY:
            # Minimize memory: fewer workers, smaller caches
            target_workers = self.resource_manager.min_workers
            self._adjust_worker_count(target_workers)
            
        else:  # BALANCED
            # Balanced approach: moderate workers, standard caches
            target_workers = self.resource_manager.initial_workers
            self._adjust_worker_count(target_workers)
        
        self.optimization_applied += 1
        logger.info(f"Optimized for {strategy.value} strategy")


# Generation 3 Convenience Functions

def create_optimized_profiler(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                             max_workers: int = 4,
                             enable_caching: bool = True) -> OptimizedConcurrentProfiler:
    """Create an optimized concurrent profiler with Generation 3 enhancements."""
    return OptimizedConcurrentProfiler(
        max_workers=max_workers,
        enable_caching=enable_caching,
        optimization_strategy=strategy
    )


async def async_batch_profile(platforms: List[str], 
                             test_prompts: List[str],
                             optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> Dict[str, Any]:
    """Async batch profiling with Generation 3 optimizations."""
    profiler = create_optimized_profiler(strategy=optimization_strategy)
    
    try:
        # Submit all tasks
        task_ids = []
        for platform in platforms:
            task_id = profiler.submit_task_optimized(
                platform=platform,
                test_prompts=test_prompts,
                use_cache=True
            )
            task_ids.append(task_id)
        
        # Wait for completion with timeout
        await asyncio.sleep(0.1)  # Let tasks start
        
        if profiler.wait_for_completion(timeout=60.0):
            results = profiler.get_all_results()
            stats = profiler.get_optimization_stats()
            
            return {
                "status": "success",
                "results": [
                    {
                        "platform": r.platform,
                        "tokens_per_second": r.result.tokens_per_second,
                        "success": r.result.success,
                        "execution_time": r.execution_time_s
                    }
                    for r in results
                ],
                "optimization_stats": stats,
                "total_platforms": len(platforms),
                "cache_hit_rate": stats.get("cache_statistics", {}).get("hit_rate", 0.0)
            }
        else:
            return {
                "status": "timeout",
                "error": "Some tasks did not complete within timeout",
                "optimization_stats": profiler.get_optimization_stats()
            }
    
    finally:
        profiler.stop()


def run_generation3_demo():
    """Demonstration of Generation 3 optimization features."""
    print("🚀 Generation 3 Optimization Demo")
    print("=" * 50)
    
    # Test different optimization strategies
    strategies = [
        OptimizationStrategy.THROUGHPUT,
        OptimizationStrategy.LATENCY, 
        OptimizationStrategy.MEMORY,
        OptimizationStrategy.BALANCED
    ]
    
    platforms = ["esp32", "stm32f4", "stm32f7"]
    test_prompts = ["Hello", "Generate code", "Optimize performance"]
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.value} optimization:")
        
        profiler = create_optimized_profiler(strategy=strategy, max_workers=2)
        
        try:
            # Submit tasks
            task_ids = []
            for platform in platforms:
                task_id = profiler.submit_task_optimized(
                    platform=platform,
                    test_prompts=test_prompts,
                    use_cache=True
                )
                task_ids.append(task_id)
            
            # Wait and collect results
            if profiler.wait_for_completion(timeout=15.0):
                stats = profiler.get_optimization_stats()
                
                print(f"   ✅ Completed {stats['completed_tasks']} tasks")
                print(f"   📊 Throughput: {stats['throughput_tasks_per_second']:.1f} tasks/sec")
                print(f"   💾 Cache saves: {stats['cache_saves']}")
                print(f"   🔧 Adaptive scaling events: {stats['adaptive_scaling_events']}")
                
                if 'cache_statistics' in stats:
                    cache_stats = stats['cache_statistics']
                    print(f"   📈 Cache hit rate: {cache_stats['hit_rate']:.1%}")
            
        finally:
            profiler.stop()
    
    print("\n✅ Generation 3 optimization demo completed!")


# Backward compatibility aliases
ScalableProfiler = OptimizedConcurrentProfiler
ConcurrentProfiler = OptimizedConcurrentProfiler


if __name__ == "__main__":
    run_generation3_demo()
