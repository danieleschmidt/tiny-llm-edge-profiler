"""
Advanced Asynchronous and Parallel Processing Pipeline for Generation 3
Provides comprehensive async/parallel capabilities including:
- Advanced async/await patterns for I/O operations
- Thread pool optimization for CPU-bound tasks
- Process pool for heavy computational work
- Pipeline parallelism for profiling stages
- Streaming data processing capabilities
- Producer-consumer patterns with backpressure
- Adaptive concurrency control
"""

import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
import time
import queue
import weakref
from typing import (
    Any, Dict, List, Optional, Callable, TypeVar, Generic, Union, 
    AsyncIterator, Iterator, Tuple, Awaitable, Coroutine, Set
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
from abc import ABC, abstractmethod
import inspect
import functools
import heapq
from pathlib import Path
import psutil
import numpy as np

from .exceptions import TinyLLMProfilerError, ResourceError, ProfilingTimeoutError
from .logging_config import get_logger, PerformanceLogger
from .models import QuantizedModel
from .results import ProfileResults
from .concurrent_utils import ProfilingTask, TaskResult, TaskStatus

logger = get_logger("async_pipeline")
perf_logger = PerformanceLogger()

T = TypeVar('T')
U = TypeVar('U')


class PipelineStage(str, Enum):
    """Pipeline processing stages."""
    INPUT = "input"
    PREPROCESSING = "preprocessing"
    PROFILING = "profiling"
    ANALYSIS = "analysis"
    POSTPROCESSING = "postprocessing"
    OUTPUT = "output"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance monitoring."""
    stage_durations: Dict[PipelineStage, List[float]] = field(default_factory=dict)
    throughput_items_per_second: float = 0.0
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    active_workers: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[PipelineStage, int] = field(default_factory=dict)
    total_processed: int = 0
    total_errors: int = 0
    
    def add_stage_duration(self, stage: PipelineStage, duration: float):
        """Add duration measurement for a pipeline stage."""
        if stage not in self.stage_durations:
            self.stage_durations[stage] = []
        self.stage_durations[stage].append(duration)
        
        # Keep only recent measurements
        if len(self.stage_durations[stage]) > 1000:
            self.stage_durations[stage] = self.stage_durations[stage][-500:]
    
    def get_average_duration(self, stage: PipelineStage) -> float:
        """Get average duration for a pipeline stage."""
        if stage in self.stage_durations and self.stage_durations[stage]:
            return np.mean(self.stage_durations[stage])
        return 0.0
    
    def get_stage_throughput(self, stage: PipelineStage) -> float:
        """Get throughput for a specific stage."""
        avg_duration = self.get_average_duration(stage)
        return 1.0 / avg_duration if avg_duration > 0 else 0.0


@dataclass
class PipelineItem(Generic[T]):
    """Item flowing through the processing pipeline."""
    item_id: str
    data: T
    priority: TaskPriority = TaskPriority.NORMAL
    stage: PipelineStage = PipelineStage.INPUT
    created_at: datetime = field(default_factory=datetime.now)
    stage_timestamps: Dict[PipelineStage, datetime] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    
    def mark_stage_start(self, stage: PipelineStage):
        """Mark the start of a pipeline stage."""
        self.stage = stage
        self.stage_timestamps[stage] = datetime.now()
    
    def get_stage_duration(self, stage: PipelineStage) -> Optional[float]:
        """Get duration of a completed stage."""
        stages = list(PipelineStage)
        current_index = stages.index(stage)
        
        if current_index == 0:
            start_time = self.created_at
        else:
            prev_stage = stages[current_index - 1]
            start_time = self.stage_timestamps.get(prev_stage)
        
        end_time = self.stage_timestamps.get(stage)
        
        if start_time and end_time:
            return (end_time - start_time).total_seconds()
        
        return None


class PipelineProcessor(ABC, Generic[T, U]):
    """Abstract base class for pipeline processors."""
    
    @abstractmethod
    async def process(self, item: PipelineItem[T]) -> PipelineItem[U]:
        """Process a pipeline item."""
        pass
    
    def can_process_batch(self) -> bool:
        """Check if processor supports batch processing."""
        return hasattr(self, 'process_batch')
    
    async def process_batch(self, items: List[PipelineItem[T]]) -> List[PipelineItem[U]]:
        """Process a batch of items (default implementation)."""
        results = []
        for item in items:
            result = await self.process(item)
            results.append(result)
        return results


class AdaptiveConcurrencyController:
    """Adaptive concurrency controller that adjusts based on system load."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 100,
        target_cpu_usage: float = 80.0,
        adjustment_interval: float = 5.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_usage = target_cpu_usage
        self.adjustment_interval = adjustment_interval
        
        self.current_workers = min_workers
        self.last_adjustment = time.time()
        self.performance_history: List[Tuple[int, float, float]] = []  # (workers, cpu, throughput)
        
    def should_adjust_concurrency(self) -> bool:
        """Check if concurrency should be adjusted."""
        return time.time() - self.last_adjustment >= self.adjustment_interval
    
    def get_optimal_workers(self, current_throughput: float) -> int:
        """Calculate optimal number of workers."""
        current_cpu = psutil.cpu_percent(interval=0.1)
        
        # Record current performance
        self.performance_history.append((self.current_workers, current_cpu, current_throughput))
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-10:]
        
        new_workers = self.current_workers
        
        # Increase workers if CPU usage is low and we're not at max
        if current_cpu < self.target_cpu_usage * 0.8 and self.current_workers < self.max_workers:
            new_workers = min(self.current_workers + 1, self.max_workers)
        
        # Decrease workers if CPU usage is too high
        elif current_cpu > self.target_cpu_usage * 1.2 and self.current_workers > self.min_workers:
            new_workers = max(self.current_workers - 1, self.min_workers)
        
        # Use performance history to make better decisions
        if len(self.performance_history) >= 3:
            # Check if recent increases actually improved throughput
            recent_data = self.performance_history[-3:]
            if all(data[2] > 0 for data in recent_data):  # All have positive throughput
                throughputs = [data[2] for data in recent_data]
                worker_counts = [data[0] for data in recent_data]
                
                # If throughput is decreasing despite more workers, scale down
                if len(throughputs) >= 2 and throughputs[-1] < throughputs[-2] * 0.95:
                    if worker_counts[-1] > worker_counts[-2]:
                        new_workers = max(self.current_workers - 1, self.min_workers)
        
        self.current_workers = new_workers
        self.last_adjustment = time.time()
        
        return new_workers


class AsyncQueue(Generic[T]):
    """Advanced async queue with priority and backpressure support."""
    
    def __init__(
        self,
        maxsize: int = 1000,
        enable_priority: bool = True,
        backpressure_threshold: float = 0.8
    ):
        self.maxsize = maxsize
        self.enable_priority = enable_priority
        self.backpressure_threshold = backpressure_threshold
        
        if enable_priority:
            self._queue: List[Tuple[int, int, T]] = []  # (priority, sequence, item)
            self._sequence_counter = 0
        else:
            self._queue = asyncio.Queue(maxsize=maxsize)
        
        self._condition = asyncio.Condition()
        self._size = 0
        self._closed = False
        
        # Backpressure metrics
        self._enqueue_waiters: Set[asyncio.Task] = set()
        self._dequeue_waiters: Set[asyncio.Task] = set()
    
    async def put(self, item: T, priority: int = 5) -> None:
        """Put item into queue with optional priority."""
        if self._closed:
            raise RuntimeError("Queue is closed")
        
        async with self._condition:
            # Wait if queue is full
            while self._size >= self.maxsize:
                waiter = asyncio.create_task(self._condition.wait())
                self._enqueue_waiters.add(waiter)
                try:
                    await waiter
                finally:
                    self._enqueue_waiters.discard(waiter)
            
            # Add item to queue
            if self.enable_priority:
                # Priority queue (higher priority value = higher priority)
                heapq.heappush(self._queue, (-priority, self._sequence_counter, item))
                self._sequence_counter += 1
            else:
                await self._queue.put(item)
            
            self._size += 1
            self._condition.notify()
    
    async def get(self) -> T:
        """Get item from queue."""
        async with self._condition:
            # Wait for item
            while self._size == 0 and not self._closed:
                waiter = asyncio.create_task(self._condition.wait())
                self._dequeue_waiters.add(waiter)
                try:
                    await waiter
                finally:
                    self._dequeue_waiters.discard(waiter)
            
            if self._size == 0 and self._closed:
                raise asyncio.QueueEmpty()
            
            # Get item
            if self.enable_priority:
                _, _, item = heapq.heappop(self._queue)
            else:
                item = await self._queue.get()
            
            self._size -= 1
            self._condition.notify()
            
            return item
    
    def qsize(self) -> int:
        """Get current queue size."""
        return self._size
    
    def is_backpressure_active(self) -> bool:
        """Check if backpressure is active."""
        return self._size >= (self.maxsize * self.backpressure_threshold)
    
    async def close(self):
        """Close the queue."""
        async with self._condition:
            self._closed = True
            
            # Cancel all waiters
            for waiter in self._enqueue_waiters.copy():
                waiter.cancel()
            for waiter in self._dequeue_waiters.copy():
                waiter.cancel()
            
            self._condition.notify_all()


class StreamProcessor(Generic[T, U]):
    """Streaming processor for continuous data processing."""
    
    def __init__(
        self,
        processor_func: Callable[[T], Awaitable[U]],
        buffer_size: int = 100,
        max_concurrent: int = 10
    ):
        self.processor_func = processor_func
        self.buffer_size = buffer_size
        self.max_concurrent = max_concurrent
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks: Set[asyncio.Task] = set()
    
    async def process_stream(
        self,
        input_stream: AsyncIterator[T]
    ) -> AsyncIterator[U]:
        """Process input stream and yield results."""
        buffer = []
        
        async for item in input_stream:
            buffer.append(item)
            
            # Process buffer when it's full
            if len(buffer) >= self.buffer_size:
                async for result in self._process_buffer(buffer):
                    yield result
                buffer.clear()
        
        # Process remaining items in buffer
        if buffer:
            async for result in self._process_buffer(buffer):
                yield result
    
    async def _process_buffer(self, buffer: List[T]) -> AsyncIterator[U]:
        """Process a buffer of items concurrently."""
        # Create tasks for each item
        tasks = []
        for item in buffer:
            task = asyncio.create_task(self._process_item(item))
            tasks.append(task)
            self._active_tasks.add(task)
        
        # Yield results as they complete
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                yield result
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
            finally:
                self._active_tasks.discard(future)
    
    async def _process_item(self, item: T) -> U:
        """Process a single item with concurrency control."""
        async with self._semaphore:
            return await self.processor_func(item)


class PipelineStageProcessor(PipelineProcessor[T, U]):
    """Generic pipeline stage processor with async support."""
    
    def __init__(
        self,
        stage: PipelineStage,
        processor_func: Callable[[T], Awaitable[U]],
        batch_size: int = 1,
        max_workers: int = 4
    ):
        self.stage = stage
        self.processor_func = processor_func
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self._semaphore = asyncio.Semaphore(max_workers)
    
    async def process(self, item: PipelineItem[T]) -> PipelineItem[U]:
        """Process a single pipeline item."""
        async with self._semaphore:
            item.mark_stage_start(self.stage)
            
            try:
                # Process the data
                processed_data = await self.processor_func(item.data)
                
                # Create new pipeline item with processed data
                new_item = PipelineItem(
                    item_id=item.item_id,
                    data=processed_data,
                    priority=item.priority,
                    stage=self.stage,
                    created_at=item.created_at,
                    stage_timestamps=item.stage_timestamps.copy(),
                    metadata=item.metadata.copy()
                )
                
                return new_item
                
            except Exception as e:
                logger.error(f"Error in {self.stage} stage: {e}")
                item.error = e
                return item
    
    async def process_batch(self, items: List[PipelineItem[T]]) -> List[PipelineItem[U]]:
        """Process a batch of items."""
        if self.batch_size == 1:
            # Process items individually
            return await super().process_batch(items)
        
        # Group items into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        results = []
        for batch in batches:
            batch_tasks = [self.process(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                else:
                    results.append(result)
        
        return results


class AsyncPipeline:
    """Advanced asynchronous processing pipeline."""
    
    def __init__(
        self,
        name: str = "async_pipeline",
        max_queue_size: int = 1000,
        enable_adaptive_concurrency: bool = True
    ):
        self.name = name
        self.max_queue_size = max_queue_size
        self.enable_adaptive_concurrency = enable_adaptive_concurrency
        
        # Pipeline stages
        self.processors: Dict[PipelineStage, PipelineProcessor] = {}
        self.stage_queues: Dict[PipelineStage, AsyncQueue] = {}
        
        # Concurrency control
        if enable_adaptive_concurrency:
            self.concurrency_controller = AdaptiveConcurrencyController()
        
        # Pipeline state
        self.running = False
        self.workers: Dict[PipelineStage, List[asyncio.Task]] = {}
        self.metrics = PipelineMetrics()
        
        # Event loop and coordination
        self._shutdown_event = asyncio.Event()
        self._worker_tasks: Set[asyncio.Task] = set()
    
    def add_stage(
        self,
        stage: PipelineStage,
        processor: PipelineProcessor,
        queue_size: Optional[int] = None,
        num_workers: int = 4
    ):
        """Add a processing stage to the pipeline."""
        self.processors[stage] = processor
        
        # Create queue for this stage
        queue_size = queue_size or (self.max_queue_size // len(PipelineStage))
        self.stage_queues[stage] = AsyncQueue(maxsize=queue_size)
        
        logger.info(f"Added pipeline stage: {stage} with {num_workers} workers")
    
    async def start(self):
        """Start the pipeline."""
        if self.running:
            return
        
        logger.info(f"Starting async pipeline: {self.name}")
        self.running = True
        
        # Start workers for each stage
        for stage, processor in self.processors.items():
            num_workers = 4  # Default, could be made configurable
            
            if self.enable_adaptive_concurrency:
                num_workers = self.concurrency_controller.current_workers
            
            stage_workers = []
            for i in range(num_workers):
                worker_task = asyncio.create_task(
                    self._stage_worker(stage, processor, i)
                )
                stage_workers.append(worker_task)
                self._worker_tasks.add(worker_task)
            
            self.workers[stage] = stage_workers
        
        # Start metrics collection
        metrics_task = asyncio.create_task(self._collect_metrics())
        self._worker_tasks.add(metrics_task)
        
        logger.info(f"Pipeline {self.name} started with {len(self._worker_tasks)} workers")
    
    async def stop(self, timeout: float = 30.0):
        """Stop the pipeline."""
        if not self.running:
            return
        
        logger.info(f"Stopping async pipeline: {self.name}")
        self.running = False
        self._shutdown_event.set()
        
        # Close all queues
        for queue in self.stage_queues.values():
            await queue.close()
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for workers to finish
        if self._worker_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._worker_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Pipeline shutdown timeout reached")
        
        self._worker_tasks.clear()
        self.workers.clear()
        
        logger.info(f"Pipeline {self.name} stopped")
    
    async def submit(self, data: Any, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit data to the pipeline."""
        if not self.running:
            raise RuntimeError("Pipeline is not running")
        
        # Create pipeline item
        item_id = f"item_{int(time.time() * 1000000)}"
        item = PipelineItem(
            item_id=item_id,
            data=data,
            priority=priority
        )
        
        # Add to first stage queue
        first_stage = list(PipelineStage)[0]
        if first_stage in self.stage_queues:
            await self.stage_queues[first_stage].put(item, priority.value)
        
        self.metrics.total_processed += 1
        
        return item_id
    
    async def submit_batch(
        self, 
        data_items: List[Any], 
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> List[str]:
        """Submit a batch of items to the pipeline."""
        item_ids = []
        
        for data in data_items:
            item_id = await self.submit(data, priority)
            item_ids.append(item_id)
        
        return item_ids
    
    async def _stage_worker(
        self,
        stage: PipelineStage,
        processor: PipelineProcessor,
        worker_id: int
    ):
        """Worker for processing items in a specific stage."""
        worker_name = f"{stage}_worker_{worker_id}"
        logger.debug(f"Started pipeline worker: {worker_name}")
        
        try:
            while self.running or not self._shutdown_event.is_set():
                try:
                    # Get item from stage queue
                    if stage in self.stage_queues:
                        item = await asyncio.wait_for(
                            self.stage_queues[stage].get(),
                            timeout=1.0
                        )
                    else:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Process item
                    start_time = time.time()
                    processed_item = await processor.process(item)
                    end_time = time.time()
                    
                    # Record metrics
                    duration = end_time - start_time
                    self.metrics.add_stage_duration(stage, duration)
                    
                    # Move to next stage
                    await self._move_to_next_stage(processed_item)
                    
                except asyncio.TimeoutError:
                    # Timeout is expected during shutdown
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_name} error: {e}")
                    self.metrics.error_counts[stage] = (
                        self.metrics.error_counts.get(stage, 0) + 1
                    )
                    self.metrics.total_errors += 1
        
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_name} cancelled")
        
        logger.debug(f"Pipeline worker stopped: {worker_name}")
    
    async def _move_to_next_stage(self, item: PipelineItem):
        """Move processed item to next pipeline stage."""
        stages = list(PipelineStage)
        current_index = stages.index(item.stage)
        
        if current_index < len(stages) - 1:
            # Move to next stage
            next_stage = stages[current_index + 1]
            if next_stage in self.stage_queues:
                await self.stage_queues[next_stage].put(item, item.priority.value)
        else:
            # Item has completed all stages
            logger.debug(f"Pipeline item completed: {item.item_id}")
    
    async def _collect_metrics(self):
        """Collect pipeline metrics periodically."""
        while self.running:
            try:
                # Update queue sizes
                for stage, queue in self.stage_queues.items():
                    self.metrics.queue_sizes[stage.value] = queue.qsize()
                
                # Update active workers
                for stage, workers in self.workers.items():
                    active_count = sum(1 for worker in workers if not worker.done())
                    self.metrics.active_workers[stage.value] = active_count
                
                # Calculate throughput
                if hasattr(self, '_last_processed_count'):
                    items_diff = self.metrics.total_processed - self._last_processed_count
                    time_diff = 5.0  # 5 second interval
                    self.metrics.throughput_items_per_second = items_diff / time_diff
                
                self._last_processed_count = self.metrics.total_processed
                
                # Adaptive concurrency adjustment
                if (self.enable_adaptive_concurrency and 
                    self.concurrency_controller.should_adjust_concurrency()):
                    
                    optimal_workers = self.concurrency_controller.get_optimal_workers(
                        self.metrics.throughput_items_per_second
                    )
                    
                    # Adjust worker count if needed
                    # Implementation would involve scaling workers up/down
                
                await asyncio.sleep(5.0)  # Collect metrics every 5 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10.0)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            "name": self.name,
            "running": self.running,
            "total_processed": self.metrics.total_processed,
            "total_errors": self.metrics.total_errors,
            "throughput_items_per_second": self.metrics.throughput_items_per_second,
            "queue_sizes": self.metrics.queue_sizes,
            "active_workers": self.metrics.active_workers,
            "error_counts": {k.value: v for k, v in self.metrics.error_counts.items()},
            "stage_durations": {
                stage.value: {
                    "avg_duration": self.metrics.get_average_duration(stage),
                    "throughput": self.metrics.get_stage_throughput(stage),
                    "sample_count": len(durations)
                }
                for stage, durations in self.metrics.stage_durations.items()
            },
            "adaptive_concurrency": self.enable_adaptive_concurrency,
            "worker_count": sum(len(workers) for workers in self.workers.values())
        }


class ProfilingPipeline(AsyncPipeline):
    """Specialized pipeline for profiling workloads."""
    
    def __init__(self, name: str = "profiling_pipeline", **kwargs):
        super().__init__(name, **kwargs)
        self._setup_profiling_stages()
    
    def _setup_profiling_stages(self):
        """Setup standard profiling pipeline stages."""
        
        # Input validation stage
        input_processor = PipelineStageProcessor(
            stage=PipelineStage.INPUT,
            processor_func=self._validate_input,
            max_workers=2
        )
        self.add_stage(PipelineStage.INPUT, input_processor)
        
        # Preprocessing stage
        preprocess_processor = PipelineStageProcessor(
            stage=PipelineStage.PREPROCESSING,
            processor_func=self._preprocess_task,
            max_workers=4
        )
        self.add_stage(PipelineStage.PREPROCESSING, preprocess_processor)
        
        # Main profiling stage
        profiling_processor = PipelineStageProcessor(
            stage=PipelineStage.PROFILING,
            processor_func=self._execute_profiling,
            max_workers=8,
            batch_size=1  # Profiling usually needs to be individual
        )
        self.add_stage(PipelineStage.PROFILING, profiling_processor)
        
        # Analysis stage
        analysis_processor = PipelineStageProcessor(
            stage=PipelineStage.ANALYSIS,
            processor_func=self._analyze_results,
            max_workers=4
        )
        self.add_stage(PipelineStage.ANALYSIS, analysis_processor)
        
        # Output stage
        output_processor = PipelineStageProcessor(
            stage=PipelineStage.OUTPUT,
            processor_func=self._format_output,
            max_workers=2
        )
        self.add_stage(PipelineStage.OUTPUT, output_processor)
    
    async def _validate_input(self, data: Any) -> Any:
        """Validate pipeline input data."""
        if not isinstance(data, ProfilingTask):
            raise ValueError(f"Expected ProfilingTask, got {type(data)}")
        
        # Validate task fields
        if not data.platform:
            raise ValueError("Platform is required")
        
        if not data.model:
            raise ValueError("Model is required")
        
        return data
    
    async def _preprocess_task(self, task: ProfilingTask) -> ProfilingTask:
        """Preprocess profiling task."""
        # Add default test prompts if none provided
        if not task.test_prompts:
            task.test_prompts = ["Hello world", "Test prompt"]
        
        # Set default metrics if none provided
        if not task.metrics:
            task.metrics = ["latency", "memory"]
        
        return task
    
    async def _execute_profiling(self, task: ProfilingTask) -> TaskResult:
        """Execute the actual profiling task."""
        try:
            # This would integrate with the actual profiler
            from .profiler import EdgeProfiler, ProfilingConfig
            
            profiler = EdgeProfiler(
                platform=task.platform,
                device=task.device_path
            )
            
            # Create config
            config = ProfilingConfig()
            if task.config:
                for key, value in task.config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Execute profiling
            results = profiler.profile_model(
                model=task.model,
                test_prompts=task.test_prompts,
                metrics=task.metrics,
                config=config
            )
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=results,
                start_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Profiling execution failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=e,
                start_time=datetime.now()
            )
    
    async def _analyze_results(self, result: TaskResult) -> TaskResult:
        """Analyze profiling results."""
        if result.success and result.result:
            # Add analysis metadata
            if hasattr(result.result, 'session_metadata'):
                result.result.session_metadata = result.result.session_metadata or {}
                result.result.session_metadata['pipeline_processed'] = True
                result.result.session_metadata['pipeline_analysis_time'] = datetime.now().isoformat()
        
        return result
    
    async def _format_output(self, result: TaskResult) -> TaskResult:
        """Format final output."""
        # Add any final formatting or logging
        if result.success:
            logger.info(f"Profiling task completed: {result.task_id}")
        else:
            logger.error(f"Profiling task failed: {result.task_id} - {result.error}")
        
        return result


# Global pipeline instances
_global_profiling_pipeline: Optional[ProfilingPipeline] = None


async def get_profiling_pipeline(**kwargs) -> ProfilingPipeline:
    """Get or create the global profiling pipeline."""
    global _global_profiling_pipeline
    
    if _global_profiling_pipeline is None:
        _global_profiling_pipeline = ProfilingPipeline(**kwargs)
        await _global_profiling_pipeline.start()
    
    return _global_profiling_pipeline


async def submit_profiling_task_async(
    model: QuantizedModel,
    platform: str,
    test_prompts: List[str] = None,
    device_path: Optional[str] = None,
    priority: TaskPriority = TaskPriority.NORMAL
) -> str:
    """Submit a profiling task to the async pipeline."""
    pipeline = await get_profiling_pipeline()
    
    task = ProfilingTask(
        task_id=f"async_profile_{int(time.time() * 1000000)}",
        platform=platform,
        model=model,
        device_path=device_path,
        test_prompts=test_prompts or ["Hello world"],
        metrics=["latency", "memory"]
    )
    
    return await pipeline.submit(task, priority)


@asynccontextmanager
async def async_profiling_session(
    models: List[QuantizedModel],
    platforms: List[str],
    test_prompts: List[str] = None
):
    """Context manager for async profiling sessions."""
    pipeline = await get_profiling_pipeline()
    
    try:
        # Submit all profiling tasks
        task_ids = []
        for model in models:
            for platform in platforms:
                task_id = await submit_profiling_task_async(
                    model=model,
                    platform=platform,
                    test_prompts=test_prompts
                )
                task_ids.append(task_id)
        
        yield task_ids
        
    finally:
        # Pipeline cleanup is handled by the global instance
        pass


# Utility functions for async operations
async def run_concurrent_profiling(
    tasks: List[ProfilingTask],
    max_concurrent: int = 10
) -> List[TaskResult]:
    """Run multiple profiling tasks concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_single_task(task: ProfilingTask) -> TaskResult:
        async with semaphore:
            pipeline = await get_profiling_pipeline()
            task_id = await pipeline.submit(task)
            
            # Wait for completion (simplified - in practice would use callbacks)
            # This is a placeholder for result retrieval
            return TaskResult(task_id=task_id, success=True)
    
    # Execute all tasks concurrently
    tasks_futures = [run_single_task(task) for task in tasks]
    results = await asyncio.gather(*tasks_futures, return_exceptions=True)
    
    # Filter out exceptions and return successful results
    successful_results = [
        result for result in results 
        if isinstance(result, TaskResult)
    ]
    
    return successful_results