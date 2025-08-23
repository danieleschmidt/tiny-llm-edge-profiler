"""
Advanced Resource Optimization System for Generation 3
Provides comprehensive resource optimization capabilities including:
- Memory pool management with intelligent allocation strategies
- Connection pooling with auto-sizing and health monitoring
- Resource recycling and efficient cleanup mechanisms
- Adaptive resource allocation based on usage patterns
- Performance monitoring and auto-tuning of resource utilization
- Resource leak detection and prevention
"""

import time
import threading
import asyncio
import weakref
import gc
import psutil
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Callable,
    TypeVar,
    Generic,
    Set,
    Tuple,
    Union,
)
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import sqlite3
from queue import Queue, Empty, Full, PriorityQueue
import pickle
import mmap
import numpy as np

from .exceptions import TinyLLMProfilerError, ResourceError
from .logging_config import get_logger, PerformanceLogger
from .resource_pool import ResourcePool, ResourceFactory, ResourceInfo

logger = get_logger("resource_optimizer")
perf_logger = PerformanceLogger()

T = TypeVar("T")


@dataclass
class ResourceUsageStats:
    """Statistics about resource usage patterns."""

    total_requests: int = 0
    successful_allocations: int = 0
    failed_allocations: int = 0
    peak_usage: int = 0
    average_hold_time_seconds: float = 0.0
    usage_pattern: str = "unknown"  # bursty, steady, idle
    last_usage: Optional[datetime] = None

    # Memory-specific stats
    total_memory_allocated_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    memory_fragmentation_ratio: float = 0.0

    # Performance metrics
    avg_allocation_time_ms: float = 0.0
    avg_cleanup_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

    def update_allocation_stats(self, success: bool, allocation_time_ms: float):
        """Update allocation statistics."""
        self.total_requests += 1
        if success:
            self.successful_allocations += 1
        else:
            self.failed_allocations += 1

        # Update average allocation time
        if self.total_requests == 1:
            self.avg_allocation_time_ms = allocation_time_ms
        else:
            # Exponential moving average
            self.avg_allocation_time_ms = (
                0.9 * self.avg_allocation_time_ms + 0.1 * allocation_time_ms
            )

    def get_success_rate(self) -> float:
        """Get allocation success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_allocations / self.total_requests

    def detect_usage_pattern(self, recent_requests: List[datetime]) -> str:
        """Detect usage pattern from recent requests."""
        if not recent_requests or len(recent_requests) < 5:
            return "insufficient_data"

        # Calculate time intervals between requests
        intervals = []
        for i in range(1, len(recent_requests)):
            interval = (recent_requests[i] - recent_requests[i - 1]).total_seconds()
            intervals.append(interval)

        if not intervals:
            return "single_request"

        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        coefficient_of_variation = (
            std_interval / avg_interval if avg_interval > 0 else 0
        )

        # Classify pattern
        if coefficient_of_variation > 1.5:
            self.usage_pattern = "bursty"
        elif coefficient_of_variation < 0.5 and avg_interval < 60:  # Less than 1 minute
            self.usage_pattern = "steady"
        elif avg_interval > 300:  # More than 5 minutes
            self.usage_pattern = "idle"
        else:
            self.usage_pattern = "irregular"

        return self.usage_pattern


class AdaptiveMemoryPool:
    """Adaptive memory pool that adjusts size based on usage patterns."""

    def __init__(
        self,
        initial_size_mb: int = 64,
        max_size_mb: int = 512,
        block_size_bytes: int = 1024,
        growth_factor: float = 1.5,
        shrink_threshold: float = 0.3,
    ):
        self.initial_size_mb = initial_size_mb
        self.max_size_mb = max_size_mb
        self.block_size_bytes = block_size_bytes
        self.growth_factor = growth_factor
        self.shrink_threshold = shrink_threshold

        # Memory management
        self.current_size_bytes = initial_size_mb * 1024 * 1024
        self.allocated_blocks: Dict[int, Tuple[memoryview, datetime]] = {}
        self.free_blocks: List[memoryview] = []
        self.memory_buffer: Optional[bytearray] = None

        # Statistics
        self.stats = ResourceUsageStats()
        self.allocation_history: deque = deque(maxlen=1000)

        # Thread safety
        self._lock = threading.RLock()

        # Initialize memory pool
        self._initialize_pool()

        # Start monitoring
        self._monitoring_thread = threading.Thread(
            target=self._monitor_usage, daemon=True
        )
        self._monitoring_thread.start()

    def _initialize_pool(self):
        """Initialize the memory pool."""
        with self._lock:
            try:
                self.memory_buffer = bytearray(self.current_size_bytes)

                # Create initial free blocks
                num_blocks = self.current_size_bytes // self.block_size_bytes
                for i in range(num_blocks):
                    start_idx = i * self.block_size_bytes
                    end_idx = start_idx + self.block_size_bytes
                    block = memoryview(self.memory_buffer[start_idx:end_idx])
                    self.free_blocks.append(block)

                logger.info(
                    f"Initialized memory pool: {self.current_size_bytes // (1024*1024)}MB, {num_blocks} blocks"
                )

            except Exception as e:
                logger.error(f"Failed to initialize memory pool: {e}")
                raise ResourceError(
                    operation="initialize_memory_pool",
                    required_mb=self.initial_size_mb,
                    available_mb=0,
                ).with_message(f"Memory pool initialization failed: {e}")

    def allocate(self, size_bytes: int) -> Optional[int]:
        """
        Allocate memory block from pool.

        Args:
            size_bytes: Size of memory to allocate

        Returns:
            Block ID if successful, None if allocation failed
        """
        start_time = time.time()

        with self._lock:
            try:
                # Check if we have suitable free blocks
                required_blocks = (
                    size_bytes + self.block_size_bytes - 1
                ) // self.block_size_bytes

                if len(self.free_blocks) < required_blocks:
                    # Try to grow pool
                    if not self._grow_pool():
                        self.stats.update_allocation_stats(
                            False, (time.time() - start_time) * 1000
                        )
                        return None

                # Allocate blocks
                allocated_blocks = []
                for _ in range(required_blocks):
                    if self.free_blocks:
                        block = self.free_blocks.pop(0)
                        allocated_blocks.append(block)
                    else:
                        # Return already allocated blocks to free pool
                        self.free_blocks.extend(allocated_blocks)
                        self.stats.update_allocation_stats(
                            False, (time.time() - start_time) * 1000
                        )
                        return None

                # Create block ID and store allocation
                block_id = id(allocated_blocks[0])
                self.allocated_blocks[block_id] = (allocated_blocks[0], datetime.now())

                # Update statistics
                allocation_time_ms = (time.time() - start_time) * 1000
                self.stats.update_allocation_stats(True, allocation_time_ms)
                self.stats.peak_usage = max(
                    self.stats.peak_usage, len(self.allocated_blocks)
                )
                self.stats.total_memory_allocated_mb += size_bytes / (1024 * 1024)
                self.stats.last_usage = datetime.now()

                # Record allocation
                self.allocation_history.append(
                    {
                        "timestamp": datetime.now(),
                        "size_bytes": size_bytes,
                        "block_id": block_id,
                        "allocation_time_ms": allocation_time_ms,
                    }
                )

                logger.debug(f"Allocated {size_bytes} bytes (block_id: {block_id})")
                return block_id

            except Exception as e:
                logger.error(f"Memory allocation error: {e}")
                self.stats.update_allocation_stats(
                    False, (time.time() - start_time) * 1000
                )
                return None

    def deallocate(self, block_id: int) -> bool:
        """
        Deallocate memory block.

        Args:
            block_id: ID of block to deallocate

        Returns:
            True if successful
        """
        start_time = time.time()

        with self._lock:
            try:
                if block_id not in self.allocated_blocks:
                    logger.warning(f"Attempted to deallocate unknown block: {block_id}")
                    return False

                # Get block and allocation time
                block, allocation_time = self.allocated_blocks[block_id]

                # Calculate hold time
                hold_time = (datetime.now() - allocation_time).total_seconds()

                # Update average hold time
                if self.stats.average_hold_time_seconds == 0:
                    self.stats.average_hold_time_seconds = hold_time
                else:
                    self.stats.average_hold_time_seconds = (
                        0.9 * self.stats.average_hold_time_seconds + 0.1 * hold_time
                    )

                # Return block to free pool
                self.free_blocks.append(block)
                del self.allocated_blocks[block_id]

                # Update cleanup time statistics
                cleanup_time_ms = (time.time() - start_time) * 1000
                if self.stats.avg_cleanup_time_ms == 0:
                    self.stats.avg_cleanup_time_ms = cleanup_time_ms
                else:
                    self.stats.avg_cleanup_time_ms = (
                        0.9 * self.stats.avg_cleanup_time_ms + 0.1 * cleanup_time_ms
                    )

                logger.debug(
                    f"Deallocated block {block_id} (held for {hold_time:.2f}s)"
                )
                return True

            except Exception as e:
                logger.error(f"Memory deallocation error: {e}")
                return False

    def _grow_pool(self) -> bool:
        """Grow the memory pool if possible."""
        if self.current_size_bytes >= self.max_size_mb * 1024 * 1024:
            logger.warning("Memory pool at maximum size, cannot grow")
            return False

        new_size_bytes = min(
            int(self.current_size_bytes * self.growth_factor),
            self.max_size_mb * 1024 * 1024,
        )

        growth_bytes = new_size_bytes - self.current_size_bytes

        try:
            # Extend memory buffer
            self.memory_buffer.extend(bytearray(growth_bytes))

            # Create new free blocks
            new_blocks = growth_bytes // self.block_size_bytes
            for i in range(new_blocks):
                start_idx = self.current_size_bytes + (i * self.block_size_bytes)
                end_idx = start_idx + self.block_size_bytes
                block = memoryview(self.memory_buffer[start_idx:end_idx])
                self.free_blocks.append(block)

            self.current_size_bytes = new_size_bytes

            logger.info(
                f"Grew memory pool to {self.current_size_bytes // (1024*1024)}MB ({new_blocks} new blocks)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to grow memory pool: {e}")
            return False

    def _shrink_pool(self) -> bool:
        """Shrink the memory pool if usage is low."""
        # Only shrink if usage is below threshold
        usage_ratio = len(self.allocated_blocks) / max(
            len(self.allocated_blocks) + len(self.free_blocks), 1
        )

        if usage_ratio > self.shrink_threshold:
            return False

        # Don't shrink below initial size
        if self.current_size_bytes <= self.initial_size_mb * 1024 * 1024:
            return False

        # Calculate new size
        target_size_bytes = max(
            int(self.current_size_bytes / self.growth_factor),
            self.initial_size_mb * 1024 * 1024,
        )

        # For simplicity, we'll just log the intention to shrink
        # Actual shrinking would require more complex memory management
        logger.info(
            f"Pool could be shrunk from {self.current_size_bytes // (1024*1024)}MB to {target_size_bytes // (1024*1024)}MB"
        )

        return False  # Not implemented for safety

    def _monitor_usage(self):
        """Monitor pool usage and adjust size as needed."""
        while True:
            try:
                time.sleep(60)  # Check every minute

                with self._lock:
                    # Update usage pattern
                    recent_allocations = [
                        entry["timestamp"]
                        for entry in list(self.allocation_history)
                        if entry["timestamp"] > datetime.now() - timedelta(minutes=10)
                    ]

                    self.stats.detect_usage_pattern(recent_allocations)

                    # Update memory fragmentation ratio
                    total_blocks = len(self.allocated_blocks) + len(self.free_blocks)
                    if total_blocks > 0:
                        self.stats.memory_fragmentation_ratio = (
                            len(self.free_blocks) / total_blocks
                        )

                    # Consider shrinking if usage is consistently low
                    if (
                        self.stats.usage_pattern == "idle"
                        and len(self.allocated_blocks) / max(total_blocks, 1)
                        < self.shrink_threshold
                    ):
                        self._shrink_pool()

            except Exception as e:
                logger.error(f"Memory pool monitoring error: {e}")
                time.sleep(300)  # Wait longer on error

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_blocks = len(self.allocated_blocks) + len(self.free_blocks)

            return {
                "current_size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_mb,
                "allocated_blocks": len(self.allocated_blocks),
                "free_blocks": len(self.free_blocks),
                "total_blocks": total_blocks,
                "usage_ratio": len(self.allocated_blocks) / max(total_blocks, 1),
                "fragmentation_ratio": self.stats.memory_fragmentation_ratio,
                "allocation_stats": asdict(self.stats),
                "block_size_bytes": self.block_size_bytes,
            }


class AdaptiveConnectionPool(Generic[T]):
    """Adaptive connection pool that auto-sizes based on demand."""

    def __init__(
        self,
        factory: ResourceFactory[T],
        initial_size: int = 2,
        max_size: int = 20,
        min_size: int = 1,
        idle_timeout_seconds: int = 300,
        health_check_interval: int = 60,
    ):
        self.factory = factory
        self.initial_size = initial_size
        self.max_size = max_size
        self.min_size = min_size
        self.idle_timeout_seconds = idle_timeout_seconds
        self.health_check_interval = health_check_interval

        # Connection management
        self.connections: Dict[str, ResourceInfo] = {}
        self.available_connections: Queue = Queue()
        self.connection_waiters: List[threading.Event] = []

        # Statistics and monitoring
        self.stats = ResourceUsageStats()
        self.demand_history: deque = deque(maxlen=100)
        self.auto_sizing_enabled = True

        # Thread safety
        self._lock = threading.RLock()

        # Background tasks
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self._auto_sizing_thread = threading.Thread(
            target=self._auto_sizing_loop, daemon=True
        )

        # Initialize pool
        self._initialize_connections()

        # Start background tasks
        self._health_check_thread.start()
        self._auto_sizing_thread.start()

    def _initialize_connections(self):
        """Initialize the connection pool with initial connections."""
        for i in range(self.initial_size):
            try:
                self._create_connection()
            except Exception as e:
                logger.error(f"Failed to create initial connection {i}: {e}")

    def _create_connection(self) -> Optional[str]:
        """Create a new connection."""
        connection_id = f"conn_{int(time.time() * 1000000)}_{id(self)}"

        try:
            start_time = time.time()
            connection = self.factory.create_resource()
            creation_time = (time.time() - start_time) * 1000

            # Validate connection
            if not self.factory.validate_resource(connection):
                logger.error(f"Created connection failed validation: {connection_id}")
                self.factory.cleanup_resource(connection)
                return None

            # Store connection info
            connection_info = ResourceInfo(
                resource_id=connection_id,
                resource=connection,
                created_at=datetime.now(),
                metadata={"creation_time_ms": creation_time},
            )

            with self._lock:
                self.connections[connection_id] = connection_info
                self.available_connections.put(connection_id)

            logger.debug(f"Created connection: {connection_id}")
            return connection_id

        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            return None

    def acquire(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Acquire a connection from the pool.

        Args:
            timeout: Maximum time to wait for a connection

        Returns:
            Connection if available, None if timeout
        """
        start_time = time.time()

        # Record demand
        self.demand_history.append(datetime.now())

        try:
            # Try to get available connection
            connection_id = None

            if timeout is None:
                connection_id = self.available_connections.get()
            else:
                try:
                    connection_id = self.available_connections.get(timeout=timeout)
                except Empty:
                    # Try to create new connection if under limit
                    with self._lock:
                        if len(self.connections) < self.max_size:
                            new_connection_id = self._create_connection()
                            if new_connection_id:
                                connection_id = new_connection_id
                                self.available_connections.get_nowait()  # Remove from queue

            if connection_id and connection_id in self.connections:
                connection_info = self.connections[connection_id]

                # Validate connection before returning
                if self.factory.validate_resource(connection_info.resource):
                    connection_info.mark_used()

                    # Update stats
                    allocation_time_ms = (time.time() - start_time) * 1000
                    self.stats.update_allocation_stats(True, allocation_time_ms)

                    return connection_info.resource
                else:
                    # Connection is unhealthy, remove and retry
                    self._remove_connection(connection_id)
                    return self.acquire(timeout)

            # No connection available
            self.stats.update_allocation_stats(False, (time.time() - start_time) * 1000)
            return None

        except Exception as e:
            logger.error(f"Connection acquisition error: {e}")
            self.stats.update_allocation_stats(False, (time.time() - start_time) * 1000)
            return None

    def release(self, connection: T) -> bool:
        """
        Release a connection back to the pool.

        Args:
            connection: Connection to release

        Returns:
            True if successful
        """
        try:
            # Find connection info
            connection_info = None
            connection_id = None

            with self._lock:
                for cid, cinfo in self.connections.items():
                    if cinfo.resource is connection:
                        connection_info = cinfo
                        connection_id = cid
                        break

            if not connection_info:
                logger.warning("Attempted to release unknown connection")
                return False

            # Reset connection if possible
            if self.factory.reset_resource(connection):
                connection_info.mark_released()
                self.available_connections.put(connection_id)
                return True
            else:
                # Connection couldn't be reset, remove it
                self._remove_connection(connection_id)
                return False

        except Exception as e:
            logger.error(f"Connection release error: {e}")
            return False

    def _remove_connection(self, connection_id: str):
        """Remove a connection from the pool."""
        with self._lock:
            if connection_id in self.connections:
                connection_info = self.connections[connection_id]
                try:
                    self.factory.cleanup_resource(connection_info.resource)
                except Exception as e:
                    logger.error(f"Connection cleanup error: {e}")

                del self.connections[connection_id]
                logger.debug(f"Removed connection: {connection_id}")

    def _health_check_loop(self):
        """Periodic health check of connections."""
        while True:
            try:
                time.sleep(self.health_check_interval)

                unhealthy_connections = []

                with self._lock:
                    for connection_id, connection_info in self.connections.items():
                        if connection_info.in_use:
                            continue

                        # Check if connection is expired
                        if connection_info.is_expired(3600):  # 1 hour max age
                            unhealthy_connections.append(connection_id)
                            continue

                        # Check if connection is idle too long
                        if connection_info.is_idle(self.idle_timeout_seconds):
                            # Only remove if above minimum size
                            if len(self.connections) > self.min_size:
                                unhealthy_connections.append(connection_id)
                                continue

                        # Validate connection health
                        try:
                            if not self.factory.validate_resource(
                                connection_info.resource
                            ):
                                unhealthy_connections.append(connection_id)
                        except Exception as e:
                            logger.error(f"Health check error for {connection_id}: {e}")
                            unhealthy_connections.append(connection_id)

                # Remove unhealthy connections
                for connection_id in unhealthy_connections:
                    self._remove_connection(connection_id)

                # Ensure minimum pool size
                with self._lock:
                    while len(self.connections) < self.min_size:
                        if not self._create_connection():
                            break

            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(300)  # Wait longer on error

    def _auto_sizing_loop(self):
        """Auto-sizing logic based on demand patterns."""
        while True:
            try:
                time.sleep(120)  # Check every 2 minutes

                if not self.auto_sizing_enabled:
                    continue

                # Analyze recent demand
                now = datetime.now()
                recent_demand = [
                    timestamp
                    for timestamp in self.demand_history
                    if timestamp > now - timedelta(minutes=5)
                ]

                demand_rate = len(recent_demand) / 5.0  # Requests per minute

                with self._lock:
                    current_size = len(self.connections)
                    available_count = self.available_connections.qsize()
                    utilization = (current_size - available_count) / max(
                        current_size, 1
                    )

                # Scale up if high utilization and recent demand
                if (
                    utilization > 0.8
                    and demand_rate > 2
                    and current_size < self.max_size
                ):
                    self._create_connection()
                    logger.info(
                        f"Scaled up connection pool to {current_size + 1} (utilization: {utilization:.2f})"
                    )

                # Scale down if low utilization
                elif (
                    utilization < 0.3
                    and demand_rate < 0.5
                    and current_size > self.min_size
                ):
                    # Find least recently used idle connection to remove
                    oldest_connection_id = None
                    oldest_time = now

                    with self._lock:
                        for connection_id, connection_info in self.connections.items():
                            if (
                                not connection_info.in_use
                                and connection_info.last_used < oldest_time
                            ):
                                oldest_time = connection_info.last_used
                                oldest_connection_id = connection_id

                    if oldest_connection_id:
                        self._remove_connection(oldest_connection_id)
                        logger.info(
                            f"Scaled down connection pool to {current_size - 1} (utilization: {utilization:.2f})"
                        )

            except Exception as e:
                logger.error(f"Auto-sizing loop error: {e}")
                time.sleep(600)  # Wait longer on error

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            available_count = self.available_connections.qsize()
            total_connections = len(self.connections)

            return {
                "total_connections": total_connections,
                "available_connections": available_count,
                "in_use_connections": total_connections - available_count,
                "utilization_ratio": (total_connections - available_count)
                / max(total_connections, 1),
                "min_size": self.min_size,
                "max_size": self.max_size,
                "auto_sizing_enabled": self.auto_sizing_enabled,
                "allocation_stats": asdict(self.stats),
                "recent_demand_rate": len(
                    [
                        ts
                        for ts in self.demand_history
                        if ts > datetime.now() - timedelta(minutes=5)
                    ]
                )
                / 5.0,
            }


class ResourceLeakDetector:
    """Detects and reports resource leaks."""

    def __init__(self):
        self.tracked_resources: Dict[str, Dict[str, Any]] = {}
        self.leak_threshold_seconds = 3600  # 1 hour
        self.check_interval_seconds = 300  # 5 minutes

        # Start monitoring
        self._monitoring_thread = threading.Thread(
            target=self._monitor_leaks, daemon=True
        )
        self._monitoring_thread.start()

    def track_resource(
        self, resource_id: str, resource_type: str, stack_trace: Optional[str] = None
    ):
        """Start tracking a resource for potential leaks."""
        import traceback

        self.tracked_resources[resource_id] = {
            "resource_type": resource_type,
            "created_at": datetime.now(),
            "stack_trace": stack_trace or "".join(traceback.format_stack()),
            "warnings_issued": 0,
        }

    def untrack_resource(self, resource_id: str):
        """Stop tracking a resource (it was properly cleaned up)."""
        if resource_id in self.tracked_resources:
            del self.tracked_resources[resource_id]

    def _monitor_leaks(self):
        """Monitor for resource leaks."""
        while True:
            try:
                time.sleep(self.check_interval_seconds)

                current_time = datetime.now()
                leaked_resources = []

                for resource_id, resource_info in self.tracked_resources.items():
                    age = (current_time - resource_info["created_at"]).total_seconds()

                    if age > self.leak_threshold_seconds:
                        leaked_resources.append((resource_id, resource_info, age))

                # Report leaks
                for resource_id, resource_info, age in leaked_resources:
                    if resource_info["warnings_issued"] == 0:
                        logger.warning(
                            f"Potential resource leak detected: {resource_info['resource_type']} "
                            f"({resource_id}) has been alive for {age:.0f} seconds"
                        )

                        if resource_info["stack_trace"]:
                            logger.debug(
                                f"Creation stack trace for {resource_id}:\n{resource_info['stack_trace']}"
                            )

                    resource_info["warnings_issued"] += 1

                    # Remove from tracking after multiple warnings
                    if resource_info["warnings_issued"] > 5:
                        del self.tracked_resources[resource_id]

            except Exception as e:
                logger.error(f"Resource leak monitoring error: {e}")
                time.sleep(600)  # Wait longer on error

    def get_leak_report(self) -> Dict[str, Any]:
        """Get current leak detection report."""
        current_time = datetime.now()

        report = {
            "tracked_resources": len(self.tracked_resources),
            "potential_leaks": 0,
            "resource_types": defaultdict(int),
            "oldest_resource_age_seconds": 0,
        }

        oldest_age = 0

        for resource_id, resource_info in self.tracked_resources.items():
            age = (current_time - resource_info["created_at"]).total_seconds()
            resource_type = resource_info["resource_type"]

            report["resource_types"][resource_type] += 1

            if age > self.leak_threshold_seconds:
                report["potential_leaks"] += 1

            oldest_age = max(oldest_age, age)

        report["oldest_resource_age_seconds"] = oldest_age
        report["resource_types"] = dict(report["resource_types"])

        return report


class ResourceOptimizer:
    """Main resource optimizer coordinating all optimization components."""

    def __init__(
        self,
        enable_memory_pooling: bool = True,
        enable_connection_pooling: bool = True,
        enable_leak_detection: bool = True,
        memory_pool_size_mb: int = 128,
        connection_pool_size: int = 10,
    ):
        self.enable_memory_pooling = enable_memory_pooling
        self.enable_connection_pooling = enable_connection_pooling
        self.enable_leak_detection = enable_leak_detection

        # Initialize optimization components
        self.memory_pool: Optional[AdaptiveMemoryPool] = None
        self.connection_pools: Dict[str, AdaptiveConnectionPool] = {}
        self.leak_detector: Optional[ResourceLeakDetector] = None

        if enable_memory_pooling:
            self.memory_pool = AdaptiveMemoryPool(
                initial_size_mb=memory_pool_size_mb // 4,
                max_size_mb=memory_pool_size_mb,
            )

        if enable_leak_detection:
            self.leak_detector = ResourceLeakDetector()

        # Resource registry
        self.resource_registry: Dict[str, ResourceUsageStats] = {}

        # Performance metrics
        self.optimization_metrics = {
            "memory_saved_mb": 0.0,
            "connection_reuse_count": 0,
            "resource_leaks_detected": 0,
            "performance_improvements": [],
        }

        logger.info("Resource optimizer initialized")

    def create_connection_pool(
        self, pool_name: str, factory: ResourceFactory[T], **pool_config
    ) -> AdaptiveConnectionPool[T]:
        """Create a connection pool for a specific resource type."""
        if not self.enable_connection_pooling:
            raise RuntimeError("Connection pooling is disabled")

        pool = AdaptiveConnectionPool(factory, **pool_config)
        self.connection_pools[pool_name] = pool

        logger.info(f"Created connection pool: {pool_name}")
        return pool

    def get_connection_pool(self, pool_name: str) -> Optional[AdaptiveConnectionPool]:
        """Get an existing connection pool."""
        return self.connection_pools.get(pool_name)

    @contextmanager
    def allocate_memory(self, size_bytes: int):
        """Context manager for memory allocation."""
        if not self.memory_pool:
            # Fallback to regular allocation
            try:
                memory_buffer = bytearray(size_bytes)
                yield memory_buffer
            finally:
                del memory_buffer
            return

        block_id = None
        try:
            block_id = self.memory_pool.allocate(size_bytes)
            if block_id:
                # Track resource if leak detection is enabled
                if self.leak_detector:
                    self.leak_detector.track_resource(str(block_id), "memory_block")

                yield block_id
            else:
                # Fallback allocation
                memory_buffer = bytearray(size_bytes)
                yield memory_buffer

        finally:
            if block_id and self.memory_pool:
                self.memory_pool.deallocate(block_id)

                if self.leak_detector:
                    self.leak_detector.untrack_resource(str(block_id))

    @contextmanager
    def acquire_connection(self, pool_name: str, timeout: Optional[float] = None):
        """Context manager for connection acquisition."""
        if pool_name not in self.connection_pools:
            raise ValueError(f"Connection pool not found: {pool_name}")

        pool = self.connection_pools[pool_name]
        connection = None

        try:
            connection = pool.acquire(timeout)
            if not connection:
                raise ResourceError(
                    operation="acquire_connection", required_mb=0, available_mb=0
                ).with_message(f"Failed to acquire connection from pool: {pool_name}")

            # Track resource if leak detection is enabled
            if self.leak_detector:
                self.leak_detector.track_resource(
                    str(id(connection)), f"connection_{pool_name}"
                )

            yield connection

        finally:
            if connection:
                pool.release(connection)

                if self.leak_detector:
                    self.leak_detector.untrack_resource(str(id(connection)))

    def optimize_garbage_collection(self):
        """Optimize garbage collection settings."""
        try:
            # Force garbage collection
            collected = gc.collect()

            # Adjust GC thresholds based on memory usage
            memory_info = psutil.virtual_memory()
            memory_pressure = memory_info.percent / 100.0

            if memory_pressure > 0.8:
                # More aggressive GC under memory pressure
                gc.set_threshold(700, 10, 10)
            elif memory_pressure < 0.5:
                # Less aggressive GC when memory is abundant
                gc.set_threshold(2000, 20, 20)
            else:
                # Default thresholds
                gc.set_threshold(700, 10, 10)

            logger.debug(
                f"Garbage collection optimization: collected {collected} objects"
            )

        except Exception as e:
            logger.error(f"Garbage collection optimization error: {e}")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "memory_pool": None,
            "connection_pools": {},
            "leak_detection": None,
            "system_resources": {},
            "optimization_metrics": self.optimization_metrics.copy(),
        }

        # Memory pool stats
        if self.memory_pool:
            stats["memory_pool"] = self.memory_pool.get_stats()

        # Connection pool stats
        for pool_name, pool in self.connection_pools.items():
            stats["connection_pools"][pool_name] = pool.get_stats()

        # Leak detection stats
        if self.leak_detector:
            stats["leak_detection"] = self.leak_detector.get_leak_report()

        # System resources
        try:
            memory = psutil.virtual_memory()
            stats["system_resources"] = {
                "memory_total_mb": memory.total / (1024 * 1024),
                "memory_available_mb": memory.available / (1024 * 1024),
                "memory_percent_used": memory.percent,
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(),
            }
        except Exception as e:
            logger.error(f"Failed to get system resource stats: {e}")
            stats["system_resources"] = {"error": str(e)}

        return stats

    def shutdown(self):
        """Shutdown the resource optimizer."""
        logger.info("Shutting down resource optimizer...")

        # Clean up memory pool
        if self.memory_pool:
            # Memory pool cleanup is handled by garbage collection
            pass

        # Clean up connection pools
        for pool_name, pool in self.connection_pools.items():
            try:
                # Close all connections in the pool
                with pool._lock:
                    for connection_id in list(pool.connections.keys()):
                        pool._remove_connection(connection_id)

                logger.debug(f"Closed connection pool: {pool_name}")
            except Exception as e:
                logger.error(f"Error closing connection pool {pool_name}: {e}")

        self.connection_pools.clear()

        logger.info("Resource optimizer shutdown complete")


# Global resource optimizer instance
_global_resource_optimizer: Optional[ResourceOptimizer] = None


def get_resource_optimizer(**kwargs) -> ResourceOptimizer:
    """Get or create the global resource optimizer."""
    global _global_resource_optimizer

    if _global_resource_optimizer is None:
        _global_resource_optimizer = ResourceOptimizer(**kwargs)

    return _global_resource_optimizer


def optimize_memory_allocation(size_bytes: int):
    """Context manager for optimized memory allocation."""
    optimizer = get_resource_optimizer()
    return optimizer.allocate_memory(size_bytes)


def optimize_connection_pooling(
    pool_name: str, factory: ResourceFactory[T], **config
) -> AdaptiveConnectionPool[T]:
    """Create an optimized connection pool."""
    optimizer = get_resource_optimizer()
    return optimizer.create_connection_pool(pool_name, factory, **config)


def acquire_pooled_connection(pool_name: str, timeout: Optional[float] = None):
    """Context manager for acquiring pooled connections."""
    optimizer = get_resource_optimizer()
    return optimizer.acquire_connection(pool_name, timeout)


@contextmanager
def resource_monitoring(resource_type: str):
    """Context manager for monitoring resource usage."""
    optimizer = get_resource_optimizer()
    resource_id = f"{resource_type}_{int(time.time() * 1000000)}"

    if optimizer.leak_detector:
        optimizer.leak_detector.track_resource(resource_id, resource_type)

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

    try:
        yield resource_id

    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        perf_logger.log_resource_usage(
            resource_type=resource_type,
            resource_id=resource_id,
            duration_seconds=duration,
            memory_delta_mb=memory_delta,
        )

        if optimizer.leak_detector:
            optimizer.leak_detector.untrack_resource(resource_id)
