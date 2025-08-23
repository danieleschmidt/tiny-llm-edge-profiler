"""
Resource pooling and management for the Tiny LLM Edge Profiler.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, ContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from queue import Queue, Empty, Full
from abc import ABC, abstractmethod
import weakref

from .exceptions import TinyLLMProfilerError, ResourceError
from .logging_config import get_logger
from .platforms import PlatformManager

logger = get_logger("resource_pool")

T = TypeVar("T")


@dataclass
class ResourceInfo:
    """Information about a pooled resource."""

    resource_id: str
    resource: Any
    created_at: datetime
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    in_use: bool = False
    healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_used(self) -> None:
        """Mark resource as used."""
        self.last_used = datetime.now()
        self.use_count += 1
        self.in_use = True

    def mark_released(self) -> None:
        """Mark resource as released."""
        self.in_use = False

    def is_expired(self, max_age_seconds: int) -> bool:
        """Check if resource has expired."""
        return (datetime.now() - self.created_at).total_seconds() > max_age_seconds

    def is_idle(self, max_idle_seconds: int) -> bool:
        """Check if resource has been idle too long."""
        return (datetime.now() - self.last_used).total_seconds() > max_idle_seconds


class ResourceFactory(ABC, Generic[T]):
    """Abstract factory for creating pooled resources."""

    @abstractmethod
    def create_resource(self, **kwargs) -> T:
        """Create a new resource instance."""
        pass

    @abstractmethod
    def validate_resource(self, resource: T) -> bool:
        """Validate that a resource is healthy and usable."""
        pass

    @abstractmethod
    def cleanup_resource(self, resource: T) -> None:
        """Clean up a resource before destroying it."""
        pass

    def reset_resource(self, resource: T) -> bool:
        """Reset a resource to a clean state. Return True if successful."""
        return True


class ResourcePool(Generic[T]):
    """
    Generic resource pool with automatic cleanup and health monitoring.
    """

    def __init__(
        self,
        factory: ResourceFactory[T],
        min_size: int = 1,
        max_size: int = 10,
        max_age_seconds: int = 3600,  # 1 hour
        max_idle_seconds: int = 300,  # 5 minutes
        health_check_interval: int = 60,  # 1 minute
        creation_timeout: float = 30.0,
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.max_idle_seconds = max_idle_seconds
        self.health_check_interval = health_check_interval
        self.creation_timeout = creation_timeout

        # Resource management
        self._resources: Dict[str, ResourceInfo] = {}
        self._available: Queue = Queue(maxsize=max_size)
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "created": 0,
            "destroyed": 0,
            "acquired": 0,
            "released": 0,
            "health_check_failures": 0,
            "creation_failures": 0,
        }

        # Background maintenance
        self._maintenance_thread: Optional[threading.Thread] = None
        self._shutdown = False

        # Initialize minimum pool size
        self._initialize_pool()

        # Start maintenance thread
        self._start_maintenance()

    def acquire(
        self, timeout: Optional[float] = None, **creation_kwargs
    ) -> ContextManager[T]:
        """
        Acquire a resource from the pool.

        Args:
            timeout: Maximum time to wait for a resource
            **creation_kwargs: Arguments for resource creation if needed

        Returns:
            Context manager for the resource
        """
        return self._ResourceContext(self, timeout, creation_kwargs)

    def _acquire_resource(
        self, timeout: Optional[float] = None, **creation_kwargs
    ) -> T:
        """Internal method to acquire a resource."""
        start_time = time.time()

        while True:
            # Try to get an available resource
            try:
                resource_id = self._available.get(timeout=0.1)

                with self._lock:
                    if resource_id in self._resources:
                        resource_info = self._resources[resource_id]

                        # Validate resource
                        if self._validate_and_prepare_resource(resource_info):
                            resource_info.mark_used()
                            self._stats["acquired"] += 1
                            logger.debug(f"Acquired resource: {resource_id}")
                            return resource_info.resource
                        else:
                            # Resource is unhealthy, remove it
                            self._destroy_resource(resource_id)
                            continue

            except Empty:
                pass

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise ResourceError(
                    operation="acquire_resource", required_mb=0, available_mb=0
                ).with_message("Timeout waiting for available resource")

            # Try to create new resource if under limit
            with self._lock:
                if len(self._resources) < self.max_size:
                    try:
                        resource = self._create_resource(**creation_kwargs)
                        if resource:
                            return resource
                    except Exception as e:
                        logger.error(f"Failed to create resource: {e}")
                        self._stats["creation_failures"] += 1

            # Brief wait before retrying
            time.sleep(0.1)

    def _release_resource(self, resource: T) -> None:
        """Internal method to release a resource."""
        with self._lock:
            # Find resource info
            resource_info = None
            for info in self._resources.values():
                if info.resource is resource:
                    resource_info = info
                    break

            if not resource_info:
                logger.warning("Attempted to release unknown resource")
                return

            resource_info.mark_released()

            # Reset resource to clean state
            try:
                if self.factory.reset_resource(resource):
                    # Resource successfully reset, return to pool
                    self._available.put(resource_info.resource_id, timeout=1.0)
                    self._stats["released"] += 1
                    logger.debug(f"Released resource: {resource_info.resource_id}")
                else:
                    # Resource couldn't be reset, destroy it
                    self._destroy_resource(resource_info.resource_id)

            except (Full, Exception) as e:
                logger.error(f"Failed to release resource: {e}")
                self._destroy_resource(resource_info.resource_id)

    def _create_resource(self, **creation_kwargs) -> Optional[T]:
        """Create a new resource."""
        resource_id = f"resource_{int(time.time() * 1000)}_{id(self)}"

        try:
            resource = self.factory.create_resource(**creation_kwargs)

            resource_info = ResourceInfo(
                resource_id=resource_id,
                resource=resource,
                created_at=datetime.now(),
                metadata=creation_kwargs,
            )

            with self._lock:
                self._resources[resource_id] = resource_info
                resource_info.mark_used()
                self._stats["created"] += 1

            logger.debug(f"Created resource: {resource_id}")
            return resource

        except Exception as e:
            logger.error(f"Resource creation failed: {e}")
            raise

    def _validate_and_prepare_resource(self, resource_info: ResourceInfo) -> bool:
        """Validate and prepare a resource for use."""
        try:
            # Check if resource is expired
            if resource_info.is_expired(self.max_age_seconds):
                logger.debug(f"Resource expired: {resource_info.resource_id}")
                return False

            # Validate resource health
            if not self.factory.validate_resource(resource_info.resource):
                logger.debug(f"Resource validation failed: {resource_info.resource_id}")
                resource_info.healthy = False
                return False

            return True

        except Exception as e:
            logger.error(f"Resource validation error: {e}")
            return False

    def _destroy_resource(self, resource_id: str) -> None:
        """Destroy a resource."""
        with self._lock:
            if resource_id not in self._resources:
                return

            resource_info = self._resources[resource_id]

            try:
                self.factory.cleanup_resource(resource_info.resource)
            except Exception as e:
                logger.error(f"Resource cleanup error: {e}")

            del self._resources[resource_id]
            self._stats["destroyed"] += 1

            logger.debug(f"Destroyed resource: {resource_id}")

    def _initialize_pool(self) -> None:
        """Initialize the pool with minimum resources."""
        for _ in range(self.min_size):
            try:
                self._create_resource()
            except Exception as e:
                logger.error(f"Failed to initialize pool resource: {e}")

    def _start_maintenance(self) -> None:
        """Start background maintenance thread."""
        if self._maintenance_thread is None:
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                name="ResourcePoolMaintenance",
                daemon=True,
            )
            self._maintenance_thread.start()

    def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while not self._shutdown:
            try:
                self._perform_maintenance()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

    def _perform_maintenance(self) -> None:
        """Perform maintenance tasks."""
        with self._lock:
            current_time = datetime.now()
            resources_to_destroy = []

            # Check each resource
            for resource_id, resource_info in self._resources.items():
                if resource_info.in_use:
                    continue

                # Check if expired or idle
                if resource_info.is_expired(
                    self.max_age_seconds
                ) or resource_info.is_idle(self.max_idle_seconds):
                    resources_to_destroy.append(resource_id)
                    continue

                # Health check
                try:
                    if not self.factory.validate_resource(resource_info.resource):
                        resources_to_destroy.append(resource_id)
                        self._stats["health_check_failures"] += 1
                except Exception as e:
                    logger.error(f"Health check error for {resource_id}: {e}")
                    resources_to_destroy.append(resource_id)
                    self._stats["health_check_failures"] += 1

            # Destroy unhealthy resources
            for resource_id in resources_to_destroy:
                self._destroy_resource(resource_id)

            # Ensure minimum pool size
            active_count = len(self._resources)
            if active_count < self.min_size:
                needed = self.min_size - active_count
                for _ in range(needed):
                    try:
                        self._create_resource()
                    except Exception as e:
                        logger.error(f"Failed to maintain minimum pool size: {e}")
                        break

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            active_resources = len(self._resources)
            in_use = sum(1 for r in self._resources.values() if r.in_use)
            available = active_resources - in_use

            return {
                "pool_size": active_resources,
                "min_size": self.min_size,
                "max_size": self.max_size,
                "in_use": in_use,
                "available": available,
                "stats": self._stats.copy(),
            }

    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the resource pool."""
        self._shutdown = True

        # Wait for maintenance thread
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=timeout)

        # Destroy all resources
        with self._lock:
            resource_ids = list(self._resources.keys())
            for resource_id in resource_ids:
                self._destroy_resource(resource_id)

        logger.info("Resource pool shutdown complete")

    class _ResourceContext:
        """Context manager for resource acquisition."""

        def __init__(
            self,
            pool: "ResourcePool",
            timeout: Optional[float],
            creation_kwargs: Dict[str, Any],
        ):
            self.pool = pool
            self.timeout = timeout
            self.creation_kwargs = creation_kwargs
            self.resource: Optional[T] = None

        def __enter__(self) -> T:
            self.resource = self.pool._acquire_resource(
                self.timeout, **self.creation_kwargs
            )
            return self.resource

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.resource:
                self.pool._release_resource(self.resource)


class DeviceConnectionFactory(ResourceFactory):
    """Factory for creating device connection resources."""

    def __init__(self, platform: str, device_path: str, **connection_kwargs):
        self.platform = platform
        self.device_path = device_path
        self.connection_kwargs = connection_kwargs
        self.platform_manager = PlatformManager(platform)

    def create_resource(self, **kwargs) -> Any:
        """Create a device connection."""
        from .profiler import EdgeProfiler

        # Merge connection kwargs
        conn_kwargs = {**self.connection_kwargs, **kwargs}

        profiler = EdgeProfiler(
            platform=self.platform, device=self.device_path, **conn_kwargs
        )

        # Test connection
        if not profiler.connect():
            raise ResourceError(
                operation="create_device_connection", required_mb=0, available_mb=0
            ).with_message(
                f"Failed to connect to {self.platform} device at {self.device_path}"
            )

        return profiler

    def validate_resource(self, resource: Any) -> bool:
        """Validate device connection."""
        try:
            return resource.is_connected
        except:
            return False

    def cleanup_resource(self, resource: Any) -> None:
        """Clean up device connection."""
        try:
            resource.disconnect()
        except:
            pass

    def reset_resource(self, resource: Any) -> bool:
        """Reset device connection."""
        try:
            if not resource.is_connected:
                return resource.connect()
            return True
        except:
            return False


class ModelCacheFactory(ResourceFactory):
    """Factory for creating cached model instances."""

    def __init__(self, model_path: str, **model_kwargs):
        self.model_path = model_path
        self.model_kwargs = model_kwargs

    def create_resource(self, **kwargs) -> Any:
        """Create a model instance."""
        from .models import QuantizedModel

        # Merge model kwargs
        model_kwargs = {**self.model_kwargs, **kwargs}

        model = QuantizedModel.from_file(self.model_path, **model_kwargs)
        return model

    def validate_resource(self, resource: Any) -> bool:
        """Validate model instance."""
        try:
            is_valid, _ = resource.validate()
            return is_valid
        except:
            return False

    def cleanup_resource(self, resource: Any) -> None:
        """Clean up model instance."""
        # Models don't need explicit cleanup
        pass

    def reset_resource(self, resource: Any) -> bool:
        """Reset model instance."""
        # Models don't need resetting
        return True


class ResourcePoolManager:
    """
    Manager for multiple resource pools.
    """

    def __init__(self):
        self._pools: Dict[str, ResourcePool] = {}
        self._lock = threading.RLock()

    def create_device_pool(
        self,
        pool_name: str,
        platform: str,
        device_path: str,
        min_size: int = 1,
        max_size: int = 5,
        **connection_kwargs,
    ) -> ResourcePool:
        """Create a device connection pool."""
        factory = DeviceConnectionFactory(platform, device_path, **connection_kwargs)

        pool = ResourcePool(factory=factory, min_size=min_size, max_size=max_size)

        with self._lock:
            self._pools[pool_name] = pool

        logger.info(f"Created device pool: {pool_name} ({platform} at {device_path})")
        return pool

    def create_model_pool(
        self,
        pool_name: str,
        model_path: str,
        min_size: int = 1,
        max_size: int = 3,
        **model_kwargs,
    ) -> ResourcePool:
        """Create a model cache pool."""
        factory = ModelCacheFactory(model_path, **model_kwargs)

        pool = ResourcePool(
            factory=factory,
            min_size=min_size,
            max_size=max_size,
            max_age_seconds=7200,  # 2 hours for models
        )

        with self._lock:
            self._pools[pool_name] = pool

        logger.info(f"Created model pool: {pool_name} ({model_path})")
        return pool

    def get_pool(self, pool_name: str) -> Optional[ResourcePool]:
        """Get a pool by name."""
        with self._lock:
            return self._pools.get(pool_name)

    def remove_pool(self, pool_name: str) -> bool:
        """Remove and shutdown a pool."""
        with self._lock:
            if pool_name in self._pools:
                pool = self._pools[pool_name]
                pool.shutdown()
                del self._pools[pool_name]
                logger.info(f"Removed pool: {pool_name}")
                return True
            return False

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        with self._lock:
            return {name: pool.get_stats() for name, pool in self._pools.items()}

    def shutdown_all(self, timeout: float = 30.0) -> None:
        """Shutdown all pools."""
        with self._lock:
            pool_names = list(self._pools.keys())

            for pool_name in pool_names:
                try:
                    self._pools[pool_name].shutdown(timeout / len(pool_names))
                    del self._pools[pool_name]
                except Exception as e:
                    logger.error(f"Error shutting down pool {pool_name}: {e}")

        logger.info("All resource pools shutdown")


# Global resource pool manager
_global_pool_manager = ResourcePoolManager()


def get_pool_manager() -> ResourcePoolManager:
    """Get the global resource pool manager."""
    return _global_pool_manager


def create_device_pool(
    pool_name: str, platform: str, device_path: str, **kwargs
) -> ResourcePool:
    """Create a device connection pool."""
    return _global_pool_manager.create_device_pool(
        pool_name, platform, device_path, **kwargs
    )


def create_model_pool(pool_name: str, model_path: str, **kwargs) -> ResourcePool:
    """Create a model cache pool."""
    return _global_pool_manager.create_model_pool(pool_name, model_path, **kwargs)


def get_pool(pool_name: str) -> Optional[ResourcePool]:
    """Get a resource pool by name."""
    return _global_pool_manager.get_pool(pool_name)


@contextmanager
def acquire_device(pool_name: str, timeout: Optional[float] = None):
    """Context manager for acquiring a device from a pool."""
    pool = get_pool(pool_name)
    if not pool:
        raise ResourceError(
            operation="acquire_device", required_mb=0, available_mb=0
        ).with_message(f"Device pool not found: {pool_name}")

    with pool.acquire(timeout=timeout) as device:
        yield device


@contextmanager
def acquire_model(pool_name: str, timeout: Optional[float] = None):
    """Context manager for acquiring a model from a pool."""
    pool = get_pool(pool_name)
    if not pool:
        raise ResourceError(
            operation="acquire_model", required_mb=0, available_mb=0
        ).with_message(f"Model pool not found: {pool_name}")

    with pool.acquire(timeout=timeout) as model:
        yield model
