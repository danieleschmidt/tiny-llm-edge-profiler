"""
Advanced Multi-Level Caching Architecture for Generation 3
Provides comprehensive caching capabilities including:
- Multi-level caching (L1/L2/L3) with intelligent promotion/demotion
- Intelligent cache invalidation strategies
- Persistent caching across sessions with compression
- Cache optimization for different access patterns
- Memory-mapped file caching for large datasets
- Predictive cache warming and preloading
"""

import time
import mmap
import hashlib
import pickle
import json
import threading
import asyncio
import weakref
import gzip
import lz4.frame
from typing import (
    Any,
    Dict,
    Optional,
    Union,
    Callable,
    TypeVar,
    Generic,
    List,
    Set,
    Tuple,
)
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import numpy as np

from .exceptions import TinyLLMProfilerError
from .logging_config import get_logger
from .security import secure_delete_file
from .cache import (
    InMemoryCache,
    PersistentCache,
    SmartCache,
)  # Import existing cache classes

logger = get_logger("advanced_cache")

T = TypeVar("T")


@dataclass
class CacheMetrics:
    """Comprehensive cache metrics and statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    promotions: int = 0  # L2->L1, L3->L2
    demotions: int = 0  # L1->L2, L2->L3
    invalidations: int = 0
    preloads: int = 0
    compression_ratio: float = 0.0
    avg_access_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def total_operations(self) -> int:
        """Get total cache operations."""
        return (
            self.hits + self.misses + self.evictions + self.promotions + self.demotions
        )


class CacheLevel(str):
    """Cache level identifiers."""

    L1 = "L1"  # Ultra-fast memory cache (small, hot data)
    L2 = "L2"  # Fast memory cache (medium, warm data)
    L3 = "L3"  # Persistent cache (large, cold data)


@dataclass
class AdvancedCacheEntry(Generic[T]):
    """Advanced cache entry with comprehensive metadata."""

    key: str
    value: T
    level: CacheLevel
    created_at: datetime
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    access_frequency: float = 0.0  # Accesses per hour
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    compression_type: Optional[str] = None
    checksum: Optional[str] = None

    # Access pattern analysis
    access_pattern: str = "unknown"  # sequential, random, bursty, etc.
    locality_score: float = 0.0  # Temporal locality score

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self) -> None:
        """Update access information with frequency tracking."""
        now = datetime.now()
        time_diff = (now - self.last_accessed).total_seconds() / 3600  # Hours

        self.last_accessed = now
        self.access_count += 1

        # Calculate access frequency (exponential moving average)
        if time_diff > 0:
            new_frequency = 1.0 / time_diff
            self.access_frequency = 0.9 * self.access_frequency + 0.1 * new_frequency

    def calculate_priority_score(self) -> float:
        """Calculate priority score for cache level placement."""
        # Factors: access frequency, recency, size efficiency
        frequency_score = min(self.access_frequency, 100.0)
        recency_score = max(
            0, 100 - (datetime.now() - self.last_accessed).total_seconds() / 3600
        )
        size_penalty = min(self.size_bytes / (1024 * 1024), 10.0)  # MB penalty

        return frequency_score + recency_score - size_penalty


class CacheInvalidationStrategy(ABC):
    """Abstract base for cache invalidation strategies."""

    @abstractmethod
    def should_invalidate(
        self, entry: AdvancedCacheEntry, context: Dict[str, Any]
    ) -> bool:
        """Determine if entry should be invalidated."""
        pass


class TimeBasedInvalidation(CacheInvalidationStrategy):
    """Time-based cache invalidation."""

    def __init__(self, max_age_seconds: int):
        self.max_age_seconds = max_age_seconds

    def should_invalidate(
        self, entry: AdvancedCacheEntry, context: Dict[str, Any]
    ) -> bool:
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > self.max_age_seconds


class DependencyInvalidation(CacheInvalidationStrategy):
    """Dependency-based cache invalidation."""

    def __init__(self, invalidated_dependencies: Set[str]):
        self.invalidated_dependencies = invalidated_dependencies

    def should_invalidate(
        self, entry: AdvancedCacheEntry, context: Dict[str, Any]
    ) -> bool:
        return bool(entry.dependencies & self.invalidated_dependencies)


class TagBasedInvalidation(CacheInvalidationStrategy):
    """Tag-based cache invalidation."""

    def __init__(self, invalidated_tags: Set[str]):
        self.invalidated_tags = invalidated_tags

    def should_invalidate(
        self, entry: AdvancedCacheEntry, context: Dict[str, Any]
    ) -> bool:
        return bool(entry.tags & self.invalidated_tags)


class CompressionHandler:
    """Handles compression and decompression of cache data."""

    COMPRESSION_TYPES = {
        "none": {"compress": lambda x: x, "decompress": lambda x: x},
        "gzip": {"compress": gzip.compress, "decompress": gzip.decompress},
        "lz4": {"compress": lz4.frame.compress, "decompress": lz4.frame.decompress},
    }

    @classmethod
    def compress(cls, data: bytes, compression_type: str = "lz4") -> Tuple[bytes, str]:
        """Compress data using specified algorithm."""
        if compression_type not in cls.COMPRESSION_TYPES:
            compression_type = "none"

        compress_func = cls.COMPRESSION_TYPES[compression_type]["compress"]
        compressed_data = compress_func(data)

        return compressed_data, compression_type

    @classmethod
    def decompress(cls, data: bytes, compression_type: str) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type not in cls.COMPRESSION_TYPES:
            return data

        decompress_func = cls.COMPRESSION_TYPES[compression_type]["decompress"]
        return decompress_func(data)

    @classmethod
    def calculate_compression_ratio(
        cls, original_size: int, compressed_size: int
    ) -> float:
        """Calculate compression ratio."""
        if original_size == 0:
            return 1.0
        return compressed_size / original_size


class MemoryMappedCache:
    """Memory-mapped file cache for large datasets."""

    def __init__(self, cache_dir: Path, max_file_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.mmap_files: Dict[str, Tuple[mmap.mmap, int]] = {}
        self._lock = threading.RLock()

    def put(self, key: str, data: bytes) -> bool:
        """Store data in memory-mapped file."""
        if len(data) > self.max_file_size:
            logger.warning(f"Data too large for memory-mapped cache: {len(data)} bytes")
            return False

        with self._lock:
            try:
                file_path = (
                    self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.mmap"
                )

                # Create and write file
                with open(file_path, "wb") as f:
                    f.write(data)

                # Open as memory-mapped file
                with open(file_path, "r+b") as f:
                    mm = mmap.mmap(f.fileno(), len(data))
                    self.mmap_files[key] = (mm, len(data))

                logger.debug(
                    f"Created memory-mapped cache entry: {key} ({len(data)} bytes)"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to create memory-mapped cache entry: {e}")
                return False

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data from memory-mapped file."""
        with self._lock:
            if key in self.mmap_files:
                try:
                    mm, size = self.mmap_files[key]
                    mm.seek(0)
                    return mm.read(size)
                except Exception as e:
                    logger.error(f"Failed to read from memory-mapped cache: {e}")
                    # Clean up corrupted entry
                    self.remove(key)

            return None

    def remove(self, key: str) -> bool:
        """Remove memory-mapped cache entry."""
        with self._lock:
            if key in self.mmap_files:
                try:
                    mm, _ = self.mmap_files[key]
                    mm.close()
                    del self.mmap_files[key]

                    # Remove file
                    file_path = (
                        self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.mmap"
                    )
                    if file_path.exists():
                        file_path.unlink()

                    return True
                except Exception as e:
                    logger.error(f"Failed to remove memory-mapped cache entry: {e}")

            return False

    def close(self):
        """Close all memory-mapped files."""
        with self._lock:
            for key in list(self.mmap_files.keys()):
                self.remove(key)


class PredictiveCache:
    """Predictive cache that learns access patterns and preloads data."""

    def __init__(self, prediction_window: int = 10):
        self.prediction_window = prediction_window
        self.access_history: List[Tuple[str, datetime]] = []
        self.patterns: Dict[str, List[str]] = defaultdict(
            list
        )  # key -> likely next keys
        self.preload_queue: asyncio.Queue = asyncio.Queue()
        self._pattern_lock = threading.RLock()

    def record_access(self, key: str):
        """Record cache access for pattern learning."""
        with self._pattern_lock:
            now = datetime.now()
            self.access_history.append((key, now))

            # Keep only recent history
            cutoff = now - timedelta(hours=1)
            self.access_history = [(k, t) for k, t in self.access_history if t > cutoff]

            # Update patterns
            self._update_patterns(key)

    def _update_patterns(self, current_key: str):
        """Update access pattern predictions."""
        if len(self.access_history) < 2:
            return

        # Look for sequential patterns
        recent_keys = [k for k, _ in self.access_history[-self.prediction_window :]]

        if len(recent_keys) >= 2:
            prev_key = recent_keys[-2]

            # Record transition
            if current_key not in self.patterns[prev_key]:
                self.patterns[prev_key].append(current_key)

            # Limit pattern size
            if len(self.patterns[prev_key]) > 5:
                self.patterns[prev_key] = self.patterns[prev_key][-5:]

    def predict_next_keys(self, current_key: str, count: int = 3) -> List[str]:
        """Predict next likely cache keys."""
        with self._pattern_lock:
            if current_key in self.patterns:
                # Sort by frequency of occurrence
                candidates = self.patterns[current_key]
                pattern_counts = defaultdict(int)

                for candidate in candidates:
                    pattern_counts[candidate] += 1

                # Return most frequent patterns
                sorted_patterns = sorted(
                    pattern_counts.items(), key=lambda x: x[1], reverse=True
                )

                return [k for k, _ in sorted_patterns[:count]]

        return []

    def should_preload(self, key: str) -> bool:
        """Determine if a key should be preloaded."""
        # Simple heuristic: preload if key appears in recent patterns
        recent_keys = [k for k, _ in self.access_history[-5:]]
        return any(
            key in self.patterns.get(recent_key, []) for recent_key in recent_keys
        )


class MultiLevelCache:
    """
    Advanced multi-level cache system with intelligent management.
    """

    def __init__(
        self,
        l1_config: Optional[Dict[str, Any]] = None,
        l2_config: Optional[Dict[str, Any]] = None,
        l3_config: Optional[Dict[str, Any]] = None,
        enable_compression: bool = True,
        enable_prediction: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        # Default configurations
        l1_config = l1_config or {"max_size": 100, "max_memory_mb": 50}
        l2_config = l2_config or {"max_size": 1000, "max_memory_mb": 200}
        l3_config = l3_config or {"max_size_gb": 2.0}

        # Initialize cache levels
        self.l1_cache = InMemoryCache(
            max_size=l1_config["max_size"],
            max_memory_mb=l1_config["max_memory_mb"],
            default_ttl_seconds=300,  # 5 minutes for L1
        )

        self.l2_cache = InMemoryCache(
            max_size=l2_config["max_size"],
            max_memory_mb=l2_config["max_memory_mb"],
            default_ttl_seconds=3600,  # 1 hour for L2
        )

        cache_dir = cache_dir or Path.home() / ".cache" / "tiny_llm_profiler" / "l3"
        self.l3_cache = PersistentCache(
            cache_dir=cache_dir,
            max_size_gb=l3_config["max_size_gb"],
            compression=enable_compression,
        )

        # Advanced features
        self.enable_compression = enable_compression
        self.enable_prediction = enable_prediction

        if enable_prediction:
            self.predictive_cache = PredictiveCache()

        # Memory-mapped cache for large objects
        self.mmap_cache = MemoryMappedCache(cache_dir / "mmap")

        # Cache management
        self.entries: Dict[str, AdvancedCacheEntry] = {}
        self.invalidation_strategies: List[CacheInvalidationStrategy] = []
        self.metrics = CacheMetrics()
        self._lock = threading.RLock()

        # Background tasks
        self._maintenance_executor = ThreadPoolExecutor(max_workers=2)
        self._running = True

        # Start background maintenance
        self._start_maintenance_tasks()

        logger.info("Multi-level cache system initialized")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from multi-level cache with intelligent promotion.

        Args:
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        start_time = time.time()

        with self._lock:
            # Record access for prediction
            if self.enable_prediction:
                self.predictive_cache.record_access(key)

            # Check cache levels in order
            value = None
            found_level = None

            # L1 Cache (fastest)
            value = self.l1_cache.get(key)
            if value is not None:
                found_level = CacheLevel.L1
                self.metrics.hits += 1
            else:
                # L2 Cache
                value = self.l2_cache.get(key)
                if value is not None:
                    found_level = CacheLevel.L2
                    self.metrics.hits += 1

                    # Consider promotion to L1
                    if key in self.entries:
                        entry = self.entries[key]
                        if self._should_promote(entry, CacheLevel.L1):
                            self.l1_cache.put(key, value, ttl_seconds=300)
                            self.metrics.promotions += 1
                            logger.debug(f"Promoted {key} from L2 to L1")
                else:
                    # L3 Cache (persistent)
                    value = self.l3_cache.get(key)
                    if value is not None:
                        found_level = CacheLevel.L3
                        self.metrics.hits += 1

                        # Consider promotion to L2
                        if key in self.entries:
                            entry = self.entries[key]
                            if self._should_promote(entry, CacheLevel.L2):
                                self.l2_cache.put(key, value, ttl_seconds=3600)
                                self.metrics.promotions += 1
                                logger.debug(f"Promoted {key} from L3 to L2")
                    else:
                        # Check memory-mapped cache
                        try:
                            data = self.mmap_cache.get(key)
                            if data:
                                value = pickle.loads(data)
                                found_level = "MMAP"
                                self.metrics.hits += 1
                        except Exception as e:
                            logger.error(
                                f"Failed to load from memory-mapped cache: {e}"
                            )

            # Update metrics and entry info
            if value is not None and key in self.entries:
                entry = self.entries[key]
                entry.touch()
                entry.level = found_level
            else:
                self.metrics.misses += 1

            # Track access time
            access_time = (time.time() - start_time) * 1000  # ms
            self.metrics.avg_access_time_ms = (
                0.9 * self.metrics.avg_access_time_ms + 0.1 * access_time
            )

            # Trigger predictive preloading
            if value is not None and self.enable_prediction:
                self._trigger_predictive_preload(key)

            return value

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
        compression_type: str = "lz4",
    ) -> bool:
        """
        Put value in multi-level cache with intelligent placement.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live
            tags: Tags for invalidation
            dependencies: Dependencies for invalidation
            compression_type: Compression algorithm

        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Serialize value to calculate size
                serialized_value = pickle.dumps(value)
                size_bytes = len(serialized_value)

                # Create advanced cache entry
                entry = AdvancedCacheEntry(
                    key=key,
                    value=value,
                    level=CacheLevel.L1,  # Start at L1, will be adjusted
                    created_at=datetime.now(),
                    ttl_seconds=ttl_seconds,
                    size_bytes=size_bytes,
                    tags=tags or set(),
                    dependencies=dependencies or set(),
                )

                # Decide placement level based on size and access patterns
                target_level = self._determine_placement_level(entry)
                entry.level = target_level

                # Store in appropriate cache level
                success = False

                if target_level == CacheLevel.L1:
                    self.l1_cache.put(key, value, ttl_seconds or 300)
                    success = True
                elif target_level == CacheLevel.L2:
                    self.l2_cache.put(key, value, ttl_seconds or 3600)
                    success = True
                elif target_level == CacheLevel.L3:
                    self.l3_cache.put(key, value, ttl_seconds, list(tags or []))
                    success = True

                # For very large objects, also store in memory-mapped cache
                if size_bytes > 10 * 1024 * 1024:  # 10MB threshold
                    compressed_data, compression_used = CompressionHandler.compress(
                        serialized_value, compression_type
                    )
                    entry.compression_type = compression_used
                    entry.compression_ratio = (
                        CompressionHandler.calculate_compression_ratio(
                            size_bytes, len(compressed_data)
                        )
                    )

                    if self.mmap_cache.put(key, compressed_data):
                        logger.debug(
                            f"Stored large object in memory-mapped cache: {key}"
                        )

                # Store entry metadata
                self.entries[key] = entry

                logger.debug(f"Cached {key} in {target_level} ({size_bytes} bytes)")
                return success

            except Exception as e:
                logger.error(f"Failed to cache {key}: {e}")
                return False

    def invalidate(
        self,
        keys: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
        strategy: Optional[CacheInvalidationStrategy] = None,
    ) -> int:
        """
        Invalidate cache entries using various strategies.

        Args:
            keys: Specific keys to invalidate
            tags: Tags to invalidate
            dependencies: Dependencies to invalidate
            strategy: Custom invalidation strategy

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            invalidated_count = 0

            # Collect invalidation strategies
            strategies = []
            if tags:
                strategies.append(TagBasedInvalidation(tags))
            if dependencies:
                strategies.append(DependencyInvalidation(dependencies))
            if strategy:
                strategies.append(strategy)

            # Find entries to invalidate
            entries_to_invalidate = []

            if keys:
                # Specific keys
                entries_to_invalidate.extend(keys)
            else:
                # Apply strategies
                for entry_key, entry in self.entries.items():
                    for invalidation_strategy in strategies:
                        if invalidation_strategy.should_invalidate(entry, {}):
                            entries_to_invalidate.append(entry_key)
                            break

            # Invalidate entries
            for key in entries_to_invalidate:
                if self._invalidate_key(key):
                    invalidated_count += 1

            self.metrics.invalidations += invalidated_count

            logger.info(f"Invalidated {invalidated_count} cache entries")
            return invalidated_count

    def _invalidate_key(self, key: str) -> bool:
        """Invalidate a specific key from all cache levels."""
        try:
            # Remove from all cache levels
            self.l1_cache.delete(key)
            self.l2_cache.delete(key)
            self.l3_cache.delete(key)
            self.mmap_cache.remove(key)

            # Remove metadata
            if key in self.entries:
                del self.entries[key]

            return True
        except Exception as e:
            logger.error(f"Failed to invalidate {key}: {e}")
            return False

    def _determine_placement_level(self, entry: AdvancedCacheEntry) -> CacheLevel:
        """Determine optimal cache level for new entry."""
        # Size-based placement
        if entry.size_bytes < 1024:  # < 1KB
            return CacheLevel.L1
        elif entry.size_bytes < 1024 * 1024:  # < 1MB
            return CacheLevel.L2
        else:
            return CacheLevel.L3

    def _should_promote(
        self, entry: AdvancedCacheEntry, target_level: CacheLevel
    ) -> bool:
        """Determine if entry should be promoted to higher cache level."""
        priority_score = entry.calculate_priority_score()

        if target_level == CacheLevel.L1:
            # Promote to L1 if high access frequency and small size
            return priority_score > 50 and entry.size_bytes < 10240  # 10KB
        elif target_level == CacheLevel.L2:
            # Promote to L2 if moderate access and reasonable size
            return priority_score > 20 and entry.size_bytes < 1024 * 1024  # 1MB

        return False

    def _trigger_predictive_preload(self, accessed_key: str):
        """Trigger predictive preloading based on access patterns."""
        if not self.enable_prediction:
            return

        # Get predicted next keys
        predicted_keys = self.predictive_cache.predict_next_keys(accessed_key)

        for predicted_key in predicted_keys:
            if predicted_key not in self.entries:
                continue

            # Check if key should be preloaded
            if self.predictive_cache.should_preload(predicted_key):
                # Submit preload task (non-blocking)
                self._maintenance_executor.submit(self._preload_key, predicted_key)

    def _preload_key(self, key: str):
        """Preload a key from lower to higher cache levels."""
        try:
            # Try to move from L3 to L2
            value = self.l3_cache.get(key)
            if value and key not in [k for k in self.l2_cache._cache.keys()]:
                self.l2_cache.put(key, value, ttl_seconds=3600)
                self.metrics.preloads += 1
                logger.debug(f"Preloaded {key} from L3 to L2")
        except Exception as e:
            logger.error(f"Failed to preload {key}: {e}")

    def _start_maintenance_tasks(self):
        """Start background maintenance tasks."""
        self._maintenance_executor.submit(self._maintenance_loop)

    def _maintenance_loop(self):
        """Background maintenance loop."""
        while self._running:
            try:
                # Perform cache maintenance
                self._perform_cache_maintenance()

                # Update metrics
                self._update_metrics()

                # Sleep between maintenance cycles
                time.sleep(60)  # 1 minute

            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                time.sleep(300)  # 5 minutes on error

    def _perform_cache_maintenance(self):
        """Perform cache maintenance tasks."""
        with self._lock:
            current_time = datetime.now()

            # Clean up expired entries
            expired_keys = [
                key for key, entry in self.entries.items() if entry.is_expired()
            ]

            for key in expired_keys:
                self._invalidate_key(key)

            # Apply invalidation strategies
            for strategy in self.invalidation_strategies:
                entries_to_check = list(self.entries.items())
                for key, entry in entries_to_check:
                    if strategy.should_invalidate(
                        entry, {"current_time": current_time}
                    ):
                        self._invalidate_key(key)

    def _update_metrics(self):
        """Update cache metrics."""
        try:
            # Memory usage
            l1_stats = self.l1_cache.get_stats()
            l2_stats = self.l2_cache.get_stats()
            l3_stats = self.l3_cache.get_stats()

            self.metrics.memory_usage_mb = l1_stats.get(
                "memory_usage_mb", 0
            ) + l2_stats.get("memory_usage_mb", 0)

            self.metrics.disk_usage_mb = l3_stats.get("total_size_mb", 0)

            # Compression ratio
            compressed_entries = [
                entry
                for entry in self.entries.values()
                if entry.compression_type and entry.compression_type != "none"
            ]

            if compressed_entries:
                self.metrics.compression_ratio = np.mean(
                    [
                        getattr(entry, "compression_ratio", 1.0)
                        for entry in compressed_entries
                    ]
                )

        except Exception as e:
            logger.error(f"Failed to update cache metrics: {e}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            l1_stats = self.l1_cache.get_stats()
            l2_stats = self.l2_cache.get_stats()
            l3_stats = self.l3_cache.get_stats()

            return {
                "metrics": {
                    "hit_rate": self.metrics.hit_rate,
                    "total_operations": self.metrics.total_operations,
                    "hits": self.metrics.hits,
                    "misses": self.metrics.misses,
                    "evictions": self.metrics.evictions,
                    "promotions": self.metrics.promotions,
                    "demotions": self.metrics.demotions,
                    "invalidations": self.metrics.invalidations,
                    "preloads": self.metrics.preloads,
                    "compression_ratio": self.metrics.compression_ratio,
                    "avg_access_time_ms": self.metrics.avg_access_time_ms,
                    "memory_usage_mb": self.metrics.memory_usage_mb,
                    "disk_usage_mb": self.metrics.disk_usage_mb,
                },
                "levels": {"L1": l1_stats, "L2": l2_stats, "L3": l3_stats},
                "entries": len(self.entries),
                "invalidation_strategies": len(self.invalidation_strategies),
                "predictive_cache": self.enable_prediction,
                "compression": self.enable_compression,
            }

    def add_invalidation_strategy(self, strategy: CacheInvalidationStrategy):
        """Add a cache invalidation strategy."""
        with self._lock:
            self.invalidation_strategies.append(strategy)

    def shutdown(self, timeout: float = 30.0):
        """Shutdown the cache system."""
        logger.info("Shutting down multi-level cache system...")

        self._running = False

        # Shutdown maintenance executor
        try:
            self._maintenance_executor.shutdown(wait=True, timeout=timeout)
        except Exception as e:
            logger.error(f"Error shutting down maintenance executor: {e}")

        # Close memory-mapped cache
        try:
            self.mmap_cache.close()
        except Exception as e:
            logger.error(f"Error closing memory-mapped cache: {e}")

        logger.info("Multi-level cache system shutdown complete")


# Global multi-level cache instance
_global_multilevel_cache: Optional[MultiLevelCache] = None


def get_multilevel_cache(
    l1_config: Optional[Dict[str, Any]] = None,
    l2_config: Optional[Dict[str, Any]] = None,
    l3_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> MultiLevelCache:
    """Get or create the global multi-level cache instance."""
    global _global_multilevel_cache

    if _global_multilevel_cache is None:
        _global_multilevel_cache = MultiLevelCache(
            l1_config=l1_config, l2_config=l2_config, l3_config=l3_config, **kwargs
        )

    return _global_multilevel_cache


def advanced_cached(
    ttl_seconds: Optional[int] = None,
    tags: Optional[Set[str]] = None,
    dependencies: Optional[Set[str]] = None,
    compression_type: str = "lz4",
    cache_instance: Optional[MultiLevelCache] = None,
):
    """
    Advanced decorator for caching function results with multi-level cache.

    Args:
        ttl_seconds: Time to live for cached result
        tags: Tags for cache invalidation
        dependencies: Dependencies for cache invalidation
        compression_type: Compression algorithm
        cache_instance: Specific cache instance to use
    """

    def decorator(func: Callable) -> Callable:
        nonlocal cache_instance
        if cache_instance is None:
            cache_instance = get_multilevel_cache()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{func.__module__}.{func.__qualname__}"
            args_str = str(args) + str(sorted(kwargs.items()))
            cache_key = f"{func_name}:{hashlib.md5(args_str.encode()).hexdigest()}"

            # Try to get from cache
            result = cache_instance.get(cache_key)
            if result is not None:
                logger.debug(f"Advanced cache hit for function: {func.__name__}")
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.put(
                cache_key,
                result,
                ttl_seconds=ttl_seconds,
                tags=tags,
                dependencies=dependencies,
                compression_type=compression_type,
            )

            logger.debug(f"Advanced cached result for function: {func.__name__}")
            return result

        # Add cache management methods
        wrapper.cache_invalidate = lambda **kwargs: cache_instance.invalidate(**kwargs)
        wrapper.cache_stats = lambda: cache_instance.get_comprehensive_stats()

        return wrapper

    return decorator
