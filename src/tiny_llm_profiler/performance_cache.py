"""
Performance caching and optimization for repeated profiling operations.
"""

import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheKey:
    """Cache key for profiling results."""

    platform: str
    model_size_mb: float
    quantization: str
    prompts_hash: str
    config_hash: str

    def to_string(self) -> str:
        """Convert to string representation."""
        return f"{self.platform}_{self.model_size_mb}_{self.quantization}_{self.prompts_hash}_{self.config_hash}"


@dataclass
class CacheEntry:
    """Cache entry with profiling results and metadata."""

    key: str
    result: Dict[str, Any]
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at


class PerformanceCache:
    """LRU cache for profiling results with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _hash_prompts(self, prompts: List[str]) -> str:
        """Create hash of prompts for caching."""
        prompts_str = "||".join(sorted(prompts))
        return hashlib.md5(prompts_str.encode()).hexdigest()[:12]

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create hash of configuration for caching."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def create_key(
        self,
        platform: str,
        model_size_mb: float,
        quantization: str,
        prompts: List[str],
        config: Dict[str, Any] = None,
    ) -> CacheKey:
        """Create cache key from profiling parameters."""
        config = config or {}

        return CacheKey(
            platform=platform,
            model_size_mb=round(model_size_mb, 2),
            quantization=quantization,
            prompts_hash=self._hash_prompts(prompts),
            config_hash=self._hash_config(config),
        )

    def get(self, key: CacheKey) -> Optional[Dict[str, Any]]:
        """Get cached result."""
        with self.lock:
            key_str = key.to_string()

            if key_str not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key_str]

            # Check TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                self._remove_entry(key_str)
                self.misses += 1
                return None

            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()

            # Update access order (move to end)
            if key_str in self.access_order:
                self.access_order.remove(key_str)
            self.access_order.append(key_str)

            self.hits += 1
            logger.debug(f"Cache hit for key: {key_str}")
            return entry.result.copy()

    def put(self, key: CacheKey, result: Dict[str, Any]):
        """Store result in cache."""
        with self.lock:
            key_str = key.to_string()

            # Create cache entry
            entry = CacheEntry(
                key=key_str, result=result.copy(), created_at=time.time()
            )

            # Add to cache
            self.cache[key_str] = entry

            # Update access order
            if key_str in self.access_order:
                self.access_order.remove(key_str)
            self.access_order.append(key_str)

            # Enforce size limit
            if len(self.cache) > self.max_size:
                self._evict_lru()

            logger.debug(f"Cache stored result for key: {key_str}")

    def _remove_entry(self, key_str: str):
        """Remove entry from cache."""
        if key_str in self.cache:
            del self.cache[key_str]

        if key_str in self.access_order:
            self.access_order.remove(key_str)

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_order:
            return

        lru_key = self.access_order[0]
        self._remove_entry(lru_key)
        self.evictions += 1
        logger.debug(f"Evicted LRU entry: {lru_key}")

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            logger.info("Cache cleared")

    def cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self.cache.items()
                if current_time - entry.created_at > self.ttl_seconds
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) * 100 if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_percent": hit_rate,
                "evictions": self.evictions,
                "ttl_seconds": self.ttl_seconds,
            }

    def save_to_disk(self, file_path: Path):
        """Save cache to disk."""
        with self.lock:
            cache_data = {
                "cache": {key: asdict(entry) for key, entry in self.cache.items()},
                "access_order": self.access_order,
                "stats": {
                    "hits": self.hits,
                    "misses": self.misses,
                    "evictions": self.evictions,
                },
                "saved_at": time.time(),
            }

            with open(file_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Cache saved to {file_path}")

    def load_from_disk(self, file_path: Path) -> bool:
        """Load cache from disk."""
        if not file_path.exists():
            return False

        try:
            with open(file_path, "r") as f:
                cache_data = json.load(f)

            with self.lock:
                # Restore cache entries
                for key_str, entry_data in cache_data["cache"].items():
                    entry = CacheEntry(**entry_data)

                    # Check if entry is still valid
                    if time.time() - entry.created_at <= self.ttl_seconds:
                        self.cache[key_str] = entry

                # Restore access order (filter out expired entries)
                self.access_order = [
                    key
                    for key in cache_data.get("access_order", [])
                    if key in self.cache
                ]

                # Restore stats
                stats = cache_data.get("stats", {})
                self.hits = stats.get("hits", 0)
                self.misses = stats.get("misses", 0)
                self.evictions = stats.get("evictions", 0)

            logger.info(f"Cache loaded from {file_path} ({len(self.cache)} entries)")
            return True

        except Exception as e:
            logger.error(f"Failed to load cache from {file_path}: {e}")
            return False


class OptimizedProfiler:
    """Profiler with performance optimizations and caching."""

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        self.cache = PerformanceCache(cache_size, cache_ttl)
        self.optimization_enabled = True

        # Performance tracking
        self.profile_times = []
        self.cache_hits = 0
        self.cache_misses = 0

    def profile_with_cache(
        self,
        platform: str,
        model_size_mb: float,
        quantization: str,
        prompts: List[str],
        config: Dict[str, Any] = None,
        profiler_func: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Profile with caching support."""
        config = config or {}

        # Create cache key
        cache_key = self.cache.create_key(
            platform=platform,
            model_size_mb=model_size_mb,
            quantization=quantization,
            prompts=prompts,
            config=config,
        )

        # Try to get from cache first
        if self.optimization_enabled:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                logger.debug(f"Using cached result for {platform}")
                return cached_result

        # Cache miss - run actual profiling
        self.cache_misses += 1
        logger.debug(f"Cache miss - profiling {platform}")

        start_time = time.time()

        # Run profiling (use provided profiler or simulate)
        if profiler_func:
            result = profiler_func(platform, prompts, config)
        else:
            # Fallback simulation
            try:
                from .core_lite import SimpleProfiler
            except ImportError:
                from core_lite import SimpleProfiler
            profiler = SimpleProfiler(platform)
            profile = profiler.simulate_profiling(prompts)
            result = asdict(profile)

        execution_time = time.time() - start_time
        self.profile_times.append(execution_time)

        # Add profiling metadata
        result["profiling_metadata"] = {
            "execution_time_s": execution_time,
            "cached": False,
            "timestamp": time.time(),
        }

        # Store in cache
        if self.optimization_enabled:
            self.cache.put(cache_key, result)

        return result

    def batch_profile_with_cache(
        self, profile_requests: List[Tuple[str, float, str, List[str], Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Batch profiling with cache optimization."""
        results = []

        for platform, model_size, quantization, prompts, config in profile_requests:
            result = self.profile_with_cache(
                platform=platform,
                model_size_mb=model_size,
                quantization=quantization,
                prompts=prompts,
                config=config,
            )
            results.append(result)

        return results

    def optimize_for_pattern(self, access_pattern: Dict[str, int]):
        """Optimize cache based on access patterns."""
        # Pre-warm cache for frequently accessed configurations
        for platform, access_count in access_pattern.items():
            if access_count > 10:  # Frequently accessed
                logger.info(f"Pre-warming cache for {platform}")
                # Could pre-compute common configurations

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_stats = self.cache.get_stats()

        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            (self.cache_hits / total_requests) * 100 if total_requests > 0 else 0
        )

        avg_profile_time = (
            sum(self.profile_times) / len(self.profile_times)
            if self.profile_times
            else 0
        )

        return {
            "cache_stats": cache_stats,
            "profiler_stats": {
                "total_requests": total_requests,
                "cache_hit_rate_percent": cache_hit_rate,
                "average_profile_time_s": avg_profile_time,
                "total_profile_operations": len(self.profile_times),
            },
            "optimization_enabled": self.optimization_enabled,
        }

    def save_cache(self, file_path: Path):
        """Save cache to disk."""
        self.cache.save_to_disk(file_path)

    def load_cache(self, file_path: Path) -> bool:
        """Load cache from disk."""
        return self.cache.load_from_disk(file_path)

    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()


def cache_performance_demo():
    """Demonstrate caching performance improvements."""
    logger.info("Starting cache performance demo")

    # Initialize optimized profiler
    profiler = OptimizedProfiler(cache_size=100, cache_ttl=300)

    # Define test cases
    platforms = ["esp32", "stm32f4", "stm32f7", "rp2040"]
    test_prompts = ["Hello world", "Generate code", "Explain AI"]

    # First run - cache misses
    logger.info("First run (cache misses expected):")
    start_time = time.time()

    results1 = []
    for platform in platforms:
        result = profiler.profile_with_cache(
            platform=platform,
            model_size_mb=2.5,
            quantization="4bit",
            prompts=test_prompts,
        )
        results1.append(result)

    first_run_time = time.time() - start_time
    logger.info(f"First run completed in {first_run_time:.2f}s")

    # Second run - cache hits expected
    logger.info("Second run (cache hits expected):")
    start_time = time.time()

    results2 = []
    for platform in platforms:
        result = profiler.profile_with_cache(
            platform=platform,
            model_size_mb=2.5,
            quantization="4bit",
            prompts=test_prompts,
        )
        results2.append(result)

    second_run_time = time.time() - start_time
    logger.info(f"Second run completed in {second_run_time:.2f}s")

    # Performance improvement
    speedup = first_run_time / second_run_time if second_run_time > 0 else float("inf")
    logger.info(f"Speedup from caching: {speedup:.2f}x")

    # Show statistics
    stats = profiler.get_performance_stats()
    logger.info(f"Performance stats: {stats}")

    # Verify results are identical (except metadata)
    for i in range(len(results1)):
        r1 = results1[i].copy()
        r2 = results2[i].copy()

        # Remove metadata for comparison
        r1.pop("profiling_metadata", None)
        r2.pop("profiling_metadata", None)

        if r1 != r2:
            logger.warning(f"Results differ for platform {platforms[i]}")
        else:
            logger.debug(f"Results identical for platform {platforms[i]}")

    logger.info("Cache performance demo completed")


if __name__ == "__main__":
    cache_performance_demo()
