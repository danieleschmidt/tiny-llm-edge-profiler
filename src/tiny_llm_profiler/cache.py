"""
Intelligent caching system for the Tiny LLM Edge Profiler.
"""

import time
import hashlib
import pickle
import json
import threading
from typing import Any, Dict, Optional, Union, Callable, TypeVar, Generic, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import weakref

from .exceptions import TinyLLMProfilerError
from .logging_config import get_logger
from .security import secure_delete_file

logger = get_logger("cache")

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    key: str
    value: T
    created_at: datetime
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class InMemoryCache:
    """
    High-performance in-memory cache with LRU eviction and TTL support.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl_seconds: Optional[int] = None,
        cleanup_interval_seconds: int = 300
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU ordering
        self._lock = threading.RLock()
        self._total_size_bytes = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # Update access info
            entry.touch()
            self._update_access_order(key)
            
            self._hits += 1
            logger.debug(f"Cache hit for key: {key}")
            return entry.value
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Fallback estimate
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                ttl_seconds=ttl_seconds or self.default_ttl_seconds,
                size_bytes=size_bytes,
                tags=tags or []
            )
            
            # Ensure we have space
            self._ensure_space(size_bytes)
            
            # Add to cache
            self._cache[key] = entry
            self._access_order.append(key)
            self._total_size_bytes += size_bytes
            
            logger.debug(f"Cache put for key: {key} (size: {size_bytes} bytes)")
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self, tag: Optional[str] = None) -> int:
        """Clear cache entries, optionally by tag."""
        with self._lock:
            if tag is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._access_order.clear()
                self._total_size_bytes = 0
                logger.info(f"Cleared entire cache ({count} entries)")
                return count
            else:
                # Clear by tag
                keys_to_remove = [
                    key for key, entry in self._cache.items()
                    if tag in entry.tags
                ]
                
                for key in keys_to_remove:
                    self._remove_entry(key)
                
                logger.info(f"Cleared {len(keys_to_remove)} entries with tag: {tag}")
                return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0
            
            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self._total_size_bytes / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions
            }
    
    def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure there's space for a new entry."""
        # Check size limit
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order[0]
            self._remove_entry(oldest_key)
            self._evictions += 1
        
        # Check memory limit
        while (self._total_size_bytes + needed_bytes > self.max_memory_bytes and 
               self._access_order):
            oldest_key = self._access_order[0]
            self._remove_entry(oldest_key)
            self._evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            del self._cache[key]
            self._total_size_bytes -= entry.size_bytes
            
            if key in self._access_order:
                self._access_order.remove(key)
    
    def _update_access_order(self, key: str) -> None:
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while True:
            try:
                time.sleep(self.cleanup_interval_seconds)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class PersistentCache:
    """
    Persistent cache using filesystem storage with compression and encryption.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: float = 1.0,
        compression: bool = True,
        encryption_key: Optional[bytes] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.compression = compression
        self.encryption_key = encryption_key
        
        self._index_file = self.cache_dir / "cache_index.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Load index
        self._load_index()
        
        # Clean up on initialization
        self._cleanup_invalid_entries()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._index:
                return None
            
            entry_info = self._index[key_hash]
            
            # Check if expired
            if self._is_expired(entry_info):
                self.delete(key)
                return None
            
            # Load from file
            try:
                file_path = self.cache_dir / f"{key_hash}.cache"
                if not file_path.exists():
                    # Remove from index if file missing
                    del self._index[key_hash]
                    self._save_index()
                    return None
                
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Decrypt if needed
                if self.encryption_key:
                    data = self._decrypt(data)
                
                # Decompress if needed
                if self.compression:
                    import gzip
                    data = gzip.decompress(data)
                
                # Deserialize
                value = pickle.loads(data)
                
                # Update access time
                entry_info['last_accessed'] = datetime.now().isoformat()
                entry_info['access_count'] = entry_info.get('access_count', 0) + 1
                self._save_index()
                
                logger.debug(f"Persistent cache hit for key: {key}")
                return value
                
            except Exception as e:
                logger.error(f"Failed to load from persistent cache: {e}")
                self.delete(key)
                return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Put value in persistent cache."""
        with self._lock:
            try:
                # Serialize value
                data = pickle.dumps(value)
                
                # Compress if enabled
                if self.compression:
                    import gzip
                    data = gzip.compress(data)
                
                # Encrypt if enabled
                if self.encryption_key:
                    data = self._encrypt(data)
                
                # Check size limits
                if len(data) > self.max_size_bytes:
                    logger.warning(f"Value too large for persistent cache: {len(data)} bytes")
                    return
                
                # Ensure space
                self._ensure_space(len(data))
                
                # Save to file
                key_hash = self._hash_key(key)
                file_path = self.cache_dir / f"{key_hash}.cache"
                
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Update index
                now = datetime.now()
                self._index[key_hash] = {
                    'original_key': key,
                    'created_at': now.isoformat(),
                    'last_accessed': now.isoformat(),
                    'access_count': 0,
                    'ttl_seconds': ttl_seconds,
                    'size_bytes': len(data),
                    'tags': tags or [],
                    'file_path': str(file_path)
                }
                
                self._save_index()
                logger.debug(f"Persistent cache put for key: {key}")
                
            except Exception as e:
                logger.error(f"Failed to save to persistent cache: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete from persistent cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._index:
                return False
            
            # Remove file
            try:
                file_path = Path(self._index[key_hash]['file_path'])
                if file_path.exists():
                    secure_delete_file(file_path)
            except Exception as e:
                logger.error(f"Failed to delete cache file: {e}")
            
            # Remove from index
            del self._index[key_hash]
            self._save_index()
            
            return True
    
    def clear(self, tag: Optional[str] = None) -> int:
        """Clear persistent cache."""
        with self._lock:
            if tag is None:
                # Clear all
                count = len(self._index)
                
                # Delete all files
                for entry_info in self._index.values():
                    try:
                        file_path = Path(entry_info['file_path'])
                        if file_path.exists():
                            secure_delete_file(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete cache file: {e}")
                
                self._index.clear()
                self._save_index()
                
                logger.info(f"Cleared persistent cache ({count} entries)")
                return count
            else:
                # Clear by tag
                keys_to_remove = [
                    key_hash for key_hash, entry_info in self._index.items()
                    if tag in entry_info.get('tags', [])
                ]
                
                for key_hash in keys_to_remove:
                    try:
                        file_path = Path(self._index[key_hash]['file_path'])
                        if file_path.exists():
                            secure_delete_file(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete cache file: {e}")
                    
                    del self._index[key_hash]
                
                self._save_index()
                logger.info(f"Cleared {len(keys_to_remove)} entries with tag: {tag}")
                return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistent cache statistics."""
        with self._lock:
            total_size = sum(entry.get('size_bytes', 0) for entry in self._index.values())
            
            return {
                "entries": len(self._index),
                "total_size_mb": total_size / (1024 * 1024),
                "max_size_gb": self.max_size_bytes / (1024 * 1024 * 1024),
                "cache_dir": str(self.cache_dir),
                "compression_enabled": self.compression,
                "encryption_enabled": self.encryption_key is not None
            }
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _is_expired(self, entry_info: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        ttl_seconds = entry_info.get('ttl_seconds')
        if ttl_seconds is None:
            return False
        
        created_at = datetime.fromisoformat(entry_info['created_at'])
        return (datetime.now() - created_at).total_seconds() > ttl_seconds
    
    def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure there's space for new entry."""
        current_size = sum(entry.get('size_bytes', 0) for entry in self._index.values())
        
        if current_size + needed_bytes <= self.max_size_bytes:
            return
        
        # Sort by last accessed (LRU)
        entries = [
            (key_hash, entry_info) 
            for key_hash, entry_info in self._index.items()
        ]
        entries.sort(key=lambda x: x[1].get('last_accessed', ''))
        
        # Remove oldest entries until we have space
        for key_hash, entry_info in entries:
            if current_size + needed_bytes <= self.max_size_bytes:
                break
            
            try:
                file_path = Path(entry_info['file_path'])
                if file_path.exists():
                    secure_delete_file(file_path)
                
                current_size -= entry_info.get('size_bytes', 0)
                del self._index[key_hash]
                
            except Exception as e:
                logger.error(f"Failed to remove cache entry during cleanup: {e}")
        
        self._save_index()
    
    def _load_index(self) -> None:
        """Load cache index from file."""
        try:
            if self._index_file.exists():
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")
            self._index = {}
    
    def _save_index(self) -> None:
        """Save cache index to file."""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _cleanup_invalid_entries(self) -> None:
        """Clean up invalid cache entries."""
        invalid_keys = []
        
        for key_hash, entry_info in self._index.items():
            file_path = Path(entry_info.get('file_path', ''))
            
            # Check if file exists
            if not file_path.exists():
                invalid_keys.append(key_hash)
                continue
            
            # Check if expired
            if self._is_expired(entry_info):
                invalid_keys.append(key_hash)
                try:
                    secure_delete_file(file_path)
                except Exception:
                    pass
        
        # Remove invalid entries
        for key_hash in invalid_keys:
            if key_hash in self._index:
                del self._index[key_hash]
        
        if invalid_keys:
            self._save_index()
            logger.info(f"Cleaned up {len(invalid_keys)} invalid cache entries")
    
    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data (placeholder - would use actual encryption)."""
        # This is a placeholder - in production, use proper encryption
        return data
    
    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data (placeholder - would use actual decryption)."""
        # This is a placeholder - in production, use proper decryption
        return data


class SmartCache:
    """
    Intelligent multi-level cache with automatic promotion/demotion.
    """
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        memory_cache_mb: int = 100,
        persistent_cache_dir: Optional[Union[str, Path]] = None,
        persistent_cache_gb: float = 1.0
    ):
        # Level 1: In-memory cache (fastest)
        self.memory_cache = InMemoryCache(
            max_size=memory_cache_size,
            max_memory_mb=memory_cache_mb,
            default_ttl_seconds=3600  # 1 hour
        )
        
        # Level 2: Persistent cache (slower but persistent)
        self.persistent_cache: Optional[PersistentCache] = None
        if persistent_cache_dir:
            self.persistent_cache = PersistentCache(
                cache_dir=persistent_cache_dir,
                max_size_gb=persistent_cache_gb
            )
        
        self._promotion_threshold = 3  # Promote to memory after 3 accesses
        self._access_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        with self._lock:
            # Try memory cache first
            value = self.memory_cache.get(key)
            if value is not None:
                return value
            
            # Try persistent cache
            if self.persistent_cache:
                value = self.persistent_cache.get(key)
                if value is not None:
                    # Track access for potential promotion
                    self._access_counts[key] = self._access_counts.get(key, 0) + 1
                    
                    # Promote to memory cache if accessed frequently
                    if self._access_counts[key] >= self._promotion_threshold:
                        self.memory_cache.put(key, value, ttl_seconds=3600)
                        logger.debug(f"Promoted key to memory cache: {key}")
                    
                    return value
            
            return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        prefer_memory: bool = True
    ) -> None:
        """Put value in multi-level cache."""
        with self._lock:
            if prefer_memory:
                # Try memory cache first
                self.memory_cache.put(key, value, ttl_seconds, tags)
                
                # Also store in persistent cache if available
                if self.persistent_cache:
                    self.persistent_cache.put(key, value, ttl_seconds, tags)
            else:
                # Store only in persistent cache
                if self.persistent_cache:
                    self.persistent_cache.put(key, value, ttl_seconds, tags)
                else:
                    # Fallback to memory cache
                    self.memory_cache.put(key, value, ttl_seconds, tags)
    
    def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        with self._lock:
            deleted = False
            
            # Remove from memory cache
            if self.memory_cache.delete(key):
                deleted = True
            
            # Remove from persistent cache
            if self.persistent_cache and self.persistent_cache.delete(key):
                deleted = True
            
            # Clean up access tracking
            if key in self._access_counts:
                del self._access_counts[key]
            
            return deleted
    
    def clear(self, tag: Optional[str] = None) -> int:
        """Clear all cache levels."""
        with self._lock:
            total_cleared = 0
            
            total_cleared += self.memory_cache.clear(tag)
            
            if self.persistent_cache:
                total_cleared += self.persistent_cache.clear(tag)
            
            if tag is None:
                self._access_counts.clear()
            
            return total_cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            stats = {
                "memory_cache": self.memory_cache.get_stats(),
                "access_tracking_entries": len(self._access_counts)
            }
            
            if self.persistent_cache:
                stats["persistent_cache"] = self.persistent_cache.get_stats()
            
            return stats


def cached(
    ttl_seconds: Optional[int] = None,
    tags: Optional[List[str]] = None,
    key_func: Optional[Callable] = None,
    cache_instance: Optional[Union[InMemoryCache, PersistentCache, SmartCache]] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl_seconds: Time to live for cached result
        tags: Tags for cache management
        key_func: Custom function to generate cache key
        cache_instance: Specific cache instance to use
    """
    def decorator(func: Callable) -> Callable:
        # Use global cache if none specified
        nonlocal cache_instance
        if cache_instance is None:
            cache_instance = _global_cache
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                func_name = f"{func.__module__}.{func.__qualname__}"
                args_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func_name}:{hashlib.md5(args_str.encode()).hexdigest()}"
            
            # Try to get from cache
            result = cache_instance.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for function: {func.__name__}")
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result, ttl_seconds, tags)
            
            logger.debug(f"Cached result for function: {func.__name__}")
            return result
        
        # Add cache management methods to wrapper
        wrapper.cache_clear = lambda: cache_instance.clear()
        wrapper.cache_stats = lambda: cache_instance.get_stats()
        
        return wrapper
    
    return decorator


# Global cache instance
_global_cache = SmartCache(
    memory_cache_size=1000,
    memory_cache_mb=50,
    persistent_cache_dir=Path.home() / ".cache" / "tiny_llm_profiler"
)


def get_cache() -> SmartCache:
    """Get the global cache instance."""
    return _global_cache


def clear_all_caches() -> int:
    """Clear all caches."""
    return _global_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics."""
    return _global_cache.get_stats()