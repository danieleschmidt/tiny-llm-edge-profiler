"""
Advanced Database and Storage Optimization System for Generation 3
Provides comprehensive storage optimization capabilities including:
- Efficient storage formats for profiling results with compression
- Database connection pooling and query optimization
- Batch processing for bulk operations with transaction management
- Data compression and archival strategies
- Index optimization for fast querying and retrieval
- Time-series database integration for metrics
- Data lifecycle management and automatic cleanup
"""

import time
import sqlite3
import threading
import asyncio
import json
import gzip
import lz4.frame
import pickle
import struct
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import numpy as np

from .exceptions import TinyLLMProfilerError, ResourceError
from .logging_config import get_logger, PerformanceLogger
from .results import ProfileResults

logger = get_logger("storage_optimizer")
perf_logger = PerformanceLogger()


class CompressionAlgorithm(str, Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


class StorageFormat(str, Enum):
    """Storage format types."""
    JSON = "json"
    BINARY = "binary"
    PARQUET = "parquet"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"


class DataLifecycleStage(str, Enum):
    """Data lifecycle stages."""
    ACTIVE = "active"        # Recently created, frequently accessed
    WARM = "warm"           # Occasionally accessed
    COLD = "cold"           # Rarely accessed, can be compressed
    ARCHIVED = "archived"   # Very old, compressed and moved to archive storage
    EXPIRED = "expired"     # Can be deleted


@dataclass
class StorageStats:
    """Storage system statistics."""
    total_records: int = 0
    total_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0
    query_count: int = 0
    avg_query_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    index_efficiency: float = 0.0
    
    # Lifecycle stats
    active_records: int = 0
    warm_records: int = 0
    cold_records: int = 0
    archived_records: int = 0
    
    def update_query_stats(self, query_time_ms: float):
        """Update query performance statistics."""
        if self.query_count == 0:
            self.avg_query_time_ms = query_time_ms
        else:
            # Exponential moving average
            self.avg_query_time_ms = (
                0.9 * self.avg_query_time_ms + 0.1 * query_time_ms
            )
        self.query_count += 1
    
    def calculate_compression_ratio(self):
        """Calculate compression ratio."""
        if self.total_size_bytes > 0:
            self.compression_ratio = self.total_size_bytes / max(self.compressed_size_bytes, 1)


class DataCompressor:
    """Handles data compression and decompression."""
    
    ALGORITHMS = {
        CompressionAlgorithm.NONE: {
            "compress": lambda data: data,
            "decompress": lambda data: data
        },
        CompressionAlgorithm.GZIP: {
            "compress": lambda data: gzip.compress(data),
            "decompress": lambda data: gzip.decompress(data)
        },
        CompressionAlgorithm.LZ4: {
            "compress": lambda data: lz4.frame.compress(data),
            "decompress": lambda data: lz4.frame.decompress(data)
        }
    }
    
    @classmethod
    def compress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4
    ) -> Tuple[bytes, CompressionAlgorithm]:
        """Compress data using specified algorithm."""
        if algorithm not in self.ALGORITHMS:
            algorithm = CompressionAlgorithm.LZ4
        
        try:
            compress_func = self.ALGORITHMS[algorithm]["compress"]
            compressed_data = compress_func(data)
            return compressed_data, algorithm
        except Exception as e:
            logger.warning(f"Compression failed: {e}, using uncompressed data")
            return data, CompressionAlgorithm.NONE
    
    @classmethod
    def decompress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm
    ) -> bytes:
        """Decompress data using specified algorithm."""
        if algorithm not in self.ALGORITHMS:
            return data
        
        try:
            decompress_func = self.ALGORITHMS[algorithm]["decompress"]
            return decompress_func(data)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    @classmethod
    def get_best_algorithm(self, data: bytes) -> CompressionAlgorithm:
        """Determine best compression algorithm for given data."""
        if len(data) < 100:  # Too small to benefit from compression
            return CompressionAlgorithm.NONE
        
        # Test different algorithms and pick the best compression ratio
        best_algorithm = CompressionAlgorithm.NONE
        best_ratio = 1.0
        
        for algorithm in [CompressionAlgorithm.LZ4, CompressionAlgorithm.GZIP]:
            try:
                compressed, _ = self.compress(data, algorithm)
                ratio = len(data) / len(compressed)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_algorithm = algorithm
            except Exception:
                continue
        
        return best_algorithm


class SerializationHandler:
    """Handles different serialization formats."""
    
    @staticmethod
    def serialize(
        data: Any,
        format_type: StorageFormat = StorageFormat.BINARY
    ) -> bytes:
        """Serialize data to bytes using specified format."""
        try:
            if format_type == StorageFormat.JSON:
                json_str = json.dumps(data, default=str, separators=(',', ':'))
                return json_str.encode('utf-8')
            
            elif format_type == StorageFormat.BINARY:
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            elif format_type == StorageFormat.MSGPACK:
                try:
                    import msgpack
                    return msgpack.packb(data)
                except ImportError:
                    # Fallback to pickle if msgpack not available
                    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            else:
                # Default to pickle
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    @staticmethod
    def deserialize(
        data: bytes,
        format_type: StorageFormat = StorageFormat.BINARY
    ) -> Any:
        """Deserialize bytes to data using specified format."""
        try:
            if format_type == StorageFormat.JSON:
                json_str = data.decode('utf-8')
                return json.loads(json_str)
            
            elif format_type == StorageFormat.BINARY:
                return pickle.loads(data)
            
            elif format_type == StorageFormat.MSGPACK:
                try:
                    import msgpack
                    return msgpack.unpackb(data, raw=False)
                except ImportError:
                    # Fallback to pickle
                    return pickle.loads(data)
            
            else:
                # Default to pickle
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise


class OptimizedDatabase:
    """Optimized SQLite database with connection pooling and query optimization."""
    
    def __init__(
        self,
        db_path: Union[str, Path],
        pool_size: int = 10,
        enable_wal_mode: bool = True,
        enable_foreign_keys: bool = True,
        cache_size_mb: int = 64
    ):
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.enable_wal_mode = enable_wal_mode
        self.enable_foreign_keys = enable_foreign_keys
        self.cache_size_mb = cache_size_mb
        
        # Connection pool
        self._connection_pool: List[sqlite3.Connection] = []
        self._pool_lock = threading.Lock()
        self._available_connections: List[sqlite3.Connection] = []
        
        # Query cache
        self._query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._query_cache_lock = threading.RLock()
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Statistics
        self.stats = StorageStats()
        
        # Initialize database and pool
        self._initialize_database()
        self._initialize_connection_pool()
        
        logger.info(f"Initialized optimized database: {db_path}")
    
    def _initialize_database(self):
        """Initialize database with optimizations."""
        # Create database directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create initial connection to set up database
        conn = sqlite3.connect(str(self.db_path))
        
        try:
            # Enable optimizations
            if self.enable_wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
            
            conn.execute(f"PRAGMA cache_size=-{self.cache_size_mb * 1024}")  # Negative for KB
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
            
            if self.enable_foreign_keys:
                conn.execute("PRAGMA foreign_keys=ON")
            
            # Create core tables
            self._create_tables(conn)
            
            conn.commit()
            
        finally:
            conn.close()
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables with optimized schema."""
        
        # Profiling results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiling_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                model_name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                data_format TEXT NOT NULL,
                compression_algorithm TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                data_blob BLOB NOT NULL,
                lifecycle_stage TEXT NOT NULL DEFAULT 'active',
                last_accessed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        
        # Create indexes for common queries
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_session_id ON profiling_results(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_platform ON profiling_results(platform)",
            "CREATE INDEX IF NOT EXISTS idx_model_name ON profiling_results(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_created_at ON profiling_results(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_lifecycle_stage ON profiling_results(lifecycle_stage)",
            "CREATE INDEX IF NOT EXISTS idx_composite_search ON profiling_results(platform, model_name, created_at)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
        
        # Time-series metrics table for detailed performance data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS time_series_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                metadata TEXT
            )
        """)
        
        # Index for time-series queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timeseries_lookup 
            ON time_series_metrics(session_id, metric_name, timestamp)
        """)
        
        # Metadata table for storage optimization tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS storage_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _initialize_connection_pool(self):
        """Initialize connection pool."""
        with self._pool_lock:
            for _ in range(self.pool_size):
                conn = self._create_optimized_connection()
                self._connection_pool.append(conn)
                self._available_connections.append(conn)
    
    def _create_optimized_connection(self) -> sqlite3.Connection:
        """Create an optimized database connection."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False  # Allow sharing between threads
        )
        
        # Enable row factory for better result handling
        conn.row_factory = sqlite3.Row
        
        # Apply optimizations
        conn.execute(f"PRAGMA cache_size=-{self.cache_size_mb * 1024}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        if self.enable_foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        
        try:
            # Get connection from pool
            with self._pool_lock:
                if self._available_connections:
                    conn = self._available_connections.pop()
                else:
                    # Create temporary connection if pool is exhausted
                    conn = self._create_optimized_connection()
            
            yield conn
            
        finally:
            # Return connection to pool
            if conn:
                try:
                    # Rollback any uncommitted transaction
                    conn.rollback()
                    
                    with self._pool_lock:
                        if len(self._available_connections) < self.pool_size:
                            self._available_connections.append(conn)
                        else:
                            # Close excess connection
                            conn.close()
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except:
                        pass
    
    def execute_query(
        self,
        query: str,
        params: Tuple = (),
        fetch_results: bool = True,
        use_cache: bool = False
    ) -> Optional[List[sqlite3.Row]]:
        """Execute SQL query with optimization."""
        start_time = time.time()
        
        # Check cache if enabled
        if use_cache and fetch_results:
            cache_key = hashlib.md5((query + str(params)).encode()).hexdigest()
            
            with self._query_cache_lock:
                if cache_key in self._query_cache:
                    cached_result, cache_time = self._query_cache[cache_key]
                    
                    # Check if cache entry is still valid
                    if (datetime.now() - cache_time).total_seconds() < self._cache_ttl_seconds:
                        self.stats.cache_hit_rate = (
                            0.9 * self.stats.cache_hit_rate + 0.1 * 1.0
                        )
                        return cached_result
        
        # Execute query
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if fetch_results:
                    results = cursor.fetchall()
                    
                    # Cache results if enabled
                    if use_cache:
                        with self._query_cache_lock:
                            self._query_cache[cache_key] = (results, datetime.now())
                            
                            # Limit cache size
                            if len(self._query_cache) > 1000:
                                # Remove oldest entries
                                sorted_cache = sorted(
                                    self._query_cache.items(),
                                    key=lambda x: x[1][1]
                                )
                                for key, _ in sorted_cache[:100]:  # Remove oldest 100
                                    del self._query_cache[key]
                    
                    return results
                else:
                    conn.commit()
                    return None
                    
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise
        
        finally:
            # Update statistics
            query_time_ms = (time.time() - start_time) * 1000
            self.stats.update_query_stats(query_time_ms)
            
            # Update cache miss rate if using cache
            if use_cache and fetch_results:
                self.stats.cache_hit_rate = (
                    0.9 * self.stats.cache_hit_rate + 0.1 * 0.0
                )
    
    def execute_batch(
        self,
        query: str,
        params_list: List[Tuple],
        batch_size: int = 1000
    ) -> None:
        """Execute batch operations efficiently."""
        try:
            with self.get_connection() as conn:
                # Process in batches to avoid memory issues
                for i in range(0, len(params_list), batch_size):
                    batch = params_list[i:i + batch_size]
                    
                    conn.executemany(query, batch)
                    
                    # Commit batch
                    conn.commit()
                    
                    logger.debug(f"Processed batch {i // batch_size + 1}: {len(batch)} records")
                
        except Exception as e:
            logger.error(f"Batch execution error: {e}")
            raise
    
    def optimize_database(self):
        """Run database optimization tasks."""
        try:
            with self.get_connection() as conn:
                # Analyze query patterns to optimize indexes
                conn.execute("ANALYZE")
                
                # Vacuum to reclaim space (only if needed)
                result = conn.execute("PRAGMA page_count").fetchone()
                page_count = result[0] if result else 0
                
                if page_count > 10000:  # Only vacuum large databases
                    logger.info("Running VACUUM to optimize database...")
                    conn.execute("VACUUM")
                
                # Update statistics
                conn.execute("PRAGMA incremental_vacuum")
                
                logger.info("Database optimization completed")
                
        except Exception as e:
            logger.error(f"Database optimization error: {e}")
    
    def close(self):
        """Close all database connections."""
        with self._pool_lock:
            for conn in self._connection_pool:
                try:
                    conn.close()
                except:
                    pass
            
            self._connection_pool.clear()
            self._available_connections.clear()
        
        logger.info("Database connections closed")


class DataLifecycleManager:
    """Manages data lifecycle and archival strategies."""
    
    def __init__(
        self,
        database: OptimizedDatabase,
        active_retention_days: int = 7,
        warm_retention_days: int = 30,
        cold_retention_days: int = 90,
        archive_retention_days: int = 365
    ):
        self.database = database
        self.active_retention_days = active_retention_days
        self.warm_retention_days = warm_retention_days
        self.cold_retention_days = cold_retention_days
        self.archive_retention_days = archive_retention_days
        
        # Background task control
        self.running = False
        self.lifecycle_thread: Optional[threading.Thread] = None
        
        logger.info("Data lifecycle manager initialized")
    
    def start_lifecycle_management(self):
        """Start background lifecycle management."""
        if self.running:
            return
        
        self.running = True
        self.lifecycle_thread = threading.Thread(
            target=self._lifecycle_management_loop,
            daemon=True
        )
        self.lifecycle_thread.start()
        
        logger.info("Data lifecycle management started")
    
    def stop_lifecycle_management(self):
        """Stop background lifecycle management."""
        if not self.running:
            return
        
        self.running = False
        
        if self.lifecycle_thread:
            self.lifecycle_thread.join(timeout=30.0)
        
        logger.info("Data lifecycle management stopped")
    
    def _lifecycle_management_loop(self):
        """Background loop for lifecycle management."""
        while self.running:
            try:
                # Run lifecycle transitions
                self._transition_data_stages()
                
                # Clean up expired data
                self._cleanup_expired_data()
                
                # Compress cold data
                self._compress_cold_data()
                
                # Update statistics
                self._update_lifecycle_stats()
                
                # Sleep for 1 hour
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Lifecycle management error: {e}")
                time.sleep(1800)  # Wait 30 minutes on error
    
    def _transition_data_stages(self):
        """Transition data between lifecycle stages."""
        now = datetime.now()
        
        transitions = [
            # Active -> Warm
            (
                DataLifecycleStage.ACTIVE,
                DataLifecycleStage.WARM,
                now - timedelta(days=self.active_retention_days)
            ),
            # Warm -> Cold
            (
                DataLifecycleStage.WARM,
                DataLifecycleStage.COLD,
                now - timedelta(days=self.warm_retention_days)
            ),
            # Cold -> Archived
            (
                DataLifecycleStage.COLD,
                DataLifecycleStage.ARCHIVED,
                now - timedelta(days=self.cold_retention_days)
            )
        ]
        
        for from_stage, to_stage, cutoff_time in transitions:
            query = """
                UPDATE profiling_results 
                SET lifecycle_stage = ?
                WHERE lifecycle_stage = ? AND created_at < ?
            """
            
            results = self.database.execute_query(
                query,
                (to_stage.value, from_stage.value, cutoff_time),
                fetch_results=False
            )
            
            logger.debug(f"Transitioned data from {from_stage.value} to {to_stage.value}")
    
    def _cleanup_expired_data(self):
        """Remove expired data that's past retention period."""
        cutoff_time = datetime.now() - timedelta(days=self.archive_retention_days)
        
        # Get count of records to be deleted
        count_query = """
            SELECT COUNT(*) FROM profiling_results 
            WHERE lifecycle_stage = ? AND created_at < ?
        """
        
        result = self.database.execute_query(
            count_query,
            (DataLifecycleStage.ARCHIVED.value, cutoff_time)
        )
        
        if result and result[0][0] > 0:
            delete_count = result[0][0]
            
            # Delete expired records
            delete_query = """
                DELETE FROM profiling_results 
                WHERE lifecycle_stage = ? AND created_at < ?
            """
            
            self.database.execute_query(
                delete_query,
                (DataLifecycleStage.ARCHIVED.value, cutoff_time),
                fetch_results=False
            )
            
            # Also clean up related time-series data
            ts_delete_query = """
                DELETE FROM time_series_metrics 
                WHERE timestamp < ?
            """
            
            self.database.execute_query(
                ts_delete_query,
                (cutoff_time,),
                fetch_results=False
            )
            
            logger.info(f"Cleaned up {delete_count} expired records")
    
    def _compress_cold_data(self):
        """Re-compress cold data with better algorithms."""
        # Find cold data that could benefit from better compression
        query = """
            SELECT id, data_blob, compression_algorithm, original_size, compressed_size
            FROM profiling_results 
            WHERE lifecycle_stage = ? AND compression_algorithm != ?
            LIMIT 100
        """
        
        results = self.database.execute_query(
            query,
            (DataLifecycleStage.COLD.value, CompressionAlgorithm.GZIP.value)
        )
        
        if results:
            for row in results:
                try:
                    # Decompress current data
                    current_algorithm = CompressionAlgorithm(row['compression_algorithm'])
                    decompressed_data = DataCompressor.decompress(
                        row['data_blob'],
                        current_algorithm
                    )
                    
                    # Re-compress with better algorithm
                    new_compressed, new_algorithm = DataCompressor.compress(
                        decompressed_data,
                        CompressionAlgorithm.GZIP
                    )
                    
                    # Update if compression improved
                    if len(new_compressed) < len(row['data_blob']):
                        update_query = """
                            UPDATE profiling_results 
                            SET data_blob = ?, compression_algorithm = ?, compressed_size = ?
                            WHERE id = ?
                        """
                        
                        self.database.execute_query(
                            update_query,
                            (new_compressed, new_algorithm.value, len(new_compressed), row['id']),
                            fetch_results=False
                        )
                        
                        logger.debug(f"Recompressed record {row['id']}: {len(row['data_blob'])} -> {len(new_compressed)} bytes")
                    
                except Exception as e:
                    logger.error(f"Error recompressing record {row['id']}: {e}")
    
    def _update_lifecycle_stats(self):
        """Update lifecycle statistics."""
        stages_query = """
            SELECT lifecycle_stage, COUNT(*) as count
            FROM profiling_results 
            GROUP BY lifecycle_stage
        """
        
        results = self.database.execute_query(stages_query)
        
        if results:
            # Reset stats
            self.database.stats.active_records = 0
            self.database.stats.warm_records = 0
            self.database.stats.cold_records = 0
            self.database.stats.archived_records = 0
            
            for row in results:
                stage = row['lifecycle_stage']
                count = row['count']
                
                if stage == DataLifecycleStage.ACTIVE.value:
                    self.database.stats.active_records = count
                elif stage == DataLifecycleStage.WARM.value:
                    self.database.stats.warm_records = count
                elif stage == DataLifecycleStage.COLD.value:
                    self.database.stats.cold_records = count
                elif stage == DataLifecycleStage.ARCHIVED.value:
                    self.database.stats.archived_records = count
            
            self.database.stats.total_records = sum([
                self.database.stats.active_records,
                self.database.stats.warm_records,
                self.database.stats.cold_records,
                self.database.stats.archived_records
            ])


class StorageOptimizer:
    """Main storage optimizer coordinating all storage optimization components."""
    
    def __init__(
        self,
        db_path: Union[str, Path] = "profiling_data.db",
        enable_compression: bool = True,
        enable_lifecycle_management: bool = True,
        default_format: StorageFormat = StorageFormat.BINARY,
        default_compression: CompressionAlgorithm = CompressionAlgorithm.LZ4
    ):
        self.db_path = Path(db_path)
        self.enable_compression = enable_compression
        self.enable_lifecycle_management = enable_lifecycle_management
        self.default_format = default_format
        self.default_compression = default_compression
        
        # Initialize components
        self.database = OptimizedDatabase(self.db_path)
        
        if enable_lifecycle_management:
            self.lifecycle_manager = DataLifecycleManager(self.database)
            self.lifecycle_manager.start_lifecycle_management()
        
        # Background optimization
        self._optimization_executor = ThreadPoolExecutor(max_workers=2)
        self._start_background_optimization()
        
        logger.info("Storage optimizer initialized")
    
    def store_profiling_result(
        self,
        result: ProfileResults,
        session_id: Optional[str] = None,
        format_type: Optional[StorageFormat] = None,
        compression: Optional[CompressionAlgorithm] = None
    ) -> str:
        """
        Store profiling result with optimization.
        
        Args:
            result: ProfileResults to store
            session_id: Optional session identifier
            format_type: Storage format to use
            compression: Compression algorithm to use
            
        Returns:
            Stored record ID
        """
        session_id = session_id or getattr(result, 'session_id', f"session_{int(time.time())}")
        format_type = format_type or self.default_format
        compression = compression or self.default_compression
        
        try:
            # Serialize result
            serialized_data = SerializationHandler.serialize(result, format_type)
            original_size = len(serialized_data)
            
            # Compress if enabled
            if self.enable_compression and compression != CompressionAlgorithm.NONE:
                compressed_data, used_compression = DataCompressor.compress(
                    serialized_data, compression
                )
            else:
                compressed_data = serialized_data
                used_compression = CompressionAlgorithm.NONE
            
            compressed_size = len(compressed_data)
            
            # Store in database
            insert_query = """
                INSERT INTO profiling_results (
                    session_id, platform, model_name, created_at,
                    data_format, compression_algorithm,
                    original_size, compressed_size, data_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.database.execute_query(
                insert_query,
                (
                    session_id,
                    result.platform,
                    result.model_name,
                    datetime.now(),
                    format_type.value,
                    used_compression.value,
                    original_size,
                    compressed_size,
                    compressed_data
                ),
                fetch_results=False
            )
            
            # Update statistics
            self.database.stats.total_records += 1
            self.database.stats.total_size_bytes += original_size
            self.database.stats.compressed_size_bytes += compressed_size
            self.database.stats.calculate_compression_ratio()
            
            logger.debug(f"Stored profiling result: {session_id} ({original_size} -> {compressed_size} bytes)")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to store profiling result: {e}")
            raise
    
    def retrieve_profiling_result(
        self,
        session_id: str,
        update_access_stats: bool = True
    ) -> Optional[ProfileResults]:
        """
        Retrieve profiling result by session ID.
        
        Args:
            session_id: Session identifier
            update_access_stats: Whether to update access statistics
            
        Returns:
            ProfileResults if found, None otherwise
        """
        try:
            query = """
                SELECT data_blob, data_format, compression_algorithm, id
                FROM profiling_results 
                WHERE session_id = ?
                ORDER BY created_at DESC 
                LIMIT 1
            """
            
            results = self.database.execute_query(query, (session_id,), use_cache=True)
            
            if not results:
                return None
            
            row = results[0]
            
            # Update access statistics if requested
            if update_access_stats:
                self._update_access_stats(row['id'])
            
            # Decompress data
            compression_algorithm = CompressionAlgorithm(row['compression_algorithm'])
            decompressed_data = DataCompressor.decompress(
                row['data_blob'], 
                compression_algorithm
            )
            
            # Deserialize result
            format_type = StorageFormat(row['data_format'])
            result = SerializationHandler.deserialize(decompressed_data, format_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve profiling result {session_id}: {e}")
            return None
    
    def query_profiling_results(
        self,
        platform: Optional[str] = None,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        include_archived: bool = False
    ) -> List[Tuple[str, ProfileResults]]:
        """
        Query profiling results with filters.
        
        Args:
            platform: Filter by platform
            model_name: Filter by model name
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum results to return
            include_archived: Whether to include archived results
            
        Returns:
            List of (session_id, ProfileResults) tuples
        """
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if platform:
                conditions.append("platform = ?")
                params.append(platform)
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            if start_date:
                conditions.append("created_at >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("created_at <= ?")
                params.append(end_date)
            
            if not include_archived:
                conditions.append("lifecycle_stage != ?")
                params.append(DataLifecycleStage.ARCHIVED.value)
            
            # Construct query
            base_query = """
                SELECT session_id, data_blob, data_format, compression_algorithm, id
                FROM profiling_results
            """
            
            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)
            
            base_query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            results = self.database.execute_query(base_query, tuple(params), use_cache=True)
            
            if not results:
                return []
            
            # Process results
            profiling_results = []
            
            for row in results:
                try:
                    # Decompress and deserialize
                    compression_algorithm = CompressionAlgorithm(row['compression_algorithm'])
                    decompressed_data = DataCompressor.decompress(
                        row['data_blob'],
                        compression_algorithm
                    )
                    
                    format_type = StorageFormat(row['data_format'])
                    result = SerializationHandler.deserialize(decompressed_data, format_type)
                    
                    profiling_results.append((row['session_id'], result))
                    
                    # Update access stats for retrieved records
                    self._update_access_stats(row['id'])
                    
                except Exception as e:
                    logger.error(f"Failed to process result {row['session_id']}: {e}")
                    continue
            
            return profiling_results
            
        except Exception as e:
            logger.error(f"Failed to query profiling results: {e}")
            return []
    
    def store_time_series_metric(
        self,
        session_id: str,
        metric_name: str,
        metric_value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store time-series metric data."""
        timestamp = timestamp or datetime.now()
        metadata_json = json.dumps(metadata) if metadata else None
        
        try:
            insert_query = """
                INSERT INTO time_series_metrics (
                    session_id, metric_name, metric_value, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?)
            """
            
            self.database.execute_query(
                insert_query,
                (session_id, metric_name, metric_value, timestamp, metadata_json),
                fetch_results=False
            )
            
        except Exception as e:
            logger.error(f"Failed to store time-series metric: {e}")
    
    def get_time_series_data(
        self,
        session_id: str,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Tuple[datetime, float, Dict[str, Any]]]:
        """Retrieve time-series data for a metric."""
        try:
            conditions = ["session_id = ?", "metric_name = ?"]
            params = [session_id, metric_name]
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            query = f"""
                SELECT timestamp, metric_value, metadata
                FROM time_series_metrics 
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp
            """
            
            results = self.database.execute_query(query, tuple(params), use_cache=True)
            
            time_series_data = []
            for row in results:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                time_series_data.append((
                    datetime.fromisoformat(row['timestamp']),
                    row['metric_value'],
                    metadata
                ))
            
            return time_series_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve time-series data: {e}")
            return []
    
    def _update_access_stats(self, record_id: int):
        """Update access statistics for a record."""
        update_query = """
            UPDATE profiling_results 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """
        
        try:
            self.database.execute_query(
                update_query,
                (datetime.now(), record_id),
                fetch_results=False
            )
        except Exception as e:
            logger.debug(f"Failed to update access stats: {e}")
    
    def _start_background_optimization(self):
        """Start background optimization tasks."""
        # Schedule periodic database optimization
        self._optimization_executor.submit(self._periodic_optimization_loop)
    
    def _periodic_optimization_loop(self):
        """Periodic optimization tasks."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                
                # Optimize database
                self.database.optimize_database()
                
                # Clear old query cache entries
                with self.database._query_cache_lock:
                    current_time = datetime.now()
                    expired_keys = [
                        key for key, (_, cache_time) in self.database._query_cache.items()
                        if (current_time - cache_time).total_seconds() > self.database._cache_ttl_seconds
                    ]
                    
                    for key in expired_keys:
                        del self.database._query_cache[key]
                
                logger.debug("Periodic optimization completed")
                
            except Exception as e:
                logger.error(f"Periodic optimization error: {e}")
                time.sleep(1800)  # Wait 30 minutes on error
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        return {
            "database_stats": asdict(self.database.stats),
            "database_path": str(self.db_path),
            "compression_enabled": self.enable_compression,
            "lifecycle_management_enabled": self.enable_lifecycle_management,
            "default_format": self.default_format.value,
            "default_compression": self.default_compression.value,
            "connection_pool_size": self.database.pool_size,
            "query_cache_size": len(self.database._query_cache),
            "background_tasks_active": not self._optimization_executor._shutdown
        }
    
    def shutdown(self):
        """Shutdown the storage optimizer."""
        logger.info("Shutting down storage optimizer...")
        
        # Stop lifecycle management
        if hasattr(self, 'lifecycle_manager'):
            self.lifecycle_manager.stop_lifecycle_management()
        
        # Shutdown optimization executor
        self._optimization_executor.shutdown(wait=True, timeout=30.0)
        
        # Close database
        self.database.close()
        
        logger.info("Storage optimizer shutdown complete")


# Global storage optimizer instance
_global_storage_optimizer: Optional[StorageOptimizer] = None


def get_storage_optimizer(**kwargs) -> StorageOptimizer:
    """Get or create the global storage optimizer."""
    global _global_storage_optimizer
    
    if _global_storage_optimizer is None:
        _global_storage_optimizer = StorageOptimizer(**kwargs)
    
    return _global_storage_optimizer


def store_profiling_result(
    result: ProfileResults,
    session_id: Optional[str] = None,
    **kwargs
) -> str:
    """Store profiling result using optimized storage."""
    optimizer = get_storage_optimizer()
    return optimizer.store_profiling_result(result, session_id, **kwargs)


def retrieve_profiling_result(session_id: str) -> Optional[ProfileResults]:
    """Retrieve profiling result by session ID."""
    optimizer = get_storage_optimizer()
    return optimizer.retrieve_profiling_result(session_id)


def query_profiling_results(**kwargs) -> List[Tuple[str, ProfileResults]]:
    """Query profiling results with filters."""
    optimizer = get_storage_optimizer()
    return optimizer.query_profiling_results(**kwargs)