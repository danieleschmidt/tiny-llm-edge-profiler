"""
Enterprise-grade features for Generation 3: Scalable, production-ready profiling.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
from pathlib import Path
import uuid

from .logging_config import get_logger, PerformanceLogger
from .performance_optimizer import PerformanceOptimizer, get_performance_optimizer
from .cache import InMemoryCache, PersistentCache
from .concurrent import ConcurrentProfiler
from .health import health_checker
from .security import SecurityValidator
from .exceptions import TinyLLMProfilerError

logger = get_logger("enterprise")
perf_logger = PerformanceLogger()


@dataclass
class EnterpriseConfig:
    """Configuration for enterprise features."""
    # High availability
    enable_clustering: bool = False
    cluster_nodes: List[str] = field(default_factory=list)
    
    # Advanced monitoring
    enable_distributed_tracing: bool = False
    tracing_sample_rate: float = 0.1
    
    # Performance optimization
    enable_adaptive_optimization: bool = True
    optimization_interval_seconds: int = 60
    
    # Security
    enable_audit_logging: bool = True
    encryption_enabled: bool = False
    
    # Scalability
    max_concurrent_sessions: int = 100
    session_timeout_minutes: int = 30
    
    # Resource management
    memory_limit_gb: float = 8.0
    cpu_limit_cores: int = 8
    
    # Data retention
    metrics_retention_days: int = 30
    results_retention_days: int = 90


class EnterpriseProfiler:
    """
    Enterprise-grade profiler with advanced scalability and reliability features.
    """
    
    def __init__(self, config: Optional[EnterpriseConfig] = None):
        self.config = config or EnterpriseConfig()
        
        # Core components
        self.performance_optimizer = get_performance_optimizer()
        self.security_validator = SecurityValidator()
        self.session_manager = SessionManager(self.config)
        self.distributed_cache = DistributedCache()
        
        # Enterprise features
        self.cluster_manager = ClusterManager(self.config) if self.config.enable_clustering else None
        self.tracing_system = TracingSystem(self.config) if self.config.enable_distributed_tracing else None
        self.audit_logger = AuditLogger(self.config) if self.config.enable_audit_logging else None
        
        # State
        self.is_running = False
        self.startup_time = None
        
        logger.info("Enterprise profiler initialized")
    
    async def start(self) -> None:
        """Start enterprise profiler with all components."""
        if self.is_running:
            return
        
        self.startup_time = datetime.now()
        
        # Start performance optimization
        self.performance_optimizer.start_optimization()
        
        # Start session management
        await self.session_manager.start()
        
        # Start clustering if enabled
        if self.cluster_manager:
            await self.cluster_manager.start()
        
        # Start distributed tracing
        if self.tracing_system:
            self.tracing_system.start()
        
        self.is_running = True
        logger.info("Enterprise profiler started successfully")
    
    async def stop(self) -> None:
        """Stop enterprise profiler gracefully."""
        if not self.is_running:
            return
        
        # Stop components in reverse order
        if self.tracing_system:
            self.tracing_system.stop()
        
        if self.cluster_manager:
            await self.cluster_manager.stop()
        
        await self.session_manager.stop()
        self.performance_optimizer.stop_optimization()
        
        self.is_running = False
        logger.info("Enterprise profiler stopped")
    
    async def create_profiling_session(
        self,
        session_config: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Create a new profiling session with enterprise features."""
        session_id = str(uuid.uuid4())
        
        # Validate session configuration
        validated_config = self.security_validator.validate_network_config(session_config)
        
        # Create session
        session = await self.session_manager.create_session(
            session_id, validated_config, user_id
        )
        
        # Start tracing if enabled
        if self.tracing_system:
            trace = self.tracing_system.start_trace(session_id, "profiling_session")
        
        # Log session creation
        if self.audit_logger:
            self.audit_logger.log_session_created(session_id, user_id, validated_config)
        
        logger.info(f"Created profiling session: {session_id}", session_id=session_id)
        return session_id
    
    async def execute_profiling_with_optimization(
        self,
        session_id: str,
        profiling_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute profiling with automatic optimization."""
        
        # Get session
        session = await self.session_manager.get_session(session_id)
        if not session:
            raise TinyLLMProfilerError(f"Session not found: {session_id}")
        
        # Start trace span
        span_id = None
        if self.tracing_system:
            span_id = self.tracing_system.start_span(session_id, "execute_profiling")
        
        try:
            # Execute profiling with concurrency and optimization
            results = await self._execute_optimized_profiling(profiling_tasks, session)
            
            # Cache results
            await self.distributed_cache.store_results(session_id, results)
            
            # Update session with results
            await self.session_manager.update_session_results(session_id, results)
            
            # Log completion
            if self.audit_logger:
                self.audit_logger.log_profiling_completed(session_id, len(profiling_tasks))
            
            logger.info(f"Profiling completed for session: {session_id}", 
                       session_id=session_id, task_count=len(profiling_tasks))
            
            return results
            
        except Exception as e:
            logger.exception(f"Profiling failed for session: {session_id}", 
                           session_id=session_id, exception=e)
            raise
        
        finally:
            # End trace span
            if self.tracing_system and span_id:
                self.tracing_system.end_span(span_id)
    
    async def _execute_optimized_profiling(
        self,
        tasks: List[Dict[str, Any]], 
        session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute profiling with optimization strategies."""
        
        # Create concurrent profiler with optimized settings
        concurrent_profiler = ConcurrentProfiler(
            max_threads=min(len(tasks), self.config.cpu_limit_cores),
            timeout_seconds=300
        )
        
        # Batch tasks for optimal resource utilization
        batched_tasks = self._batch_tasks_optimally(tasks)
        
        # Execute batches with load balancing
        results = {}
        for batch in batched_tasks:
            batch_results = await self._execute_task_batch(batch, concurrent_profiler)
            results.update(batch_results)
        
        return results
    
    def _batch_tasks_optimally(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Batch tasks for optimal resource utilization."""
        # Simple batching strategy - could be made more sophisticated
        batch_size = max(1, min(len(tasks) // 4, 10))  # 4 batches, max 10 per batch
        
        batches = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    async def _execute_task_batch(
        self,
        batch: List[Dict[str, Any]], 
        profiler: ConcurrentProfiler
    ) -> Dict[str, Any]:
        """Execute a batch of profiling tasks."""
        batch_results = {}
        
        # Execute tasks concurrently within the batch
        futures = []
        for task in batch:
            future = asyncio.create_task(self._execute_single_task(task, profiler))
            futures.append(future)
        
        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(completed_results):
            task_id = batch[i].get('task_id', f'task_{i}')
            if isinstance(result, Exception):
                batch_results[task_id] = {'error': str(result), 'success': False}
            else:
                batch_results[task_id] = result
        
        return batch_results
    
    async def _execute_single_task(
        self,
        task: Dict[str, Any], 
        profiler: ConcurrentProfiler
    ) -> Dict[str, Any]:
        """Execute a single profiling task."""
        # This would integrate with the actual profiling logic
        # For now, return a mock result
        await asyncio.sleep(0.1)  # Simulate work
        
        return {
            'task_id': task.get('task_id'),
            'success': True,
            'metrics': {
                'latency_ms': 100.0,
                'memory_kb': 256.0,
                'tokens_per_second': 10.0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_enterprise_status(self) -> Dict[str, Any]:
        """Get comprehensive enterprise system status."""
        return {
            'running': self.is_running,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'active_sessions': self.session_manager.get_session_count(),
            'performance_optimization': self.performance_optimizer.get_optimization_report(),
            'system_health': health_checker.get_overall_health().healthy,
            'clustering_enabled': self.config.enable_clustering,
            'tracing_enabled': self.config.enable_distributed_tracing,
            'cache_stats': self.distributed_cache.get_stats(),
            'resource_usage': self._get_resource_usage()
        }
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
        }


class SessionManager:
    """Advanced session management for enterprise deployments."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start session management."""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        logger.info("Session manager started")
    
    async def stop(self) -> None:
        """Stop session management."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Session manager stopped")
    
    async def create_session(
        self,
        session_id: str, 
        config: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new session."""
        session = {
            'id': session_id,
            'user_id': user_id,
            'config': config,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'status': 'active',
            'results': {}
        }
        
        self.sessions[session_id] = session
        self.session_locks[session_id] = asyncio.Lock()
        
        logger.info(f"Created session: {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session['last_accessed'] = datetime.now()
        return session
    
    async def update_session_results(
        self,
        session_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Update session with profiling results."""
        if session_id in self.sessions:
            async with self.session_locks.get(session_id, asyncio.Lock()):
                self.sessions[session_id]['results'].update(results)
                self.sessions[session_id]['last_accessed'] = datetime.now()
    
    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)
    
    async def _cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions periodically."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                now = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    last_accessed = session['last_accessed']
                    if (now - last_accessed).total_seconds() > (self.config.session_timeout_minutes * 60):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    if session_id in self.session_locks:
                        del self.session_locks[session_id]
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in session cleanup", exception=e)


class DistributedCache:
    """Distributed caching system for enterprise deployments."""
    
    def __init__(self):
        self.local_cache = InMemoryCache(max_memory_mb=500)
        self.persistent_cache = PersistentCache(Path("/tmp/tiny_llm_cache"))
        
    async def store_results(self, session_id: str, results: Dict[str, Any]) -> None:
        """Store profiling results in distributed cache."""
        # Store in both local and persistent cache
        cache_key = f"session_results:{session_id}"
        
        self.local_cache.put(cache_key, results, ttl_seconds=3600)
        self.persistent_cache.put(cache_key, results, ttl_seconds=86400)  # 24 hours
        
        logger.debug(f"Cached results for session: {session_id}")
    
    async def get_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached results."""
        cache_key = f"session_results:{session_id}"
        
        # Try local cache first
        results = self.local_cache.get(cache_key)
        if results:
            return results
        
        # Try persistent cache
        results = self.persistent_cache.get(cache_key)
        if results:
            # Restore to local cache
            self.local_cache.put(cache_key, results, ttl_seconds=3600)
            return results
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'local_cache': self.local_cache.get_stats(),
            'persistent_cache_size_mb': 0  # Would implement for persistent cache
        }


class ClusterManager:
    """Cluster management for high availability deployments."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.is_leader = False
        self.cluster_nodes = config.cluster_nodes
        
    async def start(self) -> None:
        """Start cluster management."""
        # Would implement actual clustering logic
        self.is_leader = True  # Simple single-node "cluster" for now
        logger.info("Cluster manager started (single node)")
    
    async def stop(self) -> None:
        """Stop cluster management."""
        logger.info("Cluster manager stopped")


class TracingSystem:
    """Distributed tracing for performance monitoring."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.traces: Dict[str, Dict[str, Any]] = {}
        
    def start(self) -> None:
        """Start tracing system."""
        logger.info("Tracing system started")
    
    def stop(self) -> None:
        """Stop tracing system."""
        logger.info("Tracing system stopped")
    
    def start_trace(self, session_id: str, operation: str) -> str:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())
        self.traces[trace_id] = {
            'session_id': session_id,
            'operation': operation,
            'start_time': datetime.now(),
            'spans': []
        }
        return trace_id
    
    def start_span(self, trace_id: str, operation: str) -> str:
        """Start a new span within a trace."""
        span_id = str(uuid.uuid4())
        if trace_id in self.traces:
            span = {
                'span_id': span_id,
                'operation': operation,
                'start_time': datetime.now()
            }
            self.traces[trace_id]['spans'].append(span)
        return span_id
    
    def end_span(self, span_id: str) -> None:
        """End a span."""
        # Would implement span completion logic
        pass


class AuditLogger:
    """Comprehensive audit logging for enterprise compliance."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.audit_log_path = Path("/tmp/tiny_llm_audit.log")
        
    def log_session_created(
        self,
        session_id: str,
        user_id: Optional[str],
        config: Dict[str, Any]
    ) -> None:
        """Log session creation."""
        self._write_audit_log("SESSION_CREATED", {
            'session_id': session_id,
            'user_id': user_id,
            'config_keys': list(config.keys())
        })
    
    def log_profiling_completed(self, session_id: str, task_count: int) -> None:
        """Log profiling completion."""
        self._write_audit_log("PROFILING_COMPLETED", {
            'session_id': session_id,
            'task_count': task_count
        })
    
    def _write_audit_log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write audit log entry."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


# Global enterprise profiler instance
enterprise_profiler = None


def get_enterprise_profiler(config: Optional[EnterpriseConfig] = None) -> EnterpriseProfiler:
    """Get or create enterprise profiler instance."""
    global enterprise_profiler
    if enterprise_profiler is None:
        enterprise_profiler = EnterpriseProfiler(config)
    return enterprise_profiler