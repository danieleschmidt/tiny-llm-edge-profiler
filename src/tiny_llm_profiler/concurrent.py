"""
Concurrent profiling utilities - aliases to scalable_profiler for backward compatibility.
"""

# Import all from scalable_profiler for backward compatibility
from .scalable_profiler import (
    ConcurrentProfiler,
    ProfileTask,
    ProfileTaskResult,
    ProfileTaskQueue,
    ProfileWorker,
    BatchProfiler,
    async_profile_single,
    async_profile_batch,
    run_concurrent_benchmark_demo
)

# Create aliases
ProfilingTask = ProfileTask