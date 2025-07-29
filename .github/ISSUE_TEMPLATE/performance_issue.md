---
name: Performance Issue
about: Report performance problems or optimization opportunities
title: '[PERF] Performance issue with [specific component/operation]'
labels: ['performance', 'needs-investigation']
assignees: ['terragon-labs/performance-team']
---

## Performance Issue Summary
<!-- Provide a clear description of the performance problem -->

**Component Affected:**
- [ ] Model profiling/inference
- [ ] Data transfer (host ↔ device)
- [ ] Metric collection
- [ ] Result analysis/processing
- [ ] Memory management
- [ ] Power measurement
- [ ] Communication protocols
- [ ] File I/O operations
- [ ] Other: ___________

## Current Performance

**Measured Performance:**
- Operation: 
- Current speed: 
- Current memory usage: 
- Current power consumption:
- Current latency:

**Benchmark Details:**
```
Model: 
Platform: 
Configuration: 
Test conditions:
```

**Performance Metrics:**
<!-- Include specific measurements -->
```
Tokens/second: 
First token latency: ___ ms
Inter-token latency: ___ ms  
Peak memory usage: ___ KB
Average power: ___ mW
Energy per token: ___ mJ
```

## Expected Performance

**Performance Target:**
- Expected speed: 
- Expected memory usage:
- Expected power consumption:
- Expected latency:

**Baseline Comparison:**
<!-- How does this compare to similar systems/tools? -->
- Industry benchmark: 
- Similar tools performance:
- Theoretical maximum:

**Business Impact:**
- [ ] Blocks development workflow
- [ ] Reduces profiling accuracy  
- [ ] Impacts user experience
- [ ] Increases operational costs
- [ ] Other: ___________

## Environment Details

**Hardware Configuration:**
- Host system: 
- CPU: 
- RAM: 
- Storage: 
- Target device: 
- Platform: 

**Software Environment:**
- Python version: 
- tiny-llm-profiler version: 
- OS: 
- Docker (if used): 
- Virtual environment: 

**Model Information:**
- Model name: 
- Model size: 
- Quantization: 
- Input/output format:

## Reproduction Steps

**Test Setup:**
```python
# Code to reproduce the performance issue
from tiny_llm_profiler import EdgeProfiler

profiler = EdgeProfiler(
    platform="...",
    # configuration
)

# Steps that demonstrate the performance problem
```

**Test Data:**
<!-- Describe test data or attach files if relevant -->
- Dataset: 
- Input size: 
- Test duration:

**Measurement Method:**
<!-- How are you measuring performance? -->
- [ ] Built-in profiler metrics
- [ ] External benchmarking tools
- [ ] Manual timing
- [ ] System monitoring
- [ ] Other: ___________

## Performance Analysis

**Profiling Results:**
<!-- Include profiling output, flame graphs, etc. -->
```
# Paste performance profiling output here
```

**Bottleneck Identification:**
<!-- What appears to be the bottleneck? -->
- [ ] CPU-bound operation
- [ ] Memory-bound operation
- [ ] I/O-bound operation
- [ ] Network/communication bound
- [ ] Algorithm inefficiency
- [ ] Resource contention
- [ ] Unknown

**Resource Utilization:**
- CPU usage: ___%
- Memory usage: ___MB
- Disk I/O: 
- Network usage:

## Potential Solutions

**Optimization Ideas:**
<!-- Any ideas for improving performance? -->
- [ ] Algorithm optimization
- [ ] Caching/memoization
- [ ] Parallel processing
- [ ] Memory optimization
- [ ] I/O optimization
- [ ] Hardware acceleration
- [ ] Configuration tuning
- [ ] Other: ___________

**Implementation Complexity:**
- [ ] Simple fix
- [ ] Moderate refactoring required
- [ ] Significant architecture changes
- [ ] Research needed

**Backward Compatibility:**
- [ ] Optimization can be transparent
- [ ] Requires configuration changes
- [ ] May break existing code
- [ ] Unknown impact

## Impact Assessment

**Severity:**
- [ ] Critical - Unusable in production
- [ ] High - Major productivity impact
- [ ] Medium - Noticeable but workable
- [ ] Low - Minor improvement

**Affected Users:**
- [ ] All users
- [ ] Users of specific platforms
- [ ] Users of specific models
- [ ] Power users with large datasets
- [ ] Specific use cases: ___________

**Performance Regression:**
- [ ] New issue in latest version
- [ ] Long-standing issue
- [ ] Unknown - first time using

**Previous Version Comparison:**
<!-- If this is a regression, which version worked better? -->
- Working version: 
- Performance difference:

## Additional Context

**Related Issues:**
<!-- Link to related performance issues -->
- Related to #
- Similar to #
- Depends on #

**Workarounds:**
<!-- Any temporary workarounds you've found -->
- Current workaround: 
- Effectiveness: 
- Limitations:

**References:**
<!-- Links to relevant documentation, papers, benchmarks -->
- Performance benchmarks: 
- Optimization guides:
- Research papers:

**Logs/Traces:**
<!-- Attach relevant log files or traces -->
```
# Paste relevant log output here
```

---

**For Performance Team:**

**Performance Analysis:**
- [ ] Benchmarks reproduced
- [ ] Profiling completed
- [ ] Bottlenecks identified
- [ ] Optimization plan created

**Priority Assessment:**
- [ ] P0 - Critical performance blocker
- [ ] P1 - High impact optimization
- [ ] P2 - Medium improvement opportunity
- [ ] P3 - Nice to have optimization

**Technical Approach:**
- [ ] Algorithmic optimization
- [ ] System-level optimization
- [ ] Hardware-specific tuning
- [ ] Configuration optimization

**Estimated Effort:**
- Investigation: ___ days
- Implementation: ___ days
- Testing: ___ days
- Total: ___ days