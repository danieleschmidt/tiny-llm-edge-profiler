# Incident Response Runbook

## 🚨 Emergency Response Procedures

This runbook provides step-by-step procedures for responding to incidents in the Tiny LLM Edge Profiler system.

## Incident Classification

### Severity Levels
- **P0 - Critical**: Complete service outage, data loss risk
- **P1 - High**: Major functionality impaired, user-facing issues
- **P2 - Medium**: Limited functionality impaired, workarounds available
- **P3 - Low**: Minor issues, no immediate user impact

## Common Incidents

### 1. Hardware Device Communication Failures

#### Symptoms
- Device not detected by profiler
- Communication timeouts
- Flashing failures
- Intermittent connection losses

#### Immediate Actions
1. **Check Physical Connections**
   ```bash
   # Verify device enumeration
   lsusb  # Linux
   # or
   ls /dev/tty*  # Check for new serial devices
   ```

2. **Test Communication**
   ```bash
   # Test basic communication
   tiny-profiler detect-devices
   
   # Test specific device
   tiny-profiler test-connection --device /dev/ttyUSB0
   ```

3. **Power Cycle Device**
   - Disconnect and reconnect USB
   - Check power LED indicators
   - Verify adequate power supply (especially ESP32)

#### Resolution Steps
1. **Driver Issues**
   ```bash
   # Linux: Check driver loading
   dmesg | grep -i usb
   
   # Install CH340/CP2102 drivers if needed
   sudo apt-get install ch341ser-dkms
   ```

2. **Permission Issues**
   ```bash
   # Add user to dialout group
   sudo usermod -a -G dialout $USER
   
   # Or temporary permission
   sudo chmod 666 /dev/ttyUSB0
   ```

3. **Device Reset**
   ```bash
   # Force device reset
   tiny-profiler reset-device --device /dev/ttyUSB0 --platform esp32
   ```

### 2. Model Loading Failures

#### Symptoms
- "Model too large" errors
- Memory allocation failures
- Corrupted model files
- Quantization errors

#### Immediate Actions
1. **Check Model Size**
   ```python
   from tiny_llm_profiler import ModelAnalyzer
   
   analyzer = ModelAnalyzer()
   info = analyzer.analyze_model("model.bin")
   print(f"Model size: {info.size_mb:.1f} MB")
   print(f"Memory required: {info.runtime_memory_kb:.1f} KB")
   ```

2. **Verify Model Format**
   ```bash
   # Check model file integrity
   file model.bin
   
   # Verify checksum if available
   sha256sum model.bin
   ```

#### Resolution Steps
1. **Memory Optimization**
   ```python
   from tiny_llm_profiler import MemoryOptimizer
   
   optimizer = MemoryOptimizer()
   optimized_model = optimizer.optimize_for_platform(
       model, platform="esp32"
   )
   ```

2. **Re-quantization**
   ```python
   from tiny_llm_profiler import ModelQuantizer
   
   quantizer = ModelQuantizer()
   smaller_model = quantizer.quantize(
       model_path="original_model",
       bits=2,  # More aggressive quantization
       group_size=64
   )
   ```

### 3. Performance Degradation

#### Symptoms
- Slower than expected inference
- High latency measurements
- Memory leaks
- Power consumption spikes

#### Immediate Actions
1. **Performance Baseline**
   ```bash
   # Run standard benchmark
   tiny-profiler benchmark \
     --model reference_model.bin \
     --device /dev/ttyUSB0 \
     --baseline-comparison
   ```

2. **Resource Monitoring**
   ```python
   from tiny_llm_profiler import RealtimeMonitor
   
   monitor = RealtimeMonitor("/dev/ttyUSB0", platform="esp32")
   with monitor.start_session() as session:
       for _ in range(100):
           metrics = session.get_current_metrics()
           if metrics.memory_usage_kb > 400:  # Alert threshold
               print(f"HIGH MEMORY: {metrics.memory_usage_kb} KB")
   ```

#### Resolution Steps
1. **Memory Leak Detection**
   ```python
   profiler = EdgeProfiler(platform="esp32", device="/dev/ttyUSB0")
   memory_profile = profiler.profile_memory(
       model=model,
       track_allocations=True,
       duration_seconds=300
   )
   
   # Check for memory leaks
   if memory_profile.has_leaks():
       print("Memory leak detected!")
       memory_profile.generate_leak_report()
   ```

2. **Platform Optimization**
   ```python
   from tiny_llm_profiler import PlatformOptimizer
   
   optimizer = PlatformOptimizer("esp32")
   optimized_model = optimizer.optimize(
       model,
       use_psram=True,
       use_dual_core=True,
       cpu_freq_mhz=240
   )
   ```

### 4. Test Suite Failures

#### Symptoms
- Unit tests failing
- Integration test timeouts
- Hardware test failures
- CI/CD pipeline failures

#### Immediate Actions
1. **Isolate Failure**
   ```bash
   # Run specific test categories
   pytest tests/unit/ -v
   pytest tests/integration/ -v --device /dev/ttyUSB0
   pytest tests/hardware/ -v --platform esp32
   ```

2. **Check Test Environment**
   ```bash
   # Verify test dependencies
   pip check
   
   # Check hardware availability
   tiny-profiler detect-devices
   ```

#### Resolution Steps
1. **Test Environment Reset**
   ```bash
   # Clean test environment
   pytest --cache-clear
   
   # Reinstall in clean environment
   pip uninstall tiny-llm-edge-profiler
   pip install -e ".[dev,test]"
   ```

2. **Hardware Test Recovery**
   ```bash
   # Reset test hardware
   tiny-profiler flash --device /dev/ttyUSB0 --platform esp32 --firmware test
   
   # Run hardware tests with extended timeouts
   pytest tests/hardware/ --timeout=300
   ```

## Escalation Procedures

### Internal Escalation
1. **P0/P1 Incidents**: Immediately notify on-call engineer
2. **P2 Incidents**: Create GitHub issue, assign to relevant team
3. **P3 Incidents**: Add to backlog, prioritize in next sprint

### External Communication
1. **Status Page Updates**: Update for P0/P1 incidents
2. **User Notifications**: Email/Discord for breaking changes
3. **Documentation Updates**: Post-incident documentation review

## Monitoring and Alerting

### Key Metrics to Monitor
- Device connection success rate
- Model loading success rate
- Average inference latency
- Memory usage trends
- Test suite pass rate

### Alert Thresholds
```yaml
alerts:
  device_connection_failure_rate:
    threshold: "> 5% over 5 minutes"
    severity: P2
    
  model_loading_failure_rate:
    threshold: "> 10% over 10 minutes"
    severity: P1
    
  inference_latency_p95:
    threshold: "> 2x baseline"
    severity: P2
    
  memory_usage:
    threshold: "> 90% of device capacity"
    severity: P1
```

## Post-Incident Procedures

### Immediate (Within 24 hours)
1. **Incident Summary**: Create GitHub issue with full timeline
2. **Root Cause Analysis**: Identify technical and process causes
3. **Immediate Fixes**: Implement critical fixes
4. **User Communication**: Update users on resolution

### Follow-up (Within 1 week)
1. **Process Improvements**: Update runbooks and procedures
2. **Monitoring Enhancements**: Add new alerts/dashboards
3. **Testing Improvements**: Add tests to prevent recurrence
4. **Documentation Updates**: Update troubleshooting guides

## Contact Information

### On-Call Rotation
- **Primary**: Check internal rotation schedule
- **Secondary**: Escalate to team lead
- **Emergency**: Contact engineering manager

### External Contacts
- **Hardware Partners**: Vendor support contacts
- **Cloud Providers**: Support ticket systems
- **Security Team**: security@your-org.com

## Tools and Resources

### Debugging Tools
```bash
# Device debugging
tiny-profiler debug-device --device /dev/ttyUSB0
tiny-profiler system-info --device /dev/ttyUSB0

# Log analysis
journalctl -f -u tiny-profiler
tail -f /var/log/tiny-profiler.log
```

### Useful Commands
```bash
# Emergency device reset
tiny-profiler emergency-reset --all-devices

# System health check
tiny-profiler health-check --comprehensive

# Generate incident report
tiny-profiler generate-incident-report --since "2 hours ago"
```

---

## Quick Reference

| Issue Type | First Action | Escalation Time |
|------------|--------------|-----------------|
| Device Connection | Check physical connection | 15 minutes |
| Model Loading | Verify model size/format | 30 minutes |
| Performance | Run baseline benchmark | 20 minutes |
| Test Failures | Isolate failing tests | 10 minutes |

**Remember**: When in doubt, escalate early and preserve evidence!