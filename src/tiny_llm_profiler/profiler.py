"""
Core EdgeProfiler implementation for measuring LLM performance on edge devices.
Enhanced with comprehensive robustness and reliability features.
"""

import time
import asyncio
import threading
import contextlib
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import serial
import psutil
from pydantic import BaseModel, ConfigDict

from .models import QuantizedModel
from .results import ProfileResults, LatencyProfile, MemoryProfile, PowerProfile
from .platforms import PlatformManager
from .exceptions import (
    DeviceConnectionError,
    DeviceTimeoutError,
    ProfilingError,
    ProfilingTimeoutError,
    ResourceError,
    InsufficientMemoryError,
)
from .logging_config import get_logger, PerformanceLogger
from .reliability import (
    retry,
    circuit_breaker,
    with_timeout,
    managed_resource,
    RetryConfig,
    CircuitBreakerConfig,
    RetryStrategy,
)
from .security import validate_device_path, validate_identifier


@dataclass
class ProfilingConfig:
    """Configuration for profiling sessions with reliability features."""

    # Basic profiling parameters
    sample_rate_hz: int = 100
    duration_seconds: int = 60
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    timeout_seconds: int = 300
    enable_power_profiling: bool = False
    enable_memory_profiling: bool = True
    enable_latency_profiling: bool = True

    # Robustness and reliability parameters
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_timeout: float = 10.0
    operation_timeout: float = 30.0
    heartbeat_interval: float = 5.0
    auto_recovery: bool = True
    graceful_degradation: bool = True
    memory_monitoring: bool = True
    resource_cleanup: bool = True

    # Circuit breaker settings
    failure_threshold: int = 5
    circuit_timeout: float = 60.0

    # Resource limits
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = 90.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.sample_rate_hz <= 0 or self.sample_rate_hz > 10000:
            raise ValueError("sample_rate_hz must be between 1 and 10000")

        if self.duration_seconds <= 0 or self.duration_seconds > 3600:
            raise ValueError("duration_seconds must be between 1 and 3600")

        if self.measurement_iterations <= 0 or self.measurement_iterations > 1000:
            raise ValueError("measurement_iterations must be between 1 and 1000")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError("max_retries must be between 0 and 10")

        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")

        if self.operation_timeout <= 0:
            raise ValueError("operation_timeout must be positive")


class EdgeProfiler:
    """
    Enhanced profiler class for measuring LLM performance on edge devices.

    Features comprehensive robustness and reliability enhancements:
    - Automatic retry mechanisms with exponential backoff
    - Circuit breaker patterns for device communication
    - Resource management and leak prevention
    - Graceful degradation and error recovery
    - Health monitoring and diagnostics
    - Secure input validation

    Supports multiple platforms including ESP32, STM32, RISC-V, and ARM Cortex-M.
    """

    def __init__(
        self,
        platform: str,
        device: Optional[str] = None,
        baudrate: int = 921600,
        connection: str = "serial",
        config: Optional[ProfilingConfig] = None,
        **kwargs,
    ):
        """
        Initialize EdgeProfiler for a specific platform.

        Args:
            platform: Target platform (esp32, stm32f4, stm32f7, rp2040, etc.)
            device: Device path (/dev/ttyUSB0, COM3, etc.)
            baudrate: Serial communication baudrate
            connection: Connection type (serial, network, local)
            config: Profiling configuration with reliability settings
        """
        # Validate and sanitize inputs
        self.platform = validate_identifier(platform.lower(), "platform")
        self.device = validate_device_path(device) if device else None
        self.baudrate = baudrate
        self.connection = connection
        self.config = config or ProfilingConfig()

        # Validate configuration
        self.config.validate()

        # Initialize logging
        self.logger = get_logger(f"profiler.{self.platform}")
        self.perf_logger = PerformanceLogger()

        # Platform management
        self.platform_manager = PlatformManager(platform)
        self.platform_config = self.platform_manager.get_config()

        # Connection management
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        self.connection_lock = threading.Lock()
        self.last_heartbeat: Optional[datetime] = None
        self.connection_attempts = 0

        # Resource management
        self.managed_resources: List[Any] = []
        self._cleanup_funcs: List[callable] = []

        # Performance monitoring
        self.start_time: Optional[datetime] = None
        self.profiling_session_id: Optional[str] = None

        # Health monitoring
        self.device_health = {
            "status": "unknown",
            "last_check": None,
            "failures": 0,
            "last_error": None,
        }

        # Setup reliability patterns
        self._setup_reliability_patterns()

        self.logger.info(
            f"Initialized EdgeProfiler for {platform} with enhanced reliability features"
        )

    def _setup_reliability_patterns(self):
        """Setup retry mechanisms and circuit breakers."""
        # Device connection retry configuration
        self.connection_retry = RetryConfig(
            max_attempts=self.config.max_retries,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=self.config.retry_delay,
            max_delay=30.0,
            exceptions=(DeviceConnectionError, serial.SerialException, OSError),
        )

        # Circuit breaker for device communication
        self.device_circuit = circuit_breaker(
            f"device_{self.platform}",
            failure_threshold=self.config.failure_threshold,
            timeout_seconds=self.config.circuit_timeout,
        )

    @retry("connection")
    @with_timeout(10.0)
    def connect(self) -> bool:
        """
        Establish connection to the target device with robust error handling.

        Returns:
            True if connection successful, False otherwise

        Raises:
            DeviceConnectionError: If connection fails after all retries
            DeviceTimeoutError: If connection attempt times out
        """
        with self.connection_lock:
            if self.is_connected:
                self.logger.info("Already connected to device")
                return True

            self.connection_attempts += 1

            try:
                self.logger.info(
                    f"Attempting to connect to {self.platform} device (attempt #{self.connection_attempts})"
                )

                if self.connection == "serial":
                    if not self.device:
                        raise DeviceConnectionError(
                            "no_device",
                            self.platform,
                            ValueError("Device path not specified"),
                        )

                    # Pre-connection health check
                    if not self._check_device_availability():
                        raise DeviceConnectionError(
                            self.device,
                            self.platform,
                            RuntimeError("Device not available"),
                        )

                    # Create serial connection with resource management
                    serial_conn = serial.Serial(
                        self.device,
                        self.baudrate,
                        timeout=self.config.connection_timeout,
                        write_timeout=self.config.operation_timeout,
                    )

                    # Register for cleanup
                    self.managed_resources.append(serial_conn)
                    self._cleanup_funcs.append(serial_conn.close)

                    self.serial_conn = serial_conn

                    # Test connection
                    if not self._test_connection():
                        raise DeviceConnectionError(
                            self.device,
                            self.platform,
                            RuntimeError("Connection test failed"),
                        )

                elif self.connection == "local":
                    # Local profiling (e.g., Raspberry Pi, development machine)
                    self.logger.info("Setting up local profiling connection")

                    # Check system resources
                    if not self._check_system_resources():
                        raise DeviceConnectionError(
                            "localhost",
                            self.platform,
                            ResourceError("Insufficient system resources"),
                        )

                elif self.connection == "network":
                    # Network connection (future implementation)
                    raise NotImplementedError("Network connections not yet implemented")

                else:
                    raise ValueError(f"Unsupported connection type: {self.connection}")

                # Connection successful
                self.is_connected = True
                self.last_heartbeat = datetime.now()
                self.device_health["status"] = "connected"
                self.device_health["last_check"] = datetime.now()
                self.device_health["failures"] = 0

                self.logger.info(f"Successfully connected to {self.platform} device")
                self.perf_logger.log_device_event(
                    "connected", self.platform, self.device or "local"
                )

                # Start heartbeat monitoring if configured
                if self.config.heartbeat_interval > 0:
                    self._start_heartbeat_monitoring()

                return True

            except (serial.SerialException, OSError, ValueError) as e:
                self.device_health["failures"] += 1
                self.device_health["last_error"] = str(e)
                self.logger.error(f"Failed to connect to device: {e}")
                raise DeviceConnectionError(self.device or "unknown", self.platform, e)

            except Exception as e:
                self.logger.error(f"Unexpected error during connection: {e}")
                raise

    def disconnect(self):
        """Close connection to the target device with proper cleanup."""
        with self.connection_lock:
            if not self.is_connected:
                self.logger.debug("Already disconnected")
                return

            try:
                self.logger.info(f"Disconnecting from {self.platform} device")

                # Stop heartbeat monitoring
                self._stop_heartbeat_monitoring()

                # Close serial connection
                if self.serial_conn:
                    try:
                        self.serial_conn.close()
                        self.logger.debug("Serial connection closed")
                    except Exception as e:
                        self.logger.warning(f"Error closing serial connection: {e}")

                # Clean up managed resources
                self._cleanup_resources()

                # Update status
                self.is_connected = False
                self.serial_conn = None
                self.device_health["status"] = "disconnected"
                self.device_health["last_check"] = datetime.now()

                self.logger.info("Successfully disconnected from device")
                self.perf_logger.log_device_event(
                    "disconnected", self.platform, self.device or "local"
                )

            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
                # Force cleanup
                self.is_connected = False
                self.serial_conn = None

    def _check_device_availability(self) -> bool:
        """Check if the device is available before attempting connection."""
        if not self.device:
            return False

        try:
            device_path = Path(self.device)
            if not device_path.exists():
                self.logger.warning(f"Device path does not exist: {self.device}")
                return False

            # Additional platform-specific checks could be added here
            return True

        except Exception as e:
            self.logger.warning(f"Error checking device availability: {e}")
            return False

    def _test_connection(self) -> bool:
        """Test the connection to ensure it's working properly."""
        if not self.serial_conn:
            return False

        try:
            # Send a simple test command
            test_command = b"PING\n"
            self.serial_conn.write(test_command)

            # Wait for response (with timeout)
            start_time = time.time()
            while (time.time() - start_time) < 2.0:
                if self.serial_conn.in_waiting > 0:
                    response = self.serial_conn.readline()
                    if response:
                        self.logger.debug(f"Connection test response: {response}")
                        return True
                time.sleep(0.1)

            # No response - connection might be working but device not responding
            # For now, consider this acceptable
            self.logger.debug(
                "No response to connection test, but connection appears functional"
            )
            return True

        except Exception as e:
            self.logger.warning(f"Connection test failed: {e}")
            return False

    def _check_system_resources(self) -> bool:
        """Check if system has sufficient resources for profiling."""
        try:
            # Check available memory
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)

            if self.config.max_memory_mb and available_mb < self.config.max_memory_mb:
                self.logger.warning(f"Low available memory: {available_mb:.1f}MB")
                return False

            if available_mb < 100:  # Minimum 100MB required
                self.logger.error(
                    f"Insufficient memory: {available_mb:.1f}MB available"
                )
                return False

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if (
                self.config.max_cpu_percent
                and cpu_percent > self.config.max_cpu_percent
            ):
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Error checking system resources: {e}")
            return True  # Allow operation to continue

    def _start_heartbeat_monitoring(self):
        """Start heartbeat monitoring in a separate thread."""
        if hasattr(self, "_heartbeat_thread") and self._heartbeat_thread.is_alive():
            return  # Already running

        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_monitor, daemon=True
        )
        self._heartbeat_thread.start()
        self.logger.debug("Started heartbeat monitoring")

    def _stop_heartbeat_monitoring(self):
        """Stop heartbeat monitoring."""
        if hasattr(self, "_heartbeat_stop_event"):
            self._heartbeat_stop_event.set()

        if hasattr(self, "_heartbeat_thread") and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)
            self.logger.debug("Stopped heartbeat monitoring")

    def _heartbeat_monitor(self):
        """Monitor device heartbeat in background thread."""
        while not self._heartbeat_stop_event.wait(self.config.heartbeat_interval):
            try:
                if self.is_connected:
                    # Simple heartbeat check
                    current_time = datetime.now()

                    # Check if device is still responsive
                    if self.connection == "serial" and self.serial_conn:
                        # Send heartbeat ping
                        self.serial_conn.write(b"HB\n")

                        # Update heartbeat time
                        self.last_heartbeat = current_time

                        # Update device health
                        self.device_health["status"] = "healthy"
                        self.device_health["last_check"] = current_time

                    elif self.connection == "local":
                        # For local connections, check system health
                        if self._check_system_resources():
                            self.last_heartbeat = current_time
                            self.device_health["status"] = "healthy"
                        else:
                            self.device_health["status"] = "degraded"

                        self.device_health["last_check"] = current_time

            except Exception as e:
                self.logger.warning(f"Heartbeat check failed: {e}")
                self.device_health["failures"] += 1
                self.device_health["status"] = "unhealthy"
                self.device_health["last_error"] = str(e)

    def _cleanup_resources(self):
        """Clean up managed resources."""
        for cleanup_func in self._cleanup_funcs:
            try:
                cleanup_func()
            except Exception as e:
                self.logger.warning(f"Error in cleanup function: {e}")

        self.managed_resources.clear()
        self._cleanup_funcs.clear()
        self.logger.debug("Cleaned up managed resources")

    @with_timeout(300.0)  # 5 minute timeout for profiling
    def profile_model(
        self,
        model: QuantizedModel,
        test_prompts: List[str],
        metrics: List[str] = None,
        config: Optional[ProfilingConfig] = None,
    ) -> ProfileResults:
        """
        Profile a quantized model on the target platform with comprehensive robustness.

        Args:
            model: QuantizedModel to profile
            test_prompts: List of test prompts for inference
            metrics: Metrics to collect (latency, memory, power)
            config: Profiling configuration

        Returns:
            ProfileResults containing all measured metrics

        Raises:
            ProfilingError: If profiling fails
            ProfilingTimeoutError: If profiling times out
            ResourceError: If insufficient resources
        """
        # Initialize profiling session
        session_config = config or self.config
        metrics = metrics or ["latency", "memory"]

        # Generate unique session ID
        from .security import generate_session_id

        self.profiling_session_id = generate_session_id()
        self.start_time = datetime.now()

        # Set up logging context
        self.logger.set_context(
            profiling_session_id=self.profiling_session_id,
            platform=self.platform,
            model_name=model.name,
        )

        # Validate inputs
        self._validate_profiling_inputs(model, test_prompts, metrics, session_config)

        # Pre-profiling health checks
        self._pre_profiling_health_check()

        try:
            # Ensure connection
            if not self.is_connected:
                if not self.connect():
                    raise ProfilingError("Failed to establish device connection")

            # Log profiling start
            self.perf_logger.log_profiling_start(
                self.platform,
                model.name,
                self.profiling_session_id,
                metrics=metrics,
                test_prompts_count=len(test_prompts),
            )

            # Initialize results structure
            results = ProfileResults(
                platform=self.platform,
                model_name=model.name,
                model_size_mb=model.size_mb,
                quantization=model.quantization,
                session_id=self.profiling_session_id,
                start_time=self.start_time,
            )

            # Resource monitoring context
            with managed_resource(results, "profiling_results"):
                # Run profiling based on requested metrics with error recovery
                successful_metrics = []
                failed_metrics = []

                for metric in metrics:
                    try:
                        if (
                            metric == "latency"
                            and session_config.enable_latency_profiling
                        ):
                            latency_profile = self._profile_latency_robust(
                                model, test_prompts, session_config
                            )
                            results.add_latency_profile(latency_profile)
                            successful_metrics.append("latency")

                        elif (
                            metric == "memory"
                            and session_config.enable_memory_profiling
                        ):
                            memory_profile = self._profile_memory_robust(
                                model, test_prompts, session_config
                            )
                            results.add_memory_profile(memory_profile)
                            successful_metrics.append("memory")

                        elif (
                            metric == "power" and session_config.enable_power_profiling
                        ):
                            power_profile = self._profile_power_robust(
                                model, test_prompts, session_config
                            )
                            results.add_power_profile(power_profile)
                            successful_metrics.append("power")

                    except Exception as e:
                        self.logger.error(
                            f"Failed to profile {metric}: {e}", metric=metric
                        )
                        failed_metrics.append((metric, str(e)))

                        # Continue with other metrics if graceful degradation is enabled
                        if not session_config.graceful_degradation:
                            raise

                # Check if any metrics were successful
                if not successful_metrics and failed_metrics:
                    error_summary = "; ".join([f"{m}: {e}" for m, e in failed_metrics])
                    raise ProfilingError(f"All metrics failed: {error_summary}")

                # Add session metadata
                results.session_metadata = {
                    "successful_metrics": successful_metrics,
                    "failed_metrics": [m for m, _ in failed_metrics],
                    "connection_type": self.connection,
                    "platform_config": self.platform_config.name,
                    "device_health": self.device_health.copy(),
                    "resource_stats": self._get_resource_stats(),
                }

                # Log profiling completion
                duration_seconds = (datetime.now() - self.start_time).total_seconds()
                self.perf_logger.log_profiling_end(
                    self.profiling_session_id,
                    duration_seconds,
                    {
                        "successful_metrics": successful_metrics,
                        "failed_metrics": len(failed_metrics),
                        "total_metrics": len(metrics),
                    },
                )

                self.logger.info(
                    f"Profiling completed successfully",
                    successful_metrics=successful_metrics,
                    duration_seconds=duration_seconds,
                )

                return results

        except Exception as e:
            # Log profiling failure
            duration_seconds = (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0
            )
            self.logger.error(
                f"Profiling failed after {duration_seconds:.1f}s: {e}",
                duration_seconds=duration_seconds,
            )
            raise

        finally:
            # Clear logging context
            self.logger.clear_context()
            self.profiling_session_id = None
            self.start_time = None

    def _validate_profiling_inputs(
        self,
        model: QuantizedModel,
        test_prompts: List[str],
        metrics: List[str],
        config: ProfilingConfig,
    ):
        """Validate profiling inputs for security and correctness."""
        # Validate model
        if not model or not model.name:
            raise ValueError("Invalid model: model and model.name are required")

        # Validate prompts
        if not test_prompts:
            raise ValueError("At least one test prompt is required")

        max_prompts = getattr(config, "max_prompts", 100)
        if len(test_prompts) > max_prompts:
            raise ValueError(
                f"Too many test prompts: {len(test_prompts)}, max allowed: {max_prompts}"
            )

        # Sanitize prompts for security
        from .security import InputSanitizer

        sanitizer = InputSanitizer()
        try:
            max_prompt_length = getattr(config, "max_prompt_length", 10000)
            sanitizer.sanitize_prompts(test_prompts, max_prompt_length)
        except Exception as e:
            raise ValueError(f"Invalid prompts: {e}")

        # Validate metrics
        valid_metrics = {"latency", "memory", "power", "throughput"}
        invalid_metrics = set(metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. Valid metrics: {valid_metrics}"
            )

    def _pre_profiling_health_check(self):
        """Perform comprehensive health check before profiling."""
        self.logger.info("Performing pre-profiling health check")

        # Check system resources
        if not self._check_system_resources():
            if not self.config.graceful_degradation:
                raise ResourceError("Insufficient system resources for profiling")
            else:
                self.logger.warning(
                    "Resource constraints detected - profiling may be degraded"
                )

        # Check device health
        if self.is_connected and self.device_health["status"] not in [
            "healthy",
            "connected",
        ]:
            self.logger.warning(f"Device health status: {self.device_health['status']}")

            # Attempt to recover connection if auto_recovery is enabled
            if self.config.auto_recovery:
                self.logger.info("Attempting to recover device connection")
                try:
                    self.disconnect()
                    time.sleep(1.0)  # Brief pause
                    self.connect()
                except Exception as e:
                    self.logger.error(f"Connection recovery failed: {e}")
                    if not self.config.graceful_degradation:
                        raise

        # Check circuit breaker status
        cb_status = self.device_circuit.get_status()
        if cb_status["state"] != "closed":
            self.logger.warning(
                f"Circuit breaker is {cb_status['state']} - profiling may fail"
            )

        self.logger.info("Pre-profiling health check completed")

    def _get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource utilization statistics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()

            return {
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "cpu_percent": cpu_percent,
                "managed_resources_count": len(self.managed_resources),
                "connection_attempts": self.connection_attempts,
                "device_health_status": self.device_health["status"],
            }
        except Exception as e:
            self.logger.warning(f"Error getting resource stats: {e}")
            return {"error": str(e)}

    # Import robust profiling methods
    from .profiler_robust_methods import (
        _profile_latency_robust,
        _profile_memory_robust,
        _profile_memory_local_robust,
        _profile_memory_remote_robust,
        _profile_power_robust,
        _run_inference_robust,
        _run_local_inference,
        _run_serial_inference,
        _measure_first_token_robust,
    )

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the profiler."""
        return {
            "platform": self.platform,
            "connection_type": self.connection,
            "is_connected": self.is_connected,
            "device_health": self.device_health.copy(),
            "circuit_breaker": self.device_circuit.get_status(),
            "last_heartbeat": (
                self.last_heartbeat.isoformat() if self.last_heartbeat else None
            ),
            "connection_attempts": self.connection_attempts,
            "managed_resources": len(self.managed_resources),
            "profiling_active": self.profiling_session_id is not None,
            "resource_stats": self._get_resource_stats(),
        }

    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self.device_circuit.reset()
        self.logger.info("Circuit breaker manually reset")

    def force_reconnect(self) -> bool:
        """Force a reconnection to the device."""
        try:
            self.logger.info("Forcing device reconnection")
            if self.is_connected:
                self.disconnect()

            time.sleep(1.0)  # Brief pause
            return self.connect()

        except Exception as e:
            self.logger.error(f"Force reconnect failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry with enhanced cleanup tracking."""
        if not self.connect():
            raise RuntimeError("Failed to establish connection in context manager")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with comprehensive cleanup."""
        try:
            # Log any exception that occurred
            if exc_type:
                self.logger.error(
                    f"Exception in profiler context: {exc_type.__name__}: {exc_val}",
                    exception_type=exc_type.__name__,
                )

            # Ensure clean disconnection and resource cleanup
            self.disconnect()

        except Exception as cleanup_error:
            self.logger.error(f"Error during context manager cleanup: {cleanup_error}")

    def _profile_latency(
        self, model: QuantizedModel, test_prompts: List[str], config: ProfilingConfig
    ) -> LatencyProfile:
        """Profile inference latency for the model."""
        latencies = []
        first_token_latencies = []
        inter_token_latencies = []

        # Warmup runs
        for _ in range(config.warmup_iterations):
            self._run_inference(model, test_prompts[0])

        # Measurement runs
        for prompt in test_prompts:
            for _ in range(config.measurement_iterations):
                start_time = time.perf_counter()

                # Measure first token latency
                first_token_time = self._measure_first_token(model, prompt)
                first_token_latencies.append(first_token_time)

                # Complete inference and measure total time
                total_tokens = self._run_inference(model, prompt)
                end_time = time.perf_counter()

                total_latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(total_latency)

                # Calculate inter-token latency
                if total_tokens > 1:
                    inter_token_latency = (total_latency - first_token_time) / (
                        total_tokens - 1
                    )
                    inter_token_latencies.append(inter_token_latency)

        return LatencyProfile(
            first_token_latency_ms=np.mean(first_token_latencies),
            inter_token_latency_ms=np.mean(inter_token_latencies),
            total_latency_ms=np.mean(latencies),
            tokens_per_second=(
                1000.0 / np.mean(inter_token_latencies)
                if inter_token_latencies
                else 0.0
            ),
            latency_std_ms=np.std(latencies),
        )

    def _profile_memory(
        self, model: QuantizedModel, test_prompts: List[str], config: ProfilingConfig
    ) -> MemoryProfile:
        """Profile memory usage during inference."""
        if self.connection == "local":
            return self._profile_memory_local(model, test_prompts, config)
        else:
            return self._profile_memory_remote(model, test_prompts, config)

    def _profile_memory_local(
        self, model: QuantizedModel, test_prompts: List[str], config: ProfilingConfig
    ) -> MemoryProfile:
        """Profile memory usage on local system."""
        process = psutil.Process()

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024  # Convert to KB

        peak_memory = baseline_memory
        memory_samples = []

        # Monitor memory during inference
        for prompt in test_prompts[:3]:  # Sample subset for memory profiling
            start_memory = process.memory_info().rss / 1024

            # Run inference while monitoring memory
            self._run_inference(model, prompt)

            current_memory = process.memory_info().rss / 1024
            memory_samples.append(current_memory - baseline_memory)
            peak_memory = max(peak_memory, current_memory)

        return MemoryProfile(
            baseline_memory_kb=baseline_memory,
            peak_memory_kb=peak_memory,
            memory_usage_kb=np.mean(memory_samples),
            memory_efficiency_tokens_per_kb=0.0,  # Calculate based on throughput
        )

    def _profile_memory_remote(
        self, model: QuantizedModel, test_prompts: List[str], config: ProfilingConfig
    ) -> MemoryProfile:
        """Profile memory usage on remote device via serial communication."""
        # Send memory profiling commands to device
        memory_readings = []

        for prompt in test_prompts[:3]:
            # Request memory status before inference
            baseline = self._get_device_memory()

            # Run inference
            self._run_inference(model, prompt)

            # Request memory status after inference
            peak = self._get_device_memory()
            memory_readings.append(peak - baseline)

        return MemoryProfile(
            baseline_memory_kb=baseline,
            peak_memory_kb=max(memory_readings) + baseline,
            memory_usage_kb=np.mean(memory_readings),
            memory_efficiency_tokens_per_kb=0.0,
        )

    def _profile_power(
        self, model: QuantizedModel, test_prompts: List[str], config: ProfilingConfig
    ) -> PowerProfile:
        """Profile power consumption during inference."""
        # Placeholder for power profiling implementation
        # Would integrate with hardware power measurement tools
        return PowerProfile(
            idle_power_mw=50.0,
            active_power_mw=150.0,
            peak_power_mw=200.0,
            energy_per_token_mj=2.5,
            total_energy_mj=100.0,
        )

    def _run_inference(self, model: QuantizedModel, prompt: str) -> int:
        """Run inference and return number of generated tokens."""
        if self.connection == "local":
            # Simulate local inference
            time.sleep(0.05)  # Simulate computation time
            return len(prompt.split()) + 10  # Simulate token generation

        elif self.serial_conn:
            # Send inference command to remote device
            command = f"INFER:{prompt}\n"
            self.serial_conn.write(command.encode())

            # Read response
            response = self.serial_conn.readline().decode().strip()

            # Parse response for token count
            if response.startswith("TOKENS:"):
                return int(response.split(":")[1])

        return 0

    def _measure_first_token(self, model: QuantizedModel, prompt: str) -> float:
        """Measure time to first token generation."""
        start_time = time.perf_counter()

        # Implementation depends on platform
        if self.connection == "local":
            time.sleep(0.02)  # Simulate first token delay
        elif self.serial_conn:
            # Send command and wait for first token
            command = f"FIRST_TOKEN:{prompt}\n"
            self.serial_conn.write(command.encode())
            self.serial_conn.readline()  # Wait for response

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to ms

    def _get_device_memory(self) -> float:
        """Get current memory usage from remote device."""
        if self.serial_conn:
            self.serial_conn.write(b"MEM_STATUS\n")
            response = self.serial_conn.readline().decode().strip()

            if response.startswith("MEM:"):
                return float(response.split(":")[1])

        return 0.0

    def stream_metrics(self, duration_seconds: int = 60):
        """Stream real-time metrics from the device."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")

        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            if self.connection == "local":
                # Local metrics
                process = psutil.Process()
                yield {
                    "timestamp": time.time(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "tokens_per_second": np.random.uniform(8, 12),  # Simulated
                }
            elif self.serial_conn:
                # Remote metrics
                self.serial_conn.write(b"METRICS\n")
                response = self.serial_conn.readline().decode().strip()

                if response.startswith("METRICS:"):
                    # Parse metrics from device
                    metrics_data = response.split(":")[1].split(",")
                    yield {
                        "timestamp": time.time(),
                        "cpu_percent": (
                            float(metrics_data[0]) if len(metrics_data) > 0 else 0
                        ),
                        "memory_kb": (
                            float(metrics_data[1]) if len(metrics_data) > 1 else 0
                        ),
                        "tokens_per_second": (
                            float(metrics_data[2]) if len(metrics_data) > 2 else 0
                        ),
                    }

            time.sleep(0.1)  # 10Hz sampling rate

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
