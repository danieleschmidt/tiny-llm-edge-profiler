"""
Robust profiling methods for the enhanced EdgeProfiler.
These methods replace the original profiling methods with comprehensive error handling,
retry logic, and graceful degradation.
"""

import time
import numpy as np
import psutil
from typing import List, Optional
from datetime import datetime

from .models import QuantizedModel
from .results import LatencyProfile, MemoryProfile, PowerProfile
from .exceptions import ProfilingTimeoutError, InsufficientDataError, ResourceError
from .reliability import retry, with_timeout, RetryStrategy, RetryConfig


def _profile_latency_robust(
    self, model: QuantizedModel, test_prompts: List[str], config
) -> LatencyProfile:
    """Profile inference latency with robust error handling and retry logic."""
    self.logger.info("Starting robust latency profiling")

    # Setup retry configuration for latency measurements
    latency_retry = RetryConfig(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=0.5,
        exceptions=(Exception,),  # Retry on any exception initially
    )

    @retry("latency_measurement")
    def measure_latency_sample(prompt: str, iteration: int) -> dict:
        """Measure latency for a single prompt with retry logic."""
        try:
            start_time = time.perf_counter()

            # Measure first token latency
            first_token_start = time.perf_counter()
            first_token_time = self._measure_first_token_robust(model, prompt)

            # Complete inference and measure total time
            total_tokens = self._run_inference_robust(model, prompt)
            end_time = time.perf_counter()

            total_latency = (end_time - start_time) * 1000  # Convert to ms

            # Calculate inter-token latency
            inter_token_latency = 0.0
            if total_tokens > 1:
                remaining_time = total_latency - first_token_time
                inter_token_latency = remaining_time / (total_tokens - 1)

            return {
                "total_latency_ms": total_latency,
                "first_token_latency_ms": first_token_time,
                "inter_token_latency_ms": inter_token_latency,
                "total_tokens": total_tokens,
                "tokens_per_second": (
                    total_tokens / (total_latency / 1000.0)
                    if total_latency > 0
                    else 0.0
                ),
            }

        except Exception as e:
            self.logger.warning(
                f"Latency measurement failed for iteration {iteration}: {e}"
            )
            # Log metric for monitoring
            self.perf_logger.log_metric(
                "latency_measurement_failure",
                1,
                "count",
                prompt_length=len(prompt),
                iteration=iteration,
            )
            raise

    latencies = []
    first_token_latencies = []
    inter_token_latencies = []
    tokens_per_second_samples = []

    successful_measurements = 0
    failed_measurements = 0

    try:
        # Warmup runs with error handling
        self.logger.info(f"Running {config.warmup_iterations} warmup iterations")
        for i in range(config.warmup_iterations):
            try:
                with managed_timeout(config.operation_timeout, f"warmup_{i}"):
                    self._run_inference_robust(model, test_prompts[0])
                self.logger.debug(f"Warmup iteration {i+1} completed")
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i+1} failed: {e}")
                # Continue with warmup - not critical

        # Measurement runs
        self.logger.info(
            f"Running latency measurements: {len(test_prompts)} prompts x {config.measurement_iterations} iterations"
        )

        for prompt_idx, prompt in enumerate(test_prompts):
            prompt_measurements = []

            for iteration in range(config.measurement_iterations):
                try:
                    # Use timeout for each measurement
                    with managed_timeout(
                        config.operation_timeout,
                        f"latency_measurement_{prompt_idx}_{iteration}",
                    ):
                        measurement = measure_latency_sample(prompt, iteration)
                        prompt_measurements.append(measurement)
                        successful_measurements += 1

                        # Log successful measurement
                        self.perf_logger.log_metric(
                            "latency_measurement_success",
                            measurement["total_latency_ms"],
                            "ms",
                            prompt_index=prompt_idx,
                            iteration=iteration,
                        )

                except ProfilingTimeoutError as e:
                    self.logger.error(f"Latency measurement timed out: {e}")
                    failed_measurements += 1

                    # If graceful degradation is disabled, re-raise
                    if not config.graceful_degradation:
                        raise

                except Exception as e:
                    self.logger.warning(f"Latency measurement failed: {e}")
                    failed_measurements += 1

                    # If too many failures and graceful degradation disabled, re-raise
                    if (
                        not config.graceful_degradation
                        or failed_measurements > successful_measurements
                    ):
                        if failed_measurements > 5:  # Arbitrary threshold
                            raise InsufficientDataError(
                                "latency",
                                successful_measurements,
                                config.measurement_iterations * len(test_prompts),
                            )

            # Process measurements for this prompt
            if prompt_measurements:
                latencies.extend([m["total_latency_ms"] for m in prompt_measurements])
                first_token_latencies.extend(
                    [m["first_token_latency_ms"] for m in prompt_measurements]
                )
                inter_token_latencies.extend(
                    [
                        m["inter_token_latency_ms"]
                        for m in prompt_measurements
                        if m["inter_token_latency_ms"] > 0
                    ]
                )
                tokens_per_second_samples.extend(
                    [m["tokens_per_second"] for m in prompt_measurements]
                )

        # Check if we have sufficient data
        min_required_samples = max(
            3, config.measurement_iterations // 2
        )  # At least 3 or half of intended measurements
        if len(latencies) < min_required_samples:
            raise InsufficientDataError("latency", len(latencies), min_required_samples)

        # Calculate statistics with robust handling of edge cases
        mean_latency = np.mean(latencies) if latencies else 0.0
        mean_first_token = (
            np.mean(first_token_latencies) if first_token_latencies else 0.0
        )
        mean_inter_token = (
            np.mean(inter_token_latencies) if inter_token_latencies else 0.0
        )
        mean_tokens_per_sec = (
            np.mean(tokens_per_second_samples) if tokens_per_second_samples else 0.0
        )
        latency_std = np.std(latencies) if len(latencies) > 1 else 0.0

        # Log summary statistics
        self.logger.info(
            f"Latency profiling completed: {successful_measurements} successful, {failed_measurements} failed",
            mean_latency_ms=mean_latency,
            mean_tokens_per_sec=mean_tokens_per_sec,
        )

        return LatencyProfile(
            first_token_latency_ms=mean_first_token,
            inter_token_latency_ms=mean_inter_token,
            total_latency_ms=mean_latency,
            tokens_per_second=mean_tokens_per_sec,
            latency_std_ms=latency_std,
            sample_count=len(latencies),
            success_rate=(
                successful_measurements
                / (successful_measurements + failed_measurements)
                if (successful_measurements + failed_measurements) > 0
                else 0.0
            ),
        )

    except Exception as e:
        self.logger.error(f"Latency profiling failed: {e}")
        # Return partial results if graceful degradation is enabled
        if config.graceful_degradation and latencies:
            self.logger.info(
                "Returning partial latency results due to graceful degradation"
            )
            return LatencyProfile(
                first_token_latency_ms=(
                    np.mean(first_token_latencies) if first_token_latencies else 0.0
                ),
                inter_token_latency_ms=(
                    np.mean(inter_token_latencies) if inter_token_latencies else 0.0
                ),
                total_latency_ms=np.mean(latencies),
                tokens_per_second=(
                    np.mean(tokens_per_second_samples)
                    if tokens_per_second_samples
                    else 0.0
                ),
                latency_std_ms=np.std(latencies) if len(latencies) > 1 else 0.0,
                sample_count=len(latencies),
                success_rate=(
                    successful_measurements
                    / (successful_measurements + failed_measurements)
                    if (successful_measurements + failed_measurements) > 0
                    else 0.0
                ),
                partial_results=True,
            )
        raise


def _profile_memory_robust(
    self, model: QuantizedModel, test_prompts: List[str], config
) -> MemoryProfile:
    """Profile memory usage with robust error handling."""
    self.logger.info("Starting robust memory profiling")

    if self.connection == "local":
        return self._profile_memory_local_robust(model, test_prompts, config)
    else:
        return self._profile_memory_remote_robust(model, test_prompts, config)


def _profile_memory_local_robust(
    self, model: QuantizedModel, test_prompts: List[str], config
) -> MemoryProfile:
    """Profile memory usage on local system with robustness."""

    @retry("memory_measurement")
    def get_memory_usage() -> dict:
        """Get current memory usage with retry logic."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            virtual_memory = psutil.virtual_memory()

            return {
                "rss_kb": memory_info.rss / 1024,
                "vms_kb": memory_info.vms / 1024,
                "percent": process.memory_percent(),
                "available_system_kb": virtual_memory.available / 1024,
            }
        except psutil.AccessDenied as e:
            self.logger.warning("Memory access denied, using alternative method")
            # Fallback to system-wide memory info
            virtual_memory = psutil.virtual_memory()
            return {
                "rss_kb": 0,  # Unknown
                "vms_kb": 0,  # Unknown
                "percent": 0,  # Unknown
                "available_system_kb": virtual_memory.available / 1024,
            }
        except Exception as e:
            self.logger.error(f"Memory measurement failed: {e}")
            raise

    try:
        # Baseline memory measurement
        baseline_memory = get_memory_usage()
        self.logger.debug(f"Baseline memory: {baseline_memory['rss_kb']:.1f}KB")

        peak_memory_kb = baseline_memory["rss_kb"]
        memory_samples = []
        successful_measurements = 0

        # Monitor memory during inference with a subset of prompts to avoid excessive overhead
        memory_test_prompts = test_prompts[: min(3, len(test_prompts))]

        for prompt_idx, prompt in enumerate(memory_test_prompts):
            try:
                # Pre-inference memory
                pre_memory = get_memory_usage()

                # Run inference while monitoring memory
                with managed_timeout(
                    config.operation_timeout, f"memory_inference_{prompt_idx}"
                ):
                    self._run_inference_robust(model, prompt)

                # Post-inference memory
                post_memory = get_memory_usage()

                # Calculate memory usage for this inference
                memory_delta = post_memory["rss_kb"] - baseline_memory["rss_kb"]
                memory_samples.append(memory_delta)
                peak_memory_kb = max(peak_memory_kb, post_memory["rss_kb"])

                successful_measurements += 1

                # Log memory usage
                self.perf_logger.log_metric(
                    "memory_usage_delta",
                    memory_delta,
                    "KB",
                    prompt_index=prompt_idx,
                    peak_memory_kb=peak_memory_kb,
                )

            except Exception as e:
                self.logger.warning(
                    f"Memory measurement failed for prompt {prompt_idx}: {e}"
                )
                if not config.graceful_degradation:
                    raise

        # Check if we have any measurements
        if not memory_samples and not config.graceful_degradation:
            raise InsufficientDataError("memory", 0, len(memory_test_prompts))

        # Calculate statistics
        avg_memory_usage = np.mean(memory_samples) if memory_samples else 0.0

        # Estimate memory efficiency (tokens per KB)
        # This is a rough estimate - in practice you'd track actual token counts
        estimated_tokens_per_kb = 0.0
        if avg_memory_usage > 0:
            # Rough estimate: assume 10 tokens per prompt on average
            estimated_tokens = len(memory_test_prompts) * 10
            estimated_tokens_per_kb = estimated_tokens / avg_memory_usage

        self.logger.info(
            f"Memory profiling completed: {successful_measurements}/{len(memory_test_prompts)} measurements successful",
            baseline_memory_kb=baseline_memory["rss_kb"],
            peak_memory_kb=peak_memory_kb,
            avg_memory_usage_kb=avg_memory_usage,
        )

        return MemoryProfile(
            baseline_memory_kb=baseline_memory["rss_kb"],
            peak_memory_kb=peak_memory_kb,
            memory_usage_kb=avg_memory_usage,
            memory_efficiency_tokens_per_kb=estimated_tokens_per_kb,
            sample_count=len(memory_samples),
            success_rate=(
                successful_measurements / len(memory_test_prompts)
                if memory_test_prompts
                else 0.0
            ),
        )

    except Exception as e:
        self.logger.error(f"Memory profiling failed: {e}")
        raise


def _profile_memory_remote_robust(
    self, model: QuantizedModel, test_prompts: List[str], config
) -> MemoryProfile:
    """Profile memory usage on remote device with robust communication."""

    @retry("remote_memory_query")
    @with_timeout(config.operation_timeout)
    def get_remote_memory() -> float:
        """Query remote device memory with retry and timeout."""
        if not self.serial_conn:
            raise RuntimeError("No serial connection available")

        try:
            # Send memory status command
            self.serial_conn.write(b"MEM_STATUS\n")

            # Read response with timeout
            response = self.serial_conn.readline().decode().strip()

            if response.startswith("MEM:"):
                memory_kb = float(response.split(":")[1])
                return memory_kb
            else:
                raise ValueError(f"Invalid memory response: {response}")

        except (serial.SerialException, OSError) as e:
            self.logger.warning(f"Serial communication error: {e}")
            raise
        except Exception as e:
            self.logger.warning(f"Remote memory query failed: {e}")
            raise

    try:
        # Baseline memory reading
        baseline_memory_kb = get_remote_memory()
        self.logger.debug(f"Remote baseline memory: {baseline_memory_kb}KB")

        peak_memory_kb = baseline_memory_kb
        memory_readings = []
        successful_measurements = 0

        # Test with subset of prompts
        memory_test_prompts = test_prompts[: min(3, len(test_prompts))]

        for prompt_idx, prompt in enumerate(memory_test_prompts):
            try:
                # Run inference
                with managed_timeout(
                    config.operation_timeout, f"remote_memory_inference_{prompt_idx}"
                ):
                    self._run_inference_robust(model, prompt)

                # Get memory reading after inference
                post_memory_kb = get_remote_memory()

                memory_delta = post_memory_kb - baseline_memory_kb
                memory_readings.append(memory_delta)
                peak_memory_kb = max(peak_memory_kb, post_memory_kb)

                successful_measurements += 1

                # Log measurement
                self.perf_logger.log_metric(
                    "remote_memory_usage", memory_delta, "KB", prompt_index=prompt_idx
                )

            except Exception as e:
                self.logger.warning(
                    f"Remote memory measurement failed for prompt {prompt_idx}: {e}"
                )
                if not config.graceful_degradation:
                    raise

        # Calculate results
        avg_memory_usage = np.mean(memory_readings) if memory_readings else 0.0

        self.logger.info(
            f"Remote memory profiling completed: {successful_measurements}/{len(memory_test_prompts)} measurements successful",
            baseline_memory_kb=baseline_memory_kb,
            peak_memory_kb=peak_memory_kb,
        )

        return MemoryProfile(
            baseline_memory_kb=baseline_memory_kb,
            peak_memory_kb=peak_memory_kb,
            memory_usage_kb=avg_memory_usage,
            memory_efficiency_tokens_per_kb=0.0,  # Not calculated for remote
            sample_count=len(memory_readings),
            success_rate=(
                successful_measurements / len(memory_test_prompts)
                if memory_test_prompts
                else 0.0
            ),
        )

    except Exception as e:
        self.logger.error(f"Remote memory profiling failed: {e}")
        raise


def _profile_power_robust(
    self, model: QuantizedModel, test_prompts: List[str], config
) -> PowerProfile:
    """Profile power consumption with robust measurements."""
    self.logger.info("Starting robust power profiling")

    # Note: This is a placeholder implementation
    # Real power profiling would integrate with hardware power measurement tools
    # such as INA219/INA260 current sensors, or platform-specific power APIs

    try:
        # Simulate power profiling with some realistic values
        # In a real implementation, this would:
        # 1. Set up power measurement hardware/APIs
        # 2. Measure idle power baseline
        # 3. Run inference while measuring power consumption
        # 4. Calculate energy consumption metrics

        idle_power_mw = 50.0  # Typical idle power for microcontrollers
        active_power_mw = 150.0  # Active power during inference
        peak_power_mw = 200.0  # Peak power consumption

        # Estimate energy per token based on inference time and power
        # This would be calculated from actual measurements
        estimated_tokens_total = len(test_prompts) * 10  # Rough estimate
        estimated_inference_time_sec = len(test_prompts) * 0.1  # Rough estimate
        total_energy_mj = (
            active_power_mw * estimated_inference_time_sec
        ) / 1000.0  # Convert to mJ
        energy_per_token_mj = (
            total_energy_mj / estimated_tokens_total
            if estimated_tokens_total > 0
            else 0.0
        )

        self.logger.info("Power profiling completed with simulated values")
        self.logger.warning(
            "Power profiling is using simulated values - integrate with actual power measurement hardware for real data"
        )

        return PowerProfile(
            idle_power_mw=idle_power_mw,
            active_power_mw=active_power_mw,
            peak_power_mw=peak_power_mw,
            energy_per_token_mj=energy_per_token_mj,
            total_energy_mj=total_energy_mj,
            measurement_duration_sec=estimated_inference_time_sec,
            simulated=True,  # Indicate this is simulated data
        )

    except Exception as e:
        self.logger.error(f"Power profiling failed: {e}")
        raise


# Helper methods for robust inference operations


def _run_inference_robust(self, model: QuantizedModel, prompt: str) -> int:
    """Run inference with robust error handling and communication."""

    @retry("inference_operation")
    @self.device_circuit  # Apply circuit breaker pattern
    def execute_inference() -> int:
        """Execute inference with device communication protection."""
        if self.connection == "local":
            return self._run_local_inference(model, prompt)
        elif self.connection == "serial":
            return self._run_serial_inference(model, prompt)
        else:
            raise ValueError(f"Unsupported connection type: {self.connection}")

    try:
        return execute_inference()
    except Exception as e:
        self.logger.error(f"Inference failed for prompt: {e}")
        raise


def _run_local_inference(self, model: QuantizedModel, prompt: str) -> int:
    """Run local inference simulation."""
    # Simulate local inference
    time.sleep(0.05)  # Simulate computation time
    return len(prompt.split()) + 10  # Simulate token generation


def _run_serial_inference(self, model: QuantizedModel, prompt: str) -> int:
    """Run inference via serial communication."""
    if not self.serial_conn:
        raise RuntimeError("Serial connection not available")

    try:
        # Send inference command to remote device
        command = f"INFER:{prompt}\n"
        self.serial_conn.write(command.encode())

        # Read response with timeout
        response = self.serial_conn.readline().decode().strip()

        # Parse response for token count
        if response.startswith("TOKENS:"):
            return int(response.split(":")[1])
        else:
            # Fallback: estimate tokens
            self.logger.warning(f"Unexpected inference response: {response}")
            return len(prompt.split()) + 5

    except (serial.SerialException, OSError) as e:
        self.logger.error(f"Serial inference failed: {e}")
        raise
    except Exception as e:
        self.logger.error(f"Inference parsing failed: {e}")
        raise


def _measure_first_token_robust(self, model: QuantizedModel, prompt: str) -> float:
    """Measure time to first token generation with robustness."""

    @retry("first_token_measurement")
    def measure_first_token() -> float:
        """Measure first token with retry logic."""
        start_time = time.perf_counter()

        if self.connection == "local":
            # Simulate first token delay
            time.sleep(0.02)
        elif self.connection == "serial" and self.serial_conn:
            try:
                # Send command and wait for first token
                command = f"FIRST_TOKEN:{prompt}\n"
                self.serial_conn.write(command.encode())

                # Wait for response
                response = self.serial_conn.readline()
                if not response:
                    raise RuntimeError("No response from device for first token")

            except (serial.SerialException, OSError) as e:
                self.logger.warning(
                    f"First token measurement communication failed: {e}"
                )
                raise

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to ms

    return measure_first_token()


# Context manager for timeout handling
import contextlib


@contextlib.contextmanager
def managed_timeout(timeout_seconds: float, operation_name: str):
    """Context manager for operation timeouts."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise ProfilingTimeoutError(operation_name, timeout_seconds)
