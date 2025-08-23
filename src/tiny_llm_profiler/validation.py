"""
Validation and error handling utilities for robust profiling.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import psutil
import serial

from .models import QuantizedModel
from .results import ProfileResults


class ValidationLevel(str, Enum):
    """Validation strictness levels."""

    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class ValidationCategory(str, Enum):
    """Categories of validation checks."""

    HARDWARE = "hardware"
    MODEL = "model"
    PLATFORM = "platform"
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    category: ValidationCategory
    severity: str  # "error", "warning", "info"
    message: str
    recommendation: Optional[str] = None
    auto_fix: Optional[Callable] = None


@dataclass
class ValidationResult:
    """Result of validation checks."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)

    def __post_init__(self):
        """Categorize issues by severity."""
        for issue in self.issues:
            if issue.severity == "error":
                self.errors.append(issue)
            elif issue.severity == "warning":
                self.warnings.append(issue)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


class SystemValidator:
    """Validates system requirements for profiling."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)

    def validate_system(self) -> ValidationResult:
        """Validate system requirements."""
        issues = []

        # Check memory
        issues.extend(self._check_system_memory())

        # Check disk space
        issues.extend(self._check_disk_space())

        # Check permissions
        issues.extend(self._check_permissions())

        # Check Python environment
        issues.extend(self._check_python_environment())

        return ValidationResult(
            is_valid=not any(issue.severity == "error" for issue in issues),
            issues=issues,
        )

    def _check_system_memory(self) -> List[ValidationIssue]:
        """Check available system memory."""
        issues = []
        memory = psutil.virtual_memory()

        # Check total memory
        if memory.total < 1 * 1024**3:  # Less than 1GB
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.HARDWARE,
                    severity="warning",
                    message=f"Low system memory: {memory.total / 1024**3:.1f}GB",
                    recommendation="Consider running on a system with more memory for better performance",
                )
            )

        # Check available memory
        if memory.available < 512 * 1024**2:  # Less than 512MB
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.HARDWARE,
                    severity="error",
                    message=f"Insufficient available memory: {memory.available / 1024**2:.0f}MB",
                    recommendation="Close other applications or add more RAM",
                )
            )

        return issues

    def _check_disk_space(self) -> List[ValidationIssue]:
        """Check available disk space."""
        issues = []
        disk_usage = psutil.disk_usage(".")

        # Check available space
        available_gb = disk_usage.free / (1024**3)
        if available_gb < 1:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.ENVIRONMENT,
                    severity="warning",
                    message=f"Low disk space: {available_gb:.1f}GB available",
                    recommendation="Free up disk space for profiling data storage",
                )
            )

        return issues

    def _check_permissions(self) -> List[ValidationIssue]:
        """Check file permissions."""
        issues = []

        # Check write permissions in current directory
        try:
            test_file = Path("test_write_permission.tmp")
            test_file.write_text("test")
            test_file.unlink()
        except PermissionError:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.ENVIRONMENT,
                    severity="error",
                    message="No write permission in current directory",
                    recommendation="Change to a directory with write permissions",
                )
            )

        return issues

    def _check_python_environment(self) -> List[ValidationIssue]:
        """Check Python environment."""
        issues = []

        # Check Python version
        import sys

        if sys.version_info < (3, 8):
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.ENVIRONMENT,
                    severity="error",
                    message=f"Python version {sys.version} is too old",
                    recommendation="Upgrade to Python 3.8 or newer",
                )
            )

        # Check required packages
        required_packages = ["numpy", "serial", "psutil", "plotly"]
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.ENVIRONMENT,
                        severity="error",
                        message=f"Required package '{package}' not found",
                        recommendation=f"Install with: pip install {package}",
                    )
                )

        return issues


class PlatformValidator:
    """Validates platform-specific requirements."""

    def __init__(self, platform: str):
        self.platform = platform
        self.logger = logging.getLogger(__name__)

    def validate_platform(self, device_path: Optional[str] = None) -> ValidationResult:
        """Validate platform configuration."""
        issues = []

        # Platform-specific validations
        if self.platform == "esp32":
            issues.extend(self._validate_esp32(device_path))
        elif self.platform.startswith("stm32"):
            issues.extend(self._validate_stm32(device_path))
        elif self.platform == "rp2040":
            issues.extend(self._validate_rp2040(device_path))
        elif self.platform in ["rpi_zero", "jetson_nano"]:
            issues.extend(self._validate_sbc())
        else:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PLATFORM,
                    severity="warning",
                    message=f"Unknown platform: {self.platform}",
                    recommendation="Verify platform name is correct",
                )
            )

        return ValidationResult(
            is_valid=not any(issue.severity == "error" for issue in issues),
            issues=issues,
        )

    def _validate_esp32(self, device_path: Optional[str]) -> List[ValidationIssue]:
        """Validate ESP32 setup."""
        issues = []

        if not device_path:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PLATFORM,
                    severity="error",
                    message="No device path specified for ESP32",
                    recommendation="Provide device path (e.g., /dev/ttyUSB0 or COM3)",
                )
            )
            return issues

        # Check device exists
        device = Path(device_path)
        if not device.exists():
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PLATFORM,
                    severity="error",
                    message=f"ESP32 device not found: {device_path}",
                    recommendation="Check device connection and driver installation",
                )
            )

        # Try to connect
        try:
            with serial.Serial(device_path, 921600, timeout=1) as ser:
                ser.write(b"AT\n")
                time.sleep(0.1)
                response = ser.read_all()

                if not response:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.PLATFORM,
                            severity="warning",
                            message="No response from ESP32 device",
                            recommendation="Verify firmware is loaded and device is ready",
                        )
                    )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PLATFORM,
                    severity="error",
                    message=f"Cannot connect to ESP32: {e}",
                    recommendation="Check device connection, drivers, and permissions",
                )
            )

        return issues

    def _validate_stm32(self, device_path: Optional[str]) -> List[ValidationIssue]:
        """Validate STM32 setup."""
        issues = []

        if not device_path:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PLATFORM,
                    severity="error",
                    message="No device path specified for STM32",
                    recommendation="Provide device path (e.g., /dev/ttyACM0)",
                )
            )
            return issues

        # Similar validation as ESP32 but with different parameters
        device = Path(device_path)
        if not device.exists():
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PLATFORM,
                    severity="error",
                    message=f"STM32 device not found: {device_path}",
                    recommendation="Check device connection and ST-Link driver",
                )
            )

        return issues

    def _validate_rp2040(self, device_path: Optional[str]) -> List[ValidationIssue]:
        """Validate RP2040 setup."""
        issues = []

        # RP2040 can appear as different devices
        common_paths = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"]

        if device_path:
            if not Path(device_path).exists():
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PLATFORM,
                        severity="error",
                        message=f"RP2040 device not found: {device_path}",
                        recommendation="Check device connection",
                    )
                )
        else:
            # Look for common device paths
            found_devices = [p for p in common_paths if Path(p).exists()]
            if not found_devices:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PLATFORM,
                        severity="warning",
                        message="No RP2040 device found on common paths",
                        recommendation=f"Check device connection. Common paths: {', '.join(common_paths)}",
                    )
                )

        return issues

    def _validate_sbc(self) -> List[ValidationIssue]:
        """Validate Single Board Computer setup."""
        issues = []

        # Check if we're actually running on the target platform
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

                if self.platform == "rpi_zero" and "BCM" not in cpuinfo:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.PLATFORM,
                            severity="warning",
                            message="Not running on Raspberry Pi hardware",
                            recommendation="Local profiling may not reflect actual Pi Zero performance",
                        )
                    )

        except FileNotFoundError:
            # Not on Linux, can't check CPU info
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PLATFORM,
                    severity="info",
                    message="Cannot verify SBC hardware - not on Linux system",
                    recommendation="Results may not reflect actual hardware performance",
                )
            )

        return issues


class ModelValidator:
    """Validates model compatibility and integrity."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)

    def validate_model(self, model: QuantizedModel, platform: str) -> ValidationResult:
        """Validate model for platform deployment."""
        issues = []

        # Basic model validation
        issues.extend(self._validate_model_basic(model))

        # Platform compatibility
        issues.extend(self._validate_platform_compatibility(model, platform))

        # Performance warnings
        issues.extend(self._validate_performance_expectations(model, platform))

        return ValidationResult(
            is_valid=not any(issue.severity == "error" for issue in issues),
            issues=issues,
        )

    def _validate_model_basic(self, model: QuantizedModel) -> List[ValidationIssue]:
        """Basic model validation."""
        issues = []

        # Check model file exists
        if model.model_path and not model.model_path.exists():
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.MODEL,
                    severity="error",
                    message=f"Model file not found: {model.model_path}",
                    recommendation="Check model path and ensure file exists",
                )
            )

        # Check model size
        if model.size_mb > 50:  # Very large for edge
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.MODEL,
                    severity="error",
                    message=f"Model too large for edge deployment: {model.size_mb:.1f}MB",
                    recommendation="Use more aggressive quantization or a smaller model",
                )
            )
        elif model.size_mb > 10:  # Large for edge
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.MODEL,
                    severity="warning",
                    message=f"Large model for edge deployment: {model.size_mb:.1f}MB",
                    recommendation="Consider more aggressive quantization for better performance",
                )
            )

        # Check quantization level
        if model.quantization in ["fp16", "fp32"]:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.MODEL,
                    severity="warning",
                    message=f"High precision quantization ({model.quantization}) not optimal for edge",
                    recommendation="Consider 2-bit or 4-bit quantization for edge deployment",
                )
            )

        return issues

    def _validate_platform_compatibility(
        self, model: QuantizedModel, platform: str
    ) -> List[ValidationIssue]:
        """Validate model compatibility with platform."""
        issues = []

        memory_reqs = model.get_memory_requirements(platform)

        # Platform memory limits (conservative estimates)
        platform_limits = {
            "esp32": {"ram_kb": 400, "flash_mb": 4},
            "esp32s3": {"ram_kb": 450, "flash_mb": 8},
            "stm32f4": {"ram_kb": 150, "flash_mb": 1},
            "stm32f7": {"ram_kb": 300, "flash_mb": 2},
            "rp2040": {"ram_kb": 200, "flash_mb": 2},
            "nrf52840": {"ram_kb": 200, "flash_mb": 1},
        }

        if platform in platform_limits:
            limits = platform_limits[platform]

            # Check RAM requirements
            total_ram_needed = memory_reqs["total_estimated_kb"]
            if total_ram_needed > limits["ram_kb"]:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.MODEL,
                        severity="error",
                        message=f"Model requires {total_ram_needed:.0f}KB RAM, but {platform} has {limits['ram_kb']}KB",
                        recommendation="Reduce model size, context length, or use more aggressive quantization",
                    )
                )
            elif total_ram_needed > limits["ram_kb"] * 0.8:  # Using >80% of RAM
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.MODEL,
                        severity="warning",
                        message=f"Model uses {total_ram_needed:.0f}KB ({total_ram_needed/limits['ram_kb']*100:.0f}%) of available RAM",
                        recommendation="Consider optimizations for better memory efficiency",
                    )
                )

            # Check flash requirements
            if model.size_mb > limits["flash_mb"]:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.MODEL,
                        severity="error",
                        message=f"Model size {model.size_mb:.1f}MB exceeds {platform} flash limit of {limits['flash_mb']}MB",
                        recommendation="Use external storage or more aggressive compression",
                    )
                )

        return issues

    def _validate_performance_expectations(
        self, model: QuantizedModel, platform: str
    ) -> List[ValidationIssue]:
        """Validate performance expectations."""
        issues = []

        # Estimate performance based on model and platform
        expected_performance = self._estimate_performance(model, platform)

        if expected_performance < 1.0:  # Less than 1 token/second
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.MODEL,
                    severity="warning",
                    message=f"Expected performance very low: {expected_performance:.1f} tokens/second",
                    recommendation="Consider smaller model or more aggressive optimizations",
                )
            )
        elif expected_performance < 5.0:  # Less than 5 tokens/second
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.MODEL,
                    severity="info",
                    message=f"Expected performance: {expected_performance:.1f} tokens/second",
                    recommendation="Performance may be adequate for basic use cases",
                )
            )

        return issues

    def _estimate_performance(self, model: QuantizedModel, platform: str) -> float:
        """Rough performance estimation."""
        # Platform performance factors (relative to ESP32)
        platform_factors = {
            "esp32": 1.0,
            "esp32s3": 1.2,
            "stm32f4": 0.6,
            "stm32f7": 0.8,
            "rp2040": 0.5,
            "nrf52840": 0.3,
            "rpi_zero": 2.0,
            "jetson_nano": 10.0,
        }

        # Quantization performance factors
        quant_factors = {
            "2bit": 2.0,
            "3bit": 1.6,
            "4bit": 1.3,
            "8bit": 1.0,
            "fp16": 0.8,
            "fp32": 0.4,
        }

        base_performance = 50.0 / max(
            model.size_mb, 0.1
        )  # Inverse relationship with size
        platform_factor = platform_factors.get(platform, 1.0)
        quant_factor = quant_factors.get(model.quantization, 1.0)

        return base_performance * platform_factor * quant_factor


class RobustProfilingWrapper:
    """Wrapper that adds robustness to profiling operations."""

    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        validation_level: ValidationLevel = ValidationLevel.MODERATE,
    ):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)

        # Validators
        self.system_validator = SystemValidator(validation_level)
        self.model_validator = ModelValidator(validation_level)

    def validate_before_profiling(
        self, model: QuantizedModel, platform: str, device_path: Optional[str] = None
    ) -> ValidationResult:
        """Comprehensive validation before profiling."""
        all_issues = []

        # System validation
        system_result = self.system_validator.validate_system()
        all_issues.extend(system_result.issues)

        # Platform validation
        platform_validator = PlatformValidator(platform)
        platform_result = platform_validator.validate_platform(device_path)
        all_issues.extend(platform_result.issues)

        # Model validation
        model_result = self.model_validator.validate_model(model, platform)
        all_issues.extend(model_result.issues)

        # Combined result
        has_errors = any(issue.severity == "error" for issue in all_issues)

        return ValidationResult(is_valid=not has_errors, issues=all_issues)

    def robust_profile(
        self, profiler_func: Callable, *args, **kwargs
    ) -> Union[ProfileResults, Exception]:
        """Execute profiling with retry logic and error handling."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Profiling attempt {attempt + 1}/{self.max_retries}")

                # Execute with timeout
                result = self._execute_with_timeout(profiler_func, *args, **kwargs)

                if result is not None:
                    self.logger.info("Profiling completed successfully")
                    return result
                else:
                    raise RuntimeError("Profiling returned None")

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Profiling attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Wait before retry (exponential backoff)
                    wait_time = 2**attempt
                    self.logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        self.logger.error(
            f"All profiling attempts failed. Last error: {last_exception}"
        )
        return last_exception

    def _execute_with_timeout(self, func: Callable, *args, **kwargs):
        """Execute function with timeout (simplified implementation)."""
        # In a real implementation, would use threading or multiprocessing
        # For now, just execute normally
        return func(*args, **kwargs)
