"""
Custom exceptions for the Tiny LLM Edge Profiler.
"""

from typing import Optional, Dict, Any, List


class TinyLLMProfilerError(Exception):
    """Base exception for all profiler errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class DeviceError(TinyLLMProfilerError):
    """Errors related to device communication and hardware."""

    pass


class NetworkError(TinyLLMProfilerError):
    """Errors related to network communication and connectivity."""

    pass


class ResourceError(TinyLLMProfilerError):
    """Errors related to resource allocation and management."""

    pass


class BenchmarkError(TinyLLMProfilerError):
    """Errors related to benchmarking operations."""

    pass


class ValidationError(TinyLLMProfilerError):
    """Errors related to data validation."""

    pass


class DeviceConnectionError(DeviceError):
    """Device connection failed."""

    def __init__(
        self,
        device_path: str,
        platform: str,
        original_error: Optional[Exception] = None,
    ):
        message = f"Failed to connect to {platform} device at {device_path}"
        details = {
            "device_path": device_path,
            "platform": platform,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(message, "DEVICE_CONNECTION_FAILED", details)


class DeviceTimeoutError(DeviceError):
    """Device operation timed out."""

    def __init__(self, operation: str, timeout_seconds: float):
        message = (
            f"Device operation '{operation}' timed out after {timeout_seconds} seconds"
        )
        details = {"operation": operation, "timeout_seconds": timeout_seconds}
        super().__init__(message, "DEVICE_TIMEOUT", details)


class DeviceNotSupportedError(DeviceError):
    """Platform or device not supported."""

    def __init__(self, platform: str, supported_platforms: Optional[List[str]] = None):
        message = f"Platform '{platform}' is not supported"
        details = {
            "platform": platform,
            "supported_platforms": supported_platforms or [],
        }
        super().__init__(message, "DEVICE_NOT_SUPPORTED", details)


class PlatformError(TinyLLMProfilerError):
    """Errors related to platform operations."""

    pass


class ModelError(TinyLLMProfilerError):
    """Errors related to model loading and validation."""

    pass


class ModelLoadError(ModelError):
    """Failed to load model file."""

    def __init__(
        self, model_path: str, reason: str, original_error: Optional[Exception] = None
    ):
        message = f"Failed to load model from {model_path}: {reason}"
        details = {
            "model_path": model_path,
            "reason": reason,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(message, "MODEL_LOAD_FAILED", details)


class ModelValidationError(ModelError):
    """Model validation failed."""

    def __init__(self, model_name: str, validation_issues: List[str]):
        message = (
            f"Model '{model_name}' failed validation: {'; '.join(validation_issues)}"
        )
        details = {"model_name": model_name, "validation_issues": validation_issues}
        super().__init__(message, "MODEL_VALIDATION_FAILED", details)


class ModelCompatibilityError(ModelError):
    """Model not compatible with target platform."""

    def __init__(self, model_name: str, platform: str, compatibility_issues: List[str]):
        message = f"Model '{model_name}' is not compatible with platform '{platform}'"
        details = {
            "model_name": model_name,
            "platform": platform,
            "compatibility_issues": compatibility_issues,
        }
        super().__init__(message, "MODEL_COMPATIBILITY_FAILED", details)


class ModelFormatError(ModelError):
    """Unsupported or corrupted model format."""

    def __init__(
        self,
        model_path: str,
        expected_format: Optional[str] = None,
        detected_format: Optional[str] = None,
    ):
        message = f"Invalid model format: {model_path}"
        if expected_format and detected_format:
            message += f" (expected {expected_format}, got {detected_format})"

        details = {
            "model_path": model_path,
            "expected_format": expected_format,
            "detected_format": detected_format,
        }
        super().__init__(message, "MODEL_FORMAT_ERROR", details)


class ProfilingError(TinyLLMProfilerError):
    """Errors during profiling operations."""

    pass


class ProfilingTimeoutError(ProfilingError):
    """Profiling operation timed out."""

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        partial_results: Optional[Dict[str, Any]] = None,
    ):
        message = f"Profiling operation '{operation}' timed out after {timeout_seconds} seconds"
        details = {
            "operation": operation,
            "timeout_seconds": timeout_seconds,
            "partial_results_available": partial_results is not None,
        }
        super().__init__(message, "PROFILING_TIMEOUT", details)
        self.partial_results = partial_results


class InsufficientDataError(ProfilingError):
    """Not enough data collected for reliable profiling."""

    def __init__(self, metric: str, collected_samples: int, required_samples: int):
        message = f"Insufficient data for {metric}: collected {collected_samples}, required {required_samples}"
        details = {
            "metric": metric,
            "collected_samples": collected_samples,
            "required_samples": required_samples,
        }
        super().__init__(message, "INSUFFICIENT_DATA", details)


class ProfilerConfigurationError(ProfilingError):
    """Invalid profiling configuration."""

    def __init__(self, parameter: str, value: Any, reason: str):
        message = f"Invalid configuration for '{parameter}': {value} ({reason})"
        details = {"parameter": parameter, "value": value, "reason": reason}
        super().__init__(message, "PROFILER_CONFIG_ERROR", details)


# ProfilerError alias for backward compatibility
ProfilerError = TinyLLMProfilerError


class ResourceError(TinyLLMProfilerError):
    """Errors related to system resources."""

    pass


class InsufficientMemoryError(ResourceError):
    """Not enough memory available for operation."""

    def __init__(self, required_mb: float, available_mb: float, operation: str):
        message = f"Insufficient memory for {operation}: required {required_mb:.1f}MB, available {available_mb:.1f}MB"
        details = {
            "required_mb": required_mb,
            "available_mb": available_mb,
            "operation": operation,
        }
        super().__init__(message, "INSUFFICIENT_MEMORY", details)


class DiskSpaceError(ResourceError):
    """Not enough disk space available."""

    def __init__(self, required_mb: float, available_mb: float, path: str):
        message = f"Insufficient disk space at {path}: required {required_mb:.1f}MB, available {available_mb:.1f}MB"
        details = {
            "required_mb": required_mb,
            "available_mb": available_mb,
            "path": path,
        }
        super().__init__(message, "INSUFFICIENT_DISK_SPACE", details)


class PermissionError(ResourceError):
    """Permission denied for required operation."""

    def __init__(self, resource: str, operation: str):
        message = f"Permission denied: cannot {operation} {resource}"
        details = {"resource": resource, "operation": operation}
        super().__init__(message, "PERMISSION_DENIED", details)


class SecurityError(TinyLLMProfilerError):
    """Security-related errors."""

    pass


class UnsafePathError(SecurityError):
    """Potentially unsafe file path detected."""

    def __init__(self, path: str, reason: str):
        message = f"Unsafe path detected: {path} ({reason})"
        details = {"path": path, "reason": reason}
        super().__init__(message, "UNSAFE_PATH", details)


class InputValidationError(SecurityError):
    """Input validation failed."""

    def __init__(self, parameter: str, value: Any, validation_rule: str):
        message = f"Input validation failed for '{parameter}': {value} (rule: {validation_rule})"
        details = {
            "parameter": parameter,
            "value": str(value),
            "validation_rule": validation_rule,
        }
        super().__init__(message, "INPUT_VALIDATION_FAILED", details)


class ConfigurationError(TinyLLMProfilerError):
    """Configuration and setup errors."""

    pass


class MissingDependencyError(ConfigurationError):
    """Required dependency not found."""

    def __init__(self, dependency: str, install_command: Optional[str] = None):
        message = f"Missing required dependency: {dependency}"
        if install_command:
            message += f" (install with: {install_command})"

        details = {"dependency": dependency, "install_command": install_command}
        super().__init__(message, "MISSING_DEPENDENCY", details)


class EnvironmentError(ConfigurationError):
    """Environment setup or validation error."""

    def __init__(self, environment_issue: str, suggested_fix: Optional[str] = None):
        message = f"Environment issue: {environment_issue}"
        if suggested_fix:
            message += f" (suggested fix: {suggested_fix})"

        details = {
            "environment_issue": environment_issue,
            "suggested_fix": suggested_fix,
        }
        super().__init__(message, "ENVIRONMENT_ERROR", details)


class AnalysisError(TinyLLMProfilerError):
    """Errors during result analysis."""

    pass


class InsufficientResultsError(AnalysisError):
    """Not enough results for meaningful analysis."""

    def __init__(
        self, analysis_type: str, available_results: int, required_results: int
    ):
        message = f"Insufficient results for {analysis_type}: {available_results} available, {required_results} required"
        details = {
            "analysis_type": analysis_type,
            "available_results": available_results,
            "required_results": required_results,
        }
        super().__init__(message, "INSUFFICIENT_RESULTS", details)


class DataCorruptionError(AnalysisError):
    """Detected corrupted or invalid data."""

    def __init__(self, data_type: str, corruption_details: str):
        message = f"Data corruption detected in {data_type}: {corruption_details}"
        details = {"data_type": data_type, "corruption_details": corruption_details}
        super().__init__(message, "DATA_CORRUPTION", details)


# Exception handling utilities


def handle_device_error(func):
    """Decorator to handle device-related exceptions."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "permission denied" in str(e).lower():
                raise PermissionError("device", "access") from e
            elif "timeout" in str(e).lower():
                raise DeviceTimeoutError(func.__name__, 30.0) from e
            elif "connection" in str(e).lower() or "device" in str(e).lower():
                platform = kwargs.get("platform", "unknown")
                device = kwargs.get("device", "unknown")
                raise DeviceConnectionError(device, platform, e) from e
            else:
                raise DeviceError(f"Device error in {func.__name__}: {e}") from e

    return wrapper


def handle_model_error(func):
    """Decorator to handle model-related exceptions."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            model_path = kwargs.get("model_path") or (args[0] if args else "unknown")
            raise ModelLoadError(str(model_path), "File not found", e) from e
        except PermissionError as e:
            model_path = kwargs.get("model_path") or (args[0] if args else "unknown")
            raise ModelLoadError(str(model_path), "Permission denied", e) from e
        except Exception as e:
            if "format" in str(e).lower() or "corrupt" in str(e).lower():
                model_path = kwargs.get("model_path") or (
                    args[0] if args else "unknown"
                )
                raise ModelFormatError(str(model_path)) from e
            else:
                model_name = kwargs.get("model_name", "unknown")
                raise ModelError(f"Model error in {func.__name__}: {e}") from e

    return wrapper


def validate_input(
    value: Any, validation_rules: Dict[str, Any], parameter_name: str
) -> Any:
    """
    Validate input against a set of rules.

    Args:
        value: Value to validate
        validation_rules: Dictionary of validation rules
        parameter_name: Name of the parameter being validated

    Returns:
        Validated value (potentially transformed)

    Raises:
        InputValidationError: If validation fails
    """
    # Type validation
    if "type" in validation_rules:
        expected_type = validation_rules["type"]
        if not isinstance(value, expected_type):
            raise InputValidationError(
                parameter_name, value, f"must be of type {expected_type.__name__}"
            )

    # Range validation for numbers
    if isinstance(value, (int, float)):
        if "min_value" in validation_rules and value < validation_rules["min_value"]:
            raise InputValidationError(
                parameter_name, value, f"must be >= {validation_rules['min_value']}"
            )

        if "max_value" in validation_rules and value > validation_rules["max_value"]:
            raise InputValidationError(
                parameter_name, value, f"must be <= {validation_rules['max_value']}"
            )

    # String validation
    if isinstance(value, str):
        if (
            "min_length" in validation_rules
            and len(value) < validation_rules["min_length"]
        ):
            raise InputValidationError(
                parameter_name,
                value,
                f"must be at least {validation_rules['min_length']} characters",
            )

        if (
            "max_length" in validation_rules
            and len(value) > validation_rules["max_length"]
        ):
            raise InputValidationError(
                parameter_name,
                value,
                f"must be at most {validation_rules['max_length']} characters",
            )

        if (
            "allowed_values" in validation_rules
            and value not in validation_rules["allowed_values"]
        ):
            raise InputValidationError(
                parameter_name,
                value,
                f"must be one of {validation_rules['allowed_values']}",
            )

        # Path safety validation
        if validation_rules.get("safe_path", False):
            import os

            if (
                ".." in value
                or os.path.isabs(value)
                and not validation_rules.get("allow_absolute", False)
            ):
                raise UnsafePathError(
                    value, "contains potentially unsafe path components"
                )

    # List validation
    if isinstance(value, list):
        if (
            "min_items" in validation_rules
            and len(value) < validation_rules["min_items"]
        ):
            raise InputValidationError(
                parameter_name,
                value,
                f"must have at least {validation_rules['min_items']} items",
            )

        if (
            "max_items" in validation_rules
            and len(value) > validation_rules["max_items"]
        ):
            raise InputValidationError(
                parameter_name,
                value,
                f"must have at most {validation_rules['max_items']} items",
            )

    return value


def safe_file_path(path: str, base_directory: Optional[str] = None) -> str:
    """
    Validate and sanitize a file path for security.

    Args:
        path: File path to validate
        base_directory: Optional base directory to restrict access to

    Returns:
        Sanitized absolute path

    Raises:
        UnsafePathError: If path is potentially unsafe
    """
    import os
    from pathlib import Path

    # Basic path traversal protection
    if ".." in path:
        raise UnsafePathError(path, "contains parent directory references")

    # Convert to absolute path
    abs_path = os.path.abspath(path)

    # If base directory specified, ensure path is within it
    if base_directory:
        base_abs = os.path.abspath(base_directory)
        if not abs_path.startswith(base_abs):
            raise UnsafePathError(
                path, f"path outside allowed directory {base_directory}"
            )

    # Check for suspicious patterns
    suspicious_patterns = ["../", "..\\", "/etc/", "/proc/", "/sys/", "C:\\Windows\\"]
    for pattern in suspicious_patterns:
        if pattern in abs_path:
            raise UnsafePathError(path, f"contains suspicious pattern: {pattern}")

    return abs_path
