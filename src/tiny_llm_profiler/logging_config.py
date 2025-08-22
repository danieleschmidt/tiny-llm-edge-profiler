"""
Structured logging configuration for the Tiny LLM Edge Profiler.
"""

import os
import sys
import json
import logging
import logging.handlers
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import traceback

from .exceptions import TinyLLMProfilerError


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add thread/process information if available
        if hasattr(record, "thread") and record.thread:
            log_data["thread_id"] = record.thread

        if hasattr(record, "process") and record.process:
            log_data["process_id"] = record.process

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add custom exception details for TinyLLMProfilerError
        if hasattr(record, "exception_details"):
            log_data["exception_details"] = record.exception_details

        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = [
                "platform",
                "model_name",
                "device_path",
                "operation",
                "duration",
                "metrics",
                "profiling_session_id",
            ]

            for field in extra_fields:
                if hasattr(record, field):
                    log_data[field] = getattr(record, field)

        return json.dumps(log_data, separators=(",", ":"))


class ContextAwareLogger:
    """
    Context-aware logger that maintains profiling session context.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """Set logging context that will be included in all log messages."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear the logging context."""
        self.context.clear()

    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log message with current context."""
        # Merge context with provided kwargs
        log_kwargs = {**self.context, **kwargs}

        # Create log record with extra fields
        extra = {k: v for k, v in log_kwargs.items() if k not in ["exc_info"]}

        self.logger.log(level, message, extra=extra, exc_info=kwargs.get("exc_info"))

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def exception(
        self, message: str, exception: Optional[Exception] = None, **kwargs
    ) -> None:
        """Log exception with full traceback."""
        if isinstance(exception, TinyLLMProfilerError):
            # Add custom exception details
            kwargs["exception_details"] = exception.to_dict()

        kwargs["exc_info"] = True
        self._log_with_context(logging.ERROR, message, **kwargs)


class PerformanceLogger:
    """Logger for performance metrics and profiling data."""

    def __init__(self, name: str = "tiny_llm_profiler.performance"):
        self.logger = logging.getLogger(name)

    def log_profiling_start(
        self, platform: str, model_name: str, session_id: str, **kwargs
    ) -> None:
        """Log the start of a profiling session."""
        self.logger.info(
            "Profiling session started",
            extra={
                "event_type": "profiling_start",
                "platform": platform,
                "model_name": model_name,
                "profiling_session_id": session_id,
                **kwargs,
            },
        )

    def log_profiling_end(
        self, session_id: str, duration_seconds: float, results_summary: Dict[str, Any]
    ) -> None:
        """Log the end of a profiling session."""
        self.logger.info(
            "Profiling session completed",
            extra={
                "event_type": "profiling_end",
                "profiling_session_id": session_id,
                "duration_seconds": duration_seconds,
                "results_summary": results_summary,
            },
        )

    def log_metric(self, metric_name: str, value: float, unit: str, **kwargs) -> None:
        """Log a performance metric."""
        self.logger.info(
            f"Metric recorded: {metric_name}",
            extra={
                "event_type": "metric",
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                **kwargs,
            },
        )

    def log_device_event(
        self, event_type: str, platform: str, device_path: str, **kwargs
    ) -> None:
        """Log device-related events."""
        self.logger.info(
            f"Device event: {event_type}",
            extra={
                "event_type": "device_event",
                "device_event_type": event_type,
                "platform": platform,
                "device_path": device_path,
                **kwargs,
            },
        )


def setup_logging(
    log_level: Union[str, int] = logging.INFO,
    log_dir: Optional[Path] = None,
    console_output: bool = True,
    json_format: bool = True,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
) -> None:
    """
    Setup logging configuration for the profiler.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None = no file logging)
        console_output: Enable console output
        json_format: Use JSON formatting
        max_file_size_mb: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """

    # Convert string log level to constant
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Create root logger
    root_logger = logging.getLogger("tiny_llm_profiler")
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Setup formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handlers
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main application log
        app_log_file = log_dir / "tiny_llm_profiler.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
        )
        app_handler.setLevel(log_level)
        app_handler.setFormatter(formatter)
        root_logger.addHandler(app_handler)

        # Performance metrics log
        perf_log_file = log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(formatter)

        # Performance logger gets its own handler
        perf_logger = logging.getLogger("tiny_llm_profiler.performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)

        # Error log (errors and above only)
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("serial").setLevel(logging.WARNING)


def get_logger(name: str) -> ContextAwareLogger:
    """Get a context-aware logger instance."""
    return ContextAwareLogger(f"tiny_llm_profiler.{name}")


def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger instance."""
    return PerformanceLogger()


class LoggingContextManager:
    """Context manager for setting logging context for a block of code."""

    def __init__(self, logger: ContextAwareLogger, **context):
        self.logger = logger
        self.context = context
        self.previous_context = {}

    def __enter__(self):
        # Save previous context
        self.previous_context = self.logger.context.copy()
        # Set new context
        self.logger.set_context(**self.context)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        self.logger.context = self.previous_context

        # Log any exception that occurred
        if exc_type and issubclass(exc_type, Exception):
            self.logger.exception(
                f"Exception in logging context: {exc_type.__name__}", exception=exc_val
            )


def log_function_call(logger: Optional[ContextAwareLogger] = None):
    """Decorator to log function calls with parameters and execution time."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger

            # Log function entry
            func_logger.debug(
                f"Entering function: {func.__name__}",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )

            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)

                # Log successful completion
                duration = (datetime.now() - start_time).total_seconds()
                func_logger.debug(
                    f"Function completed: {func.__name__}",
                    function=func.__name__,
                    duration_seconds=duration,
                    success=True,
                )

                return result

            except Exception as e:
                # Log exception
                duration = (datetime.now() - start_time).total_seconds()
                func_logger.exception(
                    f"Function failed: {func.__name__}",
                    exception=e,
                    function=func.__name__,
                    duration_seconds=duration,
                    success=False,
                )
                raise

        return wrapper

    return decorator


# Default logger instances
main_logger = get_logger("main")
device_logger = get_logger("device")
model_logger = get_logger("model")
profiler_logger = get_logger("profiler")
analysis_logger = get_logger("analysis")


# Auto-configure logging from environment variables
def auto_configure_logging():
    """Configure logging based on environment variables."""
    log_level = os.environ.get("TINY_LLM_LOG_LEVEL", "INFO")
    log_dir = os.environ.get("TINY_LLM_LOG_DIR")
    console_output = os.environ.get("TINY_LLM_LOG_CONSOLE", "true").lower() == "true"
    json_format = os.environ.get("TINY_LLM_LOG_JSON", "true").lower() == "true"

    setup_logging(
        log_level=log_level,
        log_dir=Path(log_dir) if log_dir else None,
        console_output=console_output,
        json_format=json_format,
    )


# Configure logging on module import if not already configured
if not logging.getLogger("tiny_llm_profiler").handlers:
    auto_configure_logging()
