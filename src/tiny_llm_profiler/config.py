"""
Configuration management system for the Tiny LLM Edge Profiler.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict

from .exceptions import ConfigurationError, InputValidationError
from .security import validate_identifier, validate_file_path, safe_file_path
from .logging_config import get_logger

logger = get_logger("config")


class LogLevel(str, Enum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OutputFormat(str, Enum):
    """Supported output formats."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    YAML = "yaml"


@dataclass
class ProfilingDefaults:
    """Default profiling configuration."""

    sample_rate_hz: int = 100
    duration_seconds: int = 60
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    timeout_seconds: int = 300
    enable_power_profiling: bool = False
    enable_memory_profiling: bool = True
    enable_latency_profiling: bool = True
    max_prompt_length: int = 10000
    max_prompts: int = 100
    default_quantization: str = "4bit"
    default_context_length: int = 2048


@dataclass
class SecurityConfig:
    """Security configuration options."""

    enable_input_validation: bool = True
    enable_path_sanitization: bool = True
    max_file_size_mb: int = 100
    allowed_model_extensions: List[str] = field(
        default_factory=lambda: [".gguf", ".ggml", ".bin", ".safetensors"]
    )
    enable_secure_temp_dirs: bool = True
    log_security_events: bool = True
    require_model_validation: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: LogLevel = LogLevel.INFO
    console_output: bool = True
    file_output: bool = True
    json_format: bool = True
    log_dir: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    performance_logging: bool = True
    security_logging: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""

    default_format: OutputFormat = OutputFormat.JSON
    output_dir: Optional[str] = None
    include_raw_data: bool = False
    compress_results: bool = False
    auto_timestamp: bool = True
    include_system_info: bool = True


class AppConfig(BaseModel):
    """Main application configuration using Pydantic for validation."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Don't allow extra fields
        use_enum_values=True,
    )

    # Application metadata
    app_name: str = Field(default="tiny-llm-edge-profiler")
    version: str = Field(default="0.1.0")

    # Profiling defaults
    profiling: ProfilingDefaults = Field(default_factory=ProfilingDefaults)

    # Security settings
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Logging configuration
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Output configuration
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Platform-specific overrides
    platform_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Device-specific settings
    device_defaults: Dict[str, Any] = Field(
        default_factory=lambda: {
            "baudrate": 921600,
            "timeout_seconds": 30,
            "connection_retries": 3,
            "retry_delay_seconds": 1,
        }
    )

    # Model repository settings
    model_repository: Dict[str, Any] = Field(
        default_factory=lambda: {
            "cache_dir": "~/.cache/tiny_llm_profiler/models",
            "auto_download": False,
            "verify_checksums": True,
            "max_cache_size_gb": 10,
        }
    )

    # Experimental features
    experimental: Dict[str, bool] = Field(
        default_factory=lambda: {
            "async_profiling": False,
            "gpu_acceleration": False,
            "distributed_profiling": False,
            "model_optimization": True,
        }
    )

    @validator("profiling")
    def validate_profiling_config(cls, v):
        """Validate profiling configuration."""
        if v.sample_rate_hz <= 0 or v.sample_rate_hz > 10000:
            raise ValueError("sample_rate_hz must be between 1 and 10000")

        if v.duration_seconds <= 0 or v.duration_seconds > 3600:
            raise ValueError("duration_seconds must be between 1 and 3600")

        if v.measurement_iterations <= 0 or v.measurement_iterations > 1000:
            raise ValueError("measurement_iterations must be between 1 and 1000")

        return v

    @validator("security")
    def validate_security_config(cls, v):
        """Validate security configuration."""
        if v.max_file_size_mb <= 0 or v.max_file_size_mb > 1000:
            raise ValueError("max_file_size_mb must be between 1 and 1000")

        # Validate allowed extensions
        for ext in v.allowed_model_extensions:
            if not ext.startswith("."):
                raise ValueError(f"Extension {ext} must start with a dot")

        return v

    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get configuration with platform-specific overrides applied."""
        base_config = asdict(self)

        if platform in self.platform_overrides:
            overrides = self.platform_overrides[platform]

            # Deep merge overrides
            def deep_merge(
                base: Dict[str, Any], override: Dict[str, Any]
            ) -> Dict[str, Any]:
                result = base.copy()
                for key, value in override.items():
                    if (
                        key in result
                        and isinstance(result[key], dict)
                        and isinstance(value, dict)
                    ):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result

            base_config = deep_merge(base_config, overrides)

        return base_config

    def export_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "profiling": asdict(self.profiling),
            "security": asdict(self.security),
            "logging": asdict(self.logging),
            "output": asdict(self.output),
            "platform_overrides": self.platform_overrides,
            "device_defaults": self.device_defaults,
            "model_repository": self.model_repository,
            "experimental": self.experimental,
        }


class ConfigManager:
    """Configuration manager with multiple source support."""

    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.config_sources: List[str] = []

        # Default configuration file locations
        self.default_config_paths = [
            Path.home() / ".config" / "tiny_llm_profiler" / "config.yaml",
            Path.cwd() / "tiny_llm_profiler_config.yaml",
            Path.cwd() / "config.yaml",
        ]

    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        load_from_env: bool = True,
    ) -> AppConfig:
        """
        Load configuration from various sources.

        Args:
            config_path: Explicit path to config file
            config_dict: Configuration dictionary to use directly
            load_from_env: Whether to load environment variable overrides

        Returns:
            Loaded and validated AppConfig
        """
        config_data = {}

        # 1. Start with defaults
        self.config = AppConfig()
        config_data = self.config.export_dict()

        # 2. Load from file if specified or found
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                file_config = self._load_config_file(config_path)
                config_data = self._deep_merge(config_data, file_config)
                self.config_sources.append(f"file:{config_path}")
            else:
                raise ConfigurationError(f"Config file not found: {config_path}")

        else:
            # Try default locations
            for default_path in self.default_config_paths:
                if default_path.exists():
                    file_config = self._load_config_file(default_path)
                    config_data = self._deep_merge(config_data, file_config)
                    self.config_sources.append(f"file:{default_path}")
                    break

        # 3. Override with provided dictionary
        if config_dict:
            config_data = self._deep_merge(config_data, config_dict)
            self.config_sources.append("dict")

        # 4. Override with environment variables
        if load_from_env:
            env_config = self._load_from_environment()
            if env_config:
                config_data = self._deep_merge(config_data, env_config)
                self.config_sources.append("environment")

        # 5. Validate and create final config
        try:
            self.config = AppConfig(**config_data)
            logger.info(
                f"Configuration loaded from sources: {', '.join(self.config_sources)}"
            )

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

        return self.config

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            # Validate file path for security
            safe_path = validate_file_path(
                config_path,
                allowed_extensions={".yaml", ".yml", ".json"},
                max_size=1024 * 1024,  # 1MB max
            )

            with open(safe_path, "r") as f:
                if safe_path.suffix.lower() in {".yaml", ".yml"}:
                    return yaml.safe_load(f) or {}
                elif safe_path.suffix.lower() == ".json":
                    return json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {safe_path.suffix}"
                    )

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file {config_path}: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {config_path}: {e}")

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        env_config = {}

        # Define environment variable mappings
        env_mappings = {
            # Logging
            "TINY_LLM_LOG_LEVEL": ("logging", "level"),
            "TINY_LLM_LOG_DIR": ("logging", "log_dir"),
            "TINY_LLM_LOG_JSON": ("logging", "json_format", bool),
            "TINY_LLM_LOG_CONSOLE": ("logging", "console_output", bool),
            # Security
            "TINY_LLM_SECURITY_VALIDATION": (
                "security",
                "enable_input_validation",
                bool,
            ),
            "TINY_LLM_MAX_FILE_SIZE": ("security", "max_file_size_mb", int),
            # Profiling
            "TINY_LLM_SAMPLE_RATE": ("profiling", "sample_rate_hz", int),
            "TINY_LLM_DURATION": ("profiling", "duration_seconds", int),
            "TINY_LLM_ITERATIONS": ("profiling", "measurement_iterations", int),
            # Output
            "TINY_LLM_OUTPUT_FORMAT": ("output", "default_format"),
            "TINY_LLM_OUTPUT_DIR": ("output", "output_dir"),
            # Model repository
            "TINY_LLM_MODEL_CACHE": ("model_repository", "cache_dir"),
            "TINY_LLM_AUTO_DOWNLOAD": ("model_repository", "auto_download", bool),
        }

        for env_var, (section, key, *type_info) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Type conversion
                if type_info:
                    value_type = type_info[0]
                    try:
                        if value_type == bool:
                            value = value.lower() in ("true", "1", "yes", "on")
                        elif value_type == int:
                            value = int(value)
                        elif value_type == float:
                            value = float(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid value for {env_var}: {value} (expected {value_type.__name__})"
                        )
                        continue

                # Set nested configuration
                if section not in env_config:
                    env_config[section] = {}
                env_config[section][key] = value

        return env_config

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self, output_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration
            format: Output format (yaml or json)
        """
        if not self.config:
            raise ConfigurationError("No configuration loaded")

        output_path = Path(output_path)
        config_data = self.config.export_dict()

        try:
            with open(output_path, "w") as f:
                if format.lower() == "yaml":
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Configuration saved to {output_path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if not self.config:
            # Load default configuration
            self.load_config()

        return self.config

    def create_default_config_file(self, output_path: Union[str, Path]) -> None:
        """Create a default configuration file with comments."""
        default_config = AppConfig()

        config_content = f"""# Tiny LLM Edge Profiler Configuration
# Generated configuration file with default values

app_name: "{default_config.app_name}"
version: "{default_config.version}"

# Profiling configuration
profiling:
  sample_rate_hz: {default_config.profiling.sample_rate_hz}  # Sampling rate for metrics collection
  duration_seconds: {default_config.profiling.duration_seconds}  # Default profiling duration
  warmup_iterations: {default_config.profiling.warmup_iterations}  # Warmup runs before measurement
  measurement_iterations: {default_config.profiling.measurement_iterations}  # Number of measurement runs
  timeout_seconds: {default_config.profiling.timeout_seconds}  # Timeout for operations
  enable_power_profiling: {str(default_config.profiling.enable_power_profiling).lower()}  # Enable power measurement
  enable_memory_profiling: {str(default_config.profiling.enable_memory_profiling).lower()}  # Enable memory profiling
  enable_latency_profiling: {str(default_config.profiling.enable_latency_profiling).lower()}  # Enable latency profiling
  max_prompt_length: {default_config.profiling.max_prompt_length}  # Maximum prompt length
  max_prompts: {default_config.profiling.max_prompts}  # Maximum number of prompts
  default_quantization: "{default_config.profiling.default_quantization}"  # Default quantization level
  default_context_length: {default_config.profiling.default_context_length}  # Default context length

# Security configuration
security:
  enable_input_validation: {str(default_config.security.enable_input_validation).lower()}  # Enable input validation
  enable_path_sanitization: {str(default_config.security.enable_path_sanitization).lower()}  # Enable path sanitization
  max_file_size_mb: {default_config.security.max_file_size_mb}  # Maximum file size for uploads
  allowed_model_extensions: {default_config.security.allowed_model_extensions}  # Allowed model file extensions
  enable_secure_temp_dirs: {str(default_config.security.enable_secure_temp_dirs).lower()}  # Use secure temporary directories
  log_security_events: {str(default_config.security.log_security_events).lower()}  # Log security events
  require_model_validation: {str(default_config.security.require_model_validation).lower()}  # Require model validation

# Logging configuration  
logging:
  level: "{default_config.logging.level.value}"  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  console_output: {str(default_config.logging.console_output).lower()}  # Enable console logging
  file_output: {str(default_config.logging.file_output).lower()}  # Enable file logging
  json_format: {str(default_config.logging.json_format).lower()}  # Use JSON log format
  log_dir: null  # Log directory (null for default)
  max_file_size_mb: {default_config.logging.max_file_size_mb}  # Maximum log file size
  backup_count: {default_config.logging.backup_count}  # Number of log file backups
  performance_logging: {str(default_config.logging.performance_logging).lower()}  # Enable performance logging
  security_logging: {str(default_config.logging.security_logging).lower()}  # Enable security logging

# Output configuration
output:
  default_format: "{default_config.output.default_format.value}"  # Default output format
  output_dir: null  # Output directory (null for current directory)
  include_raw_data: {str(default_config.output.include_raw_data).lower()}  # Include raw measurement data
  compress_results: {str(default_config.output.compress_results).lower()}  # Compress output files
  auto_timestamp: {str(default_config.output.auto_timestamp).lower()}  # Auto-add timestamps to filenames
  include_system_info: {str(default_config.output.include_system_info).lower()}  # Include system information

# Platform-specific overrides
platform_overrides: {{}}

# Device defaults
device_defaults:
  baudrate: {default_config.device_defaults['baudrate']}  # Default serial baudrate
  timeout_seconds: {default_config.device_defaults['timeout_seconds']}  # Device communication timeout
  connection_retries: {default_config.device_defaults['connection_retries']}  # Connection retry attempts
  retry_delay_seconds: {default_config.device_defaults['retry_delay_seconds']}  # Delay between retries

# Model repository settings
model_repository:
  cache_dir: "{default_config.model_repository['cache_dir']}"  # Model cache directory
  auto_download: {str(default_config.model_repository['auto_download']).lower()}  # Auto-download missing models
  verify_checksums: {str(default_config.model_repository['verify_checksums']).lower()}  # Verify model checksums
  max_cache_size_gb: {default_config.model_repository['max_cache_size_gb']}  # Maximum cache size

# Experimental features (use with caution)
experimental:
  async_profiling: {str(default_config.experimental['async_profiling']).lower()}  # Asynchronous profiling
  gpu_acceleration: {str(default_config.experimental['gpu_acceleration']).lower()}  # GPU acceleration
  distributed_profiling: {str(default_config.experimental['distributed_profiling']).lower()}  # Distributed profiling
  model_optimization: {str(default_config.experimental['model_optimization']).lower()}  # Model optimization
"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(config_content)

        logger.info(f"Default configuration file created at {output_path}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return config_manager.get_config()


def load_config(config_path: Optional[Union[str, Path]] = None, **kwargs) -> AppConfig:
    """Load configuration from file or environment."""
    return config_manager.load_config(config_path=config_path, **kwargs)


def create_default_config(output_path: Union[str, Path] = "config.yaml") -> None:
    """Create a default configuration file."""
    config_manager.create_default_config_file(output_path)
