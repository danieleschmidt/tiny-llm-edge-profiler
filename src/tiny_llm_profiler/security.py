"""
Security utilities and input validation for the Tiny LLM Edge Profiler.
"""

import os
import re
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import tempfile
import stat

from .exceptions import SecurityError, UnsafePathError, InputValidationError
from .logging_config import get_logger

logger = get_logger("security")


class SecurityValidator:
    """Centralized security validation and sanitization."""

    # Allowed file extensions for models
    ALLOWED_MODEL_EXTENSIONS = {".gguf", ".ggml", ".bin", ".safetensors", ".pt", ".pth"}

    # Allowed characters in identifiers (platform names, model names, etc.)
    IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")

    # Maximum file sizes (in bytes)
    MAX_MODEL_SIZE = 100 * 1024 * 1024  # 100MB

    @classmethod
    def validate_identifier(cls, identifier: str, name: str = "identifier") -> str:
        """Validate that an identifier is safe (platform names, model names, etc.)."""
        if not isinstance(identifier, str):
            raise InputValidationError(f"{name} must be a string")

        if not cls.IDENTIFIER_PATTERN.match(identifier):
            raise InputValidationError(
                f"{name} '{identifier}' contains invalid characters or is too long. "
                f"Only alphanumeric, underscore, dash, and dot are allowed (max 64 chars)"
            )

        return identifier

    @classmethod
    def validate_file_path(cls, file_path: str) -> str:
        """Validate a file path is safe."""
        path = Path(file_path)
        if path.is_absolute() and not str(path).startswith("/tmp/"):
            # Allow absolute paths only in /tmp for security
            if not str(path).startswith(("/home/", "/opt/", "/usr/local/")):
                raise UnsafePathError(f"Absolute path not allowed: {file_path}")
        return file_path

    @classmethod
    def generate_session_id(cls) -> str:
        """Generate a secure session ID."""
        return secrets.token_hex(16)

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize a filename by removing dangerous characters."""
        # Remove path separators and null bytes
        sanitized = re.sub(r"[/\\:\0]", "_", filename)
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)
        # Limit length
        return sanitized[:255]

    @classmethod
    def compute_file_hash(cls, file_path: str, algorithm: str = "sha256") -> str:
        """Compute hash of a file."""
        hash_obj = hashlib.new(algorithm)
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute hash for {file_path}: {e}")
            raise SecurityError(f"Failed to compute file hash: {e}")

    @classmethod
    def validate_model_file(cls, file_path: str) -> str:
        """Validate model file and return the path if safe."""
        path = Path(file_path)
        if not path.exists():
            raise SecurityError(f"Model file does not exist: {file_path}")

        # Check file size
        if path.stat().st_size > cls.MAX_MODEL_SIZE:
            raise SecurityError(f"Model file too large: {path.stat().st_size} bytes")

        # Check extension
        if path.suffix not in cls.ALLOWED_MODEL_EXTENSIONS:
            raise SecurityError(f"Invalid model file extension: {path.suffix}")

        return file_path

    @classmethod
    def validate_device_path(cls, device_path: str) -> str:
        """Validate device path is safe."""
        path = Path(device_path)

        # Allow common device paths
        allowed_prefixes = ["/dev/", "/tmp/", "COM", "local"]
        if not any(
            str(path).startswith(prefix) or str(path) == prefix
            for prefix in allowed_prefixes
        ):
            raise SecurityError(f"Device path not allowed: {device_path}")

        return device_path


class FileValidator:
    """File validation and security checking."""

    @staticmethod
    def validate_model_file(file_path: Path) -> bool:
        """Validate a model file is safe to load."""
        try:
            if not file_path.exists():
                return False

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > SecurityValidator.MAX_MODEL_SIZE:
                return False

            # Check extension
            if file_path.suffix not in SecurityValidator.ALLOWED_MODEL_EXTENSIONS:
                return False

            # Check permissions
            mode = file_path.stat().st_mode
            if mode & stat.S_IWOTH:  # World writable
                return False

            return True
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False

    @staticmethod
    def create_secure_temp_file(prefix: str = "tiny_llm_") -> Path:
        """Create a secure temporary file."""
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=".tmp")
        os.close(fd)

        # Set restrictive permissions
        temp_path = Path(path)
        temp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return temp_path


class InputSanitizer:
    """Input sanitization for user-provided data."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 255) -> str:
        """Sanitize a string input."""
        if not isinstance(value, str):
            raise InputValidationError("Input must be a string")

        # Remove null bytes and control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)

        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    @staticmethod
    def validate_platform_name(name: str) -> bool:
        """Validate platform name follows security constraints."""
        return bool(SecurityValidator.IDENTIFIER_PATTERN.match(name))

    @staticmethod
    def validate_model_name(name: str) -> bool:
        """Validate model name follows security constraints."""
        return bool(SecurityValidator.IDENTIFIER_PATTERN.match(name))

    @staticmethod
    def sanitize_prompts(prompts: list, max_length: int = 10000) -> list:
        """Sanitize a list of prompts."""
        sanitized = []
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise InputValidationError("All prompts must be strings")

            # Remove dangerous characters and limit length
            clean = InputSanitizer.sanitize_string(prompt, max_length)
            sanitized.append(clean)

        return sanitized

    MAX_CONFIG_SIZE = 1024 * 1024  # 1MB
    MAX_LOG_SIZE = 50 * 1024 * 1024  # 50MB

    @classmethod
    def validate_identifier(cls, identifier: str, name: str = "identifier") -> str:
        """
        Validate that an identifier is safe (platform names, model names, etc.).

        Args:
            identifier: The identifier to validate
            name: Human-readable name for error messages

        Returns:
            The validated identifier

        Raises:
            InputValidationError: If identifier is invalid
        """
        if not identifier:
            raise InputValidationError(name, identifier, "cannot be empty")

        if not isinstance(identifier, str):
            raise InputValidationError(name, identifier, "must be a string")

        if not cls.IDENTIFIER_PATTERN.match(identifier):
            raise InputValidationError(
                name,
                identifier,
                "must contain only alphanumeric characters, hyphens, underscores, and dots (1-64 chars)",
            )

        # Additional security checks
        if identifier.startswith(".") or identifier.endswith("."):
            raise InputValidationError(
                name, identifier, "cannot start or end with a dot"
            )

        if ".." in identifier:
            raise InputValidationError(
                name, identifier, "cannot contain consecutive dots"
            )

        logger.debug(f"Validated identifier: {name}={identifier}")
        return identifier

    @classmethod
    def validate_file_path(
        cls,
        file_path: Union[str, Path],
        allowed_extensions: Optional[set] = None,
        base_directory: Optional[Path] = None,
        max_size: Optional[int] = None,
        must_exist: bool = False,
    ) -> Path:
        """
        Validate and sanitize a file path for security.

        Args:
            file_path: Path to validate
            allowed_extensions: Set of allowed file extensions
            base_directory: Base directory to restrict access to
            max_size: Maximum file size in bytes
            must_exist: Whether the file must exist

        Returns:
            Validated Path object

        Raises:
            UnsafePathError: If path is unsafe
            InputValidationError: If path is invalid
        """
        if not file_path:
            raise InputValidationError("file_path", file_path, "cannot be empty")

        path = Path(file_path)

        # Convert to absolute path for security checks
        try:
            abs_path = path.resolve()
        except (OSError, RuntimeError) as e:
            raise UnsafePathError(str(file_path), f"cannot resolve path: {e}")

        # Check for path traversal attempts
        if ".." in str(file_path):
            raise UnsafePathError(
                str(file_path), "contains parent directory references"
            )

        # Validate against base directory if specified
        if base_directory:
            base_abs = base_directory.resolve()
            try:
                abs_path.relative_to(base_abs)
            except ValueError:
                raise UnsafePathError(
                    str(file_path), f"path outside allowed directory {base_directory}"
                )

        # Check file extension if restrictions specified
        if allowed_extensions and abs_path.suffix.lower() not in allowed_extensions:
            raise InputValidationError(
                "file_path",
                file_path,
                f"file extension must be one of: {', '.join(allowed_extensions)}",
            )

        # Check if file exists when required
        if must_exist and not abs_path.exists():
            raise InputValidationError("file_path", file_path, "file does not exist")

        # Check file size if it exists and limit is set
        if max_size and abs_path.exists():
            file_size = abs_path.stat().st_size
            if file_size > max_size:
                raise InputValidationError(
                    "file_path",
                    file_path,
                    f"file too large: {file_size} bytes (max: {max_size} bytes)",
                )

        # Check for suspicious patterns in path
        suspicious_patterns = [
            "/etc/",
            "/proc/",
            "/sys/",
            "/dev/",
            "/var/log/",
            "C:\\Windows\\",
            "C:\\System32\\",
            "/System/",
            "/Library/",
        ]

        for pattern in suspicious_patterns:
            if pattern in str(abs_path):
                raise UnsafePathError(
                    str(file_path), f"contains suspicious pattern: {pattern}"
                )

        logger.debug(f"Validated file path: {abs_path}")
        return abs_path

    @classmethod
    def validate_model_file(cls, model_path: Union[str, Path]) -> Path:
        """Validate a model file path with appropriate security checks."""
        return cls.validate_file_path(
            model_path,
            allowed_extensions=cls.ALLOWED_MODEL_EXTENSIONS,
            max_size=cls.MAX_MODEL_SIZE,
            must_exist=True,
        )

    @classmethod
    def validate_device_path(cls, device_path: str) -> str:
        """
        Validate a device path (serial port, etc.).

        Args:
            device_path: Device path to validate

        Returns:
            Validated device path

        Raises:
            InputValidationError: If device path is invalid
        """
        if not device_path:
            raise InputValidationError("device_path", device_path, "cannot be empty")

        if not isinstance(device_path, str):
            raise InputValidationError("device_path", device_path, "must be a string")

        # Common device path patterns
        valid_patterns = [
            r"^/dev/tty[A-Z]{2,4}\d+$",  # Linux/Mac serial ports
            r"^COM\d+$",  # Windows serial ports
            r"^/dev/cu\.[a-zA-Z0-9_\-\.]+$",  # Mac USB serial
            r"^/dev/serial/by-id/[a-zA-Z0-9_\-\.]+$",  # Linux by-id
        ]

        if not any(re.match(pattern, device_path) for pattern in valid_patterns):
            raise InputValidationError(
                "device_path", device_path, "invalid device path format"
            )

        logger.debug(f"Validated device path: {device_path}")
        return device_path

    @classmethod
    def validate_network_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate network configuration parameters.

        Args:
            config: Network configuration dictionary

        Returns:
            Validated configuration

        Raises:
            InputValidationError: If configuration is invalid
        """
        validated = {}

        # Validate host/IP address
        if "host" in config:
            host = config["host"]
            if not isinstance(host, str) or not host:
                raise InputValidationError("host", host, "must be a non-empty string")

            # Simple validation - could be enhanced with proper IP/hostname validation
            if not re.match(r"^[a-zA-Z0-9\.\-]+$", host):
                raise InputValidationError("host", host, "contains invalid characters")

            validated["host"] = host

        # Validate port
        if "port" in config:
            port = config["port"]
            if not isinstance(port, int) or not (1 <= port <= 65535):
                raise InputValidationError(
                    "port", port, "must be an integer between 1 and 65535"
                )

            # Warn about privileged ports
            if port < 1024:
                logger.warning(f"Using privileged port {port}")

            validated["port"] = port

        return validated

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize a filename by removing/replacing unsafe characters.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        if not filename:
            return "unnamed_file"

        # Remove path separators and other unsafe characters
        unsafe_chars = r'[<>:"/\\|?*\x00-\x1f]'
        sanitized = re.sub(unsafe_chars, "_", filename)

        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(" .")

        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed_file"

        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]

        return sanitized

    @classmethod
    def generate_session_id(cls) -> str:
        """Generate a cryptographically secure session ID."""
        return secrets.token_urlsafe(32)

    @classmethod
    def compute_file_hash(cls, file_path: Path, algorithm: str = "sha256") -> str:
        """
        Compute hash of a file for integrity checking.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, md5, etc.)

        Returns:
            Hex digest of the file hash
        """
        hasher = hashlib.new(algorithm)

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        except (OSError, IOError) as e:
            raise SecurityError(f"Cannot compute hash for {file_path}: {e}")

        return hasher.hexdigest()


class SecureTemporaryDirectory:
    """Context manager for creating secure temporary directories."""

    def __init__(self, prefix: str = "tiny_llm_profiler_"):
        self.prefix = prefix
        self.temp_dir: Optional[Path] = None

    def __enter__(self) -> Path:
        """Create secure temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))

        # Set restrictive permissions (owner only)
        os.chmod(self.temp_dir, stat.S_IRWXU)

        logger.debug(f"Created secure temporary directory: {self.temp_dir}")
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except OSError as e:
                logger.warning(
                    f"Failed to clean up temporary directory {self.temp_dir}: {e}"
                )


def secure_delete_file(file_path: Path, passes: int = 1) -> bool:
    """
    Securely delete a file by overwriting it before deletion.

    Args:
        file_path: Path to file to delete
        passes: Number of overwrite passes

    Returns:
        True if successfully deleted, False otherwise
    """
    try:
        if not file_path.exists():
            return True

        file_size = file_path.stat().st_size

        # Overwrite file with random data
        with open(file_path, "r+b") as f:
            for _ in range(passes):
                f.seek(0)
                f.write(secrets.token_bytes(file_size))
                f.flush()
                os.fsync(f.fileno())

        # Remove the file
        file_path.unlink()
        logger.info(f"Securely deleted file: {file_path}")
        return True

    except (OSError, IOError) as e:
        logger.error(f"Failed to securely delete {file_path}: {e}")
        return False


class InputSanitizer:
    """Utilities for sanitizing user inputs."""

    @staticmethod
    def sanitize_string(
        input_str: str,
        max_length: int = 1000,
        allowed_chars: Optional[str] = None,
        strip_whitespace: bool = True,
    ) -> str:
        """
        Sanitize a string input.

        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters
            strip_whitespace: Whether to strip leading/trailing whitespace

        Returns:
            Sanitized string

        Raises:
            InputValidationError: If input is invalid
        """
        if not isinstance(input_str, str):
            raise InputValidationError("input", input_str, "must be a string")

        # Strip whitespace if requested
        if strip_whitespace:
            input_str = input_str.strip()

        # Check length
        if len(input_str) > max_length:
            raise InputValidationError(
                "input",
                input_str[:50] + "...",
                f"exceeds maximum length of {max_length}",
            )

        # Check allowed characters
        if allowed_chars and not re.match(allowed_chars, input_str):
            raise InputValidationError(
                "input",
                input_str,
                f"contains characters not matching pattern: {allowed_chars}",
            )

        return input_str

    @staticmethod
    def sanitize_prompts(
        prompts: List[str], max_prompt_length: int = 10000
    ) -> List[str]:
        """
        Sanitize a list of prompts for model inference.

        Args:
            prompts: List of prompt strings
            max_prompt_length: Maximum length per prompt

        Returns:
            List of sanitized prompts
        """
        if not isinstance(prompts, list):
            raise InputValidationError("prompts", prompts, "must be a list")

        if len(prompts) > 100:  # Reasonable limit
            raise InputValidationError(
                "prompts", prompts, "too many prompts (max: 100)"
            )

        sanitized = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise InputValidationError(f"prompts[{i}]", prompt, "must be a string")

            # Basic sanitization
            prompt = prompt.strip()

            if len(prompt) > max_prompt_length:
                raise InputValidationError(
                    f"prompts[{i}]",
                    prompt[:50] + "...",
                    f"exceeds maximum length of {max_prompt_length}",
                )

            if not prompt:  # Skip empty prompts
                continue

            sanitized.append(prompt)

        if not sanitized:
            raise InputValidationError("prompts", prompts, "no valid prompts provided")

        return sanitized


def validate_environment() -> Dict[str, Any]:
    """
    Comprehensive runtime environment security validation.

    Returns:
        Dictionary with environment validation results
    """
    results = {"secure": True, "warnings": [], "errors": [], "recommendations": []}

    # Check if running as root (not recommended)
    try:
        if os.geteuid() == 0:
            results["warnings"].append(
                "Running as root user - not recommended for security"
            )
            results["recommendations"].append("Run as a non-privileged user")
    except AttributeError:
        # Windows doesn't have geteuid
        import ctypes

        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if is_admin:
                results["warnings"].append(
                    "Running as Administrator - not recommended for security"
                )
                results["recommendations"].append("Run as a standard user")
        except Exception:
            pass  # Can't determine admin status

    # Check temporary directory permissions
    temp_dir = Path(tempfile.gettempdir())
    if temp_dir.exists():
        try:
            temp_stat = temp_dir.stat()
            if temp_stat.st_mode & stat.S_IWOTH:
                results["warnings"].append("Temporary directory is world-writable")
                results["recommendations"].append(
                    "Secure temporary directory permissions"
                )
        except (OSError, AttributeError):
            pass  # Windows or permission issue

    # Check for development/debug environment variables
    debug_vars = [
        "DEBUG",
        "DEVELOPMENT",
        "TINY_LLM_DEBUG",
        "FLASK_DEBUG",
        "DJANGO_DEBUG",
        "NODE_ENV=development",
    ]
    active_debug_vars = [var for var in debug_vars if os.environ.get(var.split("=")[0])]
    if active_debug_vars:
        results["warnings"].append(
            f"Debug environment variables active: {', '.join(active_debug_vars)}"
        )
        results["recommendations"].append("Disable debug variables in production")

    # Check Python version for known security issues
    import sys

    python_version = sys.version_info
    if python_version < (3, 8):
        results["errors"].append(
            "Python version is too old and may have security vulnerabilities"
        )
        results["secure"] = False
        results["recommendations"].append("Upgrade to Python 3.8 or newer")
    elif python_version < (3, 9):
        results["warnings"].append("Python version is getting old - consider upgrading")
        results["recommendations"].append(
            "Consider upgrading to a newer Python version"
        )

    # Check for common security environment issues
    if "PATH" in os.environ:
        path_entries = os.environ["PATH"].split(os.pathsep)
        suspicious_paths = [".", "", "./"]
        risky_paths = [p for p in path_entries if p in suspicious_paths]
        if risky_paths:
            results["warnings"].append(
                "PATH contains current directory - security risk"
            )
            results["recommendations"].append("Remove current directory from PATH")

    # Check for writable directories in PATH
    if "PATH" in os.environ:
        path_entries = os.environ["PATH"].split(os.pathsep)
        writable_paths = []
        for path_entry in path_entries[
            :10
        ]:  # Check first 10 to avoid performance issues
            if path_entry and Path(path_entry).exists():
                try:
                    test_file = Path(path_entry) / f"security_test_{os.getpid()}.tmp"
                    test_file.write_text("test")
                    test_file.unlink()
                    writable_paths.append(path_entry)
                except (OSError, PermissionError):
                    pass  # Not writable, which is good

        if writable_paths:
            results["warnings"].append(
                f"Found {len(writable_paths)} writable directories in PATH"
            )
            results["recommendations"].append(
                "Ensure PATH directories are not writable by regular users"
            )

    # Check for secure random number generation
    try:
        import secrets

        # Test that secrets module works
        secrets.token_bytes(32)
    except ImportError:
        results["errors"].append("Secure random number generation not available")
        results["secure"] = False
        results["recommendations"].append("Ensure Python 'secrets' module is available")

    # Check for SSL/TLS capabilities
    try:
        import ssl

        if not hasattr(ssl, "create_default_context"):
            results["warnings"].append("SSL context creation not available")
            results["recommendations"].append("Upgrade SSL/TLS libraries")
    except ImportError:
        results["warnings"].append("SSL module not available")
        results["recommendations"].append("Install SSL/TLS support")

    # Check umask (Unix-like systems)
    try:
        current_umask = os.umask(0)
        os.umask(current_umask)  # Restore original

        if current_umask & 0o022 == 0:  # Files would be group/world writable
            results["warnings"].append(
                f"Permissive umask detected: {oct(current_umask)}"
            )
            results["recommendations"].append(
                "Set more restrictive umask (e.g., 0o022 or 0o077)"
            )
    except (AttributeError, OSError):
        pass  # Windows or other issue

    # Check for virtualenv/venv usage
    in_virtualenv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if not in_virtualenv:
        results["warnings"].append("Not running in a virtual environment")
        results["recommendations"].append(
            "Use a virtual environment for better isolation"
        )

    # Final security assessment
    if results["errors"]:
        results["secure"] = False

    logger.info(
        f"Environment validation completed: {'secure' if results['secure'] else 'has issues'} "
        f"({len(results['warnings'])} warnings, {len(results['errors'])} errors)"
    )

    return results


class SecurityAuditor:
    """Comprehensive security auditing for the profiler."""

    def __init__(self):
        self.logger = get_logger("security_auditor")
        self.audit_results: List[Dict[str, Any]] = []

    def run_security_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit."""
        self.logger.info("Starting comprehensive security audit")

        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "audit_id": secrets.token_urlsafe(16),
            "overall_security_score": 0,
            "categories": {},
            "recommendations": [],
            "critical_issues": [],
            "warnings": [],
        }

        # Environment security
        env_results = validate_environment()
        audit_results["categories"]["environment"] = env_results

        # File system security
        fs_results = self._audit_filesystem()
        audit_results["categories"]["filesystem"] = fs_results

        # Network security
        net_results = self._audit_network_security()
        audit_results["categories"]["network"] = net_results

        # Application security
        app_results = self._audit_application_security()
        audit_results["categories"]["application"] = app_results

        # Calculate overall security score (0-100)
        category_scores = []
        for category, results in audit_results["categories"].items():
            if isinstance(results, dict) and "secure" in results:
                score = 100 if results["secure"] else 50
                score -= len(results.get("warnings", [])) * 5
                score -= len(results.get("errors", [])) * 10
                category_scores.append(max(0, score))

        audit_results["overall_security_score"] = (
            int(np.mean(category_scores)) if category_scores else 0
        )

        # Collect recommendations and issues
        for results in audit_results["categories"].values():
            if isinstance(results, dict):
                audit_results["recommendations"].extend(
                    results.get("recommendations", [])
                )
                audit_results["warnings"].extend(results.get("warnings", []))
                audit_results["critical_issues"].extend(results.get("errors", []))

        self.audit_results.append(audit_results)

        self.logger.info(
            f"Security audit completed. Score: {audit_results['overall_security_score']}/100"
        )

        return audit_results

    def _audit_filesystem(self) -> Dict[str, Any]:
        """Audit filesystem security."""
        results = {"secure": True, "warnings": [], "errors": [], "recommendations": []}

        # Check current working directory permissions
        try:
            cwd = Path.cwd()
            cwd_stat = cwd.stat()

            if cwd_stat.st_mode & stat.S_IWOTH:
                results["warnings"].append(
                    "Current working directory is world-writable"
                )
                results["recommendations"].append(
                    "Use a directory with restricted permissions"
                )
        except (OSError, AttributeError):
            pass

        # Check for sensitive files in current directory
        sensitive_patterns = [
            "*.key",
            "*.pem",
            "*.p12",
            "*.pfx",
            "*.jks",
            ".env",
            ".env.*",
            "config.ini",
            "secrets.*",
            "password*",
            "*password*",
            "credentials*",
        ]

        sensitive_files = []
        try:
            for pattern in sensitive_patterns:
                sensitive_files.extend(cwd.glob(pattern))
        except Exception as e:
            self.logger.debug(f"Error checking for sensitive files: {e}")

        if sensitive_files:
            results["warnings"].append(
                f"Found {len(sensitive_files)} potentially sensitive files"
            )
            results["recommendations"].append("Review and secure sensitive files")

        return results

    def _audit_network_security(self) -> Dict[str, Any]:
        """Audit network security configuration."""
        results = {"secure": True, "warnings": [], "errors": [], "recommendations": []}

        # Check for common insecure network configurations
        try:
            import socket

            # Check if we can bind to privileged ports (indicates running as admin/root)
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.bind(("127.0.0.1", 80))
                test_sock.close()
                results["warnings"].append(
                    "Can bind to privileged ports - running with elevated privileges"
                )
                results["recommendations"].append("Run with minimal privileges")
            except (OSError, socket.error):
                pass  # Good - can't bind to privileged ports

        except ImportError:
            pass

        return results

    def _audit_application_security(self) -> Dict[str, Any]:
        """Audit application-specific security."""
        results = {"secure": True, "warnings": [], "errors": [], "recommendations": []}

        # Check for debug mode
        config = get_config()
        if hasattr(config, "logging") and config.logging.level == "DEBUG":
            results["warnings"].append("Debug logging enabled in production")
            results["recommendations"].append(
                "Use INFO or WARNING level logging in production"
            )

        # Check for insecure defaults
        if hasattr(config, "security"):
            if not config.security.enable_input_validation:
                results["errors"].append("Input validation is disabled")
                results["secure"] = False
                results["recommendations"].append("Enable input validation")

            if not config.security.require_model_validation:
                results["warnings"].append("Model validation is disabled")
                results["recommendations"].append("Enable model validation")

        return results

    def export_audit_report(self, output_path: Path):
        """Export security audit report."""
        if not self.audit_results:
            self.logger.warning("No audit results to export")
            return

        latest_audit = self.audit_results[-1]

        # Create detailed report
        report = {
            "audit_summary": latest_audit,
            "audit_history": self.audit_results,
            "generated_at": datetime.now().isoformat(),
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.platform(),
                "architecture": platform.architecture()[0],
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Security audit report exported to {output_path}")


# Global security auditor instance
security_auditor = SecurityAuditor()


# Global security validator instance
security_validator = SecurityValidator()


# Convenience functions - using classmethod approach
def validate_identifier(identifier: str, name: str = "identifier") -> str:
    return SecurityValidator.validate_identifier(identifier, name)


def validate_file_path(file_path: str) -> str:
    return SecurityValidator.validate_file_path(file_path)


def validate_model_file(file_path: str) -> str:
    return SecurityValidator.validate_model_file(file_path)


def validate_device_path(device_path: str) -> str:
    return SecurityValidator.validate_device_path(device_path)


def sanitize_filename(filename: str) -> str:
    return SecurityValidator.sanitize_filename(filename)


def generate_session_id() -> str:
    return SecurityValidator.generate_session_id()


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    return SecurityValidator.compute_file_hash(file_path, algorithm)


def safe_file_path(base_path: str, filename: str) -> str:
    """Create a safe file path by joining base path and filename securely."""
    from pathlib import Path

    base = Path(base_path).resolve()
    safe_name = sanitize_filename(filename)
    full_path = base / safe_name

    # Ensure the final path is still within base_path
    if not str(full_path.resolve()).startswith(str(base)):
        raise SecurityError(f"Path traversal attempt detected: {filename}")

    return str(full_path)
