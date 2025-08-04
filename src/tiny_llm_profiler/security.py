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
    ALLOWED_MODEL_EXTENSIONS = {'.gguf', '.ggml', '.bin', '.safetensors', '.pt', '.pth'}
    
    # Allowed characters in identifiers (platform names, model names, etc.)
    IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]{1,64}$')
    
    # Maximum file sizes (in bytes)
    MAX_MODEL_SIZE = 100 * 1024 * 1024  # 100MB
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
                "must contain only alphanumeric characters, hyphens, underscores, and dots (1-64 chars)"
            )
        
        # Additional security checks
        if identifier.startswith('.') or identifier.endswith('.'):
            raise InputValidationError(name, identifier, "cannot start or end with a dot")
        
        if '..' in identifier:
            raise InputValidationError(name, identifier, "cannot contain consecutive dots")
        
        logger.debug(f"Validated identifier: {name}={identifier}")
        return identifier
    
    @classmethod
    def validate_file_path(
        cls, 
        file_path: Union[str, Path], 
        allowed_extensions: Optional[set] = None,
        base_directory: Optional[Path] = None,
        max_size: Optional[int] = None,
        must_exist: bool = False
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
        if '..' in str(file_path):
            raise UnsafePathError(str(file_path), "contains parent directory references")
        
        # Validate against base directory if specified
        if base_directory:
            base_abs = base_directory.resolve()
            try:
                abs_path.relative_to(base_abs)
            except ValueError:
                raise UnsafePathError(str(file_path), f"path outside allowed directory {base_directory}")
        
        # Check file extension if restrictions specified
        if allowed_extensions and abs_path.suffix.lower() not in allowed_extensions:
            raise InputValidationError(
                "file_path", 
                file_path, 
                f"file extension must be one of: {', '.join(allowed_extensions)}"
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
                    f"file too large: {file_size} bytes (max: {max_size} bytes)"
                )
        
        # Check for suspicious patterns in path
        suspicious_patterns = [
            '/etc/', '/proc/', '/sys/', '/dev/', '/var/log/',
            'C:\\Windows\\', 'C:\\System32\\', '/System/', '/Library/'
        ]
        
        for pattern in suspicious_patterns:
            if pattern in str(abs_path):
                raise UnsafePathError(str(file_path), f"contains suspicious pattern: {pattern}")
        
        logger.debug(f"Validated file path: {abs_path}")
        return abs_path
    
    @classmethod
    def validate_model_file(cls, model_path: Union[str, Path]) -> Path:
        """Validate a model file path with appropriate security checks."""
        return cls.validate_file_path(
            model_path,
            allowed_extensions=cls.ALLOWED_MODEL_EXTENSIONS,
            max_size=cls.MAX_MODEL_SIZE,
            must_exist=True
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
            r'^/dev/tty[A-Z]{2,4}\d+$',  # Linux/Mac serial ports
            r'^COM\d+$',                  # Windows serial ports
            r'^/dev/cu\.[a-zA-Z0-9_\-\.]+$',  # Mac USB serial
            r'^/dev/serial/by-id/[a-zA-Z0-9_\-\.]+$',  # Linux by-id
        ]
        
        if not any(re.match(pattern, device_path) for pattern in valid_patterns):
            raise InputValidationError(
                "device_path", 
                device_path, 
                "invalid device path format"
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
        if 'host' in config:
            host = config['host']
            if not isinstance(host, str) or not host:
                raise InputValidationError("host", host, "must be a non-empty string")
            
            # Simple validation - could be enhanced with proper IP/hostname validation
            if not re.match(r'^[a-zA-Z0-9\.\-]+$', host):
                raise InputValidationError("host", host, "contains invalid characters")
            
            validated['host'] = host
        
        # Validate port
        if 'port' in config:
            port = config['port']
            if not isinstance(port, int) or not (1 <= port <= 65535):
                raise InputValidationError("port", port, "must be an integer between 1 and 65535")
            
            # Warn about privileged ports
            if port < 1024:
                logger.warning(f"Using privileged port {port}")
            
            validated['port'] = port
        
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
        sanitized = re.sub(unsafe_chars, '_', filename)
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        
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
            with open(file_path, 'rb') as f:
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
                logger.warning(f"Failed to clean up temporary directory {self.temp_dir}: {e}")


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
        with open(file_path, 'r+b') as f:
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
        strip_whitespace: bool = True
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
            raise InputValidationError("input", input_str[:50] + "...", f"exceeds maximum length of {max_length}")
        
        # Check allowed characters
        if allowed_chars and not re.match(allowed_chars, input_str):
            raise InputValidationError("input", input_str, f"contains characters not matching pattern: {allowed_chars}")
        
        return input_str
    
    @staticmethod
    def sanitize_prompts(prompts: List[str], max_prompt_length: int = 10000) -> List[str]:
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
            raise InputValidationError("prompts", prompts, "too many prompts (max: 100)")
        
        sanitized = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise InputValidationError(f"prompts[{i}]", prompt, "must be a string")
            
            # Basic sanitization
            prompt = prompt.strip()
            
            if len(prompt) > max_prompt_length:
                raise InputValidationError(f"prompts[{i}]", prompt[:50] + "...", f"exceeds maximum length of {max_prompt_length}")
            
            if not prompt:  # Skip empty prompts
                continue
            
            sanitized.append(prompt)
        
        if not sanitized:
            raise InputValidationError("prompts", prompts, "no valid prompts provided")
        
        return sanitized


def validate_environment() -> Dict[str, Any]:
    """
    Validate the runtime environment for security issues.
    
    Returns:
        Dictionary with environment validation results
    """
    results = {
        "secure": True,
        "warnings": [],
        "errors": []
    }
    
    # Check if running as root (not recommended)
    if os.geteuid() == 0:
        results["warnings"].append("Running as root user - not recommended for security")
    
    # Check temporary directory permissions
    temp_dir = Path(tempfile.gettempdir())
    if temp_dir.exists():
        temp_stat = temp_dir.stat()
        if temp_stat.st_mode & stat.S_IWOTH:
            results["warnings"].append("Temporary directory is world-writable")
    
    # Check for development/debug environment variables
    debug_vars = ['DEBUG', 'DEVELOPMENT', 'TINY_LLM_DEBUG']
    active_debug_vars = [var for var in debug_vars if os.environ.get(var)]
    if active_debug_vars:
        results["warnings"].append(f"Debug environment variables active: {', '.join(active_debug_vars)}")
    
    # Check Python version for known security issues
    import sys
    python_version = sys.version_info
    if python_version < (3, 8):
        results["errors"].append("Python version is too old and may have security vulnerabilities")
        results["secure"] = False
    
    logger.info(f"Environment validation completed: {'secure' if results['secure'] else 'has issues'}")
    
    return results


# Global security validator instance
security_validator = SecurityValidator()

# Convenience functions
validate_identifier = security_validator.validate_identifier
validate_file_path = security_validator.validate_file_path
validate_model_file = security_validator.validate_model_file
validate_device_path = security_validator.validate_device_path
sanitize_filename = security_validator.sanitize_filename
generate_session_id = security_validator.generate_session_id
compute_file_hash = security_validator.compute_file_hash