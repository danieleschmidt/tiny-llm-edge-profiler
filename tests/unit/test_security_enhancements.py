"""
Unit tests for security enhancements in Generation 2.

Tests input validation, security auditing, and other security
measures implemented for production robustness.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tiny_llm_profiler.security import (
    SecurityValidator, SecurityAuditor, validate_environment,
    validate_identifier, validate_file_path, validate_model_file,
    validate_device_path, sanitize_filename, generate_session_id,
    compute_file_hash, SecureTemporaryDirectory, secure_delete_file,
    InputSanitizer, security_validator, security_auditor
)
from tiny_llm_profiler.exceptions import (
    UnsafePathError, InputValidationError, SecurityError
)


class TestSecurityValidator:
    """Test the SecurityValidator class functionality."""
    
    def test_validate_identifier_valid(self):
        """Test validation of valid identifiers."""
        valid_identifiers = [
            "esp32",
            "stm32f4",
            "valid_name",
            "test-platform",
            "model.v1",
            "a1b2c3",
            "x" * 64  # Maximum length
        ]
        
        for identifier in valid_identifiers:
            result = validate_identifier(identifier, "test_identifier")
            assert result == identifier
    
    def test_validate_identifier_invalid(self):
        """Test validation of invalid identifiers."""
        invalid_identifiers = [
            "",  # Empty
            "  ",  # Whitespace only
            "invalid space",  # Contains space
            "invalid@symbol",  # Invalid character
            "x" * 65,  # Too long
            ".starts_with_dot",  # Starts with dot
            "ends_with_dot.",  # Ends with dot
            "consecutive..dots",  # Consecutive dots
        ]
        
        for identifier in invalid_identifiers:
            with pytest.raises(InputValidationError):
                validate_identifier(identifier, "test_identifier")
    
    def test_validate_file_path_safe(self):
        """Test validation of safe file paths."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Valid file path
            result = validate_file_path(temp_path, must_exist=True)
            assert result == temp_path.resolve()
        finally:
            temp_path.unlink()
    
    def test_validate_file_path_traversal_attack(self):
        """Test protection against path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32\\config",
        ]
        
        for path in malicious_paths:
            with pytest.raises(UnsafePathError):
                validate_file_path(path)
    
    def test_validate_file_path_with_base_directory(self):
        """Test file path validation with base directory restriction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Create a file within the base directory
            safe_file = base_dir / "safe_file.txt"
            safe_file.write_text("test")
            
            # Should succeed
            result = validate_file_path(safe_file, base_directory=base_dir, must_exist=True)
            assert result == safe_file.resolve()
            
            # Create a file outside the base directory
            with tempfile.NamedTemporaryFile(delete=False) as outside_file:
                outside_path = Path(outside_file.name)
            
            try:
                # Should fail
                with pytest.raises(UnsafePathError):
                    validate_file_path(outside_path, base_directory=base_dir)
            finally:
                outside_path.unlink()
    
    def test_validate_file_path_extensions(self):
        """Test file path validation with allowed extensions."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Should succeed with allowed extension
            result = validate_file_path(
                temp_path,
                allowed_extensions={'.txt'},
                must_exist=True
            )
            assert result == temp_path.resolve()
            
            # Should fail with disallowed extension
            with pytest.raises(InputValidationError):
                validate_file_path(
                    temp_path,
                    allowed_extensions={'.pdf'},
                    must_exist=True
                )
        finally:
            temp_path.unlink()
    
    def test_validate_file_path_size_limit(self):
        """Test file path validation with size limits."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write some data to the file
            test_data = b"x" * 1000  # 1KB
            temp_file.write(test_data)
            temp_path = Path(temp_file.name)
        
        try:
            # Should succeed with sufficient limit
            result = validate_file_path(temp_path, max_size=2000, must_exist=True)
            assert result == temp_path.resolve()
            
            # Should fail with insufficient limit
            with pytest.raises(InputValidationError):
                validate_file_path(temp_path, max_size=500, must_exist=True)
        finally:
            temp_path.unlink()
    
    def test_validate_model_file(self):
        """Test model file validation."""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as model_file:
            model_file.write(b"fake model data")
            model_path = Path(model_file.name)
        
        try:
            # Should succeed for valid model file
            result = validate_model_file(model_path)
            assert result == model_path.resolve()
        finally:
            model_path.unlink()
        
        # Should fail for non-existent file
        with pytest.raises(InputValidationError):
            validate_model_file("/nonexistent/model.gguf")
    
    def test_validate_device_path(self):
        """Test device path validation."""
        valid_paths = [
            "/dev/ttyUSB0",
            "/dev/ttyACM0", 
            "COM3",
            "/dev/cu.usbserial-123",
            "/dev/serial/by-id/usb-device-name"
        ]
        
        for path in valid_paths:
            result = validate_device_path(path)
            assert result == path
        
        invalid_paths = [
            "",
            "invalid-device",
            "/invalid/path",
            "COM999999",  # Too high COM port
        ]
        
        for path in invalid_paths:
            with pytest.raises(InputValidationError):
                validate_device_path(path)
    
    def test_validate_network_config(self):
        """Test network configuration validation."""
        valid_config = {
            "host": "192.168.1.100",
            "port": 8080
        }
        
        result = SecurityValidator.validate_network_config(valid_config)
        assert result["host"] == "192.168.1.100"
        assert result["port"] == 8080
        
        # Test invalid configurations
        with pytest.raises(InputValidationError):
            SecurityValidator.validate_network_config({"host": ""})
        
        with pytest.raises(InputValidationError):
            SecurityValidator.validate_network_config({"port": -1})
        
        with pytest.raises(InputValidationError):
            SecurityValidator.validate_network_config({"port": 70000})
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file with spaces.txt", "file with spaces.txt"),
            ("file<>:\"/\\|?*.txt", "file__________.txt"),
            ("", "unnamed_file"),
            ("   ", "unnamed_file"),
            ("." * 300, "." * 255),  # Length limitation
            (".hidden_file", "hidden_file"),  # Remove leading dots
            ("file.txt.", "file.txt"),  # Remove trailing dots
        ]
        
        for input_name, expected in test_cases:
            result = sanitize_filename(input_name)
            assert result == expected
    
    def test_generate_session_id(self):
        """Test secure session ID generation."""
        id1 = generate_session_id()
        id2 = generate_session_id()
        
        # Should be different
        assert id1 != id2
        
        # Should be reasonable length
        assert len(id1) > 20
        assert len(id2) > 20
        
        # Should be URL-safe
        import string
        allowed_chars = string.ascii_letters + string.digits + '-_'
        assert all(c in allowed_chars for c in id1)
        assert all(c in allowed_chars for c in id2)
    
    def test_compute_file_hash(self):
        """Test file hash computation."""
        test_data = b"Hello, World!"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_data)
            temp_path = Path(temp_file.name)
        
        try:
            # Test SHA256 hash
            hash_sha256 = compute_file_hash(temp_path, "sha256")
            assert len(hash_sha256) == 64  # SHA256 hex digest length
            
            # Test MD5 hash
            hash_md5 = compute_file_hash(temp_path, "md5")
            assert len(hash_md5) == 32  # MD5 hex digest length
            
            # Same file should produce same hash
            hash_sha256_2 = compute_file_hash(temp_path, "sha256")
            assert hash_sha256 == hash_sha256_2
        finally:
            temp_path.unlink()


class TestSecureTemporaryDirectory:
    """Test secure temporary directory creation."""
    
    def test_secure_temp_directory(self):
        """Test secure temporary directory context manager."""
        with SecureTemporaryDirectory("test_prefix_") as temp_dir:
            assert temp_dir.exists()
            assert temp_dir.is_dir()
            assert "test_prefix_" in temp_dir.name
            
            # Create a test file
            test_file = temp_dir / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
        
        # Directory should be cleaned up after context exit
        assert not temp_dir.exists()
    
    def test_secure_temp_directory_permissions(self):
        """Test that temporary directory has secure permissions."""
        with SecureTemporaryDirectory() as temp_dir:
            # Check permissions (Unix-like systems)
            try:
                import stat
                dir_stat = temp_dir.stat()
                # Should be readable, writable, executable by owner only
                expected_perms = stat.S_IRWXU  # 0o700
                actual_perms = dir_stat.st_mode & 0o777
                assert actual_perms == expected_perms
            except (ImportError, OSError):
                # Skip on Windows or if stat unavailable
                pass


class TestSecureDeleteFile:
    """Test secure file deletion."""
    
    def test_secure_delete_file(self):
        """Test secure file deletion with overwriting."""
        test_data = b"sensitive data that should be overwritten"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_data)
            temp_path = Path(temp_file.name)
        
        assert temp_path.exists()
        
        # Securely delete the file
        result = secure_delete_file(temp_path, passes=1)
        
        assert result is True
        assert not temp_path.exists()
    
    def test_secure_delete_nonexistent_file(self):
        """Test secure deletion of non-existent file."""
        non_existent = Path("/tmp/nonexistent_file_12345.tmp")
        result = secure_delete_file(non_existent)
        assert result is True  # Should return True for non-existent files


class TestInputSanitizer:
    """Test input sanitization functionality."""
    
    def test_sanitize_string(self):
        """Test string sanitization."""
        test_cases = [
            ("normal string", "normal string"),
            ("  padded string  ", "padded string"),  # With strip_whitespace=True
            ("x" * 500, "x" * 1000),  # Should not truncate within limit
        ]
        
        for input_str, expected in test_cases:
            if len(expected) <= 1000:  # Within default max_length
                result = InputSanitizer.sanitize_string(input_str)
                assert result == expected
    
    def test_sanitize_string_too_long(self):
        """Test string sanitization with length limits."""
        long_string = "x" * 2000
        
        with pytest.raises(InputValidationError):
            InputSanitizer.sanitize_string(long_string, max_length=1000)
    
    def test_sanitize_string_invalid_characters(self):
        """Test string sanitization with character restrictions."""
        test_string = "hello world 123"
        allowed_pattern = r'^[a-z ]+$'  # Only lowercase letters and spaces
        
        # Should pass
        result = InputSanitizer.sanitize_string(test_string, allowed_chars=allowed_pattern)
        assert result == "hello world "  # Numbers removed, but this depends on implementation
        
        # Test with invalid pattern
        with pytest.raises(InputValidationError):
            InputSanitizer.sanitize_string("HELLO", allowed_chars=r'^[a-z]+$')
    
    def test_sanitize_prompts(self):
        """Test prompt list sanitization."""
        valid_prompts = [
            "What is the capital of France?",
            "Explain quantum computing.",
            "Write a short story about robots."
        ]
        
        result = InputSanitizer.sanitize_prompts(valid_prompts)
        assert len(result) == 3
        assert all(isinstance(prompt, str) for prompt in result)
    
    def test_sanitize_prompts_invalid(self):
        """Test prompt sanitization with invalid inputs."""
        # Too many prompts
        too_many_prompts = ["prompt"] * 200
        with pytest.raises(InputValidationError):
            InputSanitizer.sanitize_prompts(too_many_prompts)
        
        # Non-string prompts
        invalid_prompts = ["valid prompt", 123, "another valid prompt"]
        with pytest.raises(InputValidationError):
            InputSanitizer.sanitize_prompts(invalid_prompts)
        
        # Empty prompts list
        empty_prompts = []
        with pytest.raises(InputValidationError):
            InputSanitizer.sanitize_prompts(empty_prompts)
        
        # Prompts that are too long
        long_prompt = "x" * 20000
        long_prompts = [long_prompt]
        with pytest.raises(InputValidationError):
            InputSanitizer.sanitize_prompts(long_prompts)


class TestEnvironmentValidation:
    """Test environment security validation."""
    
    @patch('os.geteuid')
    def test_validate_environment_root_user(self, mock_geteuid):
        """Test environment validation when running as root."""
        mock_geteuid.return_value = 0  # Root user
        
        results = validate_environment()
        
        assert not results["secure"] or len(results["warnings"]) > 0
        assert any("root" in warning.lower() for warning in results["warnings"])
    
    @patch('os.environ', {'DEBUG': '1', 'DEVELOPMENT': 'true'})
    def test_validate_environment_debug_vars(self):
        """Test environment validation with debug variables."""
        results = validate_environment()
        
        assert len(results["warnings"]) > 0
        assert any("debug" in warning.lower() for warning in results["warnings"])
    
    @patch('sys.version_info', (3, 7, 0))  # Old Python version
    def test_validate_environment_old_python(self):
        """Test environment validation with old Python version."""
        results = validate_environment()
        
        assert not results["secure"]
        assert len(results["errors"]) > 0
        assert any("python" in error.lower() for error in results["errors"])
    
    def test_validate_environment_secure(self):
        """Test environment validation in secure conditions."""
        # This test may vary depending on actual environment
        results = validate_environment()
        
        # Should have basic structure
        assert "secure" in results
        assert "warnings" in results
        assert "errors" in results
        assert "recommendations" in results
        assert isinstance(results["secure"], bool)
        assert isinstance(results["warnings"], list)
        assert isinstance(results["errors"], list)


class TestSecurityAuditor:
    """Test comprehensive security auditing."""
    
    @pytest.fixture
    def auditor(self):
        """Fixture providing a security auditor."""
        return SecurityAuditor()
    
    def test_security_auditor_initialization(self, auditor):
        """Test security auditor initialization."""
        assert auditor.audit_results == []
    
    def test_run_security_audit(self, auditor):
        """Test running a complete security audit."""
        audit_results = auditor.run_security_audit()
        
        # Check structure of audit results
        assert "timestamp" in audit_results
        assert "audit_id" in audit_results
        assert "overall_security_score" in audit_results
        assert "categories" in audit_results
        assert "recommendations" in audit_results
        assert "critical_issues" in audit_results
        assert "warnings" in audit_results
        
        # Check that categories were audited
        categories = audit_results["categories"]
        assert "environment" in categories
        assert "filesystem" in categories
        assert "network" in categories
        assert "application" in categories
        
        # Security score should be reasonable
        score = audit_results["overall_security_score"]
        assert 0 <= score <= 100
    
    def test_filesystem_audit(self, auditor):
        """Test filesystem security audit."""
        results = auditor._audit_filesystem()
        
        assert "secure" in results
        assert "warnings" in results
        assert "errors" in results
        assert "recommendations" in results
    
    def test_network_security_audit(self, auditor):
        """Test network security audit."""
        results = auditor._audit_network_security()
        
        assert "secure" in results
        assert "warnings" in results
        assert "errors" in results
        assert "recommendations" in results
    
    def test_application_security_audit(self, auditor):
        """Test application security audit."""
        results = auditor._audit_application_security()
        
        assert "secure" in results
        assert "warnings" in results
        assert "errors" in results
        assert "recommendations" in results
    
    def test_export_audit_report(self, auditor):
        """Test exporting security audit report."""
        # Run an audit first
        auditor.run_security_audit()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            auditor.export_audit_report(output_path)
            
            # Verify exported report
            with open(output_path, 'r') as f:
                report = json.load(f)
            
            assert "audit_summary" in report
            assert "audit_history" in report
            assert "generated_at" in report
            assert "system_info" in report
        finally:
            output_path.unlink()
    
    def test_multiple_audits(self, auditor):
        """Test running multiple audits and tracking history."""
        # Run multiple audits
        audit1 = auditor.run_security_audit()
        audit2 = auditor.run_security_audit()
        
        assert len(auditor.audit_results) == 2
        assert audit1["audit_id"] != audit2["audit_id"]


class TestGlobalSecurityInstances:
    """Test global security instances."""
    
    def test_global_security_validator(self):
        """Test global security validator instance."""
        assert security_validator is not None
        assert isinstance(security_validator, SecurityValidator)
    
    def test_global_security_auditor(self):
        """Test global security auditor instance."""
        assert security_auditor is not None
        assert isinstance(security_auditor, SecurityAuditor)


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def test_end_to_end_security_validation(self):
        """Test complete security validation workflow."""
        # Test identifier validation
        platform = validate_identifier("esp32", "platform")
        assert platform == "esp32"
        
        # Test file path validation
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as model_file:
            model_file.write(b"fake model data")
            model_path = Path(model_file.name)
        
        try:
            validated_path = validate_model_file(model_path)
            assert validated_path.exists()
        finally:
            model_path.unlink()
        
        # Test environment validation
        env_results = validate_environment()
        assert "secure" in env_results
        
        # Test security audit
        audit_results = security_auditor.run_security_audit()
        assert "overall_security_score" in audit_results
    
    def test_security_with_error_conditions(self):
        """Test security validation under error conditions."""
        # Test with malicious inputs
        malicious_inputs = [
            "../../../etc/passwd",
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../windows/system32/config",
        ]
        
        for malicious_input in malicious_inputs:
            # Should raise appropriate security exceptions
            with pytest.raises((UnsafePathError, InputValidationError)):
                validate_file_path(malicious_input)
    
    def test_secure_temp_operations(self):
        """Test secure temporary file operations."""
        with SecureTemporaryDirectory("security_test_") as temp_dir:
            # Create sensitive data
            sensitive_file = temp_dir / "sensitive.dat"
            sensitive_data = "SECRET_API_KEY=very_secret_value"
            sensitive_file.write_text(sensitive_data)
            
            # Verify file exists and has content
            assert sensitive_file.exists()
            assert sensitive_file.read_text() == sensitive_data
            
            # Securely delete the file
            result = secure_delete_file(sensitive_file)
            assert result is True
            assert not sensitive_file.exists()
        
        # Temp directory should also be cleaned up
        assert not temp_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__])