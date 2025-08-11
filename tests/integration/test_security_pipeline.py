"""
Security Pipeline Integration Tests

Tests that verify end-to-end security measures in profiling workflows,
including input validation, secure file handling, and threat prevention.
"""

import pytest
import tempfile
import os
import stat
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, List, Any
import json
import hashlib

from tiny_llm_profiler.profiler import EdgeProfiler, ProfilingConfig
from tiny_llm_profiler.models import QuantizedModel, QuantizationType
from tiny_llm_profiler.security import SecurityValidator, InputSanitizer, FileValidator
from tiny_llm_profiler.config import ConfigManager, AppConfig
from tiny_llm_profiler.results import ProfileResults


@pytest.mark.integration
@pytest.mark.security
class TestSecureProfilingPipeline:
    """Test complete secure profiling pipeline with all security measures enabled."""
    
    def test_end_to_end_secure_profiling_workflow(self, tmp_path):
        """Test complete secure profiling workflow with comprehensive validation."""
        
        # Initialize security components
        validator = SecurityValidator()
        sanitizer = InputSanitizer()
        file_validator = FileValidator()
        
        # Step 1: Secure model file creation and validation
        model_file = tmp_path / "secure_test_model.gguf"
        self._create_secure_model_file(model_file, 2.0)
        
        # Validate model file security
        validated_model_path = validator.validate_model_file(model_file)
        assert validated_model_path == model_file.resolve()
        
        # Additional file validation
        file_validation_result = file_validator.validate_model_file(model_file)
        assert file_validation_result["valid"] is True
        assert file_validation_result["format"] == "gguf"
        assert file_validation_result["size_mb"] == pytest.approx(2.0, rel=0.1)
        
        # Step 2: Secure model loading with validation
        model = QuantizedModel.from_file(validated_model_path, quantization=QuantizationType.INT4)
        
        # Validate model properties
        model_validation = validator.validate_model_properties(model)
        assert model_validation["valid"] is True
        
        # Step 3: Secure platform selection and validation
        platform = validator.validate_identifier("esp32", "platform")
        assert platform == "esp32"
        
        device_path = validator.validate_device_path("/dev/ttyUSB0")
        assert device_path == "/dev/ttyUSB0"
        
        # Step 4: Secure test prompts validation
        raw_prompts = [
            "Hello, how are you?",
            "Explain AI safety",
            "<script>alert('test')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection attempt
            "../../../etc/passwd",  # Path traversal attempt
            "Normal prompt about weather"
        ]
        
        secure_prompts = []
        for prompt in raw_prompts:
            try:
                sanitized = sanitizer.sanitize_string(prompt, max_length=1000)
                if sanitized:  # Only add if not rejected
                    secure_prompts.append(sanitized)
            except Exception as e:
                # Malicious input rejected - this is expected
                print(f"Rejected malicious input: {prompt[:50]}...")
        
        # Should have filtered out malicious inputs
        assert len(secure_prompts) < len(raw_prompts)
        assert len(secure_prompts) >= 3  # Should keep legitimate prompts
        
        # Step 5: Secure profiling configuration
        secure_config = ProfilingConfig(
            duration_seconds=5,
            measurement_iterations=2,
            warmup_iterations=1,
            timeout_seconds=30,  # Reasonable timeout
            max_prompt_length=1000  # Limit prompt length
        )
        
        # Step 6: Secure profiling execution
        profiler = EdgeProfiler(
            platform=platform,
            device=device_path,
            connection="local"  # Safe for testing
        )
        
        results = profiler.profile_model(
            model=model,
            test_prompts=secure_prompts,
            metrics=["latency", "memory"],
            config=secure_config
        )
        
        # Step 7: Secure results validation and export
        assert results is not None
        assert results.platform == platform
        
        # Validate results don't contain sensitive information
        results_summary = results.get_summary()
        assert "password" not in str(results_summary).lower()
        assert "token" not in str(results_summary).lower() or "tokens_per_second" in str(results_summary)
        
        # Step 8: Secure file export
        secure_export_filename = validator.sanitize_filename("secure_profiling_results.json")
        secure_export_path = tmp_path / secure_export_filename
        
        results.export_json(secure_export_path)
        
        # Validate exported file
        assert secure_export_path.exists()
        assert secure_export_path.stat().st_size > 0
        
        # Verify file permissions are secure
        file_mode = secure_export_path.stat().st_mode
        assert not (file_mode & stat.S_IWOTH)  # Not world-writable
        assert not (file_mode & stat.S_IWGRP)  # Not group-writable
        
        # Step 9: Validate exported content
        with open(secure_export_path, 'r') as f:
            exported_data = json.load(f)
            
            assert "metadata" in exported_data
            assert "profiles" in exported_data
            assert exported_data["metadata"]["platform"] == platform
            
            # Check for data sanitization in exported content
            exported_str = json.dumps(exported_data)
            assert "<script>" not in exported_str
            assert "DROP TABLE" not in exported_str
            assert "../../../" not in exported_str
        
        print(f"✓ End-to-end secure profiling pipeline test passed")
        print(f"  Original prompts: {len(raw_prompts)}, Secure prompts: {len(secure_prompts)}")
        print(f"  Results exported securely to: {secure_export_path}")
    
    def test_malicious_model_file_detection(self, tmp_path):
        """Test detection and prevention of malicious model files."""
        
        validator = SecurityValidator()
        file_validator = FileValidator()
        
        # Test various malicious file types
        malicious_files = []
        
        # 1. Executable disguised as model
        exe_file = tmp_path / "malicious_model.gguf"
        with open(exe_file, 'wb') as f:
            f.write(b"MZ")  # PE header (Windows executable)
            f.write(b"\x00" * 1000)
        malicious_files.append(("executable", exe_file))
        
        # 2. File with suspicious size
        huge_file = tmp_path / "huge_model.gguf"
        with open(huge_file, 'wb') as f:
            f.write(b"GGUF")
            f.write(b"\x00" * (200 * 1024 * 1024))  # 200MB+ file
        malicious_files.append(("oversized", huge_file))
        
        # 3. File with wrong extension/content mismatch
        wrong_content_file = tmp_path / "wrong_content.gguf"
        with open(wrong_content_file, 'wb') as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG header in .gguf file
            f.write(b"\x00" * 1000)
        malicious_files.append(("wrong_content", wrong_content_file))
        
        # 4. Symlink to system file (path traversal)
        if os.name != 'nt':  # Unix-like systems only
            system_file = "/etc/passwd" if Path("/etc/passwd").exists() else "/dev/null"
            symlink_file = tmp_path / "symlink_model.gguf"
            try:
                symlink_file.symlink_to(system_file)
                malicious_files.append(("symlink", symlink_file))
            except (OSError, NotImplementedError):
                pass  # Skip if symlinks not supported
        
        # Test detection of each malicious file type
        detection_results = {}
        
        for file_type, file_path in malicious_files:
            try:
                # Should reject during validation
                validator.validate_model_file(file_path)
                detection_results[file_type] = "not_detected"  # Bad - should have been caught
                
            except Exception as e:
                detection_results[file_type] = "detected"  # Good - caught the malicious file
                print(f"  Correctly detected {file_type}: {str(e)[:50]}...")
        
        # Should have detected most/all malicious files
        detected_count = sum(1 for result in detection_results.values() if result == "detected")
        total_count = len(detection_results)
        
        if total_count > 0:
            detection_rate = detected_count / total_count
            assert detection_rate >= 0.8, f"Malicious file detection rate too low: {detection_rate:.1%}"
        
        # Test with legitimate file for comparison
        legitimate_file = tmp_path / "legitimate_model.gguf"
        self._create_secure_model_file(legitimate_file, 1.5)
        
        # Should pass validation
        validated_path = validator.validate_model_file(legitimate_file)
        assert validated_path == legitimate_file.resolve()
        
        print(f"✓ Malicious model file detection test passed")
        print(f"  Malicious files tested: {total_count}")
        print(f"  Detection rate: {detection_rate:.1%}" if total_count > 0 else "  No malicious files to test")
    
    def test_input_sanitization_comprehensive(self, tmp_path):
        """Test comprehensive input sanitization across different attack vectors."""
        
        sanitizer = InputSanitizer()
        
        # Test various attack vectors
        attack_vectors = {
            "xss_script": "<script>alert('xss')</script>",
            "xss_img": "<img src=x onerror=alert('xss')>",
            "sql_injection": "'; DROP TABLE users; --",
            "sql_union": "' UNION SELECT * FROM passwords --",
            "path_traversal": "../../../etc/passwd",
            "path_traversal_win": "..\\..\\..\\windows\\system32\\config\\sam",
            "command_injection": "$(rm -rf /)",
            "command_injection_backtick": "`cat /etc/passwd`",
            "ldap_injection": "admin*)(uid=*))",
            "xpath_injection": "' or '1'='1",
            "template_injection": "{{7*7}}[[7*7]]",
            "ssi_injection": "<!--#exec cmd=\"cat /etc/passwd\"-->",
            "null_byte": "legitimate\x00malicious",
            "unicode_bypass": "\\u003cscript\\u003e",
            "url_encoded": "%3Cscript%3Ealert%28%27xss%27%29%3C%2Fscript%3E"
        }
        
        sanitization_results = {}
        
        for attack_type, malicious_input in attack_vectors.items():
            try:
                sanitized = sanitizer.sanitize_string(malicious_input, max_length=1000)
                
                if sanitized is None:
                    sanitization_results[attack_type] = "rejected"
                elif sanitized != malicious_input:
                    sanitization_results[attack_type] = "sanitized"
                else:
                    sanitization_results[attack_type] = "unchanged"
                    
                # Additional checks for dangerous patterns
                if sanitized and any(pattern in sanitized.lower() for pattern in 
                                   ["<script>", "drop table", "../", "$(", "`"]):
                    sanitization_results[attack_type] = "dangerous_pattern_remains"
                    
            except Exception as e:
                sanitization_results[attack_type] = "error"
                print(f"  Error sanitizing {attack_type}: {e}")
        
        # Analyze sanitization effectiveness
        safe_results = ["rejected", "sanitized"]
        safe_count = sum(1 for result in sanitization_results.values() if result in safe_results)
        total_count = len(sanitization_results)
        
        safety_rate = safe_count / total_count
        assert safety_rate >= 0.9, f"Input sanitization safety rate too low: {safety_rate:.1%}"
        
        # Test legitimate inputs are preserved
        legitimate_inputs = [
            "Hello, how are you?",
            "Explain machine learning concepts",
            "What is the weather like today?",
            "Generate a Python function",
            "Translate 'hello' to French"
        ]
        
        legitimate_preserved = 0
        for legitimate_input in legitimate_inputs:
            sanitized = sanitizer.sanitize_string(legitimate_input, max_length=1000)
            if sanitized == legitimate_input:
                legitimate_preserved += 1
        
        preservation_rate = legitimate_preserved / len(legitimate_inputs)
        assert preservation_rate >= 0.9, f"Legitimate input preservation rate too low: {preservation_rate:.1%}"
        
        print(f"✓ Comprehensive input sanitization test passed")
        print(f"  Attack vectors tested: {total_count}")
        print(f"  Safety rate: {safety_rate:.1%}")
        print(f"  Legitimate preservation rate: {preservation_rate:.1%}")
        print(f"  Sanitization results: {sanitization_results}")
    
    def test_secure_configuration_management(self, tmp_path):
        """Test secure configuration loading and validation."""
        
        # Create configuration with both secure and insecure settings
        config_file = tmp_path / "security_test_config.yaml"
        config_content = """
app_name: "security-test-profiler"
version: "0.1.0"

profiling:
  sample_rate_hz: 100
  duration_seconds: 60
  measurement_iterations: 10
  timeout_seconds: 300
  max_prompt_length: 5000
  enable_power_profiling: false

security:
  enable_input_validation: true
  enable_path_sanitization: true
  max_file_size_mb: 50
  allowed_model_extensions: [".gguf", ".ggml", ".bin"]
  enable_secure_temp_dirs: true
  log_security_events: true
  require_model_validation: true

logging:
  level: "INFO"
  console_output: true
  file_output: true
  json_format: true
  security_logging: true

output:
  default_format: "json"
  include_raw_data: false
  auto_timestamp: true
  include_system_info: false  # Disable to avoid info disclosure

# Potentially insecure settings that should be caught
debug_mode: false
expose_internal_paths: false
disable_file_validation: false
"""
        
        config_file.write_text(config_content)
        
        # Load configuration securely
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path=config_file)
        
        # Validate security settings
        assert config.security.enable_input_validation is True
        assert config.security.enable_path_sanitization is True
        assert config.security.require_model_validation is True
        assert config.security.max_file_size_mb <= 100  # Reasonable limit
        
        # Test configuration-driven secure profiling
        model_file = tmp_path / "config_test_model.gguf"
        self._create_secure_model_file(model_file, 1.5)
        
        # Validate file against security configuration
        validator = SecurityValidator(config=config)
        
        # Should pass with proper extension
        validated_path = validator.validate_model_file(model_file)
        assert validated_path is not None
        
        # Should reject file with wrong extension
        wrong_ext_file = tmp_path / "model.txt"
        wrong_ext_file.write_bytes(b"GGUF" + b"\x00" * 1000)
        
        with pytest.raises(Exception):
            validator.validate_model_file(wrong_ext_file)
        
        # Should reject oversized file
        if config.security.max_file_size_mb < 100:
            oversized_file = tmp_path / "oversized.gguf"
            oversized_file.write_bytes(b"GGUF" + b"\x00" * (config.security.max_file_size_mb * 1024 * 1024 + 1000))
            
            with pytest.raises(Exception):
                validator.validate_model_file(oversized_file)
        
        # Test secure profiling with configuration
        model = QuantizedModel.from_file(validated_path)
        profiler = EdgeProfiler(platform="esp32", connection="local")
        
        secure_config = ProfilingConfig(
            duration_seconds=config.profiling.duration_seconds,
            measurement_iterations=config.profiling.measurement_iterations,
            timeout_seconds=config.profiling.timeout_seconds
        )
        
        results = profiler.profile_model(
            model=model,
            test_prompts=["Secure configuration test"],
            metrics=["latency"],
            config=secure_config
        )
        
        # Validate secure results export based on config
        assert results is not None
        
        if not config.output.include_system_info:
            # Should not include sensitive system information
            summary = results.get_summary()
            assert "hostname" not in str(summary).lower()
            assert "username" not in str(summary).lower()
            assert "home" not in str(summary).lower()
        
        print(f"✓ Secure configuration management test passed")
        print(f"  Security validation enabled: {config.security.enable_input_validation}")
        print(f"  Max file size limit: {config.security.max_file_size_mb}MB")
        print(f"  Allowed extensions: {config.security.allowed_model_extensions}")
    
    def test_secure_temporary_file_handling(self, tmp_path):
        """Test secure temporary file creation and cleanup."""
        
        validator = SecurityValidator()
        
        # Test secure temporary directory creation
        secure_temp_dir = validator.create_secure_temp_dir()
        
        assert secure_temp_dir.exists()
        assert secure_temp_dir.is_dir()
        
        # Verify directory permissions are secure
        dir_mode = secure_temp_dir.stat().st_mode
        assert not (dir_mode & stat.S_IROTH)  # Not world-readable
        assert not (dir_mode & stat.S_IWOTH)  # Not world-writable
        assert not (dir_mode & stat.S_IXOTH)  # Not world-executable
        
        # Test secure file creation within temp directory
        secure_temp_file = validator.create_secure_temp_file(
            dir=secure_temp_dir,
            suffix=".json",
            prefix="secure_test_"
        )
        
        assert secure_temp_file.exists()
        
        # Write test data
        test_data = {"secure": True, "test": "data"}
        with open(secure_temp_file, 'w') as f:
            json.dump(test_data, f)
        
        # Verify file permissions
        file_mode = secure_temp_file.stat().st_mode
        assert not (file_mode & stat.S_IROTH)  # Not world-readable
        assert not (file_mode & stat.S_IWOTH)  # Not world-writable
        
        # Test secure cleanup
        validator.cleanup_temp_files()
        
        # Files should be securely deleted
        assert not secure_temp_file.exists()
        
        # Directory should be cleaned up
        if secure_temp_dir.exists():
            # May still exist but should be empty
            assert len(list(secure_temp_dir.iterdir())) == 0
        
        print(f"✓ Secure temporary file handling test passed")
        print(f"  Secure temp directory created and cleaned up")
        print(f"  File permissions properly restricted")
    
    def _create_secure_model_file(self, path: Path, size_mb: float):
        """Create a secure model file for testing."""
        # Create proper GGUF header
        header = b"GGUF"  # Magic number
        header += b"\x03\x00\x00\x00"  # Version 3
        
        # Add minimal metadata
        metadata = json.dumps({
            "general.architecture": "llama",
            "general.quantization_version": 2,
            "general.file_type": 2,
            "llama.vocab_size": 32000
        }).encode()
        
        # Calculate remaining size
        header_size = len(header) + len(metadata) + 100
        remaining_bytes = int(size_mb * 1024 * 1024) - header_size
        
        with open(path, 'wb') as f:
            f.write(header)
            f.write(len(metadata).to_bytes(4, 'little'))
            f.write(metadata)
            f.write(b'\x00' * 96)  # Padding
            f.write(b'\x42' * max(0, remaining_bytes))  # Model data
        
        # Set secure file permissions
        path.chmod(0o600)  # Read/write for owner only


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])