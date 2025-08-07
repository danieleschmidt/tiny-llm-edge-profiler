"""
Unit tests for the validation module.
"""

import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import psutil

from tiny_llm_profiler.validation import (
    SystemValidator,
    PlatformValidator,
    ModelValidator,
    RobustProfilingWrapper,
    ValidationLevel,
    ValidationCategory,
    ValidationIssue,
    ValidationResult
)
from tiny_llm_profiler.models import QuantizedModel


class TestValidationIssue:
    """Test the ValidationIssue dataclass."""
    
    def test_creation(self):
        """Test validation issue creation."""
        issue = ValidationIssue(
            category=ValidationCategory.HARDWARE,
            severity="error",
            message="Test error",
            recommendation="Fix the error"
        )
        
        assert issue.category == ValidationCategory.HARDWARE
        assert issue.severity == "error"
        assert issue.message == "Test error"
        assert issue.recommendation == "Fix the error"


class TestValidationResult:
    """Test the ValidationResult dataclass."""
    
    def test_creation_with_no_issues(self):
        """Test validation result with no issues."""
        result = ValidationResult(is_valid=True, issues=[])
        
        assert result.is_valid is True
        assert len(result.issues) == 0
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.has_errors is False
        assert result.has_warnings is False
    
    def test_creation_with_issues(self):
        """Test validation result with various issues."""
        error_issue = ValidationIssue(
            ValidationCategory.HARDWARE, "error", "Critical error"
        )
        warning_issue = ValidationIssue(
            ValidationCategory.PLATFORM, "warning", "Warning message"
        )
        info_issue = ValidationIssue(
            ValidationCategory.MODEL, "info", "Info message"
        )
        
        issues = [error_issue, warning_issue, info_issue]
        result = ValidationResult(is_valid=False, issues=issues)
        
        assert result.is_valid is False
        assert len(result.issues) == 3
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert result.has_errors is True
        assert result.has_warnings is True
        
        assert result.errors[0] == error_issue
        assert result.warnings[0] == warning_issue


class TestSystemValidator:
    """Test the SystemValidator class."""
    
    def test_init(self):
        """Test system validator initialization."""
        validator = SystemValidator(ValidationLevel.STRICT)
        assert validator.validation_level == ValidationLevel.STRICT
    
    def test_init_default_level(self):
        """Test system validator with default validation level."""
        validator = SystemValidator()
        assert validator.validation_level == ValidationLevel.MODERATE
    
    @patch('psutil.virtual_memory')
    def test_check_system_memory_sufficient(self, mock_memory):
        """Test memory check with sufficient memory."""
        # Mock sufficient memory
        mock_memory.return_value = Mock(
            total=2 * 1024**3,      # 2GB total
            available=1 * 1024**3   # 1GB available
        )
        
        validator = SystemValidator()
        issues = validator._check_system_memory()
        
        # Should have no memory issues
        memory_issues = [i for i in issues if "memory" in i.message.lower()]
        assert len(memory_issues) == 0
    
    @patch('psutil.virtual_memory')
    def test_check_system_memory_low_total(self, mock_memory):
        """Test memory check with low total memory."""
        # Mock low total memory
        mock_memory.return_value = Mock(
            total=512 * 1024**2,    # 512MB total
            available=256 * 1024**2 # 256MB available
        )
        
        validator = SystemValidator()
        issues = validator._check_system_memory()
        
        # Should have warning about low total memory
        memory_warnings = [i for i in issues if i.severity == "warning"]
        assert len(memory_warnings) >= 1
    
    @patch('psutil.virtual_memory')
    def test_check_system_memory_insufficient_available(self, mock_memory):
        """Test memory check with insufficient available memory."""
        # Mock insufficient available memory
        mock_memory.return_value = Mock(
            total=2 * 1024**3,      # 2GB total
            available=256 * 1024**2 # Only 256MB available
        )
        
        validator = SystemValidator()
        issues = validator._check_system_memory()
        
        # Should have error about insufficient available memory
        memory_errors = [i for i in issues if i.severity == "error"]
        assert len(memory_errors) >= 1
    
    @patch('psutil.disk_usage')
    def test_check_disk_space_sufficient(self, mock_disk):
        """Test disk space check with sufficient space."""
        # Mock sufficient disk space
        mock_disk.return_value = Mock(free=5 * 1024**3)  # 5GB free
        
        validator = SystemValidator()
        issues = validator._check_disk_space()
        
        # Should have no disk space issues
        assert len(issues) == 0
    
    @patch('psutil.disk_usage')
    def test_check_disk_space_low(self, mock_disk):
        """Test disk space check with low space."""
        # Mock low disk space
        mock_disk.return_value = Mock(free=500 * 1024**2)  # 500MB free
        
        validator = SystemValidator()
        issues = validator._check_disk_space()
        
        # Should have warning about low disk space
        disk_warnings = [i for i in issues if i.severity == "warning"]
        assert len(disk_warnings) >= 1
    
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.unlink')
    def test_check_permissions_success(self, mock_unlink, mock_write):
        """Test permission check when write permission exists."""
        # Mock successful write and delete
        mock_write.return_value = None
        mock_unlink.return_value = None
        
        validator = SystemValidator()
        issues = validator._check_permissions()
        
        # Should have no permission issues
        assert len(issues) == 0
        mock_write.assert_called_once_with("test")
        mock_unlink.assert_called_once()
    
    @patch('pathlib.Path.write_text')
    def test_check_permissions_failure(self, mock_write):
        """Test permission check when write permission fails."""
        # Mock permission error
        mock_write.side_effect = PermissionError("Access denied")
        
        validator = SystemValidator()
        issues = validator._check_permissions()
        
        # Should have permission error
        permission_errors = [i for i in issues if i.severity == "error"]
        assert len(permission_errors) >= 1
    
    @patch('sys.version_info', (3, 7))  # Mock old Python version
    def test_check_python_environment_old_version(self):
        """Test Python environment check with old version."""
        validator = SystemValidator()
        issues = validator._check_python_environment()
        
        # Should have error about old Python version
        python_errors = [i for i in issues if "python" in i.message.lower()]
        assert len(python_errors) >= 1
        assert python_errors[0].severity == "error"
    
    @patch('sys.version_info', (3, 9))  # Mock current Python version
    @patch('builtins.__import__')
    def test_check_python_environment_missing_packages(self, mock_import):
        """Test Python environment check with missing packages."""
        # Mock missing package
        def import_side_effect(name):
            if name == "numpy":
                raise ImportError("No module named numpy")
            return Mock()
        
        mock_import.side_effect = import_side_effect
        
        validator = SystemValidator()
        issues = validator._check_python_environment()
        
        # Should have error about missing package
        package_errors = [i for i in issues if "numpy" in i.message]
        assert len(package_errors) >= 1
    
    @patch.object(SystemValidator, '_check_system_memory')
    @patch.object(SystemValidator, '_check_disk_space')
    @patch.object(SystemValidator, '_check_permissions')
    @patch.object(SystemValidator, '_check_python_environment')
    def test_validate_system(self, mock_python, mock_perms, mock_disk, mock_memory):
        """Test complete system validation."""
        # Mock all check methods
        mock_memory.return_value = []
        mock_disk.return_value = []
        mock_perms.return_value = []
        mock_python.return_value = []
        
        validator = SystemValidator()
        result = validator.validate_system()
        
        assert result.is_valid is True
        assert len(result.issues) == 0
        
        # Verify all checks were called
        mock_memory.assert_called_once()
        mock_disk.assert_called_once()
        mock_perms.assert_called_once()
        mock_python.assert_called_once()


class TestPlatformValidator:
    """Test the PlatformValidator class."""
    
    def test_init(self):
        """Test platform validator initialization."""
        validator = PlatformValidator("esp32")
        assert validator.platform == "esp32"
    
    @patch.object(PlatformValidator, '_validate_esp32')
    def test_validate_platform_esp32(self, mock_validate_esp32):
        """Test ESP32 platform validation."""
        mock_validate_esp32.return_value = []
        
        validator = PlatformValidator("esp32")
        result = validator.validate_platform("/dev/ttyUSB0")
        
        assert result.is_valid is True
        mock_validate_esp32.assert_called_once_with("/dev/ttyUSB0")
    
    @patch.object(PlatformValidator, '_validate_stm32')
    def test_validate_platform_stm32(self, mock_validate_stm32):
        """Test STM32 platform validation."""
        mock_validate_stm32.return_value = []
        
        validator = PlatformValidator("stm32f4")
        result = validator.validate_platform("/dev/ttyACM0")
        
        assert result.is_valid is True
        mock_validate_stm32.assert_called_once_with("/dev/ttyACM0")
    
    def test_validate_platform_unknown(self):
        """Test unknown platform validation."""
        validator = PlatformValidator("unknown_platform")
        result = validator.validate_platform()
        
        # Should have warning about unknown platform
        unknown_warnings = [i for i in result.issues if "unknown" in i.message.lower()]
        assert len(unknown_warnings) >= 1
    
    @patch('pathlib.Path.exists')
    def test_validate_esp32_device_not_found(self, mock_exists):
        """Test ESP32 validation with missing device."""
        mock_exists.return_value = False
        
        validator = PlatformValidator("esp32")
        issues = validator._validate_esp32("/dev/ttyUSB0")
        
        # Should have error about missing device
        device_errors = [i for i in issues if "not found" in i.message]
        assert len(device_errors) >= 1
    
    @patch('pathlib.Path.exists')
    @patch('serial.Serial')
    def test_validate_esp32_connection_success(self, mock_serial, mock_exists):
        """Test ESP32 validation with successful connection."""
        mock_exists.return_value = True
        mock_serial_instance = Mock()
        mock_serial.return_value.__enter__ = Mock(return_value=mock_serial_instance)
        mock_serial.return_value.__exit__ = Mock(return_value=False)
        
        # Mock successful communication
        mock_serial_instance.write.return_value = None
        mock_serial_instance.read_all.return_value = b"OK"
        
        validator = PlatformValidator("esp32")
        issues = validator._validate_esp32("/dev/ttyUSB0")
        
        # Should have no connection issues
        connection_errors = [i for i in issues if "connect" in i.message.lower()]
        assert len(connection_errors) == 0
    
    @patch('pathlib.Path.exists')
    @patch('serial.Serial')
    def test_validate_esp32_connection_failure(self, mock_serial, mock_exists):
        """Test ESP32 validation with connection failure."""
        mock_exists.return_value = True
        mock_serial.side_effect = Exception("Connection failed")
        
        validator = PlatformValidator("esp32")
        issues = validator._validate_esp32("/dev/ttyUSB0")
        
        # Should have error about connection failure
        connection_errors = [i for i in issues if "cannot connect" in i.message.lower()]
        assert len(connection_errors) >= 1
    
    def test_validate_esp32_no_device_path(self):
        """Test ESP32 validation without device path."""
        validator = PlatformValidator("esp32")
        issues = validator._validate_esp32(None)
        
        # Should have error about missing device path
        path_errors = [i for i in issues if "no device path" in i.message.lower()]
        assert len(path_errors) >= 1


class TestModelValidator:
    """Test the ModelValidator class."""
    
    def test_init(self):
        """Test model validator initialization."""
        validator = ModelValidator(ValidationLevel.STRICT)
        assert validator.validation_level == ValidationLevel.STRICT
    
    def test_validate_model_basic_valid(self):
        """Test basic model validation with valid model."""
        model = QuantizedModel(
            name="test_model",
            size_mb=2.0,
            quantization="4bit"
        )
        
        validator = ModelValidator()
        result = validator.validate_model(model, "esp32")
        
        # Should be valid for reasonable model
        assert result.is_valid is True
    
    def test_validate_model_too_large(self):
        """Test model validation with oversized model."""
        model = QuantizedModel(
            name="huge_model",
            size_mb=60.0,  # Very large
            quantization="fp32"
        )
        
        validator = ModelValidator()
        result = validator.validate_model(model, "esp32")
        
        # Should have error about model being too large
        assert result.is_valid is False
        size_errors = [i for i in result.errors if "too large" in i.message]
        assert len(size_errors) >= 1
    
    def test_validate_model_high_precision(self):
        """Test model validation with high precision quantization."""
        model = QuantizedModel(
            name="fp16_model",
            size_mb=3.0,
            quantization="fp16"
        )
        
        validator = ModelValidator()
        result = validator.validate_model(model, "esp32")
        
        # Should have warning about high precision
        precision_warnings = [i for i in result.warnings if "precision" in i.message.lower()]
        assert len(precision_warnings) >= 1
    
    @patch.object(QuantizedModel, 'get_memory_requirements')
    def test_validate_platform_compatibility_memory_exceeded(self, mock_memory_reqs):
        """Test platform compatibility with excessive memory requirements."""
        mock_memory_reqs.return_value = {
            "total_estimated_kb": 600  # Exceeds ESP32 limits
        }
        
        model = QuantizedModel("test_model", size_mb=2.0, quantization="4bit")
        validator = ModelValidator()
        issues = validator._validate_platform_compatibility(model, "esp32")
        
        # Should have error about memory requirements
        memory_errors = [i for i in issues if "memory" in i.message.lower()]
        assert len(memory_errors) >= 1
    
    @patch.object(QuantizedModel, 'get_memory_requirements')
    def test_validate_platform_compatibility_high_usage(self, mock_memory_reqs):
        """Test platform compatibility with high memory usage."""
        mock_memory_reqs.return_value = {
            "total_estimated_kb": 350  # 350KB out of 400KB ESP32 limit = 87.5%
        }
        
        model = QuantizedModel("test_model", size_mb=2.0, quantization="4bit")
        validator = ModelValidator()
        issues = validator._validate_platform_compatibility(model, "esp32")
        
        # Should have warning about high memory usage
        memory_warnings = [i for i in issues if i.severity == "warning" and "memory" in i.message.lower()]
        assert len(memory_warnings) >= 1
    
    def test_validate_platform_compatibility_flash_exceeded(self):
        """Test platform compatibility with flash size exceeded."""
        model = QuantizedModel("large_model", size_mb=5.0, quantization="8bit")  # 5MB > ESP32 4MB limit
        
        validator = ModelValidator()
        issues = validator._validate_platform_compatibility(model, "esp32")
        
        # Should have error about flash requirements
        flash_errors = [i for i in issues if "flash" in i.message.lower()]
        assert len(flash_errors) >= 1
    
    def test_estimate_performance_small_model(self):
        """Test performance estimation for small, optimized model."""
        model = QuantizedModel("small_model", size_mb=0.8, quantization="2bit")
        
        validator = ModelValidator()
        performance = validator._estimate_performance(model, "esp32")
        
        # Should predict good performance for small, quantized model
        assert performance > 10.0  # Should be reasonably high
    
    def test_estimate_performance_large_model(self):
        """Test performance estimation for large model."""
        model = QuantizedModel("large_model", size_mb=8.0, quantization="fp16")
        
        validator = ModelValidator()
        performance = validator._estimate_performance(model, "rp2040")
        
        # Should predict poor performance for large model on limited platform
        assert performance < 5.0  # Should be low


class TestRobustProfilingWrapper:
    """Test the RobustProfilingWrapper class."""
    
    def test_init(self):
        """Test robust profiling wrapper initialization."""
        wrapper = RobustProfilingWrapper(
            max_retries=5,
            timeout_seconds=600,
            validation_level=ValidationLevel.STRICT
        )
        
        assert wrapper.max_retries == 5
        assert wrapper.timeout_seconds == 600
        assert wrapper.validation_level == ValidationLevel.STRICT
    
    @patch.object(SystemValidator, 'validate_system')
    @patch.object(PlatformValidator, 'validate_platform')
    @patch.object(ModelValidator, 'validate_model')
    def test_validate_before_profiling_success(self, mock_model_val, mock_platform_val, mock_system_val):
        """Test comprehensive validation with all checks passing."""
        # Mock all validations to pass
        mock_system_val.return_value = ValidationResult(is_valid=True, issues=[])
        mock_platform_val.return_value = ValidationResult(is_valid=True, issues=[])
        mock_model_val.return_value = ValidationResult(is_valid=True, issues=[])
        
        model = QuantizedModel("test_model", size_mb=2.0, quantization="4bit")
        
        wrapper = RobustProfilingWrapper()
        result = wrapper.validate_before_profiling(model, "esp32", "/dev/ttyUSB0")
        
        assert result.is_valid is True
        assert len(result.issues) == 0
    
    @patch.object(SystemValidator, 'validate_system')
    @patch.object(PlatformValidator, 'validate_platform')
    @patch.object(ModelValidator, 'validate_model')
    def test_validate_before_profiling_with_errors(self, mock_model_val, mock_platform_val, mock_system_val):
        """Test comprehensive validation with errors."""
        # Mock system validation to fail
        system_error = ValidationIssue(ValidationCategory.HARDWARE, "error", "System error")
        mock_system_val.return_value = ValidationResult(is_valid=False, issues=[system_error])
        mock_platform_val.return_value = ValidationResult(is_valid=True, issues=[])
        mock_model_val.return_value = ValidationResult(is_valid=True, issues=[])
        
        model = QuantizedModel("test_model", size_mb=2.0, quantization="4bit")
        
        wrapper = RobustProfilingWrapper()
        result = wrapper.validate_before_profiling(model, "esp32", "/dev/ttyUSB0")
        
        assert result.is_valid is False
        assert len(result.errors) >= 1
    
    def test_robust_profile_success(self):
        """Test robust profiling with successful execution."""
        def mock_profiler_func():
            return "success_result"
        
        wrapper = RobustProfilingWrapper(max_retries=3)
        result = wrapper.robust_profile(mock_profiler_func)
        
        assert result == "success_result"
    
    def test_robust_profile_retry_then_success(self):
        """Test robust profiling with retry then success."""
        call_count = 0
        
        def mock_profiler_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success_result"
        
        wrapper = RobustProfilingWrapper(max_retries=3)
        result = wrapper.robust_profile(mock_profiler_func)
        
        assert result == "success_result"
        assert call_count == 3  # Should have retried
    
    def test_robust_profile_all_retries_fail(self):
        """Test robust profiling with all retries failing."""
        def mock_profiler_func():
            raise Exception("Persistent failure")
        
        wrapper = RobustProfilingWrapper(max_retries=2)
        result = wrapper.robust_profile(mock_profiler_func)
        
        # Should return the exception
        assert isinstance(result, Exception)
        assert "Persistent failure" in str(result)
    
    def test_execute_with_timeout(self):
        """Test execution with timeout (simplified)."""
        def quick_func():
            return "quick_result"
        
        wrapper = RobustProfilingWrapper()
        result = wrapper._execute_with_timeout(quick_func)
        
        assert result == "quick_result"


if __name__ == "__main__":
    pytest.main([__file__])