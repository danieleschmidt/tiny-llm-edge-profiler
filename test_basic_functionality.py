#!/usr/bin/env python3
"""Basic functionality test for tiny-llm-edge-profiler core modules."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Test that core modules can be imported."""
    try:
        from tiny_llm_profiler.models import QuantizedModel, QuantizationType, ModelFormat
        from tiny_llm_profiler.platforms import PlatformManager, Architecture
        from tiny_llm_profiler.results import ProfileResults, LatencyProfile, MemoryProfile, PowerProfile
        print("✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test QuantizedModel creation and basic operations."""
    try:
        from tiny_llm_profiler.models import QuantizedModel, QuantizationType
        
        # Create a basic model
        model = QuantizedModel(
            name="test_model",
            quantization=QuantizationType.INT4,
            vocab_size=32000,
            context_length=2048
        )
        
        # Test properties
        assert model.name == "test_model"
        assert model.quantization == QuantizationType.INT4
        assert model.vocab_size == 32000
        assert model.context_length == 2048
        
        # Test size calculation (should be 0 without actual model file)
        assert model.size_mb >= 0
        
        print("✓ Model creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def test_platform_manager():
    """Test PlatformManager functionality."""
    try:
        from tiny_llm_profiler.platforms import PlatformManager
        
        # Test supported platforms
        platforms = PlatformManager.list_supported_platforms()
        assert len(platforms) > 0
        assert "esp32" in platforms
        assert "stm32f4" in platforms
        
        # Test platform configuration
        pm = PlatformManager("esp32")
        config = pm.get_config()
        
        assert config.name == "esp32"
        assert config.display_name == "ESP32"
        assert config.memory.ram_kb > 0
        
        # Test memory constraints
        constraints = pm.get_memory_constraints()
        assert "total_ram_kb" in constraints
        assert "available_ram_kb" in constraints
        
        print("✓ Platform manager test passed")
        return True
        
    except Exception as e:
        print(f"✗ Platform manager test failed: {e}")
        return False

def test_results_creation():
    """Test ProfileResults creation and operations."""
    try:
        from tiny_llm_profiler.results import (
            ProfileResults, LatencyProfile, MemoryProfile, PowerProfile
        )
        from datetime import datetime, timezone
        
        # Create profile results
        results = ProfileResults(
            platform="esp32",
            model_name="test_model",
            model_size_mb=2.5,
            quantization="4bit"
        )
        
        # Create and add profiles
        latency_profile = LatencyProfile(
            first_token_latency_ms=50.0,
            inter_token_latency_ms=25.0,
            total_latency_ms=500.0,
            tokens_per_second=10.0,
            latency_std_ms=5.0
        )
        
        memory_profile = MemoryProfile(
            baseline_memory_kb=100.0,
            peak_memory_kb=300.0,
            memory_usage_kb=250.0,
            memory_efficiency_tokens_per_kb=0.5
        )
        
        power_profile = PowerProfile(
            idle_power_mw=50.0,
            active_power_mw=150.0,
            peak_power_mw=200.0,
            energy_per_token_mj=2.5,
            total_energy_mj=100.0
        )
        
        results.add_latency_profile(latency_profile)
        results.add_memory_profile(memory_profile)
        results.add_power_profile(power_profile)
        
        # Test summary generation
        summary = results.get_summary()
        assert "platform" in summary
        assert "latency" in summary
        assert "memory" in summary
        assert "power" in summary
        
        # Test efficiency score calculation
        efficiency_score = results.calculate_efficiency_score()
        assert 0 <= efficiency_score <= 100
        
        print("✓ Results creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Results creation test failed: {e}")
        return False

def main():
    """Run all basic functionality tests."""
    print("Running basic functionality tests for tiny-llm-edge-profiler")
    print("=" * 60)
    
    tests = [
        test_core_imports,
        test_model_creation,
        test_platform_manager,
        test_results_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic functionality tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())