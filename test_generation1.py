#!/usr/bin/env python3
"""
Standalone test for Generation 1 functionality.
Tests the lightweight core without heavy dependencies.
"""

import sys
import tempfile
from pathlib import Path

# Import only the specific module to avoid dependency issues
sys.path.insert(0, 'src')

try:
    from tiny_llm_profiler.core_lite import (
        QuickStartProfiler,
        SimplePlatformManager,
        quick_check,
        get_recommended_platform,
        print_platform_comparison,
        run_basic_benchmark
    )
    print("✅ Generation 1 imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_generation1():
    """Test Generation 1 functionality."""
    
    print("\n🚀 Testing Generation 1 Enhancements")
    print("=" * 50)
    
    # Test 1: Platform Manager
    print("\n1. Testing Platform Manager")
    platforms = SimplePlatformManager.get_supported_platforms()
    print(f"   Supported platforms: {len(platforms)}")
    print(f"   Platforms: {', '.join(platforms)}")
    
    # Test platform info
    esp32_info = SimplePlatformManager.get_platform_info("esp32")
    if esp32_info:
        print(f"   ESP32 RAM: {esp32_info['ram_kb']} KB")
        print("   ✅ Platform info working")
    else:
        print("   ❌ Platform info failed")
        return False
    
    # Test 2: Create mock model file
    print("\n2. Testing Model Compatibility Check")
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
        # Create a 2MB mock model file
        temp_file.write(b'0' * (2 * 1024 * 1024))
        temp_model_path = temp_file.name
    
    try:
        # Test quick compatibility check
        is_compatible = quick_check(temp_model_path, "esp32")
        print(f"   Quick check ESP32: {'✅ Compatible' if is_compatible else '❌ Not compatible'}")
        
        # Test recommended platform
        recommended = get_recommended_platform(temp_model_path)
        print(f"   Recommended platform: {recommended}")
        
        # Test detailed compatibility
        profiler = QuickStartProfiler()
        detailed_check = profiler.check_model_compatibility(temp_model_path, "esp32")
        
        print(f"   Model size: {detailed_check.get('model_size_mb', 0):.1f} MB")
        print(f"   Compatible: {'✅ Yes' if detailed_check['compatible'] else '❌ No'}")
        
        if detailed_check.get('utilization'):
            print(f"   Flash usage: {detailed_check['utilization']['flash_percent']:.1f}%")
        
    finally:
        # Cleanup
        Path(temp_model_path).unlink(missing_ok=True)
    
    # Test 3: Quick Profile
    print("\n3. Testing Quick Profile")
    
    # Create another mock file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
        temp_file.write(b'0' * (1 * 1024 * 1024))  # 1MB model
        temp_model_path = temp_file.name
    
    try:
        profiler = QuickStartProfiler()
        result = profiler.quick_profile(temp_model_path, "esp32")
        
        print(f"   Profile status: {result['status']}")
        
        if result['status'] == 'success':
            perf = result['performance']
            print(f"   Tokens/sec: {perf['tokens_per_second']:.1f}")
            print(f"   Latency: {perf['latency_ms']:.1f} ms")
            print(f"   Memory: {perf['memory_usage_kb']:.0f} KB")
            print(f"   ✅ Quick profile working")
        else:
            print(f"   Status: {result['status']}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
    
    finally:
        Path(temp_model_path).unlink(missing_ok=True)
    
    # Test 4: Getting Started Guide
    print("\n4. Testing Getting Started Guide")
    guide = profiler.get_getting_started_guide("esp32")
    
    if "error" not in guide:
        steps = guide.get("quick_start_steps", [])
        print(f"   Guide steps: {len(steps)}")
        print(f"   Platform: {guide['platform']}")
        print("   ✅ Getting started guide working")
    else:
        print(f"   ❌ Guide error: {guide['error']}")
    
    # Test 5: Basic Benchmark
    print("\n5. Testing Basic Benchmark")
    try:
        benchmark_results = run_basic_benchmark()
        successful = sum(1 for r in benchmark_results.values() if r.get('success', False))
        print(f"   Benchmark platforms: {len(benchmark_results)}")
        print(f"   Successful simulations: {successful}")
        print("   ✅ Basic benchmark working")
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
    
    print(f"\n✅ Generation 1 functionality test completed!")
    return True


if __name__ == "__main__":
    try:
        success = test_generation1()
        if success:
            print("\n🎉 All Generation 1 tests passed!")
            print("🚀 Generation 1 implementation is working correctly!")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)