#!/usr/bin/env python3
"""
Generation 1 Enhancement: Quick Start Example
Demonstrates simple, immediate-use functionality for edge LLM profiling.
"""

import sys
from pathlib import Path

# Add the src directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_llm_profiler.core_lite import (
    QuickStartProfiler,
    quick_check,
    get_recommended_platform,
    print_platform_comparison
)


def main():
    """Main demonstration function."""
    
    print("🚀 Tiny LLM Edge Profiler - Generation 1 Quick Start Demo")
    print("=" * 60)
    
    # Demo 1: Platform Overview
    print("\n📋 1. Platform Overview")
    print("-" * 30)
    print_platform_comparison()
    
    # Demo 2: Create a mock model file for testing
    print("\n📁 2. Creating Mock Model File")
    print("-" * 30)
    
    mock_model_path = Path("mock_model.bin")
    
    # Create different sized mock models for testing
    test_models = {
        "tiny_model.bin": 1.5,  # 1.5MB - should work on all platforms
        "medium_model.bin": 3.2,  # 3.2MB - some platforms
        "large_model.bin": 8.5,  # 8.5MB - limited platforms
    }
    
    for model_name, size_mb in test_models.items():
        model_path = Path(model_name)
        # Create mock file with specified size
        with open(model_path, 'wb') as f:
            # Write specified number of MB
            f.write(b'0' * int(size_mb * 1024 * 1024))
        print(f"✅ Created {model_name} ({size_mb} MB)")
    
    # Demo 3: Quick Compatibility Checks
    print("\n🔍 3. Quick Compatibility Checks")
    print("-" * 30)
    
    for model_name in test_models.keys():
        print(f"\nTesting {model_name}:")
        
        # Check ESP32 compatibility
        esp32_compatible = quick_check(model_name, "esp32")
        print(f"  ESP32: {'✅ Compatible' if esp32_compatible else '❌ Too large'}")
        
        # Check STM32F4 compatibility  
        stm32_compatible = quick_check(model_name, "stm32f4")
        print(f"  STM32F4: {'✅ Compatible' if stm32_compatible else '❌ Too large'}")
        
        # Get recommended platform
        recommended = get_recommended_platform(model_name)
        print(f"  🎯 Recommended: {recommended}")
    
    # Demo 4: Detailed Analysis
    print("\n📊 4. Detailed Compatibility Analysis")
    print("-" * 30)
    
    profiler = QuickStartProfiler()
    
    # Analyze the tiny model in detail
    model_to_analyze = "tiny_model.bin"
    print(f"\nAnalyzing {model_to_analyze}:")
    
    for platform in ["esp32", "stm32f7", "rp2040"]:
        print(f"\n{platform.upper()}:")
        result = profiler.check_model_compatibility(model_to_analyze, platform)
        
        print(f"  Compatible: {'✅ Yes' if result['compatible'] else '❌ No'}")
        print(f"  Model size: {result.get('model_size_mb', 0):.1f} MB")
        print(f"  Flash usage: {result.get('utilization', {}).get('flash_percent', 0):.1f}%")
        
        if result.get('recommendations'):
            print("  💡 Tips:")
            for rec in result['recommendations'][:2]:  # Show first 2 recommendations
                print(f"    • {rec}")
    
    # Demo 5: Quick Profiling Simulation
    print("\n⚡ 5. Quick Profiling Simulation")
    print("-" * 30)
    
    platform = "esp32"
    model_path = "tiny_model.bin"
    
    print(f"Running quick profile for {model_path} on {platform}...")
    
    result = profiler.quick_profile(model_path, platform)
    
    if result["status"] == "success":
        perf = result["performance"]
        print(f"✅ Profiling completed successfully!")
        print(f"   🚀 Performance: {perf['tokens_per_second']:.1f} tokens/sec")
        print(f"   ⏱️  Latency: {perf['latency_ms']:.1f} ms")
        print(f"   💾 Memory: {perf['memory_usage_kb']:.0f} KB")
        print(f"   🔋 Power: {perf['power_consumption_mw']:.0f} mW")
        
        print("\n🎯 Recommendations:")
        for rec in result.get("recommendations", [])[:3]:
            print(f"   {rec}")
    
    else:
        print(f"❌ Profiling failed: {result.get('error', 'Unknown error')}")
    
    # Demo 6: Getting Started Guide
    print("\n📖 6. Getting Started Guide")
    print("-" * 30)
    
    guide = profiler.get_getting_started_guide("esp32")
    
    if "error" not in guide:
        print("ESP32 Getting Started Steps:")
        for step in guide["quick_start_steps"][:2]:  # Show first 2 steps
            print(f"\nStep {step['step']}: {step['title']}")
            print(f"  {step['description']}")
            if 'details' in step:
                for detail in step['details'][:2]:  # Show first 2 details
                    print(f"    • {detail}")
    
    # Demo 7: Advanced Usage Examples
    print("\n🔬 7. Advanced Usage Examples")
    print("-" * 30)
    
    print("For advanced profiling, you can:")
    print("✨ Use the full EdgeProfiler for hardware-in-the-loop testing")
    print("🔧 Customize profiling parameters and timeouts")
    print("📈 Run comprehensive benchmarks across multiple devices")
    print("🌍 Deploy globally with multi-region optimization")
    print("🧠 Use quantum-inspired optimization algorithms (Generation 4)")
    
    print("\n💡 Quick Commands:")
    print("   python -m tiny_llm_profiler.core_lite  # Run benchmark")
    print("   tiny-profiler-quickstart interactive   # Interactive wizard")
    print("   tiny-profiler profile --help          # Full CLI help")
    
    # Cleanup
    print("\n🧹 Cleaning up demo files...")
    for model_name in test_models.keys():
        Path(model_name).unlink(missing_ok=True)
        print(f"🗑️  Removed {model_name}")
    
    print("\n✅ Demo completed successfully!")
    print("🚀 You're ready to start profiling your own models!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your setup and try again.")
        sys.exit(1)