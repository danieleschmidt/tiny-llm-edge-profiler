"""
Unit tests for the analyzer module.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import numpy as np

from tiny_llm_profiler.analyzer import (
    MetricsAnalyzer, 
    ComparativeAnalyzer,
    BottleneckInfo,
    ComparisonResult
)
from tiny_llm_profiler.results import (
    ProfileResults,
    LatencyProfile,
    MemoryProfile,
    PowerProfile
)
from tiny_llm_profiler.models import QuantizedModel


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return QuantizedModel(
        name="test_model",
        size_mb=2.5,
        quantization="4bit"
    )


@pytest.fixture
def sample_latency_profile():
    """Create a sample latency profile."""
    return LatencyProfile(
        first_token_latency_ms=150.0,
        inter_token_latency_ms=50.0,
        total_latency_ms=500.0,
        tokens_per_second=10.0,
        latency_std_ms=25.0
    )


@pytest.fixture
def sample_memory_profile():
    """Create a sample memory profile."""
    return MemoryProfile(
        baseline_memory_kb=100.0,
        peak_memory_kb=350.0,
        memory_usage_kb=250.0,
        memory_efficiency_tokens_per_kb=0.05
    )


@pytest.fixture
def sample_power_profile():
    """Create a sample power profile."""
    return PowerProfile(
        idle_power_mw=50.0,
        active_power_mw=150.0,
        peak_power_mw=200.0,
        energy_per_token_mj=3.0,
        total_energy_mj=100.0
    )


@pytest.fixture
def sample_profile_results(sample_model, sample_latency_profile, sample_memory_profile, sample_power_profile):
    """Create sample profile results."""
    results = ProfileResults(
        platform="esp32",
        model_name=sample_model.name,
        model_size_mb=sample_model.size_mb,
        quantization=sample_model.quantization
    )
    
    results.add_latency_profile(sample_latency_profile)
    results.add_memory_profile(sample_memory_profile)
    results.add_power_profile(sample_power_profile)
    
    return results


class TestMetricsAnalyzer:
    """Test the MetricsAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = MetricsAnalyzer()
        assert analyzer.results_history == []
    
    def test_analyze_complete_results(self, sample_profile_results):
        """Test analyzing complete profile results."""
        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze(sample_profile_results)
        
        # Check basic fields
        assert analysis["platform"] == "esp32"
        assert analysis["model_name"] == "test_model"
        assert analysis["model_size_mb"] == 2.5
        assert analysis["quantization"] == "4bit"
        
        # Check latency metrics
        assert analysis["first_token_ms"] == 150.0
        assert analysis["inter_token_ms"] == 50.0
        assert analysis["tokens_per_second"] == 10.0
        assert analysis["total_latency_ms"] == 500.0
        assert "latency_variability" in analysis
        
        # Check memory metrics
        assert analysis["peak_memory_kb"] == 350.0
        assert analysis["memory_usage_kb"] == 250.0
        assert analysis["memory_overhead_kb"] == 250.0  # peak - baseline
        assert analysis["tokens_per_kb"] == pytest.approx(0.04, abs=0.001)  # 10 tps / 250KB
        
        # Check power metrics
        assert analysis["active_power_mw"] == 150.0
        assert analysis["energy_per_token_mj"] == 3.0
        assert analysis["power_efficiency"] == pytest.approx(333.33, abs=0.1)  # 1000/3.0
    
    def test_analyze_partial_results(self):
        """Test analyzing results with missing profiles."""
        results = ProfileResults(
            platform="stm32f4",
            model_name="partial_model",
            model_size_mb=1.0,
            quantization="2bit"
        )
        
        # Only add latency profile
        latency = LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=30.0,
            total_latency_ms=300.0,
            tokens_per_second=15.0,
            latency_std_ms=10.0
        )
        results.add_latency_profile(latency)
        
        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze(results)
        
        # Should have latency metrics
        assert analysis["tokens_per_second"] == 15.0
        
        # Should not have memory or power metrics
        assert "peak_memory_kb" not in analysis
        assert "active_power_mw" not in analysis
    
    def test_find_bottlenecks_high_latency(self):
        """Test bottleneck detection for high latency."""
        results = ProfileResults("esp32", "slow_model", 3.0, "8bit")
        
        # High latency profile
        latency = LatencyProfile(
            first_token_latency_ms=250.0,  # High first token
            inter_token_latency_ms=80.0,   # High inter-token
            total_latency_ms=1000.0,
            tokens_per_second=5.0,
            latency_std_ms=200.0  # High variability
        )
        results.add_latency_profile(latency)
        
        analyzer = MetricsAnalyzer()
        bottlenecks = analyzer.find_bottlenecks(results)
        
        # Should identify multiple latency bottlenecks
        bottleneck_types = [b.component for b in bottlenecks]
        assert "First Token Generation" in bottleneck_types
        assert "Token Generation Speed" in bottleneck_types
        assert "Latency Consistency" in bottleneck_types
        
        # Check bottlenecks are sorted by impact
        assert bottlenecks[0].impact >= bottlenecks[-1].impact
    
    def test_find_bottlenecks_high_memory(self):
        """Test bottleneck detection for high memory usage."""
        results = ProfileResults("rp2040", "memory_hungry", 5.0, "fp16")
        
        # High memory usage
        memory = MemoryProfile(
            baseline_memory_kb=100.0,
            peak_memory_kb=450.0,  # High memory usage
            memory_usage_kb=400.0,
            memory_efficiency_tokens_per_kb=0.02  # Low efficiency
        )
        results.add_memory_profile(memory)
        
        analyzer = MetricsAnalyzer()
        bottlenecks = analyzer.find_bottlenecks(results)
        
        bottleneck_types = [b.component for b in bottlenecks]
        assert "Memory Usage" in bottleneck_types
        assert "Memory Efficiency" in bottleneck_types
    
    def test_find_bottlenecks_high_power(self):
        """Test bottleneck detection for high power consumption."""
        results = ProfileResults("esp32", "power_hungry", 2.0, "fp32")
        
        # High power consumption
        power = PowerProfile(
            idle_power_mw=30.0,
            active_power_mw=250.0,  # High power
            peak_power_mw=300.0,
            energy_per_token_mj=8.0,  # High energy per token
            total_energy_mj=200.0
        )
        results.add_power_profile(power)
        
        analyzer = MetricsAnalyzer()
        bottlenecks = analyzer.find_bottlenecks(results)
        
        bottleneck_types = [b.component for b in bottlenecks]
        assert "Power Consumption" in bottleneck_types
        assert "Energy Efficiency" in bottleneck_types
    
    def test_find_bottlenecks_optimal_performance(self, sample_profile_results):
        """Test bottleneck detection for optimal performance."""
        # Modify results to be optimal
        sample_profile_results.latency_profile.first_token_latency_ms = 50.0
        sample_profile_results.latency_profile.inter_token_latency_ms = 20.0
        sample_profile_results.latency_profile.latency_std_ms = 5.0
        
        sample_profile_results.memory_profile.peak_memory_kb = 200.0
        sample_profile_results.memory_profile.memory_efficiency_tokens_per_kb = 0.2
        
        sample_profile_results.power_profile.active_power_mw = 80.0
        sample_profile_results.power_profile.energy_per_token_mj = 1.5
        
        analyzer = MetricsAnalyzer()
        bottlenecks = analyzer.find_bottlenecks(sample_profile_results)
        
        # Should find few or no bottlenecks
        assert len(bottlenecks) <= 1
    
    @patch('tiny_llm_profiler.analyzer.make_subplots')
    @patch('builtins.open', create=True)
    def test_generate_report(self, mock_open, mock_subplots, sample_profile_results):
        """Test report generation."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        
        analyzer = MetricsAnalyzer()
        analyzer.generate_report(sample_profile_results, "test_report.html")
        
        # Check that figure methods were called
        mock_subplots.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called_once()
        mock_fig.write_html.assert_called_once_with("test_report.html")


class TestComparativeAnalyzer:
    """Test the ComparativeAnalyzer class."""
    
    def test_init(self):
        """Test comparative analyzer initialization."""
        analyzer = ComparativeAnalyzer()
        assert analyzer.comparisons == []
    
    def test_compare_single_result(self, sample_profile_results):
        """Test comparison with single result."""
        analyzer = ComparativeAnalyzer()
        
        comparison_data = [("model1", "esp32", sample_profile_results)]
        results = analyzer.compare(comparison_data)
        
        assert len(results) == 1
        assert results[0].model_name == "model1"
        assert results[0].platform == "esp32"
        assert results[0].relative_performance == 1.0  # Only one result, so it's the best
    
    def test_compare_multiple_results(self):
        """Test comparison with multiple results."""
        analyzer = ComparativeAnalyzer()
        
        # Create results with different performance characteristics
        good_results = ProfileResults("esp32", "good_model", 2.0, "4bit")
        good_latency = LatencyProfile(50.0, 20.0, 200.0, 20.0, 10.0)
        good_memory = MemoryProfile(100.0, 200.0, 150.0, 0.1)
        good_power = PowerProfile(40.0, 100.0, 150.0, 2.0, 50.0)
        good_results.add_latency_profile(good_latency)
        good_results.add_memory_profile(good_memory)
        good_results.add_power_profile(good_power)
        
        poor_results = ProfileResults("stm32f4", "poor_model", 5.0, "fp16")
        poor_latency = LatencyProfile(200.0, 80.0, 800.0, 5.0, 50.0)
        poor_memory = MemoryProfile(200.0, 600.0, 500.0, 0.02)
        poor_power = PowerProfile(60.0, 300.0, 400.0, 10.0, 200.0)
        poor_results.add_latency_profile(poor_latency)
        poor_results.add_memory_profile(poor_memory)
        poor_results.add_power_profile(poor_power)
        
        comparison_data = [
            ("good_model", "esp32", good_results),
            ("poor_model", "stm32f4", poor_results)
        ]
        
        results = analyzer.compare(comparison_data)
        
        assert len(results) == 2
        # Results should be sorted by performance (best first)
        assert results[0].relative_performance >= results[1].relative_performance
        
        # The good model should be the best performer
        best_result = results[0]
        assert best_result.model_name == "good_model"
        assert best_result.relative_performance == 1.0
    
    def test_calculate_performance_score(self, sample_profile_results):
        """Test performance score calculation."""
        analyzer = ComparativeAnalyzer()
        score = analyzer._calculate_performance_score(sample_profile_results)
        
        # Score should be positive for valid results
        assert score > 0
        
        # Test with no profiles
        empty_results = ProfileResults("test", "empty", 1.0, "4bit")
        empty_score = analyzer._calculate_performance_score(empty_results)
        assert empty_score == 0.0
    
    @patch('tiny_llm_profiler.analyzer.make_subplots')
    def test_plot_comparison(self, mock_subplots):
        """Test comparison plotting."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        
        analyzer = ComparativeAnalyzer()
        
        # Create some comparison results
        results = [
            ComparisonResult("model1", "esp32", Mock(), 1.0),
            ComparisonResult("model2", "stm32f4", Mock(), 0.8)
        ]
        
        # Mock the profile attributes
        for result in results:
            result.profile.latency_profile = Mock()
            result.profile.latency_profile.tokens_per_second = 10.0
            result.profile.memory_profile = Mock()
            result.profile.memory_profile.memory_usage_kb = 250.0
            result.profile.power_profile = Mock()
            result.profile.power_profile.energy_per_token_mj = 3.0
        
        analyzer.plot_comparison(results, ["latency", "memory", "energy"], "comparison.html")
        
        mock_subplots.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called_once()
        mock_fig.write_html.assert_called_once_with("comparison.html")
    
    def test_recommend_deployment_with_constraints(self):
        """Test deployment recommendations with constraints."""
        analyzer = ComparativeAnalyzer()
        
        # Create results that meet and don't meet constraints
        good_results = ProfileResults("esp32", "efficient_model", 1.5, "2bit")
        good_latency = LatencyProfile(80.0, 25.0, 300.0, 15.0, 15.0)
        good_memory = MemoryProfile(100.0, 250.0, 200.0, 0.08)
        good_power = PowerProfile(30.0, 120.0, 180.0, 2.5, 75.0)
        good_results.add_latency_profile(good_latency)
        good_results.add_memory_profile(good_memory)
        good_results.add_power_profile(good_power)
        
        poor_results = ProfileResults("stm32f4", "inefficient_model", 4.0, "8bit")
        poor_latency = LatencyProfile(300.0, 100.0, 1200.0, 3.0, 80.0)  # High latency
        poor_memory = MemoryProfile(200.0, 800.0, 600.0, 0.01)  # High memory
        poor_power = PowerProfile(80.0, 400.0, 500.0, 15.0, 300.0)  # High power
        poor_results.add_latency_profile(poor_latency)
        poor_results.add_memory_profile(poor_memory)
        poor_results.add_power_profile(poor_power)
        
        comparison_results = [
            ComparisonResult("efficient_model", "esp32", good_results, 1.0),
            ComparisonResult("inefficient_model", "stm32f4", poor_results, 0.3)
        ]
        analyzer.comparisons = comparison_results
        
        # Test with constraints that only the good model meets
        constraints = {
            "max_latency_ms": 500,
            "max_memory_kb": 400,
            "max_power_mw": 200
        }
        
        recommendations = analyzer.recommend_deployment(constraints)
        
        assert len(recommendations) == 1
        assert recommendations[0].model_name == "efficient_model"
        
        # Test with very strict constraints that no model meets
        strict_constraints = {
            "max_latency_ms": 100,
            "max_memory_kb": 150,
            "max_power_mw": 50
        }
        
        strict_recommendations = analyzer.recommend_deployment(strict_constraints)
        assert len(strict_recommendations) == 0
    
    def test_recommend_deployment_empty_comparisons(self):
        """Test recommendations with no comparisons."""
        analyzer = ComparativeAnalyzer()
        
        constraints = {"max_latency_ms": 500}
        recommendations = analyzer.recommend_deployment(constraints)
        
        assert len(recommendations) == 0


class TestBottleneckInfo:
    """Test the BottleneckInfo dataclass."""
    
    def test_creation(self):
        """Test bottleneck info creation."""
        bottleneck = BottleneckInfo(
            component="Test Component",
            impact=0.7,
            description="Test description",
            recommendation="Test recommendation"
        )
        
        assert bottleneck.component == "Test Component"
        assert bottleneck.impact == 0.7
        assert bottleneck.description == "Test description"
        assert bottleneck.recommendation == "Test recommendation"


class TestComparisonResult:
    """Test the ComparisonResult dataclass."""
    
    def test_creation(self, sample_profile_results):
        """Test comparison result creation."""
        result = ComparisonResult(
            model_name="test_model",
            platform="esp32",
            profile=sample_profile_results,
            relative_performance=0.85
        )
        
        assert result.model_name == "test_model"
        assert result.platform == "esp32"
        assert result.profile == sample_profile_results
        assert result.relative_performance == 0.85


if __name__ == "__main__":
    pytest.main([__file__])