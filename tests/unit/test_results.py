"""
Unit tests for the results module.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timezone
import json
import tempfile
from pathlib import Path

from tiny_llm_profiler.results import (
    LatencyProfile,
    MemoryProfile,
    PowerProfile,
    AccuracyProfile,
    ThroughputProfile,
    ProfileResults
)


class TestLatencyProfile:
    """Test the LatencyProfile dataclass."""
    
    def test_creation_basic(self):
        """Test latency profile creation with basic parameters."""
        profile = LatencyProfile(
            first_token_latency_ms=120.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1200.0,
            tokens_per_second=12.5,
            latency_std_ms=15.0
        )
        
        assert profile.first_token_latency_ms == 120.0
        assert profile.inter_token_latency_ms == 80.0
        assert profile.total_latency_ms == 1200.0
        assert profile.tokens_per_second == 12.5
        assert profile.latency_std_ms == 15.0
        assert profile.percentile_50_ms == 0.0  # Default
        assert profile.percentile_90_ms == 0.0  # Default
        assert profile.percentile_95_ms == 0.0  # Default
        assert profile.percentile_99_ms == 0.0  # Default
    
    def test_creation_with_percentiles(self):
        """Test latency profile creation with percentile data."""
        profile = LatencyProfile(
            first_token_latency_ms=120.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1200.0,
            tokens_per_second=12.5,
            latency_std_ms=15.0,
            percentile_50_ms=75.0,
            percentile_90_ms=95.0,
            percentile_95_ms=105.0,
            percentile_99_ms=140.0
        )
        
        assert profile.percentile_50_ms == 75.0
        assert profile.percentile_90_ms == 95.0
        assert profile.percentile_95_ms == 105.0
        assert profile.percentile_99_ms == 140.0
    
    def test_post_init_calculates_tokens_per_second(self):
        """Test that post_init calculates tokens_per_second from inter_token_latency."""
        profile = LatencyProfile(
            first_token_latency_ms=120.0,
            inter_token_latency_ms=100.0,  # 100ms per token
            total_latency_ms=1200.0,
            tokens_per_second=0,  # Should be calculated
            latency_std_ms=15.0
        )
        
        # Should calculate: 1000ms / 100ms = 10 tokens per second
        assert profile.tokens_per_second == 10.0
    
    def test_post_init_preserves_existing_tokens_per_second(self):
        """Test that post_init doesn't override existing tokens_per_second."""
        profile = LatencyProfile(
            first_token_latency_ms=120.0,
            inter_token_latency_ms=100.0,
            total_latency_ms=1200.0,
            tokens_per_second=15.0,  # Already set
            latency_std_ms=15.0
        )
        
        # Should preserve existing value
        assert profile.tokens_per_second == 15.0


class TestMemoryProfile:
    """Test the MemoryProfile dataclass."""
    
    def test_creation_basic(self):
        """Test memory profile creation with basic parameters."""
        profile = MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=450.0,
            memory_usage_kb=350.0,
            memory_efficiency_tokens_per_kb=0.28
        )
        
        assert profile.baseline_memory_kb == 200.0
        assert profile.peak_memory_kb == 450.0
        assert profile.memory_usage_kb == 350.0
        assert profile.memory_efficiency_tokens_per_kb == 0.28
        assert profile.fragmentation_percent == 0.0  # Default
        assert profile.gc_overhead_percent == 0.0     # Default
        assert profile.stack_usage_kb == 0.0          # Default
        assert profile.heap_usage_kb == 0.0           # Default
    
    def test_creation_with_all_fields(self):
        """Test memory profile creation with all fields."""
        profile = MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=450.0,
            memory_usage_kb=350.0,
            memory_efficiency_tokens_per_kb=0.28,
            fragmentation_percent=15.0,
            gc_overhead_percent=5.0,
            stack_usage_kb=32.0,
            heap_usage_kb=300.0
        )
        
        assert profile.fragmentation_percent == 15.0
        assert profile.gc_overhead_percent == 5.0
        assert profile.stack_usage_kb == 32.0
        assert profile.heap_usage_kb == 300.0


class TestPowerProfile:
    """Test the PowerProfile dataclass."""
    
    def test_creation_basic(self):
        """Test power profile creation with basic parameters."""
        profile = PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=85.0,
            peak_power_mw=120.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6
        )
        
        assert profile.idle_power_mw == 15.0
        assert profile.active_power_mw == 85.0
        assert profile.peak_power_mw == 120.0
        assert profile.energy_per_token_mj == 6.8
        assert profile.total_energy_mj == 81.6
        # Current will be calculated from active_power_mw / voltage_v = 85.0 / 3.3 ≈ 25.76
        assert abs(profile.average_current_ma - 25.76) < 0.1
        assert profile.voltage_v == 3.3           # Default
        assert profile.thermal_info is None       # Default
    
    def test_creation_with_all_fields(self):
        """Test power profile creation with all fields."""
        thermal_info = {"temperature_c": 45.0, "thermal_throttling": False}
        
        profile = PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=85.0,
            peak_power_mw=120.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6,
            average_current_ma=25.7,
            voltage_v=3.3,
            thermal_info=thermal_info
        )
        
        assert profile.average_current_ma == 25.7
        assert profile.voltage_v == 3.3
        assert profile.thermal_info == thermal_info
    
    def test_post_init_calculates_current(self):
        """Test that post_init calculates current from power and voltage."""
        profile = PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=99.0,  # 99mW at 3.3V should be 30mA
            peak_power_mw=120.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6,
            average_current_ma=0,  # Should be calculated
            voltage_v=3.3
        )
        
        # Should calculate: 99mW / 3.3V = 30mA
        assert abs(profile.average_current_ma - 30.0) < 0.1
    
    def test_post_init_preserves_existing_current(self):
        """Test that post_init doesn't override existing current."""
        profile = PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=99.0,
            peak_power_mw=120.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6,
            average_current_ma=35.0,  # Already set
            voltage_v=3.3
        )
        
        # Should preserve existing value
        assert profile.average_current_ma == 35.0


class TestAccuracyProfile:
    """Test the AccuracyProfile dataclass."""
    
    def test_creation_basic(self):
        """Test accuracy profile creation with basic parameters."""
        profile = AccuracyProfile(perplexity=2.5)
        
        assert profile.perplexity == 2.5
        assert profile.bleu_score == 0.0      # Default
        assert profile.rouge_score == {}      # Default
        assert profile.semantic_similarity == 0.0  # Default
        assert profile.task_specific_metrics == {}  # Default
    
    def test_creation_with_all_fields(self):
        """Test accuracy profile creation with all fields."""
        rouge_scores = {"rouge-1": 0.65, "rouge-2": 0.45, "rouge-l": 0.58}
        task_metrics = {"accuracy": 0.85, "f1_score": 0.82}
        
        profile = AccuracyProfile(
            perplexity=2.1,
            bleu_score=0.72,
            rouge_score=rouge_scores,
            semantic_similarity=0.88,
            task_specific_metrics=task_metrics
        )
        
        assert profile.perplexity == 2.1
        assert profile.bleu_score == 0.72
        assert profile.rouge_score == rouge_scores
        assert profile.semantic_similarity == 0.88
        assert profile.task_specific_metrics == task_metrics


class TestThroughputProfile:
    """Test the ThroughputProfile dataclass."""
    
    def test_creation_basic(self):
        """Test throughput profile creation with basic parameters."""
        profile = ThroughputProfile(
            tokens_per_second=12.5,
            requests_per_second=2.1
        )
        
        assert profile.tokens_per_second == 12.5
        assert profile.requests_per_second == 2.1
        assert profile.batch_processing_efficiency == 1.0  # Default
        assert profile.queue_time_ms == 0.0                # Default
        assert profile.processing_time_ms == 0.0           # Default
    
    def test_creation_with_all_fields(self):
        """Test throughput profile creation with all fields."""
        profile = ThroughputProfile(
            tokens_per_second=12.5,
            requests_per_second=2.1,
            batch_processing_efficiency=0.85,
            queue_time_ms=45.0,
            processing_time_ms=480.0
        )
        
        assert profile.batch_processing_efficiency == 0.85
        assert profile.queue_time_ms == 45.0
        assert profile.processing_time_ms == 480.0


class TestProfileResults:
    """Test the ProfileResults class."""
    
    def test_initialization_basic(self):
        """Test profile results initialization with basic parameters."""
        results = ProfileResults(
            platform="esp32",
            model_name="test_model",
            model_size_mb=2.5,
            quantization="4bit"
        )
        
        assert results.platform == "esp32"
        assert results.model_name == "test_model"
        assert results.model_size_mb == 2.5
        assert results.quantization == "4bit"
        assert isinstance(results.timestamp, datetime)
        
        # All profiles should be None initially
        assert results.latency_profile is None
        assert results.memory_profile is None
        assert results.power_profile is None
        assert results.accuracy_profile is None
        assert results.throughput_profile is None
        
        # Metadata should be empty
        assert results.test_configuration == {}
        assert results.environment_info == {}
        assert results.raw_measurements == {}
    
    def test_initialization_with_timestamp(self):
        """Test profile results initialization with custom timestamp."""
        custom_timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        
        results = ProfileResults(
            platform="stm32f4",
            model_name="custom_model",
            model_size_mb=1.8,
            quantization="2bit",
            timestamp=custom_timestamp
        )
        
        assert results.timestamp == custom_timestamp
    
    def test_add_latency_profile(self):
        """Test adding latency profile."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        latency_profile = LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1000.0,
            tokens_per_second=12.5,
            latency_std_ms=10.0
        )
        
        results.add_latency_profile(latency_profile)
        
        assert results.latency_profile == latency_profile
        # Should also create/update throughput profile
        assert results.throughput_profile is not None
        assert results.throughput_profile.tokens_per_second == 12.5
        assert results.throughput_profile.requests_per_second == 0.0
    
    def test_add_latency_profile_updates_existing_throughput(self):
        """Test that adding latency profile updates existing throughput profile."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        # Add initial throughput profile
        throughput_profile = ThroughputProfile(
            tokens_per_second=10.0,
            requests_per_second=2.5
        )
        results.throughput_profile = throughput_profile
        
        # Add latency profile
        latency_profile = LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1000.0,
            tokens_per_second=15.0,
            latency_std_ms=10.0
        )
        
        results.add_latency_profile(latency_profile)
        
        # Should update tokens_per_second but preserve requests_per_second
        assert results.throughput_profile.tokens_per_second == 15.0
        assert results.throughput_profile.requests_per_second == 2.5
    
    def test_add_memory_profile(self):
        """Test adding memory profile."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        memory_profile = MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=450.0,
            memory_usage_kb=350.0,
            memory_efficiency_tokens_per_kb=0.28
        )
        
        results.add_memory_profile(memory_profile)
        
        assert results.memory_profile == memory_profile
    
    def test_add_power_profile(self):
        """Test adding power profile."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        power_profile = PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=85.0,
            peak_power_mw=120.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6
        )
        
        results.add_power_profile(power_profile)
        
        assert results.power_profile == power_profile
    
    def test_add_accuracy_profile(self):
        """Test adding accuracy profile."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        accuracy_profile = AccuracyProfile(
            perplexity=2.5,
            bleu_score=0.72
        )
        
        results.add_accuracy_profile(accuracy_profile)
        
        assert results.accuracy_profile == accuracy_profile
    
    def test_add_raw_measurements(self):
        """Test adding raw measurements."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        measurements = [120.0, 118.5, 122.3, 119.8, 121.1]
        results.add_raw_measurements("latency_ms", measurements)
        
        assert "latency_ms" in results.raw_measurements
        assert results.raw_measurements["latency_ms"] == measurements
    
    def test_get_summary_empty_profiles(self):
        """Test getting summary with no profiles added."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        summary = results.get_summary()
        
        assert summary["platform"] == "esp32"
        assert summary["model_name"] == "model"
        assert summary["model_size_mb"] == 2.0
        assert summary["quantization"] == "4bit"
        assert "timestamp" in summary
        
        # No profile data should be present
        assert "latency" not in summary
        assert "memory" not in summary
        assert "power" not in summary
        assert "accuracy" not in summary
    
    def test_get_summary_with_all_profiles(self):
        """Test getting summary with all profiles added."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        # Add all profile types
        results.add_latency_profile(LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1000.0,
            tokens_per_second=12.5,
            latency_std_ms=10.0
        ))
        
        results.add_memory_profile(MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=450.0,
            memory_usage_kb=350.0,
            memory_efficiency_tokens_per_kb=0.28
        ))
        
        results.add_power_profile(PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=85.0,
            peak_power_mw=120.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6
        ))
        
        results.add_accuracy_profile(AccuracyProfile(
            perplexity=2.5,
            bleu_score=0.72
        ))
        
        summary = results.get_summary()
        
        # Check that all sections are present
        assert "latency" in summary
        assert "memory" in summary
        assert "power" in summary
        assert "accuracy" in summary
        
        # Check latency data
        latency = summary["latency"]
        assert latency["first_token_ms"] == 100.0
        assert latency["inter_token_ms"] == 80.0
        assert latency["tokens_per_second"] == 12.5
        assert latency["total_latency_ms"] == 1000.0
        
        # Check memory data
        memory = summary["memory"]
        assert memory["peak_memory_kb"] == 450.0
        assert memory["memory_usage_kb"] == 350.0
        assert memory["efficiency_tokens_per_kb"] == 0.28
        
        # Check power data
        power = summary["power"]
        assert power["active_power_mw"] == 85.0
        assert power["energy_per_token_mj"] == 6.8
        assert power["total_energy_mj"] == 81.6
        
        # Check accuracy data
        accuracy = summary["accuracy"]
        assert accuracy["perplexity"] == 2.5
        assert accuracy["bleu_score"] == 0.72
    
    def test_calculate_efficiency_score_no_profiles(self):
        """Test efficiency score calculation with no profiles."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        score = results.calculate_efficiency_score()
        assert score == 0.0
    
    @patch('numpy.mean')
    def test_calculate_efficiency_score_all_profiles(self, mock_mean):
        """Test efficiency score calculation with all profiles."""
        mock_mean.return_value = 75.0  # Mock average score
        
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        # Add profiles that will contribute to efficiency score
        results.add_latency_profile(LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1000.0,
            tokens_per_second=15.0,  # Good performance
            latency_std_ms=10.0
        ))
        
        results.add_memory_profile(MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=1024.0,  # 1MB - reasonable for edge
            memory_usage_kb=800.0,
            memory_efficiency_tokens_per_kb=0.28
        ))
        
        results.add_power_profile(PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=150.0,  # 150mW - reasonable consumption
            peak_power_mw=200.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6
        ))
        
        score = results.calculate_efficiency_score()
        
        # Should call np.mean with three scores (latency, memory, power)
        mock_mean.assert_called_once()
        call_args = mock_mean.call_args[0][0]
        assert len(call_args) == 3
        assert score == 75.0
    
    def test_compare_with_other_results(self):
        """Test comparing profile results with another set of results."""
        # Create first results
        results1 = ProfileResults("esp32", "model1", 2.0, "4bit")
        results1.add_latency_profile(LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1000.0,
            tokens_per_second=12.5,
            latency_std_ms=10.0
        ))
        results1.add_memory_profile(MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=400.0,
            memory_usage_kb=300.0,
            memory_efficiency_tokens_per_kb=0.3
        ))
        results1.add_power_profile(PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=80.0,
            peak_power_mw=120.0,
            energy_per_token_mj=6.4,
            total_energy_mj=76.8
        ))
        
        # Create second results (worse performance)
        results2 = ProfileResults("esp32", "model2", 2.0, "4bit")
        results2.add_latency_profile(LatencyProfile(
            first_token_latency_ms=150.0,
            inter_token_latency_ms=100.0,
            total_latency_ms=1500.0,
            tokens_per_second=10.0,
            latency_std_ms=15.0
        ))
        results2.add_memory_profile(MemoryProfile(
            baseline_memory_kb=250.0,
            peak_memory_kb=600.0,
            memory_usage_kb=500.0,
            memory_efficiency_tokens_per_kb=0.2
        ))
        results2.add_power_profile(PowerProfile(
            idle_power_mw=20.0,
            active_power_mw=120.0,
            peak_power_mw=160.0,
            energy_per_token_mj=12.0,
            total_energy_mj=120.0
        ))
        
        comparison = results1.compare_with(results2)
        
        # results1 should be better in all metrics
        assert comparison["tokens_per_second_ratio"] == 12.5 / 10.0  # 1.25
        assert comparison["latency_improvement"] == 1500.0 / 1000.0  # 1.5
        assert comparison["memory_efficiency_ratio"] == 600.0 / 400.0  # 1.5
        assert comparison["power_efficiency_ratio"] == 120.0 / 80.0    # 1.5
        assert comparison["energy_efficiency_ratio"] == 12.0 / 6.4     # ~1.875
    
    def test_get_recommendations_good_performance(self):
        """Test getting recommendations for good performance."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        
        # Add good performance profiles
        results.add_latency_profile(LatencyProfile(
            first_token_latency_ms=80.0,   # Good first token latency
            inter_token_latency_ms=40.0,   # Good inter-token latency
            total_latency_ms=800.0,
            tokens_per_second=25.0,        # Good throughput
            latency_std_ms=5.0
        ))
        
        results.add_memory_profile(MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=512.0,         # 0.5MB - reasonable
            memory_usage_kb=400.0,
            memory_efficiency_tokens_per_kb=0.5,
            fragmentation_percent=10.0    # Low fragmentation
        ))
        
        results.add_power_profile(PowerProfile(
            idle_power_mw=10.0,
            active_power_mw=100.0,        # 100mW - reasonable
            peak_power_mw=130.0,
            energy_per_token_mj=4.0,      # Good energy efficiency
            total_energy_mj=100.0
        ))
        
        recommendations = results.get_recommendations()
        
        # Should have minimal recommendations for good performance
        assert len(recommendations) == 0 or len(recommendations) == 1
    
    def test_get_recommendations_poor_performance(self):
        """Test getting recommendations for poor performance."""
        results = ProfileResults("esp32", "model", 4.0, "8bit")  # Large model, high precision
        
        # Add poor performance profiles
        results.add_latency_profile(LatencyProfile(
            first_token_latency_ms=300.0,  # Poor first token latency
            inter_token_latency_ms=250.0,  # Poor inter-token latency
            total_latency_ms=3000.0,
            tokens_per_second=4.0,         # Poor throughput
            latency_std_ms=50.0
        ))
        
        results.add_memory_profile(MemoryProfile(
            baseline_memory_kb=500.0,
            peak_memory_kb=2048.0,         # 2MB - high for edge device
            memory_usage_kb=1800.0,
            memory_efficiency_tokens_per_kb=0.1,
            fragmentation_percent=25.0     # High fragmentation
        ))
        
        results.add_power_profile(PowerProfile(
            idle_power_mw=20.0,
            active_power_mw=200.0,         # 200mW - high consumption
            peak_power_mw=300.0,
            energy_per_token_mj=50.0,      # Poor energy efficiency
            total_energy_mj=200.0
        ))
        
        recommendations = results.get_recommendations()
        
        # Should have multiple recommendations for poor performance
        assert len(recommendations) >= 4
        
        # Check for specific recommendations
        rec_text = " ".join(recommendations).lower()
        assert "quantization" in rec_text
        assert "latency" in rec_text
        assert "memory" in rec_text
        assert "power" in rec_text or "energy" in rec_text
    
    def test_get_recommendations_platform_specific(self):
        """Test platform-specific recommendations."""
        # Test ESP32 recommendations
        results_esp32 = ProfileResults("esp32", "model", 3.0, "4bit")
        results_esp32.add_memory_profile(MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=500.0,  # Above ESP32 typical RAM
            memory_usage_kb=450.0,
            memory_efficiency_tokens_per_kb=0.2
        ))
        
        recommendations_esp32 = results_esp32.get_recommendations()
        rec_text_esp32 = " ".join(recommendations_esp32).lower()
        assert "psram" in rec_text_esp32 or "esp32" in rec_text_esp32
        
        # Test STM32 recommendations
        results_stm32 = ProfileResults("stm32f4", "model", 3.0, "4bit")
        results_stm32.add_memory_profile(MemoryProfile(
            baseline_memory_kb=100.0,
            peak_memory_kb=200.0,  # Above STM32 typical SRAM
            memory_usage_kb=180.0,
            memory_efficiency_tokens_per_kb=0.2
        ))
        
        recommendations_stm32 = results_stm32.get_recommendations()
        rec_text_stm32 = " ".join(recommendations_stm32).lower()
        assert "stm32" in rec_text_stm32 or "sram" in rec_text_stm32
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_export_json(self, mock_json_dump, mock_file_open):
        """Test exporting results to JSON."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        results.add_latency_profile(LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1000.0,
            tokens_per_second=12.5,
            latency_std_ms=10.0
        ))
        results.test_configuration = {"batch_size": 1}
        results.environment_info = {"temperature": 25}
        results.add_raw_measurements("latency", [100.0, 102.0, 98.0])
        
        results.export_json("test_output.json")
        
        # Path object gets converted to string when passed to open
        call_args = mock_file_open.call_args
        assert str(call_args[0][0]).endswith("test_output.json")
        assert call_args[0][1] == 'w'
        mock_json_dump.assert_called_once()
        
        # Check the data structure passed to json.dump
        exported_data = mock_json_dump.call_args[0][0]
        assert "metadata" in exported_data
        assert "profiles" in exported_data
        assert "raw_measurements" in exported_data
        assert "summary" in exported_data
        assert "efficiency_score" in exported_data
        assert "recommendations" in exported_data
        
        # Check metadata
        metadata = exported_data["metadata"]
        assert metadata["platform"] == "esp32"
        assert metadata["model_name"] == "model"
        assert metadata["test_configuration"] == {"batch_size": 1}
        assert metadata["environment_info"] == {"temperature": 25}
        
        # Check profiles
        profiles = exported_data["profiles"]
        assert "latency" in profiles
        
        # Check raw measurements
        assert exported_data["raw_measurements"]["latency"] == [100.0, 102.0, 98.0]
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_from_json(self, mock_json_load, mock_file_open):
        """Test loading results from JSON."""
        # Mock JSON data
        json_data = {
            "metadata": {
                "platform": "esp32",
                "model_name": "test_model",
                "model_size_mb": 2.0,
                "quantization": "4bit",
                "timestamp": "2024-01-15T10:30:45+00:00",
                "test_configuration": {"batch_size": 1},
                "environment_info": {"temperature": 25}
            },
            "profiles": {
                "latency": {
                    "first_token_latency_ms": 100.0,
                    "inter_token_latency_ms": 80.0,
                    "total_latency_ms": 1000.0,
                    "tokens_per_second": 12.5,
                    "latency_std_ms": 10.0,
                    "percentile_50_ms": 0.0,
                    "percentile_90_ms": 0.0,
                    "percentile_95_ms": 0.0,
                    "percentile_99_ms": 0.0
                },
                "memory": {
                    "baseline_memory_kb": 200.0,
                    "peak_memory_kb": 450.0,
                    "memory_usage_kb": 350.0,
                    "memory_efficiency_tokens_per_kb": 0.28,
                    "fragmentation_percent": 0.0,
                    "gc_overhead_percent": 0.0,
                    "stack_usage_kb": 0.0,
                    "heap_usage_kb": 0.0
                }
            },
            "raw_measurements": {
                "latency": [100.0, 102.0, 98.0]
            }
        }
        
        mock_json_load.return_value = json_data
        
        results = ProfileResults.from_json("test_input.json")
        
        # Path object gets converted to string when passed to open
        call_args = mock_file_open.call_args
        assert str(call_args[0][0]).endswith("test_input.json")
        assert call_args[0][1] == 'r'
        mock_json_load.assert_called_once()
        
        # Check loaded data
        assert results.platform == "esp32"
        assert results.model_name == "test_model"
        assert results.model_size_mb == 2.0
        assert results.quantization == "4bit"
        assert results.test_configuration == {"batch_size": 1}
        assert results.environment_info == {"temperature": 25}
        assert results.raw_measurements["latency"] == [100.0, 102.0, 98.0]
        
        # Check profiles were loaded
        assert results.latency_profile is not None
        assert results.latency_profile.first_token_latency_ms == 100.0
        assert results.memory_profile is not None
        assert results.memory_profile.peak_memory_kb == 450.0
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('csv.DictWriter')
    def test_export_csv(self, mock_csv_writer, mock_file_open):
        """Test exporting results to CSV."""
        results = ProfileResults("esp32", "model", 2.0, "4bit")
        results.add_latency_profile(LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1000.0,
            tokens_per_second=12.5,
            latency_std_ms=10.0
        ))
        
        mock_writer_instance = Mock()
        mock_csv_writer.return_value = mock_writer_instance
        
        results.export_csv("test_output.csv")
        
        # Path object gets converted to string when passed to open
        call_args = mock_file_open.call_args
        assert str(call_args[0][0]).endswith("test_output.csv")
        assert call_args[0][1] == 'w'
        assert call_args[1]['newline'] == ''
        mock_csv_writer.assert_called_once()
        mock_writer_instance.writeheader.assert_called_once()
        mock_writer_instance.writerow.assert_called_once()
        
        # Check that flattened data was written
        written_data = mock_writer_instance.writerow.call_args[0][0]
        assert "platform" in written_data
        assert "model_name" in written_data
        assert "latency_first_token_ms" in written_data
        assert "efficiency_score" in written_data
    
    def test_repr(self):
        """Test string representation."""
        results = ProfileResults("esp32", "test_model", 2.5, "4bit")
        results.add_latency_profile(LatencyProfile(
            first_token_latency_ms=100.0,
            inter_token_latency_ms=80.0,
            total_latency_ms=1000.0,
            tokens_per_second=12.5,
            latency_std_ms=10.0
        ))
        results.add_memory_profile(MemoryProfile(
            baseline_memory_kb=200.0,
            peak_memory_kb=450.0,
            memory_usage_kb=350.0,
            memory_efficiency_tokens_per_kb=0.28
        ))
        results.add_power_profile(PowerProfile(
            idle_power_mw=15.0,
            active_power_mw=85.0,
            peak_power_mw=120.0,
            energy_per_token_mj=6.8,
            total_energy_mj=81.6
        ))
        
        repr_str = repr(results)
        
        assert "ProfileResults(test_model on esp32)" in repr_str
        assert "2.5MB 4bit quantization" in repr_str
        assert "12.5 tok/s" in repr_str
        assert "100ms first token" in repr_str
        assert "450KB peak" in repr_str
        assert "85mW" in repr_str
        assert "6.8mJ/token" in repr_str
        assert "Efficiency Score:" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])