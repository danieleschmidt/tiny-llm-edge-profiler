"""Tests for profiler.py."""

import pytest
from edge_profiler.model_spec import LayerSpec, ModelSpec
from edge_profiler.profiler import ModelProfiler, DTYPE_BYTES


def _simple_model():
    """A simple model with known parameter counts."""
    layers = [
        LayerSpec(name="fc1", layer_type="linear", input_dim=4, output_dim=8, has_bias=True),
        LayerSpec(name="fc2", layer_type="linear", input_dim=8, output_dim=2, has_bias=False),
    ]
    return ModelSpec(name="simple", layers=layers)


def test_profiler_count_parameters():
    """Test parameter counting is correct."""
    spec = _simple_model()
    profiler = ModelProfiler(spec)
    # fc1: 4*8 + 8 (bias) = 40
    # fc2: 8*2 = 16
    assert profiler.count_parameters() == 56


def test_profiler_count_parameters_no_bias():
    """Test param counting with no bias."""
    layers = [LayerSpec(name="fc", layer_type="linear", input_dim=10, output_dim=5, has_bias=False)]
    spec = ModelSpec(name="m", layers=layers)
    assert ModelProfiler(spec).count_parameters() == 50


def test_profiler_memory_bytes_float32():
    """Test memory estimation for float32."""
    spec = _simple_model()
    profiler = ModelProfiler(spec)
    # 56 params * 4 bytes = 224
    assert profiler.estimate_memory_bytes("float32") == 224


def test_profiler_memory_bytes_float16():
    """Test memory estimation for float16."""
    spec = _simple_model()
    profiler = ModelProfiler(spec)
    # 56 params * 2 bytes = 112
    assert profiler.estimate_memory_bytes("float16") == 112


def test_profiler_memory_bytes_int8():
    """Test memory estimation for int8."""
    spec = _simple_model()
    profiler = ModelProfiler(spec)
    # 56 params * 1 byte = 56
    assert profiler.estimate_memory_bytes("int8") == 56


def test_profiler_memory_bytes_int4():
    """Test memory estimation for int4."""
    spec = _simple_model()
    profiler = ModelProfiler(spec)
    # 56 params * 0.5 bytes = 28
    assert profiler.estimate_memory_bytes("int4") == 28


def test_profiler_unknown_dtype():
    """Test unknown dtype raises ValueError."""
    spec = _simple_model()
    profiler = ModelProfiler(spec)
    with pytest.raises(ValueError, match="Unknown dtype"):
        profiler.estimate_memory_bytes("float64")


def test_profiler_flops():
    """Test FLOPs estimation for linear layers."""
    layers = [
        LayerSpec(name="fc", layer_type="linear", input_dim=4, output_dim=8, has_bias=False),
    ]
    spec = ModelSpec(name="m", layers=layers, context_len=16)
    profiler = ModelProfiler(spec)
    # 2 * batch(1) * seq(16) * 4 * 8 = 1024
    assert profiler.estimate_flops(batch_size=1, seq_len=16) == 1024


def test_profiler_flops_batch():
    """Test FLOPs scales with batch size."""
    layers = [
        LayerSpec(name="fc", layer_type="linear", input_dim=4, output_dim=8, has_bias=False),
    ]
    spec = ModelSpec(name="m", layers=layers, context_len=16)
    profiler = ModelProfiler(spec)
    single = profiler.estimate_flops(batch_size=1, seq_len=16)
    four = profiler.estimate_flops(batch_size=4, seq_len=16)
    assert four == single * 4


def test_profiler_profile_keys():
    """Test profile() returns expected keys."""
    spec = ModelSpec.llama_like(hidden_dim=128, n_layers=2)
    profiler = ModelProfiler(spec)
    report = profiler.profile()
    assert "total_parameters" in report
    assert "memory" in report
    assert "flops" in report
    assert "model_name" in report


def test_profiler_per_layer_breakdown():
    """Test per_layer_breakdown returns list with correct fields."""
    spec = _simple_model()
    profiler = ModelProfiler(spec)
    breakdown = profiler.per_layer_breakdown()
    assert len(breakdown) == 2
    for row in breakdown:
        assert "name" in row
        assert "params" in row
        assert "flops" in row
        assert "memory_float32_bytes" in row
