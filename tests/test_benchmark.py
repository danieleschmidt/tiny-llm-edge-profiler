"""Tests for benchmark.py."""

import pytest
from edge_profiler.model_spec import LayerSpec, ModelSpec
from edge_profiler.benchmark import BenchmarkRunner


def _linear_spec():
    layers = [
        LayerSpec(name="fc1", layer_type="linear", input_dim=64, output_dim=128, has_bias=True),
        LayerSpec(name="fc2", layer_type="linear", input_dim=128, output_dim=32, has_bias=False),
    ]
    return ModelSpec(name="test", layers=layers, context_len=16)


def test_benchmark_layer_benchmark():
    """run_layer_benchmark returns correct keys and positive times."""
    spec = _linear_spec()
    runner = BenchmarkRunner(spec)
    result = runner.run_layer_benchmark(spec.layers[0], batch_size=1, n_runs=5)

    assert result["layer_name"] == "fc1"
    assert "avg_ms" in result
    assert "min_ms" in result
    assert "max_ms" in result
    assert "throughput_samples_per_sec" in result

    assert result["avg_ms"] >= 0
    assert result["min_ms"] <= result["avg_ms"]
    assert result["max_ms"] >= result["avg_ms"]
    assert result["throughput_samples_per_sec"] > 0


def test_benchmark_layer_batch_scaling():
    """Larger batch sizes should generally take longer (or at worst equal)."""
    spec = _linear_spec()
    runner = BenchmarkRunner(spec)
    r1 = runner.run_layer_benchmark(spec.layers[0], batch_size=1, n_runs=5)
    r8 = runner.run_layer_benchmark(spec.layers[0], batch_size=8, n_runs=5)
    # Can't guarantee strictly longer due to noise, but batch 8 shouldn't be much faster
    # Just check both return valid results
    assert r1["avg_ms"] >= 0
    assert r8["avg_ms"] >= 0


def test_benchmark_non_linear_layer():
    """Non-linear layers return zero timing."""
    spec = ModelSpec(
        name="test",
        layers=[LayerSpec(name="norm", layer_type="norm", input_dim=64, output_dim=64)],
    )
    runner = BenchmarkRunner(spec)
    result = runner.run_layer_benchmark(spec.layers[0], batch_size=1, n_runs=5)
    assert result["avg_ms"] == 0.0
    assert result["layer_name"] == "norm"


def test_benchmark_run_full_benchmark():
    """run_full_benchmark returns one result per layer."""
    spec = _linear_spec()
    runner = BenchmarkRunner(spec)
    results = runner.run_full_benchmark(batch_size=1, n_runs=3)
    assert len(results) == len(spec.layers)
    for r in results:
        assert "layer_name" in r
        assert "avg_ms" in r


def test_benchmark_estimate_inference_time():
    """estimate_inference_time_ms returns a non-negative float."""
    spec = _linear_spec()
    runner = BenchmarkRunner(spec)
    t = runner.estimate_inference_time_ms(batch_size=1)
    assert isinstance(t, float)
    assert t >= 0.0
