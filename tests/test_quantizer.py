"""Tests for quantizer.py."""

import math
import numpy as np
import pytest
from edge_profiler.model_spec import LayerSpec, ModelSpec
from edge_profiler.quantizer import QuantizationAnalyzer


def _simple_spec():
    layers = [
        LayerSpec(name="fc", layer_type="linear", input_dim=64, output_dim=32, has_bias=False),
    ]
    return ModelSpec(name="test", layers=layers)


def test_quantizer_simulate_int8():
    """INT8 quantization should produce output close to input."""
    spec = _simple_spec()
    analyzer = QuantizationAnalyzer(spec)
    rng = np.random.default_rng(0)
    weights = rng.standard_normal(100).astype(np.float32)
    dequant = analyzer.simulate_quantization(weights, bits=8)

    assert dequant.shape == weights.shape
    assert dequant.dtype == np.float32
    # Max error for int8 should be very small relative to signal range
    max_err = np.max(np.abs(weights - dequant))
    signal_range = float(np.max(np.abs(weights)))
    assert max_err < signal_range * 0.05  # within 5%


def test_quantizer_simulate_int4():
    """INT4 quantization has more error than int8 but still bounded."""
    spec = _simple_spec()
    analyzer = QuantizationAnalyzer(spec)
    rng = np.random.default_rng(0)
    weights = rng.standard_normal(100).astype(np.float32)

    dequant8 = analyzer.simulate_quantization(weights, bits=8)
    dequant4 = analyzer.simulate_quantization(weights, bits=4)

    mse8 = float(np.mean((weights - dequant8) ** 2))
    mse4 = float(np.mean((weights - dequant4) ** 2))

    # INT4 should have strictly more error than INT8 for typical weights
    assert mse4 > mse8


def test_quantizer_simulate_int32_noop():
    """bits=32 should return unchanged copy."""
    spec = _simple_spec()
    analyzer = QuantizationAnalyzer(spec)
    weights = np.array([1.0, 2.5, -3.7], dtype=np.float32)
    result = analyzer.simulate_quantization(weights, bits=32)
    np.testing.assert_array_almost_equal(result, weights)


def test_quantizer_simulate_zeros():
    """Zero weights should come back as zeros."""
    spec = _simple_spec()
    analyzer = QuantizationAnalyzer(spec)
    weights = np.zeros(50, dtype=np.float32)
    result = analyzer.simulate_quantization(weights, bits=8)
    np.testing.assert_array_equal(result, weights)


def test_quantizer_error_metrics():
    """quantization_error should return correct keys and finite values."""
    spec = _simple_spec()
    analyzer = QuantizationAnalyzer(spec)
    rng = np.random.default_rng(42)
    weights = rng.standard_normal(500).astype(np.float32)

    err = analyzer.quantization_error(weights, bits=8)
    assert set(err.keys()) == {"mse", "max_error", "snr_db", "bits"}
    assert err["bits"] == 8
    assert err["mse"] >= 0
    assert err["max_error"] >= 0
    assert math.isfinite(err["snr_db"])
    # SNR for int8 should be reasonably high (>20 dB)
    assert err["snr_db"] > 20


def test_quantizer_analyze_model():
    """analyze_model should return dict keyed by bit widths with expected fields."""
    spec = ModelSpec.llama_like(hidden_dim=128, n_layers=2)
    analyzer = QuantizationAnalyzer(spec)
    result = analyzer.analyze_model(bits_list=[4, 8])

    assert 4 in result
    assert 8 in result

    for bits, data in result.items():
        assert "memory_bytes" in data
        assert "savings_ratio" in data
        assert "est_snr_db" in data
        assert data["memory_bytes"] > 0
        assert 0 < data["savings_ratio"] < 1

    # INT4 should have higher savings than INT8
    assert result[4]["savings_ratio"] > result[8]["savings_ratio"]


def test_quantizer_compression_ratio():
    """compression_ratio should return correct ratios."""
    spec = _simple_spec()
    analyzer = QuantizationAnalyzer(spec)
    assert analyzer.compression_ratio(32, 8) == 4.0
    assert analyzer.compression_ratio(32, 4) == 8.0
    assert analyzer.compression_ratio(16, 4) == 4.0


def test_quantizer_invalid_bits():
    """simulate_quantization should raise on unsupported bit widths."""
    spec = _simple_spec()
    analyzer = QuantizationAnalyzer(spec)
    weights = np.ones(10, dtype=np.float32)
    with pytest.raises(ValueError):
        analyzer.simulate_quantization(weights, bits=3)
