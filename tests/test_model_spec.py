"""Tests for model_spec.py."""

import pytest
from edge_profiler.model_spec import LayerSpec, ModelSpec


def test_layer_spec_creation():
    """Test LayerSpec basic creation and field access."""
    layer = LayerSpec(
        name="fc1",
        layer_type="linear",
        input_dim=512,
        output_dim=1024,
        has_bias=True,
    )
    assert layer.name == "fc1"
    assert layer.layer_type == "linear"
    assert layer.input_dim == 512
    assert layer.output_dim == 1024
    assert layer.has_bias is True
    assert layer.seq_len is None


def test_layer_spec_defaults():
    """Test LayerSpec default values."""
    layer = LayerSpec(name="emb", layer_type="embedding", input_dim=32000, output_dim=512)
    assert layer.has_bias is True
    assert layer.seq_len is None


def test_model_spec_llama_like():
    """Test ModelSpec.llama_like builds a valid model."""
    spec = ModelSpec.llama_like(hidden_dim=512, n_layers=6, vocab_size=32000, context_len=512)
    assert spec.name.startswith("llama-like")
    assert spec.vocab_size == 32000
    assert spec.context_len == 512
    assert len(spec.layers) > 0

    # Should have token_embedding and lm_head
    names = [l.name for l in spec.layers]
    assert "token_embedding" in names
    assert "lm_head" in names

    # All layers should be LayerSpec instances
    for layer in spec.layers:
        assert isinstance(layer, LayerSpec)
        assert layer.input_dim > 0
        assert layer.output_dim > 0


def test_model_spec_from_config():
    """Test ModelSpec.from_config parses a dict correctly."""
    config = {
        "name": "test-model",
        "vocab_size": 1000,
        "context_len": 128,
        "layers": [
            {"name": "emb", "layer_type": "embedding", "input_dim": 1000, "output_dim": 64},
            {"name": "fc1", "layer_type": "linear", "input_dim": 64, "output_dim": 128, "has_bias": True},
            {"name": "fc2", "layer_type": "linear", "input_dim": 128, "output_dim": 1000, "has_bias": False},
        ],
    }
    spec = ModelSpec.from_config(config)
    assert spec.name == "test-model"
    assert spec.vocab_size == 1000
    assert spec.context_len == 128
    assert len(spec.layers) == 3

    assert spec.layers[0].layer_type == "embedding"
    assert spec.layers[1].has_bias is True
    assert spec.layers[2].has_bias is False


def test_model_spec_to_dict_roundtrip():
    """Test ModelSpec round-trips through to_dict / from_config."""
    original = ModelSpec.llama_like(hidden_dim=128, n_layers=2)
    d = original.to_dict()
    restored = ModelSpec.from_config(d)

    assert restored.name == original.name
    assert restored.vocab_size == original.vocab_size
    assert restored.context_len == original.context_len
    assert len(restored.layers) == len(original.layers)

    for orig_l, rest_l in zip(original.layers, restored.layers):
        assert orig_l.name == rest_l.name
        assert orig_l.layer_type == rest_l.layer_type
        assert orig_l.input_dim == rest_l.input_dim
        assert orig_l.output_dim == rest_l.output_dim
