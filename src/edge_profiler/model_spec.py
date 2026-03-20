"""Model specification dataclasses — pure Python, no torch needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class LayerSpec:
    """Describes a single layer in a model architecture."""

    name: str
    layer_type: str  # "linear", "embedding", "attention", "norm", "other"
    input_dim: int
    output_dim: int
    has_bias: bool = True
    seq_len: Optional[int] = None  # for attention layers


class ModelSpec:
    """Describes a model architecture without weights."""

    def __init__(
        self,
        name: str,
        layers: List[LayerSpec],
        vocab_size: Optional[int] = None,
        context_len: Optional[int] = None,
    ):
        self.name = name
        self.layers = layers
        self.vocab_size = vocab_size
        self.context_len = context_len

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelSpec":
        """Parse a ModelSpec from a config dict.

        Expected keys:
            name (str)
            vocab_size (int, optional)
            context_len (int, optional)
            layers (list of dicts with keys: name, layer_type, input_dim, output_dim,
                    has_bias, seq_len)
        """
        layers = []
        for l in config.get("layers", []):
            layers.append(
                LayerSpec(
                    name=l["name"],
                    layer_type=l["layer_type"],
                    input_dim=l["input_dim"],
                    output_dim=l["output_dim"],
                    has_bias=l.get("has_bias", True),
                    seq_len=l.get("seq_len", None),
                )
            )
        return cls(
            name=config["name"],
            layers=layers,
            vocab_size=config.get("vocab_size"),
            context_len=config.get("context_len"),
        )

    @classmethod
    def llama_like(
        cls,
        hidden_dim: int = 512,
        n_layers: int = 6,
        vocab_size: int = 32000,
        context_len: int = 512,
    ) -> "ModelSpec":
        """Build a LLaMA-style transformer ModelSpec.

        Each transformer block contains:
          - q, k, v, o projections (linear, hidden_dim -> hidden_dim)
          - gate, up, down projections for FFN (hidden_dim -> 4*hidden_dim or back)
          - RMSNorm layers
        Plus token embedding and output lm_head.
        """
        ffn_dim = hidden_dim * 4
        layers: List[LayerSpec] = []

        # Token embedding
        layers.append(
            LayerSpec(
                name="token_embedding",
                layer_type="embedding",
                input_dim=vocab_size,
                output_dim=hidden_dim,
                has_bias=False,
            )
        )

        for i in range(n_layers):
            prefix = f"layer{i}"
            # Self-attention projections
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                layers.append(
                    LayerSpec(
                        name=f"{prefix}.attn.{proj}",
                        layer_type="linear",
                        input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        has_bias=False,
                        seq_len=context_len,
                    )
                )
            # FFN
            layers.append(
                LayerSpec(
                    name=f"{prefix}.ffn.gate_proj",
                    layer_type="linear",
                    input_dim=hidden_dim,
                    output_dim=ffn_dim,
                    has_bias=False,
                )
            )
            layers.append(
                LayerSpec(
                    name=f"{prefix}.ffn.up_proj",
                    layer_type="linear",
                    input_dim=hidden_dim,
                    output_dim=ffn_dim,
                    has_bias=False,
                )
            )
            layers.append(
                LayerSpec(
                    name=f"{prefix}.ffn.down_proj",
                    layer_type="linear",
                    input_dim=ffn_dim,
                    output_dim=hidden_dim,
                    has_bias=False,
                )
            )
            # Norms
            layers.append(
                LayerSpec(
                    name=f"{prefix}.input_norm",
                    layer_type="norm",
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    has_bias=False,
                )
            )
            layers.append(
                LayerSpec(
                    name=f"{prefix}.post_attn_norm",
                    layer_type="norm",
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    has_bias=False,
                )
            )

        # Final norm + LM head
        layers.append(
            LayerSpec(
                name="final_norm",
                layer_type="norm",
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                has_bias=False,
            )
        )
        layers.append(
            LayerSpec(
                name="lm_head",
                layer_type="linear",
                input_dim=hidden_dim,
                output_dim=vocab_size,
                has_bias=False,
            )
        )

        return cls(
            name=f"llama-like-h{hidden_dim}-l{n_layers}",
            layers=layers,
            vocab_size=vocab_size,
            context_len=context_len,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "context_len": self.context_len,
            "layers": [
                {
                    "name": l.name,
                    "layer_type": l.layer_type,
                    "input_dim": l.input_dim,
                    "output_dim": l.output_dim,
                    "has_bias": l.has_bias,
                    "seq_len": l.seq_len,
                }
                for l in self.layers
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ModelSpec(name={self.name!r}, n_layers={len(self.layers)}, "
            f"vocab_size={self.vocab_size}, context_len={self.context_len})"
        )
