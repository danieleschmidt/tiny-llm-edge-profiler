"""ModelProfiler: parameter count, memory estimation, FLOPs analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

from .model_spec import LayerSpec, ModelSpec

DTYPE_BYTES: Dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def _layer_params(layer: LayerSpec) -> int:
    """Count parameters for a single layer."""
    lt = layer.layer_type
    if lt == "linear":
        params = layer.input_dim * layer.output_dim
        if layer.has_bias:
            params += layer.output_dim
        return params
    elif lt == "embedding":
        # vocab_size * embedding_dim stored in input_dim / output_dim
        return layer.input_dim * layer.output_dim
    elif lt in ("norm",):
        # typically 2 learnable vectors of size hidden_dim (scale + shift)
        # but some norms (RMSNorm) have only scale
        return layer.output_dim  # conservative: just scale
    else:
        return 0


def _layer_flops(layer: LayerSpec, batch_size: int, seq_len: Optional[int]) -> int:
    """Estimate FLOPs for a single layer forward pass."""
    lt = layer.layer_type
    effective_seq = seq_len or layer.seq_len or 1
    if lt == "linear":
        # multiply-add = 2 * batch * seq * in_dim * out_dim
        return 2 * batch_size * effective_seq * layer.input_dim * layer.output_dim
    elif lt == "embedding":
        # lookup: no FLOPs (just indexing)
        return 0
    elif lt in ("norm",):
        # mean + variance + scale: ~5 ops per element
        return 5 * batch_size * effective_seq * layer.output_dim
    else:
        return 0


class ModelProfiler:
    """Profile a model architecture described by a ModelSpec."""

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    def count_parameters(self) -> int:
        """Sum parameters across all layers."""
        return sum(_layer_params(l) for l in self.model_spec.layers)

    def estimate_memory_bytes(self, dtype: str = "float32") -> int:
        """Estimate total bytes needed to store model weights."""
        if dtype not in DTYPE_BYTES:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Supported: {list(DTYPE_BYTES.keys())}"
            )
        bytes_per_param = DTYPE_BYTES[dtype]
        total_params = self.count_parameters()
        return int(total_params * bytes_per_param)

    def estimate_flops(
        self, batch_size: int = 1, seq_len: Optional[int] = None
    ) -> int:
        """Estimate total FLOPs for one forward pass."""
        effective_seq = seq_len or self.model_spec.context_len or 1
        return sum(
            _layer_flops(l, batch_size, effective_seq)
            for l in self.model_spec.layers
        )

    def per_layer_breakdown(self) -> List[Dict[str, Any]]:
        """Return per-layer parameter and FLOPs breakdown."""
        effective_seq = self.model_spec.context_len or 1
        result = []
        for l in self.model_spec.layers:
            params = _layer_params(l)
            flops = _layer_flops(l, 1, effective_seq)
            result.append(
                {
                    "name": l.name,
                    "layer_type": l.layer_type,
                    "input_dim": l.input_dim,
                    "output_dim": l.output_dim,
                    "params": params,
                    "flops": flops,
                    "memory_float32_bytes": int(params * DTYPE_BYTES["float32"]),
                }
            )
        return result

    def profile(self, batch_size: int = 1) -> Dict[str, Any]:
        """Return a full profile dict with all metrics."""
        total_params = self.count_parameters()
        mem_f32 = self.estimate_memory_bytes("float32")
        mem_f16 = self.estimate_memory_bytes("float16")
        mem_int8 = self.estimate_memory_bytes("int8")
        mem_int4 = self.estimate_memory_bytes("int4")
        total_flops = self.estimate_flops(batch_size=batch_size)

        return {
            "model_name": self.model_spec.name,
            "total_parameters": total_params,
            "total_parameters_millions": round(total_params / 1e6, 3),
            "memory": {
                "float32_bytes": mem_f32,
                "float16_bytes": mem_f16,
                "int8_bytes": mem_int8,
                "int4_bytes": mem_int4,
                "float32_mb": round(mem_f32 / (1024 ** 2), 3),
                "float16_mb": round(mem_f16 / (1024 ** 2), 3),
                "int8_mb": round(mem_int8 / (1024 ** 2), 3),
                "int4_mb": round(mem_int4 / (1024 ** 2), 3),
            },
            "flops": {
                "total": total_flops,
                "gflops": round(total_flops / 1e9, 6),
                "batch_size": batch_size,
            },
            "vocab_size": self.model_spec.vocab_size,
            "context_len": self.model_spec.context_len,
            "n_layers": len(self.model_spec.layers),
        }
