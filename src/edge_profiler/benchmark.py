"""BenchmarkRunner: time forward passes on random inputs using numpy."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any

import numpy as np

from .model_spec import LayerSpec, ModelSpec


class BenchmarkRunner:
    """Benchmark layer-wise inference timing using numpy matmul simulation."""

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    def _simulate_linear_forward(
        self, layer: LayerSpec, batch_size: int
    ) -> float:
        """Create random numpy arrays and time a matmul. Returns elapsed seconds."""
        seq_len = layer.seq_len or self.model_spec.context_len or 1
        # Input: (batch * seq_len, in_dim)  Weight: (in_dim, out_dim)
        x = np.random.randn(batch_size * seq_len, layer.input_dim).astype(np.float32)
        w = np.random.randn(layer.input_dim, layer.output_dim).astype(np.float32)

        start = time.perf_counter()
        _ = x @ w
        end = time.perf_counter()
        return end - start

    def run_layer_benchmark(
        self,
        layer: LayerSpec,
        batch_size: int = 1,
        n_runs: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark a single layer.

        Returns:
            {layer_name, avg_ms, min_ms, max_ms, throughput_samples_per_sec}
        """
        if layer.layer_type not in ("linear", "attention"):
            # Non-linear layers: return near-zero timing
            return {
                "layer_name": layer.name,
                "layer_type": layer.layer_type,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "throughput_samples_per_sec": float("inf"),
            }

        times_s = [
            self._simulate_linear_forward(layer, batch_size) for _ in range(n_runs)
        ]

        avg_s = sum(times_s) / len(times_s)
        min_s = min(times_s)
        max_s = max(times_s)

        avg_ms = avg_s * 1000
        min_ms = min_s * 1000
        max_ms = max_s * 1000

        throughput = batch_size / avg_s if avg_s > 0 else float("inf")

        return {
            "layer_name": layer.name,
            "layer_type": layer.layer_type,
            "avg_ms": round(avg_ms, 4),
            "min_ms": round(min_ms, 4),
            "max_ms": round(max_ms, 4),
            "throughput_samples_per_sec": round(throughput, 2),
        }

    def run_full_benchmark(
        self, batch_size: int = 1, n_runs: int = 10
    ) -> List[Dict[str, Any]]:
        """Benchmark all layers in the model."""
        return [
            self.run_layer_benchmark(layer, batch_size=batch_size, n_runs=n_runs)
            for layer in self.model_spec.layers
        ]

    def estimate_inference_time_ms(self, batch_size: int = 1) -> float:
        """Estimate total inference time as sum of layer avg times."""
        results = self.run_full_benchmark(batch_size=batch_size, n_runs=3)
        return round(sum(r["avg_ms"] for r in results), 4)
