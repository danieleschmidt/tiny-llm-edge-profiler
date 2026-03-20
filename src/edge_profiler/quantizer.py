"""QuantizationAnalyzer: simulate INT4/INT8 precision loss."""

from __future__ import annotations

import math
from typing import Dict, List, Any

import numpy as np

from .model_spec import ModelSpec
from .profiler import DTYPE_BYTES, _layer_params


class QuantizationAnalyzer:
    """Analyze quantization effects on a model."""

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    def simulate_quantization(self, weights: np.ndarray, bits: int) -> np.ndarray:
        """Simulate INT4 or INT8 quantization by symmetric min-max rounding.

        Steps:
          1. Find the absmax of the input.
          2. Scale to [-2^(bits-1), 2^(bits-1)-1].
          3. Round to integer.
          4. Scale back to float.
        """
        if bits not in (4, 8, 16, 32):
            raise ValueError(f"bits must be 4, 8, 16, or 32; got {bits}")
        if bits == 32:
            return weights.copy()

        max_val = float(np.max(np.abs(weights)))
        if max_val == 0.0:
            return weights.copy()

        qmax = 2 ** (bits - 1) - 1  # e.g. 127 for int8, 7 for int4
        scale = max_val / qmax

        quantized = np.round(weights / scale)
        quantized = np.clip(quantized, -(qmax + 1), qmax)
        dequantized = quantized * scale
        return dequantized.astype(weights.dtype)

    def quantization_error(self, weights: np.ndarray, bits: int) -> Dict[str, Any]:
        """Compute quantization error metrics.

        Returns:
            dict with keys: mse, max_error, snr_db, bits
        """
        dequant = self.simulate_quantization(weights, bits)
        error = weights - dequant

        mse = float(np.mean(error ** 2))
        max_error = float(np.max(np.abs(error)))

        # SNR: signal power / noise power
        signal_power = float(np.mean(weights ** 2))
        noise_power = mse
        if noise_power == 0.0:
            snr_db = float("inf")
        elif signal_power == 0.0:
            snr_db = float("-inf")
        else:
            snr_db = 10.0 * math.log10(signal_power / noise_power)

        return {
            "mse": mse,
            "max_error": max_error,
            "snr_db": snr_db,
            "bits": bits,
        }

    def analyze_model(
        self, bits_list: List[int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """For each bit width, estimate memory savings and theoretical precision loss.

        Uses random sample weights to estimate SNR.

        Returns:
            {bits: {memory_bytes, savings_ratio, est_snr_db}}
        """
        if bits_list is None:
            bits_list = [4, 8]

        total_params = sum(_layer_params(l) for l in self.model_spec.layers)
        float32_bytes = total_params * DTYPE_BYTES["float32"]

        # Use a fixed random sample for reproducible estimates
        rng = np.random.default_rng(42)
        sample_weights = rng.standard_normal(min(total_params, 10000)).astype(
            np.float32
        )

        result = {}
        for bits in bits_list:
            bytes_per_param = DTYPE_BYTES.get(f"int{bits}", bits / 8)
            memory_bytes = int(total_params * bytes_per_param)
            savings_ratio = 1.0 - (memory_bytes / float32_bytes) if float32_bytes > 0 else 0.0
            err = self.quantization_error(sample_weights, bits)
            result[bits] = {
                "memory_bytes": memory_bytes,
                "savings_ratio": round(savings_ratio, 4),
                "est_snr_db": round(err["snr_db"], 2),
            }

        return result

    def compression_ratio(self, from_bits: int = 32, to_bits: int = 8) -> float:
        """Compute simple compression ratio from_bits -> to_bits."""
        if to_bits == 0:
            raise ValueError("to_bits cannot be 0")
        return from_bits / to_bits
