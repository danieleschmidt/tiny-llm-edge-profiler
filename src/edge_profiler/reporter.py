"""ReportGenerator: combine profiler, quantizer, benchmark into CSV/JSON reports."""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Dict

from .benchmark import BenchmarkRunner
from .profiler import ModelProfiler
from .quantizer import QuantizationAnalyzer


class ReportGenerator:
    """Combine profiler, quantizer, and benchmark results into reports."""

    def __init__(
        self,
        profiler: ModelProfiler,
        quantizer: QuantizationAnalyzer,
        benchmark: BenchmarkRunner,
    ):
        self.profiler = profiler
        self.quantizer = quantizer
        self.benchmark = benchmark

    def generate_report(self, batch_size: int = 1) -> Dict[str, Any]:
        """Build a comprehensive report dict."""
        profile = self.profiler.profile(batch_size=batch_size)
        quant_analysis = self.quantizer.analyze_model()
        per_layer = self.profiler.per_layer_breakdown()

        # Run benchmark for linear layers only (skip heavy computation for report)
        bench_results = self.benchmark.run_full_benchmark(batch_size=batch_size, n_runs=5)
        estimated_inference_ms = sum(r["avg_ms"] for r in bench_results)

        return {
            "model": profile,
            "quantization": quant_analysis,
            "benchmark": {
                "batch_size": batch_size,
                "estimated_inference_ms": round(estimated_inference_ms, 4),
                "layers": bench_results,
            },
            "per_layer_breakdown": per_layer,
        }

    def to_json(self, batch_size: int = 1) -> str:
        """Serialize the full report as JSON."""
        report = self.generate_report(batch_size=batch_size)
        return json.dumps(report, indent=2, default=str)

    def to_csv(self, batch_size: int = 1) -> str:
        """Serialize per-layer breakdown as CSV."""
        per_layer = self.profiler.per_layer_breakdown()
        bench = self.benchmark.run_full_benchmark(batch_size=batch_size, n_runs=5)
        bench_map = {r["layer_name"]: r for r in bench}

        output = io.StringIO()
        fieldnames = [
            "name",
            "layer_type",
            "input_dim",
            "output_dim",
            "params",
            "flops",
            "memory_float32_bytes",
            "avg_ms",
            "min_ms",
            "max_ms",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for row in per_layer:
            bench_row = bench_map.get(row["name"], {})
            writer.writerow(
                {
                    "name": row["name"],
                    "layer_type": row["layer_type"],
                    "input_dim": row["input_dim"],
                    "output_dim": row["output_dim"],
                    "params": row["params"],
                    "flops": row["flops"],
                    "memory_float32_bytes": row["memory_float32_bytes"],
                    "avg_ms": bench_row.get("avg_ms", 0.0),
                    "min_ms": bench_row.get("min_ms", 0.0),
                    "max_ms": bench_row.get("max_ms", 0.0),
                }
            )

        return output.getvalue()

    def save(self, path: str, format: str = "json", batch_size: int = 1):
        """Save report to a file.

        Args:
            path: output file path
            format: "json" or "csv"
            batch_size: batch size for benchmarking
        """
        format = format.lower()
        if format == "json":
            content = self.to_json(batch_size=batch_size)
        elif format == "csv":
            content = self.to_csv(batch_size=batch_size)
        else:
            raise ValueError(f"Unknown format {format!r}. Use 'json' or 'csv'.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
