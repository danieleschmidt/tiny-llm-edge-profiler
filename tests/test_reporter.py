"""Tests for reporter.py."""

import csv
import io
import json
import os
import tempfile

import pytest
from edge_profiler.model_spec import LayerSpec, ModelSpec
from edge_profiler.profiler import ModelProfiler
from edge_profiler.quantizer import QuantizationAnalyzer
from edge_profiler.benchmark import BenchmarkRunner
from edge_profiler.reporter import ReportGenerator


def _build_reporter():
    spec = ModelSpec.llama_like(hidden_dim=128, n_layers=2)
    profiler = ModelProfiler(spec)
    analyzer = QuantizationAnalyzer(spec)
    runner = BenchmarkRunner(spec)
    return ReportGenerator(profiler, analyzer, runner)


def test_reporter_json_output():
    """to_json returns valid JSON with expected top-level keys."""
    reporter = _build_reporter()
    raw = reporter.to_json(batch_size=1)

    # Must be valid JSON
    data = json.loads(raw)

    assert "model" in data
    assert "quantization" in data
    assert "benchmark" in data
    assert "per_layer_breakdown" in data

    # model section
    assert "total_parameters" in data["model"]
    assert "memory" in data["model"]

    # benchmark section
    assert "layers" in data["benchmark"]
    assert isinstance(data["benchmark"]["layers"], list)


def test_reporter_csv_output():
    """to_csv returns valid CSV with expected columns."""
    reporter = _build_reporter()
    raw = reporter.to_csv(batch_size=1)

    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)

    assert len(rows) > 0

    expected_cols = {"name", "layer_type", "input_dim", "output_dim", "params", "flops", "memory_float32_bytes"}
    for col in expected_cols:
        assert col in reader.fieldnames, f"Missing column: {col}"

    # All params should be non-negative integers
    for row in rows:
        assert int(row["params"]) >= 0


def test_reporter_generate_report_structure():
    """generate_report returns expected nested structure."""
    reporter = _build_reporter()
    report = reporter.generate_report(batch_size=2)
    assert isinstance(report, dict)
    assert "model" in report
    assert report["benchmark"]["batch_size"] == 2


def test_reporter_save_json(tmp_path):
    """save() writes valid JSON to file."""
    reporter = _build_reporter()
    out_path = str(tmp_path / "report.json")
    reporter.save(out_path, format="json")

    assert os.path.exists(out_path)
    with open(out_path) as f:
        data = json.load(f)
    assert "model" in data


def test_reporter_save_csv(tmp_path):
    """save() writes valid CSV to file."""
    reporter = _build_reporter()
    out_path = str(tmp_path / "report.csv")
    reporter.save(out_path, format="csv")

    assert os.path.exists(out_path)
    with open(out_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0


def test_reporter_save_invalid_format(tmp_path):
    """save() raises ValueError on unknown format."""
    reporter = _build_reporter()
    with pytest.raises(ValueError, match="Unknown format"):
        reporter.save(str(tmp_path / "x.txt"), format="xml")
