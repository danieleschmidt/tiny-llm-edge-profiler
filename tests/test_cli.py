"""Tests for cli.py."""

import json
import os
import pytest
from edge_profiler.cli import build_parser, main


def test_cli_profile_stdout(capsys):
    """edge-profiler profile should print JSON to stdout."""
    main(["profile", "--model", "llama-small"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "total_parameters" in data


def test_cli_profile_to_file(tmp_path):
    """edge-profiler profile --output saves to file."""
    out = str(tmp_path / "profile.json")
    main(["profile", "--model", "llama-small", "--output", out])
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert "total_parameters" in data


def test_cli_benchmark_stdout(capsys):
    """edge-profiler benchmark should print a JSON list."""
    main(["benchmark", "--model", "llama-tiny", "--n-runs", "3"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) > 0


def test_cli_quantize_stdout(capsys):
    """edge-profiler quantize should print quantization analysis."""
    main(["quantize", "--model", "llama-tiny", "--bits", "4", "8"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "4" in data or 4 in data


def test_cli_full_report_json(capsys):
    """edge-profiler full-report --format json should print full report."""
    main(["full-report", "--model", "llama-tiny", "--format", "json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "model" in data
    assert "quantization" in data


def test_cli_full_report_csv(capsys):
    """edge-profiler full-report --format csv should print CSV."""
    main(["full-report", "--model", "llama-tiny", "--format", "csv"])
    captured = capsys.readouterr()
    assert "name" in captured.out
    assert "layer_type" in captured.out


def test_cli_unknown_model_exits():
    """Unknown model name should exit with code 1."""
    with pytest.raises(SystemExit) as exc_info:
        main(["profile", "--model", "does-not-exist-xyz"])
    assert exc_info.value.code == 1


def test_cli_from_json_config(tmp_path, capsys):
    """CLI should accept a JSON config file as model."""
    config = {
        "name": "custom",
        "vocab_size": 100,
        "context_len": 32,
        "layers": [
            {"name": "fc", "layer_type": "linear", "input_dim": 16, "output_dim": 8},
        ],
    }
    cfg_path = str(tmp_path / "model.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f)

    main(["profile", "--model", cfg_path])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["model_name"] == "custom"
