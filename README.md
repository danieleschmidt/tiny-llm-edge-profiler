# tiny-llm-edge-profiler

A lightweight profiling toolkit for quantized LLMs on edge devices. No PyTorch required — pure Python + NumPy.

## Features

- **ModelProfiler**: parameter count, memory estimation (float32/float16/int8/int4), theoretical FLOPs
- **QuantizationAnalyzer**: INT4/INT8 precision simulation, SNR estimation, compression ratios
- **BenchmarkRunner**: layer-wise timing using NumPy matmul simulation
- **ReportGenerator**: JSON and CSV report output
- **CLI**: `edge-profiler` command with profile/benchmark/quantize/full-report subcommands

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Profile a model
edge-profiler profile --model llama-small

# Run benchmarks
edge-profiler benchmark --model llama-small --batch-size 4

# Analyze quantization
edge-profiler quantize --model llama-small --bits 4 8

# Full report
edge-profiler full-report --model llama-small --output report.json --format json
```

## Preset Models

- `llama-small`: 512 hidden dim, 6 layers, 32K vocab
- `llama-tiny`: 256 hidden dim, 4 layers, 16K vocab
- `llama-medium`: 1024 hidden dim, 12 layers, 32K vocab

## Custom Models

Pass a JSON config file as `--model`:

```json
{
  "name": "my-model",
  "vocab_size": 1000,
  "context_len": 128,
  "layers": [
    {"name": "fc1", "layer_type": "linear", "input_dim": 64, "output_dim": 128}
  ]
}
```

## Testing

```bash
python -m pytest tests/ -v
```
