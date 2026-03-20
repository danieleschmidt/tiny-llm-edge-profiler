"""CLI entry point for edge-profiler."""

from __future__ import annotations

import argparse
import json
import sys

from .model_spec import ModelSpec
from .profiler import ModelProfiler
from .quantizer import QuantizationAnalyzer
from .benchmark import BenchmarkRunner
from .reporter import ReportGenerator


PRESET_MODELS = {
    "llama-small": dict(hidden_dim=512, n_layers=6, vocab_size=32000, context_len=512),
    "llama-tiny": dict(hidden_dim=256, n_layers=4, vocab_size=16000, context_len=256),
    "llama-medium": dict(hidden_dim=1024, n_layers=12, vocab_size=32000, context_len=1024),
}


def _build_model(model_name: str) -> ModelSpec:
    if model_name in PRESET_MODELS:
        return ModelSpec.llama_like(**PRESET_MODELS[model_name])
    # Try as a JSON config file
    try:
        with open(model_name, "r") as f:
            config = json.load(f)
        return ModelSpec.from_config(config)
    except FileNotFoundError:
        print(
            f"Error: model '{model_name}' not found. "
            f"Use a preset ({', '.join(PRESET_MODELS)}) or a JSON config file.",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_profile(args: argparse.Namespace):
    spec = _build_model(args.model)
    profiler = ModelProfiler(spec)
    report = profiler.profile(batch_size=args.batch_size)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Profile saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))


def cmd_benchmark(args: argparse.Namespace):
    spec = _build_model(args.model)
    runner = BenchmarkRunner(spec)
    results = runner.run_full_benchmark(batch_size=args.batch_size, n_runs=args.n_runs)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


def cmd_quantize(args: argparse.Namespace):
    spec = _build_model(args.model)
    analyzer = QuantizationAnalyzer(spec)
    bits_list = list(args.bits)
    result = analyzer.analyze_model(bits_list=bits_list)

    output = {str(k): v for k, v in result.items()}
    output["compression_ratios"] = {
        f"32->int{b}": analyzer.compression_ratio(32, b) for b in bits_list
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Quantization analysis saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


def cmd_full_report(args: argparse.Namespace):
    spec = _build_model(args.model)
    profiler = ModelProfiler(spec)
    analyzer = QuantizationAnalyzer(spec)
    runner = BenchmarkRunner(spec)
    reporter = ReportGenerator(profiler, analyzer, runner)

    fmt = args.format.lower()
    if args.output:
        reporter.save(args.output, format=fmt, batch_size=args.batch_size)
        print(f"Full report saved to {args.output}")
    else:
        if fmt == "json":
            print(reporter.to_json(batch_size=args.batch_size))
        else:
            print(reporter.to_csv(batch_size=args.batch_size))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-profiler",
        description="Edge LLM Profiling Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # profile
    p_profile = subparsers.add_parser("profile", help="Profile model architecture")
    p_profile.add_argument("--model", default="llama-small", help="Model preset or JSON config path")
    p_profile.add_argument("--output", "-o", default=None, help="Output file")
    p_profile.add_argument("--batch-size", type=int, default=1)
    p_profile.set_defaults(func=cmd_profile)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run layer benchmarks")
    p_bench.add_argument("--model", default="llama-small")
    p_bench.add_argument("--output", "-o", default=None)
    p_bench.add_argument("--batch-size", type=int, default=1)
    p_bench.add_argument("--n-runs", type=int, default=10)
    p_bench.set_defaults(func=cmd_benchmark)

    # quantize
    p_quant = subparsers.add_parser("quantize", help="Analyze quantization effects")
    p_quant.add_argument("--model", default="llama-small")
    p_quant.add_argument("--output", "-o", default=None)
    p_quant.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    p_quant.set_defaults(func=cmd_quantize)

    # full-report
    p_full = subparsers.add_parser("full-report", help="Generate full report")
    p_full.add_argument("--model", default="llama-small")
    p_full.add_argument("--output", "-o", default=None)
    p_full.add_argument("--format", default="json", choices=["json", "csv"])
    p_full.add_argument("--batch-size", type=int, default=1)
    p_full.set_defaults(func=cmd_full_report)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
