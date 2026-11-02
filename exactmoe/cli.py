"""Command-line entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import TensorIndex


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exactmoe",
        description="Byte-preserving MoE weight indexing and correctness experiments",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect", help="inspect safetensors headers without loading weights")
    inspect.add_argument("model_dir", type=Path)
    inspect.add_argument("--expected-experts", type=int)
    inspect.add_argument("--show-candidates", type=int, default=0,
                         help="print up to N packed expert candidates (0 disables)")
    subs.add_parser("demo", help="run streaming-vs-dense correctness check on tiny synthetic MoE")
    bench = subs.add_parser("bench", help="benchmark tiny synthetic CPU MoE")
    bench.add_argument("--steps", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "inspect":
        index = TensorIndex.from_directory(args.model_dir)
        result = index.summary()
        if args.show_candidates > 0:
            result["candidate_samples"] = [
                {"name": rec.name, "shape": list(rec.shape), "dtype": rec.dtype}
                for rec in index.packed_expert_candidates(args.expected_experts)[
                    :args.show_candidates
                ]
            ]
    else:
        from .demo import run_demo
        if args.command == "bench" and args.steps < 1:
            raise SystemExit("--steps must be positive")
        result = run_demo(bench_steps=args.steps if args.command == "bench" else 0)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
