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
    profile = subs.add_parser("profile", help="inspect local or Hugging Face model JSON metadata")
    group = profile.add_mutually_exclusive_group(required=True)
    group.add_argument("--local", type=Path, help="directory holding config and index JSON")
    group.add_argument("--hf", help="Hugging Face repository ID; downloads two JSON files only")
    profile.add_argument("--revision", help="optional pinned HF revision")
    remote = subs.add_parser(
        "remote-header",
        help="inspect one remote safetensors shard header using bounded HTTP Range, no weight download",
    )
    remote.add_argument("--hf", required=True, help="model repository owner/name")
    remote.add_argument("--revision", required=True, help="pinned 40-character checkpoint commit SHA")
    remote.add_argument("--shard", required=True, help="exact shard filename")
    remote.add_argument("--block-size", type=int, default=128)
    remote.add_argument("--samples", type=int, default=8)
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
    elif args.command == "profile":
        from .profile import profile_hf, profile_local
        if args.revision is not None and args.local is not None:
            raise SystemExit("--revision can be used only with --hf")
        result = profile_local(args.local) if args.local else profile_hf(
            args.hf, revision=args.revision
        )
    elif args.command == "remote-header":
        from .remote_headers import inspect_remote_header
        result = inspect_remote_header(
            args.hf, args.revision, args.shard,
            block_size=args.block_size, sample_limit=args.samples,
        )
    else:
        from .demo import run_demo
        if args.command == "bench" and args.steps < 1:
            raise SystemExit("--steps must be positive")
        result = run_demo(bench_steps=args.steps if args.command == "bench" else 0)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
