"""Command-line entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import TensorIndex



def _expert_names(layer: int, expert: int) -> list[str]:
    if layer < 0 or expert < 0:
        raise ValueError("layer and expert must be nonnegative")
    names = []
    for part in ("gate_proj", "up_proj", "down_proj"):
        weight = f"model.layers.{layer}.mlp.experts.{expert}.{part}.weight"
        names.extend((weight, weight + "_scale_inv"))
    return names

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
    plan = subs.add_parser(
        "expert-plan", help="list only the shard files needed for one original FP8 expert"
    )
    plan.add_argument("model_dir", type=Path)
    plan.add_argument("--layer", type=int, default=3)
    plan.add_argument("--expert", type=int, default=0)
    expert = subs.add_parser(
        "expert-smoke",
        help="compute ONE native FP8 routed expert from local checkpoint; NOT full GLM inference",
    )
    expert.add_argument("model_dir", type=Path)
    expert.add_argument("--layer", type=int, default=3)
    expert.add_argument("--expert", type=int, default=0)
    expert.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    expert.add_argument("--seed", type=int, default=42)
    expert.add_argument("--block-size", type=int, default=128)
    layer = subs.add_parser(
        "layer-smoke",
        help="run one ORIGINAL FP8 GLM sparse MoE FFN (router+shared); no attention",
    )
    layer.add_argument("model_dir", type=Path)
    layer.add_argument("--layer", type=int, default=3)
    layer.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    layer.add_argument("--seed", type=int, default=42)
    layer.add_argument("--block-size", type=int, default=128)
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
    elif args.command == "expert-plan":
        names = _expert_names(args.layer, args.expert)
        result = {"layer": args.layer, "expert": args.expert,
                  "shards": TensorIndex.shard_plan(args.model_dir, names),
                  "note": "Metadata only; downloading shards is NOT performed."}
    elif args.command == "expert-smoke":
        from .fp8_reference import forward_expert
        try:
            import numpy as np
        except ImportError as exc:
            raise SystemExit('Install demo dependencies: pip install -e ".[demo]"') from exc
        if args.layer < 0 or args.expert < 0:
            raise SystemExit("--layer and --expert must be nonnegative")
        names = _expert_names(args.layer, args.expert)
        index = TensorIndex.from_selected(args.model_dir, names)
        name = f"model.layers.{args.layer}.mlp.experts.{args.expert}.gate_proj.weight"
        input_size = index[name].shape[1]
        x = np.random.default_rng(args.seed).normal(0, 0.1, input_size).astype(np.float32)
        result = forward_expert(index, args.layer, args.expert, x,
                                device=args.device, block_size=args.block_size)
        result["input_seed"] = args.seed
    elif args.command == "layer-smoke":
        from .moe_layer import forward_moe_ffn
        import numpy as np
        from pathlib import Path
        config_path = args.model_dir / "config.json"
        if not config_path.is_file():
            raise SystemExit(f"Missing model config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        from .moe_layer import read_dense_vector_or_matrix, route_noaux_tc
        if args.layer < 0:
            raise SystemExit("--layer must be nonnegative")
        hidden = int(config["hidden_size"])
        x = np.random.default_rng(args.seed).normal(0, 0.1, hidden).astype(np.float32)
        prefix = f"model.layers.{args.layer}.mlp."
        names = [prefix + "gate.weight",
                 prefix + "gate.e_score_correction_bias"]
        gate_index = TensorIndex.from_selected(args.model_dir, names)
        route = route_noaux_tc(x,
            read_dense_vector_or_matrix(gate_index[names[0]]),
            read_dense_vector_or_matrix(gate_index[names[1]]), config)
        for selected_expert in route.expert_ids:
            names.extend(_expert_names(args.layer, selected_expert))
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shared = prefix + f"shared_experts.{proj}.weight"
            names.append(shared)
            # FP8 shared matrices have scales; BF16/F32 matrices do not.
        manifest = json.loads((args.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        mapped = manifest["weight_map"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shared = prefix + f"shared_experts.{proj}.weight"
            if shared + "_scale_inv" in mapped:
                names.append(shared + "_scale_inv")
        index = TensorIndex.from_selected(args.model_dir, names)
        result = forward_moe_ffn(index, args.layer, x, config,
                                 device=args.device, block_size=args.block_size)
        result["input_seed"] = args.seed
    else:
        from .demo import run_demo
        if args.command == "bench" and args.steps < 1:
            raise SystemExit("--steps must be positive")
        result = run_demo(bench_steps=args.steps if args.command == "bench" else 0)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
