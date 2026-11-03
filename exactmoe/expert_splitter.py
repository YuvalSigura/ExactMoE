"""Expert splitting by intermediate-dimension partition.

This module performs an actual mathematical transformation:
it takes the three weight matrices of one SwiGLU expert
(gate_proj, up_proj, down_proj) and splits them into N smaller experts
by partitioning the intermediate dimension.

What it does:
- Accepts weight tensors (NumPy) of shape matching a real MoE expert
- Splits the intermediate axis into contiguous chunks
- Returns N new experts, each with reduced intermediate size
- Produces a deterministic mapping from original expert -> list of sub-experts

What it does NOT do:
- Download or modify a real GLM-5.3 checkpoint
- Retrain the router (the caller must decide how to route to the new experts)
- Guarantee that the split model has the same quality as the original
- Run any inference

This is the concrete implementation of the idea
"turn one fat expert into several smaller experts and use the right one".

Typical shapes for a GLM-style expert (hidden=6144, intermediate=16384 example):
    gate_proj: (intermediate, hidden)
    up_proj:   (intermediate, hidden)
    down_proj: (hidden, intermediate)

After splitting into N parts the i-th sub-expert has intermediate size
≈ original_intermediate // N.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required for expert_splitter") from exc


def split_expert(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    n_splits: int,
    *,
    name: str = "expert",
) -> dict[str, Any]:
    """Split one SwiGLU expert into n_splits smaller experts.

    Parameters
    ----------
    gate, up : (intermediate, hidden)
    down     : (hidden, intermediate)
    n_splits : number of sub-experts to create (must divide intermediate evenly
               or the last chunk absorbs the remainder)

    Returns
    -------
    dict with:
      - sub_experts: list of dicts containing the three sliced arrays
      - original_intermediate, new_intermediate_sizes
      - mapping description
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if gate.shape != up.shape:
        raise ValueError("gate and up must have identical shape")
    if gate.ndim != 2 or down.ndim != 2:
        raise ValueError("weights must be 2-D")
    intermediate, hidden = gate.shape
    if down.shape != (hidden, intermediate):
        raise ValueError(
            f"down shape {down.shape} does not match gate/up "
            f"(expected ({hidden}, {intermediate}))"
        )

    # Compute chunk boundaries along the intermediate axis
    base = intermediate // n_splits
    remainder = intermediate % n_splits
    sizes = [base + (1 if i < remainder else 0) for i in range(n_splits)]
    if any(s == 0 for s in sizes):
        raise ValueError("n_splits is too large for the intermediate dimension")

    boundaries = [0]
    for s in sizes:
        boundaries.append(boundaries[-1] + s)

    sub_experts = []
    for i in range(n_splits):
        start, end = boundaries[i], boundaries[i + 1]
        sub_experts.append({
            "sub_id": i,
            "name": f"{name}.sub{i}",
            "intermediate_size": end - start,
            "gate": gate[start:end].copy(),
            "up": up[start:end].copy(),
            "down": down[:, start:end].copy(),
            "slice": [start, end],
        })

    return {
        "status": "SPLIT_COMPLETE",
        "original_name": name,
        "n_splits": n_splits,
        "original_intermediate": intermediate,
        "original_hidden": hidden,
        "sub_experts": [
            {
                "sub_id": se["sub_id"],
                "name": se["name"],
                "intermediate_size": se["intermediate_size"],
                "slice": se["slice"],
                "gate_shape": list(se["gate"].shape),
                "up_shape": list(se["up"].shape),
                "down_shape": list(se["down"].shape),
            }
            for se in sub_experts
        ],
        "note": (
            "Weights have been partitioned. The original expert is no longer "
            "needed if the router is updated to select among the sub-experts. "
            "Quality impact is not evaluated here."
        ),
        # Keep the actual arrays only if the caller wants them;
        # the public summary above is JSON-serializable.
        "_arrays": sub_experts,
    }


def split_expert_summary(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    n_splits: int,
    name: str = "expert",
) -> dict[str, Any]:
    """Same as split_expert but returns only the JSON-safe summary."""
    result = split_expert(gate, up, down, n_splits, name=name)
    result.pop("_arrays", None)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split one SwiGLU expert into N smaller experts by intermediate dimension"
    )
    parser.add_argument("--intermediate", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-arrays", type=Path, help="optional .npz path to store the sliced weights")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    gate = rng.normal(0, 0.02, (args.intermediate, args.hidden)).astype(np.float32)
    up = rng.normal(0, 0.02, (args.intermediate, args.hidden)).astype(np.float32)
    down = rng.normal(0, 0.02, (args.hidden, args.intermediate)).astype(np.float32)

    full = split_expert(gate, up, down, args.splits, name="synthetic_expert")
    summary = {k: v for k, v in full.items() if k != "_arrays"}

    if args.save_arrays:
        arrays = {}
        for se in full["_arrays"]:
            prefix = f"sub{se['sub_id']}"
            arrays[f"{prefix}_gate"] = se["gate"]
            arrays[f"{prefix}_up"] = se["up"]
            arrays[f"{prefix}_down"] = se["down"]
        np.savez(args.save_arrays, **arrays)
        summary["saved_arrays"] = str(args.save_arrays)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
