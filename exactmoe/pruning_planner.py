"""Expert pruning planner.

Takes the output of expert_profiler and produces a keep/drop mask per layer
under a target retention fraction. Inspired by:

- ISTA-DASLab GSQ-RCO Coder builds that delete ~50% of experts while retaining
  high SWE-bench scores on coding-specialized models
- EASY-EP / few-shot expert localization papers
- DERN (Dropping Experts, Recombining Neurons)
- General observation that MoE routers are highly sparse and domain-skewed

This is still a planning tool. It does not rewrite checkpoints.
Any real pruned model must be re-validated for numerical agreement and task
performance against the original full GLM-5.3.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


def plan_pruning(
    profile: dict[str, Any],
    keep_fraction: float = 0.5,
    min_keep_per_layer: int = 4,
) -> dict[str, Any]:
    """Produce a per-layer keep set from a profiler report."""
    if not 0.05 <= keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in [0.05, 1.0]")
    if profile.get("status") != "PROFILING_ONLY":
        raise ValueError("expected a profile produced by expert_profiler")

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for expert in profile.get("experts", []):
        by_layer[expert["layer"]].append(expert)

    keep_sets: dict[str, list[int]] = {}
    dropped = 0
    kept = 0
    for layer, experts in sorted(by_layer.items()):
        experts.sort(key=lambda e: e["activation_count"], reverse=True)
        n = len(experts)
        target = max(min_keep_per_layer, round(n * keep_fraction))
        target = min(target, n)
        keep_ids = [e["expert"] for e in experts[:target]]
        keep_sets[str(layer)] = keep_ids
        kept += target
        dropped += n - target

    return {
        "status": "PLANNING_ONLY",
        "keep_fraction_requested": keep_fraction,
        "min_keep_per_layer": min_keep_per_layer,
        "layers_processed": len(by_layer),
        "experts_kept": kept,
        "experts_dropped": dropped,
        "keep_sets": keep_sets,
        "warning": (
            "Frequency-based pruning can remove specialists. "
            "ISTA-DASLab and other groups report that domain-specific calibration "
            "data is required, and that final quality must be measured on the target "
            "tasks (CyberGym, ExploitGym, SWE-bench, etc.). This plan does not modify "
            "any checkpoint."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan expert keep/drop sets from an expert_profiler JSON report"
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--min-keep", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.profile.read_text(encoding="utf-8"))
        result = plan_pruning(payload, args.keep_fraction, args.min_keep)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
