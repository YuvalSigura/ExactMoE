"""Expert importance profiler for MoE domain specialization planning.

This module computes per-expert activation frequency and optional output-variance
proxies from a routed-expert trace. It is intended as a *planning tool* for
research-style expert pruning or specialization (inspired by EASY-EP, DERN,
ISTA-DASLab coding-specialized pruning, and related papers).

It does NOT:
- Modify any model weights
- Run GLM-5.3 or any real model
- Guarantee that pruning will preserve accuracy
- Authenticate that a trace came from the official router

Use the output only as a starting point for offline analysis. Any actual pruning
must be validated with independent numerical and task-level checks against the
original model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from .traffic_sim import MAX_TRACE_BYTES, validate_trace


def profile_experts(trace: dict[str, Any]) -> dict[str, Any]:
    """Return per-(layer, expert) frequency and basic statistics from a valid trace."""
    tokens, kind = validate_trace(trace)
    freq: Counter[tuple[int, int]] = Counter()
    layer_counts: Counter[int] = Counter()
    total_demands = 0
    for token in tokens:
        for layer, expert, _size in token:
            key = (layer, expert)
            freq[key] += 1
            layer_counts[layer] += 1
            total_demands += 1

    # Simple importance score: frequency normalized by layer activity
    scores: list[dict[str, Any]] = []
    for (layer, expert), count in sorted(freq.items()):
        layer_total = layer_counts[layer]
        rel = count / layer_total if layer_total else 0.0
        scores.append({
            "layer": layer,
            "expert": expert,
            "activation_count": count,
            "relative_in_layer": round(rel, 6),
            "global_share": round(count / total_demands, 6) if total_demands else 0.0,
        })

    # Rank within each layer
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        by_layer[s["layer"]].append(s)
    for layer_scores in by_layer.values():
        layer_scores.sort(key=lambda x: x["activation_count"], reverse=True)
        for rank, s in enumerate(layer_scores, 1):
            s["rank_in_layer"] = rank

    canonical = json.dumps(trace, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "status": "PROFILING_ONLY",
        "source_kind_declared": kind,
        "source_authenticated": False,
        "model_executed": False,
        "token_positions": len(tokens),
        "total_expert_demands": total_demands,
        "unique_experts": len(freq),
        "experts": scores,
        "trace_sha256": hashlib.sha256(canonical).hexdigest(),
        "warning": (
            "Activation frequency is a weak proxy for expert importance. "
            "Papers (e.g. Black Sheep AI research, EASY-EP, DERN) show that "
            "low-frequency experts can still be critical for niche tasks. "
            "Any pruning decision requires full task evaluation on the original model."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile expert activation frequency from a routed trace (simulation/planning only)"
    )
    parser.add_argument("--trace", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.trace.stat().st_size > MAX_TRACE_BYTES:
            raise ValueError("trace too large")
        payload = json.loads(args.trace.read_text(encoding="utf-8"))
        result = profile_experts(payload)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
