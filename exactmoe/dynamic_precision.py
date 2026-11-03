"""Token-level dynamic precision allocation simulator for MoE experts.

Inspired by research on runtime precision control for MoE:
- PagedMoE (arXiv:2611.03173) — cross-precision expert caching with prefix bitplanes
- DPA-MoE — any-precision routed experts via SharedScale AnyBCQ + live router scores
- DynaExq / DyMoE — hotness-aware and importance-aware bit-width allocation

This is a pure Python planning simulator. It does NOT:
- Quantize real weights
- Execute matrix multiplies
- Touch GLM-5.3 checkpoints
- Measure quality or latency

It takes a routed trace + optional per-expert error profiles and assigns a bit-width
(1–8) to each expert demand according to router-style importance and a global budget.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from .traffic_sim import MAX_TRACE_BYTES, validate_trace


def allocate_precision(
    trace: dict[str, Any],
    target_avg_bits: float = 2.5,
    min_bits: int = 1,
    max_bits: int = 8,
) -> dict[str, Any]:
    """Assign integer bit-widths to each expert demand so the average is near target.

    Importance is derived from activation frequency (higher frequency → more bits).
    This is a crude proxy; real systems use router gate scores + offline residual profiles.
    """
    if not (1.0 <= target_avg_bits <= 8.0):
        raise ValueError("target_avg_bits must be in [1, 8]")
    if not (1 <= min_bits <= max_bits <= 8):
        raise ValueError("invalid bit range")

    tokens, kind = validate_trace(trace)
    freq: dict[tuple[int, int], int] = defaultdict(int)
    demands: list[tuple[int, int]] = []
    for token in tokens:
        for layer, expert, _ in token:
            key = (layer, expert)
            freq[key] += 1
            demands.append(key)

    if not demands:
        raise ValueError("no demands")

    # Rank experts by frequency; map rank to bits
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    n_unique = len(ranked)
    bit_of: dict[tuple[int, int], int] = {}
    for i, (key, _) in enumerate(ranked):
        # Linear interpolation from max_bits (hottest) to min_bits (coldest)
        frac = i / max(n_unique - 1, 1)
        bits = round(max_bits - frac * (max_bits - min_bits))
        bit_of[key] = max(min_bits, min(max_bits, bits))

    # Adjust to hit target average (simple scale)
    current_avg = sum(bit_of[k] for k in demands) / len(demands)
    if current_avg > 0:
        scale = target_avg_bits / current_avg
        for key in list(bit_of):
            bit_of[key] = max(min_bits, min(max_bits, round(bit_of[key] * scale)))

    total_bits = sum(bit_of[k] for k in demands)
    avg_bits = total_bits / len(demands)

    canonical = json.dumps(trace, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "status": "SIMULATION_ONLY",
        "source_kind_declared": kind,
        "model_executed": False,
        "weights_modified": False,
        "token_positions": len(tokens),
        "expert_demands": len(demands),
        "unique_experts": n_unique,
        "target_avg_bits": target_avg_bits,
        "achieved_avg_bits": round(avg_bits, 4),
        "min_bits_assigned": min(bit_of.values()),
        "max_bits_assigned": max(bit_of.values()),
        "bit_assignment_sample": [
            {"layer": l, "expert": e, "bits": bit_of[(l, e)]}
            for (l, e) in list(bit_of)[:16]
        ],
        "trace_sha256": hashlib.sha256(canonical).hexdigest(),
        "warning": (
            "This is a frequency-based planning sketch only. Real dynamic precision "
            "(PagedMoE, DPA-MoE, DynaExq) uses router scores, offline quantization-error "
            "profiles, and prefix-compatible encodings. Quality impact is unknown here."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate token-level dynamic bit allocation for MoE experts (planning only)"
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--target-avg-bits", type=float, default=2.5)
    parser.add_argument("--min-bits", type=int, default=1)
    parser.add_argument("--max-bits", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.trace.stat().st_size > MAX_TRACE_BYTES:
            raise ValueError("trace too large")
        payload = json.loads(args.trace.read_text(encoding="utf-8"))
        result = allocate_precision(
            payload,
            target_avg_bits=args.target_avg_bits,
            min_bits=args.min_bits,
            max_bits=args.max_bits,
        )
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
