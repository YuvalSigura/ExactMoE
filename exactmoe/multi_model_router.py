"""Multi-model routing skeleton ("Cyber Model Orchestra" style).

The idea, discussed for approximating full GLM-5.3 capabilities on constrained
hardware, is to route different task phases to different models:

- Fast small models for discovery / triage
- Medium specialized models (possibly pruned + quantized GLM-5.3-Flash or Qwen) for deep analysis
- Full GLM-5.3 (or API) only for the hardest exploit-chaining / long-horizon steps

This module provides a pure-Python decision function and a small CLI that can
be extended. It does not call any model, does not download weights, and does
not claim quality parity with GLM-5.3.

Research context: hybrid routing + distillation + domain pruning (ISTA-DASLab
style) + dynamic offloading (Colibri, FreeToken, KTransformers, Swap-MoE).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PHASES = ("discovery", "triage", "deep_analysis", "exploit_chain", "verification")


def route_phase(phase: str, difficulty: float = 0.5, prefer_local: bool = True) -> dict[str, Any]:
    """Return a recommended model tier for a named task phase.

    difficulty: 0.0 (easy) … 1.0 (hardest / longest horizon)
    prefer_local: if True, avoid recommending the full remote model unless necessary
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")

    # Simple policy; real systems would use a learned router or LLM-as-judge
    if phase in ("discovery", "triage") or difficulty < 0.35:
        tier = "fast_local"
        suggestion = "9B–14B dense or small MoE (e.g. Qwen3.6-14B / 27B class) at Q4–Q5"
    elif phase == "deep_analysis" or difficulty < 0.7:
        tier = "specialized_local"
        suggestion = (
            "Pruned + low-bit GLM-5.3-Flash or Qwen3.8-Flash-Next style MoE "
            "with Colibri/FreeToken/KTransformers offload (target ~100–150 GiB resident)"
        )
    else:
        if prefer_local and difficulty < 0.85:
            tier = "heavy_local"
            suggestion = (
                "Full GLM-5.3 with extreme 1.5–2-bit + expert pruning + NVMe streaming "
                "(Colibri / PagedMoE-style); expect very low tok/s"
            )
        else:
            tier = "full_remote"
            suggestion = "Official GLM-5.3 API or a verified full-precision local run on larger hardware"

    return {
        "status": "ROUTING_SKELETON",
        "phase": phase,
        "difficulty": difficulty,
        "prefer_local": prefer_local,
        "recommended_tier": tier,
        "model_suggestion": suggestion,
        "note": (
            "This is a policy sketch only. Actual quality depends on the concrete models, "
            "quantization, pruning, and verification against original GLM-5.3 outputs."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cyber Model Orchestra style phase router (decision skeleton only)"
    )
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--prefer-local", action="store_true", default=True)
    parser.add_argument("--no-prefer-local", dest="prefer_local", action="store_false")
    args = parser.parse_args(argv)
    try:
        result = route_phase(args.phase, args.difficulty, args.prefer_local)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
