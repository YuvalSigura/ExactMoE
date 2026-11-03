"""Full Layer 3 MoE FFN arithmetic test on acquired original weights.

Loads the router + shared expert from the control sample and the 8 experts
that were selected by a fixed synthetic hidden state. Runs the original
routing, expert FFNs and shared expert. Not a production trace and not
full-model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .expert_validation import names as expert_names, _silu
from .fp8_reference import expert_values, block_fp8_matvec
from .moe_layer import route_noaux_tc, read_dense_vector_or_matrix
from .storage import TensorIndex


def main(argv=None):
    p = argparse.ArgumentParser(description="Layer 3 MoE FFN on acquired experts (controlled test)")
    p.add_argument("control_dir", type=Path)
    p.add_argument("experts_root", type=Path)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    config = json.loads((a.control_dir / "config.json").read_text())
    hidden = int(config["hidden_size"])
    x = np.random.default_rng(a.seed).normal(0, 0.1, hidden).astype(np.float32)

    control = TensorIndex.from_directory(a.control_dir)
    prefix = "model.layers.3.mlp."
    w = read_dense_vector_or_matrix(control[prefix + "gate.weight"])
    b = read_dense_vector_or_matrix(control[prefix + "gate.e_score_correction_bias"])
    route = route_noaux_tc(x, w, b, config)

    result = np.zeros_like(x)
    for eid, rw in zip(route.expert_ids, route.weights):
        edir = a.experts_root / f"expert-{eid}"
        idx = TensorIndex.from_selected(
            edir, [n for pair in expert_names(3, eid).values() for n in pair]
        )
        result += np.float32(rw) * expert_values(idx, 3, eid, x, device="cpu")

    sp = prefix + "shared_experts."
    gate = block_fp8_matvec(control[sp + "gate_proj.weight"],
                            control[sp + "gate_proj.weight_scale_inv"], x)
    up = block_fp8_matvec(control[sp + "up_proj.weight"],
                          control[sp + "up_proj.weight_scale_inv"], x)
    product = (_silu(gate) * up).astype(np.float32)
    shared = block_fp8_matvec(control[sp + "down_proj.weight"],
                              control[sp + "down_proj.weight_scale_inv"], product)
    result += shared

    report = {
        "status": "LAYER3_MOE_FFN_ARITHMETIC_TEST",
        "layer": 3,
        "seed": a.seed,
        "selected_experts": list(route.expert_ids),
        "routing_weights": list(route.weights),
        "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "output_norm": float(np.linalg.norm(result)),
        "official_glm53_kernel_parity": "NOT_TESTED",
        "full_glm53_inference": False,
        "note": "Controlled synthetic input. Not a captured production trace.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
