"""Controlled arithmetic test of Layer 3 router + one acquired expert + shared expert.

This is NOT a captured production trace. It uses a fixed synthetic hidden state
to exercise the original router weights, selection logic, the already-acquired
expert 0 (if selected), and the shared expert. It does not invent expert
activation frequencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .moe_layer import forward_moe_ffn, route_noaux_tc, read_dense_vector_or_matrix
from .storage import TensorIndex


def main(argv=None):
    p = argparse.ArgumentParser(description="Layer 3 MoE arithmetic smoke (not a production trace)")
    p.add_argument("expert_dir", type=Path, help="directory from expert_acquire (expert 0)")
    p.add_argument("control_dir", type=Path, help="directory from layer3_control_acquire")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    # Merge the two samples into a temporary view by loading both indexes.
    # For this smoke we only need the control tensors + expert 0.
    config = json.loads((a.control_dir / "config.json").read_text())
    hidden = int(config["hidden_size"])
    x = np.random.default_rng(a.seed).normal(0, 0.1, hidden).astype(np.float32)

    # Load router from control sample
    control_index = TensorIndex.from_directory(a.control_dir)
    prefix = "model.layers.3.mlp."
    w = read_dense_vector_or_matrix(control_index[prefix + "gate.weight"])
    b = read_dense_vector_or_matrix(control_index[prefix + "gate.e_score_correction_bias"])
    route = route_noaux_tc(x, w, b, config)

    result = {
        "status": "CONTROLLED_ARITHMETIC_TEST",
        "layer": 3,
        "seed": a.seed,
        "selected_experts": list(route.expert_ids),
        "routing_weights": list(route.weights),
        "expert_0_selected": 0 in route.expert_ids,
        "note": (
            "Fixed synthetic hidden state. Not a captured production trace. "
            "Only expert 0 weights are present; other selected experts are not executed."
        ),
    }

    if 0 in route.expert_ids:
        # Run the already-acquired expert 0
        from .expert_validation import names
        from .fp8_reference import expert_values
        expert_index = TensorIndex.from_selected(
            a.expert_dir,
            [n for pair in names(3, 0).values() for n in pair],
        )
        out0 = expert_values(expert_index, 3, 0, x, device="cpu")
        result["expert_0_output_sha256"] = __import__("hashlib").sha256(out0.tobytes()).hexdigest()
        result["expert_0_output_norm"] = float(np.linalg.norm(out0))
    else:
        result["expert_0_skipped"] = "not selected by this synthetic input"

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
