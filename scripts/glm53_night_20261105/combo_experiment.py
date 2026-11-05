"""Combinations of size-reduction ideas on the 8 REAL layer-3 experts of GLM-5.3 (router-selected), against exact FP8.

Each variant reports: bytes relative to FP8, and the relative error of the routed MoE output (8 experts, routing weights).
INPUT IS A PROXY (synthetic N(0, 0.1), the only proxy that routes to the 8 experts we have). Real hidden states may differ.
Hot/cold neuron sets are chosen per input here (an oracle); a real system would have to PREDICT them.
"""
import json
import runpy
from pathlib import Path

import numpy as np

S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
q = runpy.run_path(S + r"\colibri_int4_quality.py", run_name="lib")        # loads experts (exact FP8 dequant) + colibri int4 gs64
experts, router_w, router_b, config, route_fn, silu = q["experts"], q["router_w"], q["router_b"], q["config"], q["route_noaux_tc"], q["silu"]
quant_g, dequant_g = q["quant_int4_grouped"], q["dequant_int4_grouped"]
x = q["inputs"]["A_gaussian"]
route = route_fn(x, router_w, router_b, config)
FP8_BYTES_PER_W = 1.0 + 4.0 / (128 * 128)


def q_int_rowwise(w, bits):
    qmax = (1 << (bits - 1)) - 1
    s = np.maximum(np.abs(w).max(axis=1, keepdims=True) / qmax, 1e-8)
    return (np.clip(np.rint(w / s), -qmax - 1, qmax) * s).astype(np.float32), bits / 8 + 4 / w.shape[1]


def q_int_grouped(w, bits, gs):
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    wr = w.reshape(O, I // gs, gs)
    s = np.maximum(np.abs(wr).max(axis=2, keepdims=True) / qmax, 1e-8)
    return (np.clip(np.rint(wr / s), -qmax - 1, qmax) * s).reshape(O, I).astype(np.float32), bits / 8 + 4 / gs


def colibri_int4(w, gs=64):
    qb, s = quant_g(w, 4, gs)
    return dequant_g(qb, s, *w.shape, gs), 0.5 + 4 / gs


def build(variant):
    """Return {eid: (G, U, D)} as used for computation, and the stored bytes per FP8 byte."""
    out, stored, total = {}, 0.0, 0.0
    for eid, m in experts.items():
        mats = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            w = m[name][0]
            bpw = FP8_BYTES_PER_W
            if variant.get(name) == "fp8":
                wq = w
            elif variant.get(name) == "int8":
                wq, bpw = q_int_rowwise(w, 8)
            elif variant.get(name) == "int4g64":
                wq, bpw = colibri_int4(w, 64)
            elif variant.get(name) == "int4g32":
                wq, bpw = q_int_grouped(w, 4, 32)
            elif variant.get(name) == "int3g64":
                wq, bpw = q_int_grouped(w, 3, 64)
            elif variant.get(name) == "int2g64":
                wq, bpw = q_int_grouped(w, 2, 64)
            mats.append(wq)
            stored += bpw * w.size
            total += FP8_BYTES_PER_W * w.size
        out[eid] = mats
    return out, stored / total


exact = {eid: (m["gate_proj"][0], m["up_proj"][0], m["down_proj"][0]) for eid, m in experts.items()}


def moe(weights, skip_frac=0.0, hot_fp8_frac=0.0):
    y = np.zeros_like(x)
    for eid, rw in zip(route.expert_ids, route.weights):
        G, U, D = weights[eid]
        h = silu(G @ x) * (U @ x)
        if hot_fp8_frac:                                   # hot neurons computed from the exact FP8 rows
            Ge, Ue, De = exact[eid]
            he = silu(Ge @ x) * (Ue @ x)
            hot = np.argsort(-np.abs(he) * np.linalg.norm(De, axis=0))[: int(hot_fp8_frac * 2048)]
            h = h.copy(); h[hot] = he[hot]
            out = D @ h + (De[:, hot] - D[:, hot]) @ h[hot]
        else:
            out = D @ h
        if skip_frac:                                      # drop the weakest neurons (by contribution)
            contrib = np.abs(h) * np.linalg.norm(D, axis=0)
            cold = np.argsort(contrib)[: int(skip_frac * 2048)]
            out = out - D[:, cold] @ h[cold]
        y += np.float32(rw) * out
    return y


ref = moe(exact)
variants = {
    "FP8 original": {},
    "int8 rowwise (all)": {"gate_proj": "int8", "up_proj": "int8", "down_proj": "int8"},
    "int4 g64 (Colibri default)": {"gate_proj": "int4g64", "up_proj": "int4g64", "down_proj": "int4g64"},
    "int4 g32": {"gate_proj": "int4g32", "up_proj": "int4g32", "down_proj": "int4g32"},
    "gate/up int4g64 + down int8": {"gate_proj": "int4g64", "up_proj": "int4g64", "down_proj": "int8"},
    "gate/up int8 + down int4g64": {"gate_proj": "int8", "up_proj": "int8", "down_proj": "int4g64"},
    "int3 g64": {"gate_proj": "int3g64", "up_proj": "int3g64", "down_proj": "int3g64"},
    "gate/up int3g64 + down int4g64": {"gate_proj": "int3g64", "up_proj": "int3g64", "down_proj": "int4g64"},
    "int2 g64": {"gate_proj": "int2g64", "up_proj": "int2g64", "down_proj": "int2g64"},
}
rows = []
for name, v in variants.items():
    w, ratio = build(v) if v else (exact, 1.0)
    for skip in (0.0, 0.25, 0.5):
        for hot in (0.0, 0.25):
            if v == {} and hot:
                continue
            y = moe(w, skip_frac=skip, hot_fp8_frac=hot)
            size = ratio * (1 - skip * 0.0) + (hot * (1.0 - ratio) if hot else 0.0)   # hot neurons stored twice: +hot*(fp8 - quant)
            eff = size * (1 - skip / 3 if skip else 1)                                 # skipping saves down_proj columns only (1/3 of bytes)
            rows.append({"variant": name, "skip_weak": skip, "hot_fp8": hot, "stored_vs_fp8": round(size, 3),
                         "read_per_token_vs_fp8": round(eff, 3),
                         "moe_rel_err": round(float(np.linalg.norm(y - ref) / np.linalg.norm(ref)), 4)})
for r in sorted(rows, key=lambda r: (r["moe_rel_err"])):
    print(f"{r['variant']:32s} skip={r['skip_weak']:.2f} hot_fp8={r['hot_fp8']:.2f}  stored={r['stored_vs_fp8']:.3f}  read/token={r['read_per_token_vs_fp8']:.3f}  err={r['moe_rel_err']:.4f}")
Path(S + r"\combo_experiment.json").write_text(json.dumps(rows, indent=1))
