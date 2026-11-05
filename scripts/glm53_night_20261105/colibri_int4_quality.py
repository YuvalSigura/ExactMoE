"""Quality of Colibri's int4 gs64 expert quantization on REAL GLM-5.3 weights, against the exact FP8 reference.

quant_int4_grouped is copied verbatim (math only) from colibri c/tools/convert_fp8_to_int4.py @ 0a5504a1; the dequantization
is its inverse (nibble - 8, times the group scale). The reference is the original block-FP8 expert, dequantized exactly
(cross-checked against ExactMoE in split_expert_experiment.py: 5e-7). Inputs are PROXIES (no real hidden states yet).
"""
import json
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from exactmoe.storage import TensorIndex, scan_safetensors  # noqa: E402
from exactmoe.moe_layer import route_noaux_tc, read_dense_vector_or_matrix  # noqa: E402

S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
lib = runpy.run_path(S + r"\split_expert_experiment.py", run_name="lib")
dequant, silu, inputs = lib["dequant"], lib["silu"], lib["inputs"]


def quant_int4_grouped(w, bits, gs=128):                    # verbatim math from colibri
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    ngroups = (I + gs - 1) // gs
    Ipad = ngroups * gs
    wpad = np.zeros((O, Ipad), np.float32)
    wpad[:, :I] = w
    wr = wpad.reshape(O, ngroups, gs)
    amax = np.abs(wr).max(axis=2, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(wr / s), -8, qmax).astype(np.int32)
    q = q.reshape(O, Ipad)[:, :I]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, :, 0].astype(np.float32).reshape(-1)


def dequant_int4_grouped(qb, s, O, I, gs):
    packed = qb.reshape(O, (I + 1) // 2)
    q = np.empty((O, I), np.int32)
    q[:, 0::2] = (packed & 0x0F).astype(np.int32)[:, : (I + 1) // 2] - 8
    q[:, 1::2] = (packed >> 4).astype(np.int32)[:, : I // 2] - 8
    scale = np.repeat(s.reshape(O, -1), gs, axis=1)[:, :I]
    return (q * scale).astype(np.float32)


def roundtrip(w, gs=64):
    qb, s = quant_int4_grouped(w, 4, gs)
    return dequant_int4_grouped(qb, s, *w.shape, gs), qb.nbytes + s.nbytes


root = Path(r"C:\AI-Models\GLM53-experts")
control = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM53-layer3-control"))
config = json.loads(Path(r"C:\AI-Models\GLM53-layer3-control\config.json").read_text())
pre = "model.layers.3.mlp."
router_w = read_dense_vector_or_matrix(control[pre + "gate.weight"])
router_b = read_dense_vector_or_matrix(control[pre + "gate.e_score_correction_bias"])

experts, matrix_stats, int4_bytes, fp8_bytes = {}, [], 0, 0
for d in sorted(root.glob("expert-*")):
    eid = int(d.name.split("-")[1])
    recs = scan_safetensors(d / "expert.safetensors")
    p = f"model.layers.3.mlp.experts.{eid}."
    mats = {}
    for m in ("gate_proj", "up_proj", "down_proj"):
        w = dequant(recs[p + m + ".weight"], recs[p + m + ".weight_scale_inv"])
        wq, nb = roundtrip(w)
        int4_bytes += nb
        fp8_bytes += recs[p + m + ".weight"].nbytes + recs[p + m + ".weight_scale_inv"].nbytes
        matrix_stats.append(float(np.linalg.norm(w - wq) / np.linalg.norm(w)))
        mats[m] = (w, wq)
    experts[eid] = mats


def ffn(m, x, which):
    g, u, dn = (m[k][which] for k in ("gate_proj", "up_proj", "down_proj"))
    return dn @ (silu(g @ x) * (u @ x))


report = {"experts": sorted(experts), "int4_gs64_bytes": int4_bytes, "fp8_bytes": fp8_bytes,
          "size_ratio": int4_bytes / fp8_bytes,
          "matrix_rel_frobenius_err": {"mean": float(np.mean(matrix_stats)), "max": float(np.max(matrix_stats))}}
for name, x in inputs.items():
    route = route_noaux_tc(x, router_w, router_b, config)
    exact = np.zeros_like(x)
    quant = np.zeros_like(x)
    per_expert = []
    for eid, rw in zip(route.expert_ids, route.weights):
        if eid not in experts:
            continue
        ye, yq = ffn(experts[eid], x, 0), ffn(experts[eid], x, 1)
        exact += np.float32(rw) * ye
        quant += np.float32(rw) * yq
        per_expert.append(float(np.linalg.norm(ye - yq) / np.linalg.norm(ye)))
    cos = float(exact @ quant / (np.linalg.norm(exact) * np.linalg.norm(quant)))
    report[name] = {"routed_to": list(route.expert_ids), "experts_available": len(per_expert),
                    "expert_out_rel_err_mean": float(np.mean(per_expert)) if per_expert else None,
                    "moe_out_rel_err": float(np.linalg.norm(exact - quant) / np.linalg.norm(exact)) if per_expert else None,
                    "moe_out_cosine": cos if per_expert else None}
print(json.dumps(report, indent=1))
