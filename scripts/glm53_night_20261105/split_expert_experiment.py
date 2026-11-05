"""Splitting one GLM-5.3 expert into sub-experts: exactness, concentration, certified skipping.

Read-only on the original FP8 bytes of layer 3 expert 222 (C:\\AI-Models\\GLM53-experts). Nothing is written to the model.
INPUTS ARE PROXIES, not real layer-3 hidden states (those need the attention path, which does not exist yet):
  A: N(0, 0.1) as in ExactMoE's own layer-3 test;  B: RMS-normalized token-embedding rows of the real checkpoint.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from exactmoe.fp8_reference import e4m3fn_table, expert_values  # noqa: E402
from exactmoe.storage import TensorIndex, scan_safetensors  # noqa: E402
from exactmoe.expert_validation import names as expert_names  # noqa: E402

EXPERT_DIR = Path(r"C:\AI-Models\GLM53-experts\expert-222")
FULL = Path(r"C:\AI-Models\GLM-5.3-full")
LAYER, EXPERT, BLOCK = 3, 222, 128
TABLE = e4m3fn_table()


def raw(rec) -> bytes:
    with open(rec.path, "rb") as fh:
        fh.seek(rec.start)
        return fh.read(rec.nbytes)


def dequant(weight, scale) -> np.ndarray:
    w = TABLE[np.frombuffer(raw(weight), dtype=np.uint8)].reshape(weight.shape)
    s = np.frombuffer(raw(scale), dtype=np.float32).reshape(scale.shape)
    return (w * np.kron(s, np.ones((BLOCK, BLOCK), dtype=np.float32))[: w.shape[0], : w.shape[1]]).astype(np.float32)


recs = scan_safetensors(EXPERT_DIR / "expert.safetensors")
p = f"model.layers.{LAYER}.mlp.experts.{EXPERT}."
G = dequant(recs[p + "gate_proj.weight"], recs[p + "gate_proj.weight_scale_inv"])   # (2048, 6144)
U = dequant(recs[p + "up_proj.weight"], recs[p + "up_proj.weight_scale_inv"])       # (2048, 6144)
D = dequant(recs[p + "down_proj.weight"], recs[p + "down_proj.weight_scale_inv"])   # (6144, 2048)
silu = lambda z: z / (1.0 + np.exp(-z))                                             # noqa: E731
dcol = np.linalg.norm(D, axis=0)                                                    # ||D[:, i]|| per neuron (2048 floats: 8 KiB)


def embedding_rows(ids):
    index = json.loads((FULL / "model.safetensors.index.json").read_text())["weight_map"]
    shard = FULL / index["model.embed_tokens.weight"]
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        meta = header["model.embed_tokens.weight"]
        rows, cols = meta["shape"]
        base = 8 + n + meta["data_offsets"][0]
        out = []
        for t in ids:
            fh.seek(base + t * cols * 2)
            bf = np.frombuffer(fh.read(cols * 2), dtype=np.uint16).astype(np.uint32) << 16
            v = bf.view(np.float32)
            out.append(v / np.sqrt(np.mean(v * v)))
    return out


inputs = {"A_gaussian": np.random.default_rng(42).normal(0, 0.1, 6144).astype(np.float32)}
for i, row in enumerate(embedding_rows([1000, 23456, 77777, 120001])):
    inputs[f"B_embedding_{i}"] = row.astype(np.float32)

# 0) my dequantization agrees with ExactMoE's own expert computation (same original bytes)
idx = TensorIndex.from_selected(EXPERT_DIR, [n for pair in expert_names(LAYER, EXPERT).values() for n in pair])
x0 = inputs["A_gaussian"]
mine = D @ (silu(G @ x0) * (U @ x0))
theirs = expert_values(idx, LAYER, EXPERT, x0, device="cpu")
report = {"crosscheck_vs_exactmoe_rel_err": float(np.linalg.norm(mine - theirs) / np.linalg.norm(theirs))}

# 1) exactness of the split into 16 sub-experts of 128 neurons
h = silu(G @ x0) * (U @ x0)
full = D @ h
parts = sum(D[:, c * BLOCK:(c + 1) * BLOCK] @ h[c * BLOCK:(c + 1) * BLOCK] for c in range(2048 // BLOCK))
report["split16_max_abs_diff"] = float(np.max(np.abs(full - parts)))
report["split16_rel_diff"] = float(np.linalg.norm(full - parts) / np.linalg.norm(full))
report["sub_expert_bytes"] = (3 * 6144 * BLOCK) + 3 * 48  # gate+up+down slices of one 128-neuron group (FP8) + its scales

# 2) concentration and 3) certified skipping (exact bound: ||y - y_kept|| <= sum_skipped |h_i| * ||D[:, i]||)
for name, x in inputs.items():
    h = silu(G @ x) * (U @ x)
    y = D @ h
    ny = float(np.linalg.norm(y))
    contrib = np.abs(h) * dcol
    order = np.argsort(contrib)
    total = contrib.sum()
    srt = np.sort(contrib)[::-1]
    row = {"top10pct_share": float(srt[:205].sum() / total), "top25pct_share": float(srt[:512].sum() / total)}
    for eps in (0.01, 0.05, 0.10):
        cum = np.cumsum(contrib[order])
        k = int(np.searchsorted(cum, eps * ny, side="right"))
        keep = np.ones(2048, bool)
        keep[order[:k]] = False
        err = float(np.linalg.norm(y - D[:, keep] @ h[keep]) / ny)
        group_bound = np.array([contrib[c * BLOCK:(c + 1) * BLOCK].sum() for c in range(16)])
        gk = int(np.searchsorted(np.cumsum(np.sort(group_bound)), eps * ny, side="right"))
        row[f"eps{int(eps * 100)}"] = {"down_neurons_skipped": k, "down_skipped_frac": k / 2048, "actual_rel_err": err,
                                       "bound_rel": float(cum[k - 1] / ny) if k else 0.0, "whole_groups_skipped_of16": gk}
    report[name] = row

print(json.dumps(report, indent=1))
