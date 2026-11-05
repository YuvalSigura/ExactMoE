"""Extreme ideas measured offline on 798 real layer-3 MoE inputs, top-16 experts, vs original FP8.

X1 frozen tunnel: 2/3-bit RTN g64 experts; and the bit-plane idea -- keep only the top 2 bits of the 4-bit
    GPTQ code (offset nibble q in 0..15 -> top plane t = q >> 2, reconstruct (t*4 + 1.5) - 8) times the scale.
X2 gate-first: compute gate (and up) fully? No: read gate only, rank neurons by |silu(gate)|, read up/down
    only for the kept fraction. Error and bytes read (gate 1/3 + kept * 2/3).
X3 merging: cosine similarity between experts' flattened weights (could two experts share one copy?).
Bytes are relative to one 4-bit expert."""
import io, contextlib, json, runpy, sys
from collections import Counter
from pathlib import Path
import numpy as np

S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from exactmoe.storage import TensorIndex  # noqa: E402
X = np.fromfile(S + r"\l3_in.f32", dtype=np.float32).reshape(-1, 6144)
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(S + r"\colibri_int4_quality.py", run_name="lib")
    bd = runpy.run_path(S + r"\bitdeer_probe.py", run_name="lib_bd")
    sp = runpy.run_path(S + r"\split_expert_experiment.py", run_name="lib_split")
silu, route_fn, fetch = q["silu"], q["route_noaux_tc"], bd["fetch"]
routes = [route_fn(x, q["router_w"], q["router_b"], q["config"]) for x in X]
use = Counter(int(e) for r in routes for e in r.expert_ids)
chosen = [e for e, _ in use.most_common(16)]
full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
M = ("gate_proj", "up_proj", "down_proj")

def fp8(e):
    p = f"model.layers.3.mlp.experts.{e}."
    return [sp["dequant"](full[p + m + ".weight"], full[p + m + ".weight_scale_inv"]) for m in M]
def rtn(w, bits, gs=64):
    O, I = w.shape; qmax = (1 << (bits - 1)) - 1
    wr = w.reshape(O, I // gs, gs); s = np.maximum(np.abs(wr).max(axis=2, keepdims=True) / qmax, 1e-8)
    return (np.clip(np.rint(wr / s), -qmax - 1, qmax) * s).reshape(O, I).astype(np.float32)
def gptq_planes(e, m):
    """Return (full 4-bit GPTQ, top-2-bit-plane approximation)."""
    packed = fetch(f"model.layers.3.mlp.experts.{e}.{m}.weight").astype(np.uint8)
    sc = fetch(f"model.layers.3.mlp.experts.{e}.{m}.weight_scale_inv")
    off = packed ^ np.uint8(0x88)                                  # offset nibbles 0..15
    lo, hi = (off & 0xF).astype(np.float32), (off >> 4).astype(np.float32)
    O = packed.shape[0]; qv = np.empty((O, packed.shape[1] * 2), np.float32); qv[:, 0::2] = lo; qv[:, 1::2] = hi
    scale = np.repeat(sc, qv.shape[1] // sc.shape[1], axis=1)
    full4 = (qv - 8) * scale
    top2 = (np.floor(qv / 4) * 4 + 1.5 - 8) * scale
    return full4, top2
def fwd(W, x):
    G, U, D = W; return D @ (silu(G @ x) * (U @ x))

res = {k: [] for k in ("rtn2", "rtn3", "rtn4", "gptq4", "gptq_top2plane", "gate50", "gate30", "gate20")}
sims = []
for e in chosen:
    We = fp8(e)
    W2 = [rtn(w, 2) for w in We]; W3 = [rtn(w, 3) for w in We]; W4 = [rtn(w, 4) for w in We]
    pl = [gptq_planes(e, m) for m in M]; Wg = [p[0] for p in pl]; Wt = [p[1] for p in pl]
    rows = [i for i, r in enumerate(routes) if e in [int(v) for v in r.expert_ids]]
    for i in rows:
        x = X[i]; ref = fwd(We, x); n = float(np.linalg.norm(ref)) or 1.0
        for k, W in (("rtn2", W2), ("rtn3", W3), ("rtn4", W4), ("gptq4", Wg), ("gptq_top2plane", Wt)):
            res[k].append(float(np.linalg.norm(fwd(W, x) - ref)) / n)
        G, U, D = Wg; g = G @ x; a = np.abs(silu(g))                 # gate-first on top of GPTQ 4-bit
        for frac, k in ((0.5, "gate50"), (0.3, "gate30"), (0.2, "gate20")):
            keep = np.argsort(-a)[: int(frac * a.size)]
            y = D[:, keep] @ (silu(g[keep]) * (U[keep] @ x))
            res[k].append(float(np.linalg.norm(y - ref)) / n)
    sims.append(We[0].reshape(-1))
sims = np.array(sims); sims /= np.linalg.norm(sims, axis=1, keepdims=True); C = sims @ sims.T
off_diag = C[~np.eye(len(chosen), dtype=bool)]
out = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
out["bytes_vs_4bit"] = {"rtn2": 0.5 + 0.0, "rtn3": 0.75, "gptq_top2plane": 0.5, "gate50": round(1/3 + 0.5*2/3, 3), "gate30": round(1/3 + 0.3*2/3, 3), "gate20": round(1/3 + 0.2*2/3, 3)}
out["expert_gate_cosine"] = {"mean_abs": round(float(np.mean(np.abs(off_diag))), 4), "max": round(float(off_diag.max()), 4)}
print(json.dumps(out, indent=1))
Path(S + r"\extreme_offline.json").write_text(json.dumps(out, indent=1))
