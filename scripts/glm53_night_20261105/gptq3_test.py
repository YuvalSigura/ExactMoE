"""Own GPTQ at 3 and 4 bits (g64, symmetric) for gate/up of layer-3 experts, calibrated on the first 600 real
inputs (layer-level Hessian H = X^T X: every expert sees the same input distribution) and scored on the
other 198 real inputs only. down_proj stays Bitdeer GPTQ 4-bit (its inputs are per-expert and too few here).
Arms are compared on the same held-out expert-token pairs against the original FP8 expert."""
import io, contextlib, json, runpy, sys
from collections import Counter
from pathlib import Path
import numpy as np

S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from exactmoe.storage import TensorIndex  # noqa: E402
X = np.fromfile(S + r"\l3_in.f32", dtype=np.float32).reshape(-1, 6144)
CAL, HOLD = X[:600], X[600:]
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(S + r"\colibri_int4_quality.py", run_name="lib")
    bd = runpy.run_path(S + r"\bitdeer_probe.py", run_name="lib_bd")
    sp = runpy.run_path(S + r"\split_expert_experiment.py", run_name="lib_split")
silu, route_fn, fetch, deq = q["silu"], q["route_noaux_tc"], bd["fetch"], bd["deq"]
full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))

def gptq(W, H, bits, gs=64, block=128, damp=0.01):
    """Standard GPTQ (Frantar et al.) with per-row symmetric group scales chosen from the original weights."""
    W = W.astype(np.float64).copy(); O, I = W.shape; qmax = (1 << (bits - 1)) - 1
    H = H.astype(np.float64).copy(); dead = np.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * np.mean(np.diag(H)) * np.eye(I)
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T              # upper Cholesky of H^-1
    scale = np.maximum(np.abs(W.reshape(O, I // gs, gs)).max(axis=2) / qmax, 1e-12)   # [O, I/gs]
    Q = np.zeros_like(W)
    for i1 in range(0, I, block):
        i2 = min(i1 + block, I); W1 = W[:, i1:i2].copy(); Err = np.zeros_like(W1); Hb = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j; s = scale[:, col // gs]
            qcol = np.clip(np.rint(W1[:, j] / s), -qmax - 1, qmax) * s
            Q[:, col] = qcol; err = (W1[:, j] - qcol) / Hb[j, j]
            W1[:, j:] -= np.outer(err, Hb[j, j:]); Err[:, j] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q.astype(np.float32)
def rtn(w, bits, gs=64):
    O, I = w.shape; qmax = (1 << (bits - 1)) - 1
    wr = w.reshape(O, I // gs, gs); s = np.maximum(np.abs(wr).max(axis=2, keepdims=True) / qmax, 1e-8)
    return (np.clip(np.rint(wr / s), -qmax - 1, qmax) * s).reshape(O, I).astype(np.float32)
def fwd(G, U, D, x): return D @ (silu(G @ x) * (U @ x))

H = CAL.T.astype(np.float64) @ CAL.astype(np.float64)
routes = [route_fn(x, q["router_w"], q["router_b"], q["config"]) for x in X]
hold_routes = routes[600:]
use = Counter(int(e) for r in hold_routes for e in r.expert_ids)
chosen = [e for e, _ in use.most_common(8)]
res = {k: [] for k in ("rtn4_all", "rtn3_gu", "gptq3_gu_own", "gptq4_gu_own", "bitdeer_gptq4")}
for e in chosen:
    p = f"model.layers.3.mlp.experts.{e}."
    G, U, D = (sp["dequant"](full[p + m + ".weight"], full[p + m + ".weight_scale_inv"]) for m in ("gate_proj", "up_proj", "down_proj"))
    bdw = {m: deq(fetch(p + m + ".weight"), fetch(p + m + ".weight_scale_inv"), True, True) for m in ("gate_proj", "up_proj", "down_proj")}
    arms = {
        "rtn4_all": (rtn(G, 4), rtn(U, 4), rtn(D, 4)),
        "rtn3_gu": (rtn(G, 3), rtn(U, 3), bdw["down_proj"]),
        "gptq3_gu_own": (gptq(G, H, 3), gptq(U, H, 3), bdw["down_proj"]),
        "gptq4_gu_own": (gptq(G, H, 4), gptq(U, H, 4), bdw["down_proj"]),
        "bitdeer_gptq4": (bdw["gate_proj"], bdw["up_proj"], bdw["down_proj"]),
    }
    rows = [i for i, r in enumerate(hold_routes) if e in [int(v) for v in r.expert_ids]]
    for i in rows:
        x = HOLD[i]; ref = fwd(G, U, D, x); n = float(np.linalg.norm(ref)) or 1.0
        for k, (g, u, d) in arms.items():
            res[k].append(float(np.linalg.norm(fwd(g, u, d, x) - ref)) / n)
    print(f"expert {e}: {len(rows)} held-out rows", {k: round(float(np.mean(v[-len(rows):])), 4) for k, v in res.items()}, flush=True)
out = {k: {"mean_rel_err": round(float(np.mean(v)), 4), "n": len(v)} for k, v in res.items()}
out["bytes_vs_4bit"] = {"rtn3_gu": round(2/3 * 0.75 + 1/3, 3), "gptq3_gu_own": round(2/3 * 0.75 + 1/3, 3)}
print(json.dumps(out, indent=1))
Path(S + r"\gptq3_test.json").write_text(json.dumps(out, indent=1))
