"""Exp A (sparse repair, plain vs activation-aware) and a NON-oracle hot-neuron selection, layer-3 real experts.
Calibration inputs (seeds 1-4) choose what to repair; holdout inputs (seeds 5-8) score it. Proxy inputs, fixed routing."""
import runpy, io, contextlib, numpy as np
S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(S + r"\colibri_int4_quality.py", run_name="lib")
experts, silu = q["experts"], q["silu"]
x0 = q["inputs"]["A_gaussian"]
route = q["route_noaux_tc"](x0, q["router_w"], q["router_b"], q["config"])
RW = {int(e): float(w) for e, w in zip(route.expert_ids, route.weights)}
M = ("gate_proj", "up_proj", "down_proj")
cal = [np.random.default_rng(s).normal(0, 0.1, 6144).astype(np.float32) for s in (1, 2, 3, 4)]
hold = [np.random.default_rng(s).normal(0, 0.1, 6144).astype(np.float32) for s in (5, 6, 7, 8)]

def moe(W, x, hot=None):
    y = np.zeros(6144, np.float32)
    for e, rw in RW.items():
        G, U, D = W[e]
        h = silu(G @ x) * (U @ x)
        if hot:  # NON-oracle: choose hot neurons from the int4 activations, then recompute them exactly from FP8
            Ge, Ue, De = (experts[e][m][0] for m in M)
            k = int(hot * 2048)
            idx = np.argsort(-np.abs(h) * np.linalg.norm(D, axis=0))[:k]
            he = silu(Ge[idx] @ x) * (Ue[idx] @ x)
            out = D @ h - D[:, idx] @ h[idx] + De[:, idx] @ he
        else:
            out = D @ h
        y += rw * out
    return y

exact = {e: [experts[e][m][0] for m in M] for e in RW}
quant = {e: [experts[e][m][1] for m in M] for e in RW}
def score(W, hot=None):
    return float(np.mean([np.linalg.norm(moe(W, x, hot) - moe(exact, x)) / np.linalg.norm(moe(exact, x)) for x in hold]))

# input-side importance for each matrix from calibration: E|input_j| (x for gate/up, h for down)
imp = {}
for e in RW:
    G, U, D = exact[e]
    hs = [np.abs(silu(G @ x) * (U @ x)) for x in cal]
    imp[e] = [np.mean([np.abs(x) for x in cal], 0), np.mean([np.abs(x) for x in cal], 0), np.mean(hs, 0)]

print(f"baseline int4 g64              read=0.562  holdout err={score(quant):.4f}")
for frac in (0.005, 0.01, 0.02, 0.05):
    for aware in (False, True):
        W = {}
        for e in RW:
            mats = []
            for i, m in enumerate(M):
                w, wq = experts[e][m]
                err = w - wq
                crit = np.abs(err) * (imp[e][i][None, :] if aware else 1.0)
                k = int(frac * w.size)
                flat = np.argpartition(crit.ravel(), -k)[-k:]
                r = wq.ravel().copy(); r[flat] = w.ravel()[flat]
                mats.append(r.reshape(w.shape))
            W[e] = mats
        size = 0.5625 + frac * 6            # int32 index + fp16 value per kept weight, relative to 1 B/weight FP8
        print(f"sparse {frac*100:4.1f}% {'act-aware' if aware else 'plain    '}   read={size:.3f}  holdout err={score(W):.4f}")
for hot in (0.10, 0.25):
    print(f"two-stage hot {int(hot*100)}% (non-oracle)   read={0.5625 + hot:.3f}  holdout err={score(quant, hot):.4f}")
