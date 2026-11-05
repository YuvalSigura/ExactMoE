"""Expert pruning by HOMELAB usage, as a router-bias edit: how much routed-MoE output error per pruned fraction?

Pruned experts get bias -1e4 (selection only; GLM's weights come from the unbiased sigmoid scores), so the router
picks the next-best live experts. Reference: full routing with the original FP8 experts. Error = ||y_pruned - y|| /
||y|| over the routed part only (the shared expert, unchanged, is left out: conservative).
usage: prune_test.py <layer> <eval_rows_file or 'l3'> [n_tokens]
"""
import io, contextlib, json, runpy, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from exactmoe.storage import TensorIndex
from exactmoe.moe_layer import route_noaux_tc, read_dense_vector_or_matrix
O = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
with contextlib.redirect_stdout(io.StringIO()):
    sp = runpy.run_path(O + r"\split_expert_experiment.py", run_name="lib_split")
L = int(sys.argv[1]); src = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
config = json.loads(Path(r"C:\AI-Models\GLM-5.3-full\config.json").read_text())
pre = f"model.layers.{L}.mlp."
W = read_dense_vector_or_matrix(full[pre + "gate.weight"]).reshape(256, -1); b0 = read_dense_vector_or_matrix(full[pre + "gate.e_score_correction_bias"])
X = np.fromfile(O + r"\l3_in.f32" if src == "l3" else src, np.float32).reshape(-1, 6144)
X = X[np.linspace(0, len(X) - 1, min(N, len(X))).astype(int)]
heat = defaultdict(int)
for line in Path(r"C:\AI-Models\GLM-5.3-colibri-x4g64-r8\.coli_usage").read_text().splitlines():
    p = line.split()
    if len(p) == 3 and int(p[0]) == L: heat[int(p[1])] += int(p[2])
order = sorted(range(256), key=lambda e: heat[e])          # coldest first
cache = {}
def ex(e):
    if e not in cache:
        cache[e] = tuple(sp["dequant"](full[pre + f"experts.{e}.{m}.weight"], full[pre + f"experts.{e}.{m}.weight_scale_inv"]).astype(np.float16)
                         for m in ("gate_proj", "up_proj", "down_proj"))
    return cache[e]
def y(x, r):
    out = np.zeros(6144, np.float32)
    for e, w in zip(r.expert_ids, r.weights):
        G, U, D = ex(e); g = G.astype(np.float32) @ x; u = U.astype(np.float32) @ x
        out += w * (D.astype(np.float32) @ (g / (1 + np.exp(-g)) * u))
    return out
ref = [y(x, route_noaux_tc(x, W, b0, config)) for x in X]
tot = sum(heat.values())
print(f"layer {L}: {len(X)} tokens from {src}; usage selections {tot}")
for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    pruned = order[:int(256 * frac)]; b = b0.copy(); b[pruned] = -1e4
    errs = []; moved = 0
    for x, yr in zip(X, ref):
        r = route_noaux_tc(x, W, b, config); r0 = route_noaux_tc(x, W, b0, config)
        moved += len(set(r0.expert_ids) - set(r.expert_ids))
        errs.append(float(np.linalg.norm(y(x, r) - yr) / np.linalg.norm(yr)))
    share = sum(heat[e] for e in pruned) / max(1, tot)
    print(f"prune {frac:.0%} ({len(pruned)} experts, {share:.1%} of usage): routed-output err mean {np.mean(errs):.4f} "
          f"median {np.median(errs):.4f} p90 {np.percentile(errs, 90):.4f}; slots moved {moved / (8 * len(X)):.1%}", flush=True)
