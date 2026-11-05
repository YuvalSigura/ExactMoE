"""Fast lane vs substitution, offline, routed-MoE output error against the original FP8 experts.

RAM residency is approximated by the K hottest experts of the layer (Colibri capped the RAM cache at 54/layer here).
For each token the router picks 8. A selected expert that is not resident costs a disk load. Policies:
  full        load every missing expert                         (reference quality, most loads)
  sub_J2M16   CACHE_ROUTE-like: keep the router's top-2; each other missing slot is replaced by the best resident
              expert from the router's top-16; weights renormalized over the new set
  skip_t      fast lane: a missing expert with routing weight < t is skipped (no substitute, no renormalization);
              heavier missing experts are loaded
  skipn_t     same, but the kept weights are renormalized to the original total
Reported: mean / p90 relative error of the routed output, and the share of disk loads avoided vs 'full'.
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
L = int(sys.argv[1]); N = int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 54
full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
config = json.loads(Path(r"C:\AI-Models\GLM-5.3-full\config.json").read_text())
pre = f"model.layers.{L}.mlp."
W = read_dense_vector_or_matrix(full[pre + "gate.weight"]).reshape(256, -1); b0 = read_dense_vector_or_matrix(full[pre + "gate.e_score_correction_bias"])
X = np.fromfile(O + r"\l3_in.f32", np.float32).reshape(-1, 6144)
X = X[np.linspace(0, len(X) - 1, min(N, len(X))).astype(int)]
heat = defaultdict(int)
for line in Path(r"C:\AI-Models\GLM-5.3-colibri-x4g64-r8\.coli_usage").read_text().splitlines():
    p = line.split()
    if len(p) == 3 and int(p[0]) == L: heat[int(p[1])] += int(p[2])
resident = set(sorted(range(256), key=lambda e: -heat[e])[:K])
cache = {}
def out_e(e, x):
    if e not in cache:
        cache[e] = tuple(sp["dequant"](full[pre + f"experts.{e}.{m}.weight"], full[pre + f"experts.{e}.{m}.weight_scale_inv"]).astype(np.float16)
                         for m in ("gate_proj", "up_proj", "down_proj"))
    G, U, D = cache[e]; g = G.astype(np.float32) @ x; u = U.astype(np.float32) @ x
    return D.astype(np.float32) @ (g / (1 + np.exp(-g)) * u)
sf = float(config.get("routed_scaling_factor", 1.0))
from collections import OrderedDict
class LRU:
    """Per-policy RAM cache: starts with the K hottest, then least-recently-used eviction (as Colibri's cache)."""
    def __init__(self): self.d = OrderedDict((e, 1) for e in sorted(range(256), key=lambda e: heat[e])[-K:])
    def __contains__(self, e): return e in self.d
    def use(self, e):
        self.d[e] = 1; self.d.move_to_end(e)
        while len(self.d) > K: self.d.popitem(last=False)
caches = defaultdict(LRU)
res = defaultdict(list); loads = defaultdict(int)
X = np.fromfile(O + r"\l3_in.f32", np.float32).reshape(-1, 6144)[:N]      # contiguous text order for the LRU
for x in X:
    r = route_noaux_tc(x, W, b0, config); ids, ws = list(r.expert_ids), list(r.weights); S = sum(ws)
    lg = W @ x; sc = 1 / (1 + np.exp(-lg)); sel = sc + b0
    top16 = [int(e) for e in np.argsort(-sel)[:16]]
    ys = {e: out_e(e, x) for e in ids}
    yref = sum(w * ys[e] for e, w in zip(ids, ws)); nref = np.linalg.norm(yref)
    c = caches["full"]; loads["full"] += sum(1 for e in ids if e not in c); [c.use(e) for e in ids]; res["full"].append(0.0)
    c = caches["sub_J2M16"]
    chosen = ids[:2] + [e if e in c else None for e in ids[2:]]
    pool = [e for e in top16 if e in c and e not in ids]
    for i, e in enumerate(chosen):
        if e is None: chosen[i] = pool.pop(0) if pool else ids[i]
    loads["sub_J2M16"] += sum(1 for e in chosen if e not in c); [c.use(e) for e in chosen]
    wsum = sum(sc[e] for e in chosen); y = sum((sc[e] / wsum * sf) * (ys[e] if e in ys else out_e(e, x)) for e in chosen)
    res["sub_J2M16"].append(float(np.linalg.norm(y - yref) / nref))
    for t in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
        for norm in (False, True):
            k = f"skip{'n' if norm else ''}_{t}"; c = caches[k]
            keep = [(e, w) for e, w in zip(ids, ws) if e in c or w / S >= t]
            loads[k] += sum(1 for e, _ in keep if e not in c); [c.use(e) for e, _ in keep]
            y = sum(w * ys[e] for e, w in keep) if keep else np.zeros_like(yref)
            if norm and keep: y = y * (S / sum(w for _, w in keep))
            res[k].append(float(np.linalg.norm(y - yref) / nref))
base = loads["full"]
print(f"layer {L}, {len(X)} tokens (text order), LRU {K}/256, full-routing misses/token {base/len(X):.2f} of 8")
for k, v in res.items():
    print(f"{k:12s} err mean {np.mean(v):.4f} p90 {np.percentile(v, 90):.4f}  loads avoided {1 - loads[k]/max(1, base):6.1%}")
