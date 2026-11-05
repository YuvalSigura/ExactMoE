"""Real layer-3 MoE inputs (dumped by Colibri COLI_DUMP_MOE=3) -> exact routing -> per-expert error of
RTN int4 g64 (Colibri), Bitdeer GPTQ int4 g128, and both with the two-stage hot-neuron repair, against
the original FP8 experts read from C:\\AI-Models\\GLM-5.3-full. Only the most-used experts are compared."""
import io, contextlib, json, runpy, sys
from collections import Counter
from pathlib import Path
import numpy as np

S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from exactmoe.storage import TensorIndex  # noqa: E402

TOP_EXPERTS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
X = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 6144)
print("dumped rows:", X.shape[0])

with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(S + r"\colibri_int4_quality.py", run_name="lib")
    bd = runpy.run_path(S + r"\bitdeer_probe.py", run_name="lib_bd")   # fetch(), deq() and cached layer-3 downloads
silu, route_fn = q["silu"], q["route_noaux_tc"]
quant_g, dequant_g = q["quant_int4_grouped"], q["dequant_int4_grouped"]
fetch, deq = bd["fetch"], bd["deq"]
dequant_fp8 = runpy.run_path(S + r"\split_expert_experiment.py", run_name="lib_split")["dequant"]

routes = [route_fn(x, q["router_w"], q["router_b"], q["config"]) for x in X]
use = Counter(int(e) for r in routes for e in r.expert_ids)
chosen = [e for e, _ in use.most_common(TOP_EXPERTS)]
print("experts compared:", chosen, "share of selections:", round(sum(use[e] for e in chosen) / sum(use.values()), 3))

full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
M = ("gate_proj", "up_proj", "down_proj")
def fp8(e):
    p = f"model.layers.3.mlp.experts.{e}."
    return [dequant_fp8(full[p + m + ".weight"], full[p + m + ".weight_scale_inv"]) for m in M]
def rtn(ws):
    out = []
    for w in ws:
        qb, s = quant_g(w, 4, 64); out.append(dequant_g(qb, s, *w.shape, 64))
    return out
def gptq(e):
    return [deq(fetch(f"model.layers.3.mlp.experts.{e}.{m}.weight"), fetch(f"model.layers.3.mlp.experts.{e}.{m}.weight_scale_inv"), True, True) for m in M]
def fwd(W, x):
    G, U, D = W; return D @ (silu(G @ x) * (U @ x))
def two_stage(Wq, We, x, frac):
    G, U, D = Wq; Ge, Ue, De = We
    h = silu(G @ x) * (U @ x)
    k = int(frac * h.shape[0]); idx = np.argsort(-np.abs(h) * np.linalg.norm(D, axis=0))[:k]
    he = silu(Ge[idx] @ x) * (Ue[idx] @ x)
    return D @ h - D[:, idx] @ h[idx] + De[:, idx] @ he

arms = {"rtn_g64": [], "gptq_g128": [], "rtn+2stage10": [], "gptq+2stage10": [], "rtn+2stage25": [], "gptq+2stage25": []}
moe_err = {k: [0.0, 0.0] for k in arms}
for e in chosen:
    We = fp8(e); Wr = rtn(We); Wg = gptq(e)
    rows = [i for i, r in enumerate(routes) if e in [int(v) for v in r.expert_ids]]
    for i in rows:
        x = X[i]; ref = fwd(We, x); n = float(np.linalg.norm(ref)) or 1.0
        outs = {"rtn_g64": fwd(Wr, x), "gptq_g128": fwd(Wg, x),
                "rtn+2stage10": two_stage(Wr, We, x, 0.10), "gptq+2stage10": two_stage(Wg, We, x, 0.10),
                "rtn+2stage25": two_stage(Wr, We, x, 0.25), "gptq+2stage25": two_stage(Wg, We, x, 0.25)}
        for k, y in outs.items():
            arms[k].append(float(np.linalg.norm(y - ref)) / n)
    print(f"expert {e}: {len(rows)} rows  rtn {np.mean(arms['rtn_g64'][-len(rows):]):.4f}  gptq {np.mean(arms['gptq_g128'][-len(rows):]):.4f}", flush=True)
res = {k: {"mean_rel_err": round(float(np.mean(v)), 4), "median": round(float(np.median(v)), 4), "n": len(v)} for k, v in arms.items()}
print(json.dumps(res, indent=1))
Path(S + r"\real_act_compare.json").write_text(json.dumps({"experts": chosen, "results": res}, indent=1))
