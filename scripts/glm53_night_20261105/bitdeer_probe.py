"""Fetch layer-3 experts from BitdeerAI/GLM-5.3-W4AFP8-EP-GPTQ (HTTP range reads), find the int4 packing
convention by comparison with the original FP8 experts, then compare GPTQ g128 vs Colibri RTN g64."""
import io, contextlib, json, runpy, struct, sys, urllib.request
from pathlib import Path
import numpy as np
S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
OUT = Path(S) / "bitdeer_l3"
REPO = "BitdeerAI/GLM-5.3-W4AFP8-EP-GPTQ"
REV = "598aab6620eb4806ec839bc5be6774e0715d19a7"
def url(sh): return f"https://huggingface.co/{REPO}/resolve/{REV}/{sh}"
def rng(sh, a, b):
    return urllib.request.urlopen(urllib.request.Request(url(sh), headers={"Range": f"bytes={a}-{b}"}), timeout=300).read()
headers = {}
def header(sh):
    if sh not in headers:
        n = struct.unpack("<Q", rng(sh, 0, 7))[0]
        headers[sh] = (8 + n, json.loads(rng(sh, 8, 8 + n - 1)))
    return headers[sh]
wmap = json.loads(urllib.request.urlopen(url("model.safetensors.index.json")).read())["weight_map"]
def fetch(name):
    f = OUT / (name + ".npy")
    sh = wmap[name]; base, h = header(sh); meta = h[name]
    if f.exists(): raw = np.load(f)
    else:
        a, b = meta["data_offsets"]; raw = np.frombuffer(rng(sh, base + a, base + b - 1), dtype=np.uint8).copy(); np.save(f, raw)
    if meta["dtype"] == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
    return raw.reshape(meta["shape"])
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(S + r"\colibri_int4_quality.py", run_name="lib")
experts, silu = q["experts"], q["silu"]
def deq(packed, scale, lo_first, signed):
    p = packed.astype(np.uint8)
    lo, hi = (p & 0xF).astype(np.int32), (p >> 4).astype(np.int32)
    a, b = (lo, hi) if lo_first else (hi, lo)
    if signed: a = np.where(a > 7, a - 16, a); b = np.where(b > 7, b - 16, b)
    else: a, b = a - 8, b - 8
    O = p.shape[0]; w = np.empty((O, p.shape[1] * 2), np.float32); w[:, 0::2] = a; w[:, 1::2] = b
    gs = w.shape[1] // scale.shape[1]
    return w * np.repeat(scale, gs, axis=1)
res = {}
first = next(iter(experts))
for conv in [(True, True), (False, True), (True, False), (False, False)]:
    errs = []
    for m in ("gate_proj", "up_proj", "down_proj"):
        p = f"model.layers.3.mlp.experts.{first}.{m}."
        W = experts[first][m][0]
        Wd = deq(fetch(p + "weight"), fetch(p + "weight_scale_inv"), *conv)
        errs.append(float(np.linalg.norm(Wd - W) / np.linalg.norm(W)))
    res[str(conv)] = errs
print("conventions (lo_first, signed) -> rel err gate/up/down:", json.dumps(res))
best = min(res, key=lambda k: sum(res[k])); conv = eval(best)
print("best", best)
gp = {}; rtn = {}; ex = {}
for e in experts:
    ex[e] = [experts[e][m][0] for m in ("gate_proj", "up_proj", "down_proj")]
    rtn[e] = [experts[e][m][1] for m in ("gate_proj", "up_proj", "down_proj")]
    gp[e] = [deq(fetch(f"model.layers.3.mlp.experts.{e}.{m}.weight"), fetch(f"model.layers.3.mlp.experts.{e}.{m}.weight_scale_inv"), *conv)
             for m in ("gate_proj", "up_proj", "down_proj")]
x0 = q["inputs"]["A_gaussian"]
route = q["route_noaux_tc"](x0, q["router_w"], q["router_b"], q["config"])
RW = {int(e): float(w) for e, w in zip(route.expert_ids, route.weights)}
def moe(W, x):
    y = np.zeros(6144, np.float32)
    for e, rw in RW.items():
        G, U, D = W[e]; y += rw * (D @ (silu(G @ x) * (U @ x)))
    return y
mat = {"gptq_g128": [], "rtn_g64": []}
for e in experts:
    for i in range(3):
        n = np.linalg.norm(ex[e][i])
        mat["gptq_g128"].append(float(np.linalg.norm(gp[e][i] - ex[e][i]) / n))
        mat["rtn_g64"].append(float(np.linalg.norm(rtn[e][i] - ex[e][i]) / n))
print("weight rel err mean: gptq_g128 %.4f  rtn_g64 %.4f" % (np.mean(mat["gptq_g128"]), np.mean(mat["rtn_g64"])))
xs = [np.random.default_rng(300 + s).normal(0, 0.1, 6144).astype(np.float32) for s in range(8)]
for name, W in (("gptq_g128", gp), ("rtn_g64", rtn)):
    print(name, "MoE out rel err (proxy):", round(float(np.mean([np.linalg.norm(moe(W, x) - moe(ex, x)) / np.linalg.norm(moe(ex, x)) for x in xs])), 4))
