"""Independent check: calibrate on the all-layer dump (calib text), evaluate on l3_in.f32 (a different text),
against what the container holds now (Bitdeer GPTQ for migrated shards) and the original FP8."""
import io, contextlib, runpy, sys, json, struct
from collections import Counter
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from gptq_gpu import quantize_layer_batch, silu, DEV
from exactmoe.storage import TensorIndex
O = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(O + r"\colibri_int4_quality.py", run_name="lib")
    sp = runpy.run_path(O + r"\split_expert_experiment.py", run_name="lib_split")
L = int(sys.argv[1]) if len(sys.argv) > 1 else 3; N = 32
C = Path(r"C:\AI-Models\GLM-5.3-colibri-x4g64-r8")
ix = json.loads((C / "model.safetensors.index.json").read_text())["weight_map"]
SH = {}
for p in C.glob("out-[0-9]*.safetensors"):
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    for k in h:
        if k.startswith(f"model.layers.{L}.mlp.experts."): SH[k] = (p, 8 + n, h[k], h.get("__metadata__", {}).get("homelab.gptq", "rtn"))
def cont(name):
    p, base, m, gm = SH[name]; _, _, ms, _ = SH[name + ".qs"]
    with open(p, "rb") as fh:
        a, b = m["data_offsets"]; fh.seek(base + a); w = np.frombuffer(fh.read(b - a), np.uint8)
        a, b = ms["data_offsets"]; fh.seek(base + a); s = np.frombuffer(fh.read(b - a), np.float32)
    return w, s, gm
def deq4(w, s, O_, I_):
    c = np.empty(w.size * 2, np.int8); c[0::2] = (w & 15).astype(np.int8) - 8; c[1::2] = (w >> 4).astype(np.int8) - 8
    return (c.reshape(O_, I_ // 64, 64).astype(np.float32) * s.reshape(O_, I_ // 64, 1)).reshape(O_, I_)
full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
Xe = np.fromfile(O + r"\l3_in.f32", np.float32).reshape(-1, 6144)
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0; useh = len(sys.argv) > 3 and sys.argv[3] == "h"
parts = [np.fromfile(rf"C:\AI-Models\calib\l.{L}", np.float32).reshape(-1, 6144)[skip:]]
if useh: parts.append(np.fromfile(rf"C:\AI-Models\calib\h.{L}", np.float32).reshape(-1, 6144))
Xc = torch.from_numpy(np.concatenate(parts)).to(DEV); Hx = Xc.T @ Xc; print("calibration rows", Xc.shape[0])
routes = [[int(v) for v in q["route_noaux_tc"](x, q["router_w"], q["router_b"], q["config"]).expert_ids] for x in Xe]
chosen = [e for e, _ in Counter(e for r in routes for e in r).most_common(N)]
def fp8(e, m):
    p = f"model.layers.{L}.mlp.experts.{e}.{m}"
    return torch.from_numpy(np.ascontiguousarray(sp["dequant"](full[p + ".weight"], full[p + ".weight_scale_inv"]), dtype=np.float32))
Gs, Us, Ds = (torch.stack([fp8(e, m) for e in chosen]) for m in ("gate_proj", "up_proj", "down_proj"))
Gq, Uq, Dq = [], [], []
for b0 in range(0, N, 16):     # 16 per batch: a 32-expert stack spills out of 16 GB VRAM into shared memory
    (g, u, d), _ = quantize_layer_batch(Gs[b0:b0+16].to(DEV), Us[b0:b0+16].to(DEV), Ds[b0:b0+16].to(DEV), Xc, Hx)
    Gq.append(g.cpu()); Uq.append(u.cpu()); Dq.append(d.cpu()); del g, u, d; torch.cuda.empty_cache()
Gq, Uq, Dq = torch.cat(Gq).to(DEV), torch.cat(Uq).to(DEV), torch.cat(Dq).to(DEV); Gs, Us, Ds = Gs.to(DEV), Us.to(DEV), Ds.to(DEV)
X = torch.from_numpy(Xe).to(DEV)
def fwd(g, u, d, x): return (silu(x @ g.T) * (x @ u.T)) @ d.T
res = {"own": [], "container": []}; kinds = Counter()
for b, e in enumerate(chosen):
    rows = [i for i, r in enumerate(routes) if e in r]; xh = X[rows]; ref = fwd(Gs[b], Us[b], Ds[b], xh)
    mats = []
    for m, shp in (("gate_proj", (2048, 6144)), ("up_proj", (2048, 6144)), ("down_proj", (6144, 2048))):
        w, s, gm = cont(f"model.layers.{L}.mlp.experts.{e}.{m}.weight"); mats.append(torch.from_numpy(deq4(w, s, *shp)).to(DEV))
    kinds[gm[:20]] += 1
    for k, (g, u, d) in (("own", (Gq[b], Uq[b], Dq[b])), ("container", mats)):
        res[k] += (torch.linalg.norm(fwd(g, u, d, xh) - ref, dim=1) / torch.linalg.norm(ref, dim=1)).cpu().numpy().tolist()
print(f"layer {L}: experts {N}, rows {len(res['own'])}, container kinds {dict(kinds)}")
print({k: round(float(np.mean(v)), 4) for k, v in res.items()})
