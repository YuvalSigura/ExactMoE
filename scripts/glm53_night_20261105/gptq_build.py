"""Own GPTQ (int4 g64, act-order + MSE clip, real per-layer calibration) for every routed expert, in place.

usage: gptq_build.py <container> <dump_base> [max_shards]
  dump_base: COLI_DUMP_MOE=all output prefix; layer L inputs are <dump_base>.<L> (float32 rows, 6144 wide)
Source weights: the original FP8 checkpoint (C:\\AI-Models\\GLM-5.3-full). Shards are rewritten hottest first,
each one atomically (tmp + os.replace), so the container is valid after every step. Resumable through
<container>/.own-gptq.json. No engine may have the container open while this runs.
"""
import gc, json, os, shutil, struct, sys, time, runpy, io, contextlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import os as _os
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from gptq_gpu import pack_fmt4, DEV                              # noqa: E402
from gptq_lean import factor, quantize_layer_lean               # noqa: E402
# Hard VRAM cap: past 16 GB the Windows driver pages GPU memory into system RAM ("sysmem fallback"); two GPU
# faults (nvlddmkm 153) and a HYPERVISOR_ERROR bugcheck followed that on 2026-11-05. Fail with OOM instead.
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPTQ_VRAM_FRACTION", "0.6")), 0)
from exactmoe.storage import TensorIndex                          # noqa: E402
OLD = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
with contextlib.redirect_stdout(io.StringIO()):
    sp = runpy.run_path(OLD + r"\split_expert_experiment.py", run_name="lib_split")
C = Path(sys.argv[1]); DUMP = sys.argv[2]; LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
STATE = C / ".own-gptq.json"; MIN_FREE = 8e9; BATCH = 16; MAX_CAL = 20000
full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def read_header(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; return 8 + n, json.loads(fh.read(n))
def expert_of(k):
    s = k.split("."); return (int(s[2]), int(s[5])) if ".mlp.experts." in k else None
state = json.loads(STATE.read_text()) if STATE.exists() else {"done": [], "experts": 0}
heat = defaultdict(int)
for line in (C / ".coli_usage").read_text().splitlines():
    p = line.split()
    if len(p) == 3 and int(p[0]) >= 0: heat[(int(p[0]), int(p[1]))] += int(p[2])
shards = []
for p in sorted(C.glob("out-[0-9]*.safetensors")):
    if p.name in state["done"]: continue
    _, h = read_header(p); ex = {expert_of(k) for k in h if k != "__metadata__" and expert_of(k)}
    if ex: shards.append((sum(heat[e] for e in ex), p, sorted(ex)))
shards.sort(key=lambda t: -t[0])
log(f"{len(shards)} shards to build (done {len(state['done'])}, experts {state['experts']})")
cal_cache = {}
def calib(layer):
    if layer not in cal_cache:
        if len(cal_cache) > 1: cal_cache.clear(); torch.cuda.empty_cache()
        X = np.concatenate([np.fromfile(f"{d}.{layer}", dtype=np.float32).reshape(-1, 6144) for d in DUMP.split(";")])[:MAX_CAL]
        Xt = torch.from_numpy(X).to(DEV); cal_cache[layer] = (Xt, factor((Xt.T @ Xt).unsqueeze(0)))
    return cal_cache[layer]
def fp8(layer, e, m):
    p = f"model.layers.{layer}.mlp.experts.{e}.{m}"
    return torch.from_numpy(np.ascontiguousarray(sp["dequant"](full[p + ".weight"], full[p + ".weight_scale_inv"]), dtype=np.float32))
for n_done, (hs, p, experts) in enumerate(shards[:LIMIT]):
    if shutil.disk_usage(C).free < MIN_FREE: log("stop: disk"); break
    t0 = time.time(); new = {}
    shard_keys = set(read_header(p)[1])
    by_layer = defaultdict(list)
    for (L, e) in experts: by_layer[L].append(e)
    for L, es in sorted(by_layer.items()):
        X, fac = calib(L)
        for b0 in range(0, len(es), BATCH):
            grp = es[b0:b0 + BATCH]
            Gs, Us, Ds = (torch.stack([fp8(L, e, m) for e in grp]).to(DEV) for m in ("gate_proj", "up_proj", "down_proj"))
            packed = quantize_layer_lean(Gs, Us, Ds, X, fac)
            for i, e in enumerate(grp):
                for (codes, sc), m in zip(packed, ("gate_proj", "up_proj", "down_proj")):
                    k = f"model.layers.{L}.mlp.experts.{e}.{m}.weight"
                    if k in shard_keys:             # an expert can straddle two shards: write only what this one holds
                        new[k] = pack_fmt4(codes[i], sc[i])
            del Gs, Us, Ds, packed
    base, h = read_header(p); meta = h.pop("__metadata__", None); names = list(h)
    for k in new:
        if k not in h or h[k]["data_offsets"][1] - h[k]["data_offsets"][0] != new[k][0].nbytes:
            raise SystemExit(f"geometry mismatch {k}: in shard={k in h} have={(h[k]['data_offsets'][1]-h[k]['data_offsets'][0]) if k in h else None} new={new[k][0].nbytes} {new[k][0].shape}")
        if (k + ".qs") not in h or h[k + ".qs"]["data_offsets"][1] - h[k + ".qs"]["data_offsets"][0] != new[k][1].nbytes:
            raise SystemExit(f"scale geometry mismatch {k}")
    mm = np.memmap(p, dtype=np.uint8, mode="r"); out, blobs, off = {}, [], 0
    for k in names:
        if k in new: arr, dt = new[k][0], "U8"
        elif k.endswith(".qs") and k[:-3] in new: arr, dt = new[k[:-3]][1].view(np.uint8), "F32"
        else:
            a, b = h[k]["data_offsets"]; arr, dt = np.array(mm[base + a: base + b]), None
        nb = arr.nbytes
        if dt is None: out[k] = dict(h[k]); out[k]["data_offsets"] = [off, off + nb]
        else: out[k] = {"dtype": dt, "shape": [int(arr.size if dt == "U8" else arr.size // 4)], "data_offsets": [off, off + nb]}
        blobs.append(arr); off += nb
    md = dict(meta or {}); md["homelab.gptq"] = "own GPTQ int4 g64 act-order+mse, real per-layer calibration (gptq_build.py)"
    out["__metadata__"] = md
    hj = json.dumps(out, separators=(",", ":")).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hj))); fo.write(hj)
        for arr in blobs: fo.write(arr.tobytes())
    del mm, blobs; gc.collect()
    vb, vh = read_header(tmp); k = next(iter(new)); a, b = vh[k]["data_offsets"]
    with open(tmp, "rb") as fh: fh.seek(vb + a); got = np.frombuffer(fh.read(b - a), np.uint8)
    if not np.array_equal(got, new[k][0]): raise SystemExit(f"verify failed {k}")
    os.replace(tmp, p); state["done"].append(p.name); state["experts"] += len(experts); STATE.write_text(json.dumps(state))
    log(f"[{n_done+1}/{min(len(shards), LIMIT)}] {p.name}: {len(experts)} experts in {time.time()-t0:.0f}s, free {shutil.disk_usage(C).free/1e9:.1f} GB")
log("done:", state["experts"], "experts")
