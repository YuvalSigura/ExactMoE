"""In-place migration of the Colibri container's routed experts from RTN int4 g64 to Bitdeer GPTQ int4 g128.

Shard by shard, hottest experts first (heat from <container>/.coli_usage). For each container shard:
download the GPTQ weight + scale bytes of every routed expert it holds (HTTP range reads, parallel), repack
losslessly (nibbles XOR 0x88, bf16 -> f32 scales), copy every other tensor byte-for-byte, write <shard>.tmp
and atomically replace the shard. The container is valid after every step (each expert is either the old
RTN or the new GPTQ one). Resumable via <container>/.gptq-migration.json. Stops when free disk < MIN_FREE.
No engine may have the container open while this runs (Windows cannot replace an open file).
"""
import json, os, shutil, struct, sys, threading, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

CONTAINER = Path(sys.argv[1])
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9           # max shards this run
THREADS = 24
MIN_FREE = 8 * 10**9
REPO, REV = "BitdeerAI/GLM-5.3-W4AFP8-EP-GPTQ", "598aab6620eb4806ec839bc5be6774e0715d19a7"
STATE = CONTAINER / ".gptq-migration.json"
def url(sh): return f"https://huggingface.co/{REPO}/resolve/{REV}/{sh}"
def get(u, hdr=None, tries=6):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=hdr or {}), timeout=180).read()
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(3 * (i + 1))
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

state = json.loads(STATE.read_text()) if STATE.exists() else {"done": [], "experts": 0}
wmap = json.loads(get(url("model.safetensors.index.json")))["weight_map"]
bd_headers, bd_lock = {}, threading.Lock()
def bd_header(sh):
    with bd_lock:
        if sh in bd_headers: return bd_headers[sh]
    raw = get(url(sh), {"Range": "bytes=0-7"}); n = struct.unpack("<Q", raw)[0]
    h = json.loads(get(url(sh), {"Range": f"bytes=8-{8 + n - 1}"}))
    with bd_lock: bd_headers[sh] = (8 + n, h)
    return bd_headers[sh]
def bd_tensor(name):
    sh = wmap[name]; base, h = bd_header(sh); m = h[name]; a, b = m["data_offsets"]
    data = get(url(sh), {"Range": f"bytes={base + a}-{base + b - 1}"})
    if len(data) != b - a: raise IOError(f"short read {name}")
    return m, np.frombuffer(data, dtype=np.uint8)

def read_header(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    return 8 + n, h

# heat per (layer, expert)
heat = defaultdict(int)
for line in (CONTAINER / ".coli_usage").read_text().splitlines():
    p = line.split()
    if len(p) == 3 and int(p[0]) >= 0: heat[(int(p[0]), int(p[1]))] += int(p[2])
def expert_of(name):
    s = name.split(".")
    return (int(s[2]), int(s[5])) if ".mlp.experts." in name else None

shards = []
for p in sorted(CONTAINER.glob("out-[0-9]*.safetensors")):
    if p.name in state["done"]: continue
    _, h = read_header(p)
    ex = {expert_of(k) for k in h if k != "__metadata__" and expert_of(k)}
    if ex: shards.append((sum(heat[e] for e in ex), p, len(ex)))
shards.sort(reverse=True)
log(f"{len(shards)} shards to migrate (done so far {len(state['done'])}, experts {state['experts']})")

def convert(name):
    """GPTQ weight + qs for one container weight tensor name."""
    mw, w = bd_tensor(name); ms, s = bd_tensor(name + "_scale_inv")
    if mw["dtype"] not in ("I8", "U8") or ms["dtype"] != "BF16": raise ValueError(f"{name}: dtypes")
    O, half = mw["shape"]; sO, ng = ms["shape"]
    if sO != O or (2 * half) // ng != 128: raise ValueError(f"{name}: shapes {mw['shape']} {ms['shape']}")
    qs = (s.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(O, ng)
    qs = np.repeat(qs, 2, axis=1).reshape(-1)          # g128 -> identical g64 layout (each scale twice): lossless,
    return name, (w ^ np.uint8(0x88)), qs              # same geometry as the RTN container (engine assumes one gs)

for n_done, (h_sum, p, nex) in enumerate(shards[:LIMIT]):
    free = shutil.disk_usage(CONTAINER).free
    if free < MIN_FREE: log(f"stop: only {free/1e9:.1f} GB free"); break
    t0 = time.time()
    base, h = read_header(p)
    meta = h.pop("__metadata__", None)
    names = list(h.keys())
    weights = [k for k in names if expert_of(k) and k.endswith(".weight")]
    with ThreadPoolExecutor(THREADS) as ex:
        new = {n: (w, qs) for n, w, qs in ex.map(convert, weights)}
    mm = np.memmap(p, dtype=np.uint8, mode="r")
    out_hdr, blobs, off = {}, [], 0
    for k in names:
        if k in new:
            arr, dt = new[k][0], "U8"
        elif k.endswith(".qs") and k[:-3] in new:
            arr, dt = new[k[:-3]][1], "F32"
        else:
            a, b = h[k]["data_offsets"]; arr, dt = np.array(mm[base + a: base + b]), None   # a copy: no view keeps the file mapped
        nb = arr.nbytes
        if dt is None:
            out_hdr[k] = dict(h[k]); out_hdr[k]["data_offsets"] = [off, off + nb]
        else:
            out_hdr[k] = {"dtype": dt, "shape": [int(arr.size)], "data_offsets": [off, off + nb]}
        blobs.append(arr); off += nb
    md = dict(meta or {}); md["homelab.gptq"] = f"{REPO}@{REV[:8]} routed experts (xor 0x88, bf16->f32, g128 scales duplicated to g64)"
    out_hdr["__metadata__"] = md
    hj = json.dumps(out_hdr, separators=(",", ":")).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hj))); fo.write(hj)
        for arr in blobs: fo.write(arr.tobytes())
    del mm, blobs
    import gc; gc.collect()
    # verify: header re-read and one converted tensor byte-equal
    vb, vh = read_header(tmp)
    if weights:
        k = weights[0]; a, b = vh[k]["data_offsets"]
        with open(tmp, "rb") as fh: fh.seek(vb + a); got = np.frombuffer(fh.read(b - a), np.uint8)
        if not np.array_equal(got, new[k][0]): raise SystemExit(f"verify failed for {k} in {tmp}")
    os.replace(tmp, p)
    state["done"].append(p.name); state["experts"] += nex
    STATE.write_text(json.dumps(state))
    log(f"[{n_done+1}/{min(len(shards), LIMIT)}] {p.name}: {nex} experts (heat {h_sum}) in {time.time()-t0:.0f}s, free {shutil.disk_usage(CONTAINER).free/1e9:.1f} GB")
log("migration pass finished:", state["experts"], "experts on GPTQ")
