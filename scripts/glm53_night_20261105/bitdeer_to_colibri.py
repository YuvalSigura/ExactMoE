"""Repack the routed experts of one BitdeerAI/GLM-5.3-W4AFP8-EP-GPTQ shard into a Colibri fmt=4 overlay shard.

Bitdeer: <name>.weight I8 [O, I/2], two signed int4 per byte, low nibble first; <name>.weight_scale_inv BF16
[O, I/128] (one scale per 128 inputs). Colibri fmt=4: <name>.weight U8 flat, nibble = q + 8 (offset binary),
low nibble first; <name>.weight.qs F32 flat [O * I/128]; the group size (128) follows from the scale count.
signed -> offset per nibble is +8 mod 16 == XOR 0x8, so each byte is XOR 0x88. Lossless: no recomputation.
Only routed experts of layers < n_layers are written (no MTP, no shared experts, no dense tensors).
"""
import json, struct, sys
from pathlib import Path
import numpy as np

src, out_path, n_layers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 78

with open(src, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(n))
base = 8 + n
header.pop("__metadata__", None)
mm = np.memmap(src, dtype=np.uint8, mode="r")

def routed(name):
    if ".mlp.experts." not in name or not name.endswith(".weight"):
        return False
    layer = int(name.split(".")[2])
    return layer < n_layers

tensors, order = {}, []
for name, meta in header.items():
    if not routed(name):
        continue
    sname = name + "_scale_inv"
    if sname not in header:
        raise SystemExit(f"{name}: no scale tensor")
    w, s = meta, header[sname]
    if w["dtype"] not in ("I8", "U8") or s["dtype"] != "BF16":
        raise SystemExit(f"{name}: unexpected dtypes {w['dtype']}/{s['dtype']}")
    O, half = w["shape"]; I = half * 2
    sO, ng = s["shape"]
    if sO != O or I % ng or I // ng != 128:
        raise SystemExit(f"{name}: scale shape {s['shape']} does not match [{O}, {I}//128]")
    a, b = w["data_offsets"]; wb = np.asarray(mm[base + a: base + b]) ^ np.uint8(0x88)
    a, b = s["data_offsets"]; sb = np.asarray(mm[base + a: base + b]).view(np.uint16).astype(np.uint32) << 16
    tensors[name] = wb
    tensors[name + ".qs"] = sb.view(np.float32)
    order += [name, name + ".qs"]

if not tensors:
    print(f"{src.name}: no routed experts"); sys.exit(0)

hdr, off = {}, 0
for k in order:
    arr = tensors[k]
    dt = "U8" if arr.dtype == np.uint8 else "F32"
    hdr[k] = {"dtype": dt, "shape": [int(arr.size)], "data_offsets": [off, off + arr.nbytes]}
    off += arr.nbytes
hdr["__metadata__"] = {"source": f"BitdeerAI/GLM-5.3-W4AFP8-EP-GPTQ@598aab66 {src.name}", "repack": "xor 0x88 nibbles, bf16->f32 scales"}
hj = json.dumps(hdr, separators=(",", ":")).encode()
hj += b" " * ((8 - len(hj) % 8) % 8)
tmp = out_path.with_suffix(".tmp")
with open(tmp, "wb") as fo:
    fo.write(struct.pack("<Q", len(hj))); fo.write(hj)
    for k in order:
        fo.write(tensors[k].tobytes())
tmp.replace(out_path)
print(f"{src.name} -> {out_path.name}: {len(order)//2} expert matrices, {off/1e9:.2f} GB")
