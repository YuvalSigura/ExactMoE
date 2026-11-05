"""Write <container>/model.safetensors.index.json: every tensor -> its shard; names that also exist in the
overlay dir map to the overlay shard (Colibri st_init_multi #1479 then reads the overlay copy when the
overlay dir is in COLI_MODEL_DIRS, and ignores it otherwise)."""
import json, struct, sys
from pathlib import Path
base, overlay = Path(sys.argv[1]), Path(sys.argv[2])
def names(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    return [k for k in h if k != "__metadata__"]
wm = {}
for f in sorted(base.glob("out-*.safetensors")):
    for k in names(f):
        if k in wm: raise SystemExit(f"duplicate inside the base container: {k}")
        wm[k] = f.name
over = 0; new = 0
for f in sorted(overlay.glob("out-fp8pass-*.safetensors")):
    for k in names(f):
        if k in wm: over += 1
        else: new += 1
        wm[k] = f.name
out = base / "model.safetensors.index.json"
if out.exists(): raise SystemExit(f"{out} exists; not overwriting")
out.write_text(json.dumps({"metadata": {"note": "HOMELAB overlay index (fp8 dense)"}, "weight_map": wm}), encoding="utf-8")
print(f"{len(wm)} tensors; overlay replaces {over}, adds {new}; wrote {out}")
