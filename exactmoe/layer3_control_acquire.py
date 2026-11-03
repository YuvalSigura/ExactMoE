"""Bounded acquisition of Layer 3 router + shared-expert tensors.

Downloads only the small control tensors needed for a single-layer MoE FFN
arithmetic test. Does not download routed experts (those are acquired separately)
and does not download full shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct

from .expert_acquire import (
    MODEL, REVISION, _get_json, _get_header, _request, _source_url,
    _validate_entry, MAX_RANGE,
)
from .expert_validation import names as expert_names


def control_tensor_names(layer: int) -> list[str]:
    if layer != 3:
        raise ValueError("this helper currently supports only layer 3")
    prefix = f"model.layers.{layer}.mlp."
    return [
        prefix + "gate.weight",
        prefix + "gate.e_score_correction_bias",
        prefix + "shared_experts.gate_proj.weight",
        prefix + "shared_experts.gate_proj.weight_scale_inv",
        prefix + "shared_experts.up_proj.weight",
        prefix + "shared_experts.up_proj.weight_scale_inv",
        prefix + "shared_experts.down_proj.weight",
        prefix + "shared_experts.down_proj.weight_scale_inv",
    ]


def acquire_layer3_control(out: Path, *, cap_bytes: int = 64 * 1024 * 1024) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    manifest, _ = _get_json("model.safetensors.index.json")
    config, _ = _get_json("config.json")
    weight_map = manifest["weight_map"]
    names = control_tensor_names(3)
    selected = {}
    for n in names:
        if n not in weight_map:
            raise KeyError(f"missing {n} in official manifest")
        selected[n] = weight_map[n]

    shard_headers = {}
    for shard in sorted(set(selected.values())):
        hdr, base, total, etag, _ = _get_header(shard)
        shard_headers[shard] = (hdr, base, total, etag)

    source = []
    payload = {}
    total_bytes = 0
    for name in names:
        shard = selected[name]
        hdr, base, total, etag = shard_headers[shard]
        entry = hdr[name]
        lo, hi = _validate_entry(name, entry)
        n = hi - lo
        total_bytes += n
        if total_bytes > cap_bytes or n > MAX_RANGE:
            raise MemoryError("control tensors exceed cap")
        raw, received_total, seen_etag = _request(
            _source_url(shard), start=base + lo, end=base + hi - 1)
        if received_total != total:
            raise ValueError("source changed during range read")
        payload[name] = raw
        source.append({
            "tensor": name,
            "source_shard": shard,
            "original_absolute_offset": base + lo,
            "nbytes": n,
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    tmp = out.with_name(out.name + ".tmp-" + str(os.getpid()))
    tmp.mkdir(parents=True)
    try:
        header = {}
        data = bytearray()
        for item in source:
            raw = payload[item["tensor"]]
            start = len(data)
            data.extend(raw)
            header[item["tensor"]] = {
                "dtype": item["dtype"],
                "shape": item["shape"],
                "data_offsets": [start, len(data)],
            }
        h = json.dumps(header, separators=(",", ":")).encode("utf-8")
        (tmp / "control.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + data)
        (tmp / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {x["tensor"]: "control.safetensors" for x in source}
        }, indent=2))
        (tmp / "config.json").write_text(json.dumps(config, indent=2))
        provenance = {
            "kind": "bounded-layer3-control",
            "repo": MODEL,
            "revision": REVISION,
            "layer": 3,
            "source": source,
            "source_tensor_bytes": total_bytes,
        }
        (tmp / "sample-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return provenance


def main(argv=None):
    p = argparse.ArgumentParser(description="Bounded Layer 3 router + shared expert")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-source-mib", type=int, default=64)
    a = p.parse_args(argv)
    result = acquire_layer3_control(a.out, cap_bytes=a.max_source_mib * 1024 * 1024)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
