"""Bounded acquisition of multiple Layer 3 experts by ID.

Reuses expert_acquire.acquire_remote. Each expert is written to its own
subdirectory. Does not download full shards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .expert_acquire import acquire_remote


def main(argv=None):
    p = argparse.ArgumentParser(description="Acquire multiple Layer 3 experts (bounded)")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--experts", required=True, help="comma-separated expert IDs, e.g. 222,191,59")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--max-source-mib", type=int, default=48)
    a = p.parse_args(argv)

    ids = [int(x.strip()) for x in a.experts.split(",") if x.strip()]
    results = []
    for eid in ids:
        out = a.out_root / f"expert-{eid}"
        print(f"acquiring layer {a.layer} expert {eid} -> {out}", flush=True)
        prov = acquire_remote(out, a.layer, eid, cap_bytes=a.max_source_mib * 1024 * 1024)
        results.append({"expert": eid, "bytes": prov["source_tensor_bytes"], "out": str(out)})
    print(json.dumps({"acquired": results}, indent=2))


if __name__ == "__main__":
    main()
