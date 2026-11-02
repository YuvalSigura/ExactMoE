"""Strict capped HTTPS Range extraction of one original pinned GLM-5.3 expert.

Does not download whole checkpoint or whole shard. A compact local safetensors
container repacks the exact six original tensor payloads without FP8 conversion.
A whole-shard LFS digest CANNOT be independently verified from partial ranges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .expert_validation import MODEL, REVISION, names, validate_layout, DEFAULT_READ_CAP
from .storage import TensorIndex

MAX_HEADER = 16 * 1024 * 1024
MAX_JSON = 32 * 1024 * 1024
MAX_RANGE = 16 * 1024 * 1024
CR = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
DTYPE_BYTES = {"F8_E4M3": 1, "F32": 4}


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (host == "huggingface.co" or
                    host.endswith(".huggingface.co") or host.endswith(".hf.co")):
            raise ValueError("source redirects outside trusted Hugging Face HTTPS hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = build_opener(_HTTPSRedirect)


def _request(url: str, *, start=None, end=None, limit=MAX_RANGE):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ValueError("source URL must be official huggingface.co HTTPS")
    headers = {"Accept-Encoding": "identity",
               "User-Agent": "ExactMoE/0.1 bounded-original-expert"}
    if start is not None:
        if start < 0 or end < start or end - start + 1 > limit:
            raise ValueError("byte-range exceeds configured limit")
        headers["Range"] = f"bytes={start}-{end}"
    with _OPENER.open(Request(url, headers=headers), timeout=60) as response:
        status = response.status
        if start is not None and status != 206:
            # Never read any data from a server that ignored Range.
            raise RuntimeError(f"HTTP 206 required; got {status}")
        if start is None and status != 200:
            raise RuntimeError(f"HTTP 200 expected for bounded JSON; got {status}")
        length = end - start + 1 if start is not None else limit
        content_range = response.headers.get("Content-Range", "")
        total = None
        if start is not None:
            match = CR.fullmatch(content_range)
            if match is None:
                raise ValueError("missing or malformed Content-Range")
            lo, hi, total = map(int, match.groups())
            if lo != start or hi != end or total <= end:
                raise ValueError("incorrect remote byte range")
        payload = response.read(length + 1)
        if start is not None:
            if len(payload) != length:
                raise ValueError("short/oversized source response")
        elif len(payload) > limit:
            raise ValueError("oversized metadata response")
        return payload, total, response.headers.get("ETag")


def _json_unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"duplicate JSON key: {k}")
        result[k] = v
    return result


def _source_url(filename: str):
    return f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}"


def _get_json(filename: str) -> tuple[dict, int]:
    raw, _, _ = _request(_source_url(filename), limit=MAX_JSON)
    obj = json.loads(raw, object_pairs_hook=_json_unique)
    if not isinstance(obj, dict):
        raise ValueError("invalid source JSON object")
    return obj, len(raw)


def _get_header(shard: str):
    if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", shard):
        raise ValueError("not an official shard basename")
    url = _source_url(shard)
    prefix, total, etag = _request(url, start=0, end=7)
    hlen = struct.unpack("<Q", prefix)[0]
    if not 2 <= hlen <= MAX_HEADER or 8 + hlen > total:
        raise ValueError("invalid source header length")
    raw, again, etag2 = _request(url, start=8, end=7 + hlen)
    if again != total or (etag and etag2 and etag != etag2):
        raise ValueError("source shard changed during metadata read")
    hdr = json.loads(raw, object_pairs_hook=_json_unique)
    if not isinstance(hdr, dict):
        raise ValueError("invalid source safetensors header")
    return hdr, 8 + hlen, total, etag or etag2, 8 + hlen


def _validate_entry(name: str, entry: dict, expected_shape=None):
    if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
        raise ValueError(f"invalid tensor header: {name}")
    shape = entry["shape"]
    offs = entry.get("data_offsets")
    dtype = entry.get("dtype")
    if len(shape) != 2 or any(type(i) is not int or i <= 0 for i in shape):
        raise ValueError(f"incorrect tensor shape: {name}")
    if dtype not in DTYPE_BYTES or not isinstance(offs, list) or len(offs) != 2 or any(type(i) is not int for i in offs):
        raise ValueError(f"incorrect tensor type/offset: {name}")
    lo, hi = offs
    if lo < 0 or hi - lo != math.prod(shape) * DTYPE_BYTES[dtype]:
        raise ValueError(f"noncontiguous or inconsistent tensor payload: {name}")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"wrong tensor shape: {name}")
    return lo, hi


def _compile_selection(layer: int, expert: int, manifest: dict, config: dict):
    mp = manifest.get("weight_map")
    if not isinstance(mp, dict):
        raise ValueError("official manifest missing weight_map")
    chosen = {}
    hidden = config.get("hidden_size")
    intermediate = config.get("moe_intermediate_size")
    if type(hidden) is not int or type(intermediate) is not int:
        raise ValueError("official config lacks hidden_size/moe_intermediate_size")
    for p, (w, s) in names(layer, expert).items():
        for n in (w, s):
            if n not in mp:
                raise KeyError(f"official manifest missing {n}")
            chosen[n] = mp[n]
    return chosen, hidden, intermediate


def acquire_remote(out: Path, layer: int = 3, expert: int = 0,
                   *, cap_bytes: int = DEFAULT_READ_CAP, fetch_json=_get_json,
                   fetch_header=_get_header, fetch_range=_request):
    """Read only the requested six tensor ranges and publish atomically.

    Function injection points support offline mock-server adversarial tests.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing sample: {out}")
    if cap_bytes <= 0 or cap_bytes > 64 * 1024 * 1024:
        raise ValueError("sample cap must be in (0, 64 MiB]")
    names(layer, expert)
    manifest, man_bytes = fetch_json("model.safetensors.index.json")
    config, cfg_bytes = fetch_json("config.json")
    selected, hidden, intermediate = _compile_selection(layer, expert, manifest, config)
    shard_headers = {}
    download_bytes = man_bytes + cfg_bytes
    for shard in sorted(set(selected.values())):
        hdr, base, total, etag, read_size = fetch_header(shard)
        download_bytes += read_size
        shard_headers[shard] = (hdr, base, total, etag)
    source = []
    payload = {}
    count = 0
    for p, (w, s) in names(layer, expert).items():
        for tensor in (w, s):
            shard = selected[tensor]
            hdr, base, total, etag = shard_headers[shard]
            if tensor not in hdr:
                raise ValueError(f"official manifest/header tensor identity mismatch: {tensor}")
            entry = hdr[tensor]
            expected = [intermediate, hidden] if p != "down_proj" else [hidden, intermediate]
            if tensor == s:
                expected = [math.ceil(expected[0] / 128), math.ceil(expected[1] / 128)]
            lo, hi = _validate_entry(tensor, entry, expected)
            required = "F32" if tensor == s else "F8_E4M3"
            if entry["dtype"] != required or base + hi > total:
                raise ValueError(f"incorrect source dtype/offset {tensor}")
            n = hi - lo
            count += n
            if count > cap_bytes or n > MAX_RANGE:
                raise MemoryError("source tensor exceeds bounded sample cap")
            raw, received_total, seen_etag = fetch_range(
                _source_url(shard), start=base + lo, end=base + hi - 1)
            download_bytes += len(raw)
            if received_total != total or (etag and seen_etag and etag != seen_etag):
                raise ValueError("source shard changed between ranged reads")
            payload[tensor] = raw
            source.append({"tensor": tensor, "source_shard": shard,
                           "original_absolute_offset": base + lo, "nbytes": n,
                           "shape": entry["shape"], "dtype": required,
                           "sha256": hashlib.sha256(raw).hexdigest(),
                           "etag": seen_etag or etag})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp-" + str(os.getpid()))
    if tmp.exists():
        raise FileExistsError(tmp)
    tmp.mkdir()
    try:
        header = {}
        data = bytearray()
        for item in source:
            name = item["tensor"]
            raw = payload[name]
            start = len(data)
            data.extend(raw)
            header[name] = {"dtype": item["dtype"], "shape": item["shape"],
                            "data_offsets": [start, len(data)]}
        h = json.dumps(header, separators=(",", ":")).encode("utf-8")
        (tmp / "expert.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + data)
        (tmp / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {x["tensor"]: "expert.safetensors" for x in source}}, indent=2))
        (tmp / "config.json").write_text(json.dumps(config, indent=2))
        provenance = {
            "kind": "bounded-repacked-source-byte-sample", "repo": MODEL,
            "revision": REVISION, "layer": layer, "expert": expert,
            "source_tensor_bytes": count,
            "downloaded_bytes_including_metadata": download_bytes,
            "source": source, "whole_original_shard_sha256_verified": False,
            "pinned_https_range_origin": True,
            "warning": "ETag and HTTPS pinned Range establish bounded provenance; not a whole-shard content hash",
        }
        (tmp / "sample-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        validate_sample(tmp, layer, expert)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return provenance


def validate_sample(folder: Path, layer: int, expert: int) -> dict:
    folder = Path(folder)
    evidence = json.loads((folder / "sample-provenance.json").read_text())
    if (evidence.get("repo"), evidence.get("revision"), evidence.get("layer"),
        evidence.get("expert")) != (MODEL, REVISION, layer, expert):
        raise ValueError("source revision/expert identity mismatch")
    expected = {n for pair in names(layer, expert).values() for n in pair}
    entries = evidence.get("source")
    if not isinstance(entries, list) or {x.get("tensor") for x in entries} != expected or len(entries) != 6:
        raise ValueError("sample provenance tensor identities wrong/missing")
    idx = TensorIndex.from_selected(folder, list(expected))
    cfg = json.loads((folder / "config.json").read_text())
    layout = validate_layout(idx, layer, expert, hidden_size=cfg["hidden_size"],
                             intermediate_size=cfg["moe_intermediate_size"])
    for item in entries:
        rec = idx[item["tensor"]]
        from .storage import read_bytes
        if (rec.dtype != item["dtype"] or list(rec.shape) != item["shape"] or
            rec.nbytes != item["nbytes"] or
            hashlib.sha256(read_bytes(rec)).hexdigest() != item["sha256"]):
            raise ValueError(f"source sample hash/shape/dtype mismatch: {rec.name}")
    return {"verified_sample_hashes": True,
            "verified_local_original_payload_bytes": layout["source_payload_bytes"],
            "source_revision": REVISION, "remote_full_shard_digest_verified": False,
            "source_trust": "pinned HF HTTPS ranges (not whole-shard SHA256)"}


def main(argv=None):
    p = argparse.ArgumentParser(description="Bounded original FP8 expert extraction; never full checkpoint")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--max-source-mib", type=int, default=48)
    a = p.parse_args(argv)
    result = acquire_remote(a.out, a.layer, a.expert,
                            cap_bytes=a.max_source_mib * 1024 * 1024)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
