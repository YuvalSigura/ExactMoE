"""Bounded remote safetensors header inspection.

Makes exactly two HTTPS Range requests to the Hugging Face resolve endpoint.
Never requests the data section and REJECTS servers that ignore Range (HTTP 200).
No model weight data is read or downloaded. Metadata and endpoints are untrusted.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import re
import struct
from urllib.request import Request, urlopen

from .glm_catalog import EXPERT_RE

# This upper bound applies to BOTH the advertised header and bytes read.
MAX_REMOTE_HEADER_BYTES = 16 * 1024 * 1024
_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_SHARD = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REV = re.compile(r"^[0-9a-f]{40}$")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate tensor header key: {key}")
        result[key] = value
    return result


def _fetch_range(url: str, first: int, last: int, total_expected: int | None = None):
    if first < 0 or last < first or last - first + 1 > MAX_REMOTE_HEADER_BYTES:
        raise ValueError("invalid or oversized HTTP range")
    request = Request(url, headers={
        "Range": f"bytes={first}-{last}",
        "Accept-Encoding": "identity",
        "User-Agent": "ExactMoE/0.1 (bounded header metadata inspection)",
    })
    with urlopen(request, timeout=45) as response:
        status = response.status
        cr = response.headers.get("Content-Range", "")
        # Do not read even one byte from a full-file (HTTP 200) response.
        if status != 206:
            raise RuntimeError(f"HTTP Range required (206); server returned {status}")
        match = _RANGE.fullmatch(cr)
        if not match:
            raise ValueError(f"invalid Content-Range: {cr!r}")
        lo, hi, total = map(int, match.groups())
        if (lo, hi) != (first, last) or total <= last:
            raise ValueError("server returned incorrect byte range")
        if total_expected is not None and total != total_expected:
            raise ValueError("file size changed between Range requests")
        length = last - first + 1
        data = response.read(length + 1)
        if len(data) != length:
            raise ValueError("truncated or oversized byte-range response")
        return data, total


def inspect_remote_header(repo: str, revision: str, shard: str, block_size: int = 128,
                          sample_limit: int = 8) -> dict:
    """Inspect one official shard without downloading any model weight payload.

    Only 40-character pinned SHA revisions are accepted to prevent mutable
    metadata from changing between requests. The caller supplies the shard.
    """
    if not _REPO.fullmatch(repo) or not _REV.fullmatch(revision) or not _SHARD.fullmatch(shard):
        raise ValueError("expected owner/repo, 40-character revision SHA and model-NNNNN-of-NNNNN.safetensors")
    if block_size <= 0 or sample_limit < 0:
        raise ValueError("block_size must be positive and sample_limit nonnegative")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
    prefix, total = _fetch_range(url, 0, 7)
    (header_size,) = struct.unpack("<Q", prefix)
    if not 2 <= header_size <= MAX_REMOTE_HEADER_BYTES:
        raise ValueError(f"untrusted shard header length: {header_size}")
    if header_size + 8 > total:
        raise ValueError("advertised header exceeds shard length")
    header_data, repeated_total = _fetch_range(url, 8, 7 + header_size, total)
    metadata = json.loads(header_data, object_pairs_hook=_unique_keys)
    if not isinstance(metadata, dict):
        raise ValueError("safetensors header is not a dictionary")
    result = {}
    data_bytes = total - header_size - 8
    for name, entry in metadata.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"bad tensor entry: {name}")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        dtype = entry.get("dtype")
        if (
            not isinstance(name, str) or not isinstance(dtype, str)
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(v) is not int for v in offsets)
        ):
            raise ValueError(f"bad tensor shape or offsets: {name}")
        lo, hi = offsets
        if not (0 <= lo <= hi <= data_bytes):
            raise ValueError(f"tensor {name} exceeds shard")
        result[name] = {"shape": shape, "dtype": dtype, "data_offsets": offsets}
    dtype_counts = Counter(rec["dtype"] for rec in result.values())

    # Verify quantized expert matrix/scaling shapes only when both are
    # present in this single shard. Absence may mean another shard owns it.
    paired = failures = 0
    missing_in_shard = 0
    mismatch_samples = []
    examples = []
    for name, rec in result.items():
        match = EXPERT_RE.fullmatch(name)
        if match is None or match["kind"] != "weight":
            continue
        scale_name = name + "_scale_inv"
        scale = result.get(scale_name)
        if scale is None:
            missing_in_shard += 1
            continue
        paired += 1
        shape, scaled = rec["shape"], scale["shape"]
        expected = [math.ceil(n / block_size) for n in shape]
        valid = (
            len(shape) == 2 and rec["dtype"] == "F8_E4M3"
            and scale["dtype"] == "F32" and scaled == expected
        )
        if not valid:
            failures += 1
            if len(mismatch_samples) < sample_limit:
                mismatch_samples.append({
                    "weight": name, "weight_shape": shape, "weight_dtype": rec["dtype"],
                    "scale_shape": scaled, "scale_dtype": scale["dtype"],
                    "expected_scale_shape": expected,
                })
        if len(examples) < sample_limit:
            examples.append({
                "weight": name, "weight_shape": shape,
                "weight_bytes": rec["data_offsets"][1] - rec["data_offsets"][0],
                "scale": scale_name, "scale_shape": scaled,
            })
    return {
        "model": repo, "revision": revision, "shard": shard,
        "shard_size_bytes": total, "header_size_bytes": header_size,
        "downloaded_bytes": 8 + header_size,
        "tensor_count": len(result), "dtype_counts": dict(dtype_counts),
        "fp8_expert_weight_scale_pairs_in_shard": paired,
        "fp8_pair_shape_mismatches": failures,
        "expert_weights_without_scale_in_same_shard": missing_in_shard,
        "pair_shape_samples": examples,
        "mismatch_samples": mismatch_samples,
        "note": (
            "Only safetensors header metadata was transferred. No weights were read. "
            "This cannot prove actual weight bytes, router parity or inference quality."
        ),
    }
