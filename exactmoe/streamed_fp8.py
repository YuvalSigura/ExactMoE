"""Original-byte, CPU-only GLM-5.3 routed-expert streaming execution.

Uses the existing E4M3 decoder and original FP8 block-inverse-scale math;
streams each weight matrix through a SINGLE open file descriptor, at most one
bounded group of rows per callback. No copied Kimi/Strata/Maya code.

Authenticity cannot be inferred from locally computed hashes. Pinned revision
and provenance must be independently verified outside this routine. Not a
full decoder, GPU kernel, true SSD bandwidth measurement or official parity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .fp8_reference import E4M3FN, _check_weight_and_scale
from .storage import TensorIndex, TensorRecord
from .streaming import stream_record, StreamReceipt, MAX_CHUNK_BYTES

MAX_TENSOR_BYTES = 64 * 1024 * 1024
MAX_SCALE_BYTES = 2 * 1024 * 1024


def _row_group(rows: int, cols: int, chunk_limit: int, block_size: int) -> int:
    """Select a power-of-two row group dividing the weight scale-row block."""
    if type(chunk_limit) is not int or chunk_limit < cols or chunk_limit > MAX_CHUNK_BYTES:
        raise ValueError("chunk budget must fit one full weight row and be <=16MiB")
    # Each chunk starts in one row of the scale grid. Use divisors of 128
    # to avoid mixing scale rows in a callback.
    batch = 1
    while batch * 2 <= block_size and (batch * 2) * cols <= chunk_limit:
        batch *= 2
    return min(batch, rows)


def _load_scales(scale: TensorRecord, *, expected_hash: str | None) -> tuple[np.ndarray, StreamReceipt]:
    if scale.nbytes > MAX_SCALE_BYTES:
        raise MemoryError("FP8 inverse-scale tensor exceeds bounded scale budget")
    data = bytearray()

    def collect(_offset: int, chunk: bytes) -> None:
        data.extend(chunk)

    receipt = stream_record(
        scale, collect, chunk_bytes=min(MAX_CHUNK_BYTES, max(scale.nbytes, 1)),
        max_total_bytes=MAX_SCALE_BYTES, expected_sha256=expected_hash,
    )
    values = np.frombuffer(bytes(data), dtype="<f4").reshape(scale.shape)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("inverse scales contain nonpositive or nonfinite entries")
    return values, receipt


def streamed_fp8_matvec(
    weight: TensorRecord,
    scale: TensorRecord,
    vector: Any,
    *,
    block_size: int = 128,
    max_chunk_bytes: int = 2 * 1024 * 1024,
    max_tensor_bytes: int = MAX_TENSOR_BYTES,
    expected_weight_sha256: str | None = None,
    expected_scale_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute original FP8 W@x from verified streamed ranges; no whole-W load.

    A numeric output only returns after the source file stability and supplied
    content-digest checks pass. F32 accumulation differs in reduction order from
    native production FP8 TensorCore kernels and needs independent calibration.
    """
    rows, cols = _check_weight_and_scale(weight, scale, block_size)
    if type(max_tensor_bytes) is not int or max_tensor_bytes < 1:
        raise ValueError("max_tensor_bytes must be positive")
    if weight.nbytes > max_tensor_bytes:
        raise MemoryError("FP8 weight tensor exceeds explicit streamed byte cap")
    x = np.asarray(vector, dtype=np.float32)
    if x.shape != (cols,) or not np.isfinite(x).all():
        raise ValueError("input vector must match columns and contain finite numbers")
    row_batch = _row_group(rows, cols, max_chunk_bytes, block_size)
    # A separately validated tiny scale tensor is resident; matrix rows remain
    # streamed, and output remains unpublished until the weight checksum passes.
    scales, scale_receipt = _load_scales(scale, expected_hash=expected_scale_sha256)
    y = np.empty(rows, dtype=np.float32)
    ncol_blocks = math.ceil(cols / block_size)
    if scales.shape[1] != ncol_blocks:
        raise ValueError("unexpected scale column grid")
    staged_peak = 0

    def compute(offset: int, raw: bytes) -> None:
        nonlocal staged_peak
        if offset % cols or len(raw) % cols or not raw:
            raise ValueError("weight stream chunk not aligned to complete FP8 rows")
        first = offset // cols
        nrows = len(raw) // cols
        if first + nrows > rows or first // block_size != (first + nrows - 1) // block_size:
            raise ValueError("FP8 chunk straddles inverse-scale block row")
        bits = np.frombuffer(raw, dtype=np.uint8).reshape(nrows, cols)
        sr = scales[first // block_size]
        accum = np.zeros(nrows, dtype=np.float32)
        for start in range(0, cols, block_size):
            end = min(start + block_size, cols)
            # Reconstruction is transient and bounded by the selected row block.
            tile = E4M3FN[bits[:, start:end]] * sr[start // block_size]
            if not np.isfinite(tile).all():
                raise ValueError("FP8 weight contains E4M3FN NaN")
            accum += tile @ x[start:end]
            staged_peak = max(staged_peak, int(tile.nbytes + len(raw)))
        y[first:first + nrows] = accum

    weight_receipt = stream_record(
        weight, compute, chunk_bytes=row_batch * cols,
        max_total_bytes=max_tensor_bytes,
        expected_sha256=expected_weight_sha256,
    )
    return y, {
        "weight": weight.name, "scales": scale.name, "rows": rows, "cols": cols,
        "weight_sha256": weight_receipt.content_sha256,
        "scale_sha256": scale_receipt.content_sha256,
        "expected_weight_hash_verified": weight_receipt.expected_hash_verified,
        "expected_scale_hash_verified": scale_receipt.expected_hash_verified,
        "bounded_row_group": row_batch,
        "weight_chunks": weight_receipt.chunks,
        "logical_read_bytes": weight_receipt.nbytes + scale_receipt.nbytes,
        "peak_io_chunk_bytes": max(weight_receipt.peak_payload_buffer_bytes,
                                   scale_receipt.peak_payload_buffer_bytes),
        "peak_weight_chunk_plus_fp32_tile_bytes": staged_peak,
        "source_stable_during_read": (
            weight_receipt.source_stable_during_read and
            scale_receipt.source_stable_during_read
        ),
        "physical_ssd_bytes_measured": False,
        "official_production_fp8_kernel_parity": "NOT_TESTED",
    }


def _silu(x: np.ndarray) -> np.ndarray:
    # Same stable mathematical SiLU definition as the existing CPU reference.
    pos = x >= 0
    sig = np.empty_like(x)
    sig[pos] = 1 / (1 + np.exp(-x[pos]))
    exp_neg = np.exp(x[~pos])
    sig[~pos] = exp_neg / (1 + exp_neg)
    return (x * sig).astype(np.float32)


def streamed_expert(
    index: TensorIndex,
    layer: int,
    expert: int,
    hidden: Any,
    *,
    expected_sha256_by_tensor: Mapping[str, str] | None = None,
    max_chunk_bytes: int = 2 * 1024 * 1024,
    max_tensor_bytes: int = MAX_TENSOR_BYTES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Three original routed expert FP8 projections, no routing/DSA/shared expert.

    If expected hashes are supplied, require all six, not a selected subset.
    Source provenance is only as trusted as those externally supplied hashes.
    """
    from .expert_validation import names, validate_layout

    ids = names(layer, expert)
    names_flat = {name for pair in ids.values() for name in pair}
    if expected_sha256_by_tensor is not None:
        if set(expected_sha256_by_tensor) != names_flat:
            raise ValueError("expected digest map must cover exactly all six tensors")
        if any(not isinstance(h, str) or len(h) != 64
               or any(c not in "0123456789abcdef" for c in h)
               for h in expected_sha256_by_tensor.values()):
            raise ValueError("invalid digest in six-tensor commitment")
    layout = validate_layout(index, layer, expert)
    x = np.asarray(hidden, dtype=np.float32)
    if x.shape != (layout["hidden"],) or not np.isfinite(x).all():
        raise ValueError("input must be finite 1-D GLM hidden state")
    timings: dict[str, float] = {}
    receipts: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for p in ("gate_proj", "up_proj", "down_proj"):
        weight_name, scale_name = ids[p]
        inp = x if p != "down_proj" else _silu(outputs["gate_proj"]) * outputs["up_proj"]
        start = time.perf_counter()
        vec, rec = streamed_fp8_matvec(
            index[weight_name], index[scale_name], inp,
            max_chunk_bytes=max_chunk_bytes,
            max_tensor_bytes=max_tensor_bytes,
            expected_weight_sha256=(None if expected_sha256_by_tensor is None
                                    else expected_sha256_by_tensor[weight_name]),
            expected_scale_sha256=(None if expected_sha256_by_tensor is None
                                   else expected_sha256_by_tensor[scale_name]),
        )
        outputs[p] = vec
        receipts[p] = rec
        timings[p] = time.perf_counter() - start
    result = outputs["down_proj"]
    if result.shape != x.shape or not np.isfinite(result).all():
        raise ValueError("non-finite or dimensionally incorrect expert output")
    return result, {
        "status": "CPU_SINGLE_EXPERT_ONLY",
        "layer": layer, "expert": expert, "model": "GLM-5.3 routed FFN projection family",
        "tensor_count": 6,
        "source_payload_bytes_expected": layout["source_payload_bytes"],
        "logical_bytes_streamed": sum(v["logical_read_bytes"] for v in receipts.values()),
        "expected_local_hashes_verified": expected_sha256_by_tensor is not None,
        "external_origin_attested_by_this_routine": False,
        "full_model_inference": False, "gpu_native_fp8_kernel": False,
        "physical_ssd_bytes_measured": False,
        "projection_seconds": timings, "projection_receipts": receipts,
        "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "warning": "Synthetic or locally committed content hashes are not publisher authentication; independent production oracle remains required.",
    }


def main(argv: list[str] | None = None) -> int:
    """Evaluate existing bounded PR #18 six-tensor sample (no network or weights download)."""
    import argparse
    from .expert_acquire import validate_sample
    from .expert_validation import names as expert_names, oracle_matvec

    parser = argparse.ArgumentParser(description="CPU streamed single original-shape FP8 expert; not full GLM")
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--chunk-kib", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-local-oracle", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_kib < 1 or args.chunk_kib > 16 * 1024:
        parser.error("--chunk-kib must be in [1,16384]")
    # Validate source and local payload hashes before processing any tensor.
    source_validation = validate_sample(args.sample_dir, args.layer, args.expert)
    prov = json.loads((args.sample_dir / "sample-provenance.json").read_text())
    hashes = {entry["tensor"]: entry["sha256"] for entry in prov["source"]}
    ids = expert_names(args.layer, args.expert)
    index = TensorIndex.from_selected(args.sample_dir, [
        name for pair in ids.values() for name in pair
    ])
    from .expert_validation import validate_layout, _silu
    layout = validate_layout(index, args.layer, args.expert)
    hidden = np.random.default_rng(args.seed).normal(0, .1, layout["hidden"]).astype(np.float32)
    output, report = streamed_expert(
        index, args.layer, args.expert, hidden,
        expected_sha256_by_tensor=hashes,
        max_chunk_bytes=args.chunk_kib * 1024,
    )
    if args.check_local_oracle:
        a = [oracle_matvec(index[ids[p][0]], index[ids[p][1]], hidden)
             for p in ("gate_proj", "up_proj")]
        oracle = oracle_matvec(index[ids["down_proj"][0]], index[ids["down_proj"][1]],
                              _silu(a[0]) * a[1])
        error = np.abs(oracle.astype(np.float64) - output.astype(np.float64))
        passed = bool(np.allclose(oracle, output, rtol=2e-5, atol=2e-4))
        report["local_independent_f32_oracle"] = {
            "passed": passed, "max_absolute_error": float(error.max()),
            "production_fp8_kernel_parity": "NOT_TESTED",
        }
    else:
        passed = True
        report["local_independent_f32_oracle"] = {"status": "NOT_RUN"}
    # Revalidate after all computations, prior to publishing ANY output/report.
    validate_sample(args.sample_dir, args.layer, args.expert)
    report["post_run_source_hashes_verified"] = True
    report["source_provenance"] = source_validation
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
