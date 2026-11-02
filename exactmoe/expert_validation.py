"""Fail-closed measured FP8 expert gate; never a full GLM-5.3 inference claim.

Two separate comparisons: dequantized FP8 mathematical F32 oracle, and a
CANDIDATE dynamic per-128 activation FP8 experiment. The latter is NOT proof
of parity with official Triton/DeepGEMM kernels until independently tested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from threading import Event, Thread
import time
import tracemalloc

import numpy as np

from .fp8_reference import block_fp8_matvec, E4M3FN
from .storage import TensorIndex, read_bytes, meter_reads

REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
MODEL = "zai-org/GLM-5.3"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_READ_CAP = 48 * 1024 * 1024


def names(layer: int, expert: int) -> dict[str, tuple[str, str]]:
    if type(layer) is not int or not 3 <= layer < 78 or type(expert) is not int or not 0 <= expert < 256:
        raise ValueError("expected GLM-5.3 sparse layer 3..77 and routed expert 0..255")
    prefix = f"model.layers.{layer}.mlp.experts.{expert}."
    return {p: (prefix + p + ".weight", prefix + p + ".weight_scale_inv") for p in PROJECTIONS}


def validate_layout(index: TensorIndex, layer: int, expert: int, *, block_size: int = 128,
                    hidden_size: int | None = None, intermediate_size: int | None = None) -> dict:
    """Reject crossed identities, shapes, unexpected dtype, invalid inverse scales."""
    if block_size != 128:
        raise ValueError("GLM original block size must be 128")
    actual = names(layer, expert)
    shapes = {}
    read_total = 0
    for projection, (wn, sn) in actual.items():
        w, s = index[wn], index[sn]
        if w.name != wn or s.name != sn:
            raise ValueError("wrong tensor/expert identity")
        if w.dtype != "F8_E4M3" or s.dtype != "F32" or len(w.shape) != 2 or len(s.shape) != 2:
            raise ValueError("wrong weight/scale dtype or rank")
        rows, cols = w.shape
        if rows < 1 or cols < 1 or w.nbytes != rows * cols:
            raise ValueError("invalid FP8 weight shape/byte size")
        if s.shape != (math.ceil(rows / 128), math.ceil(cols / 128)) or s.nbytes != 4 * math.prod(s.shape):
            raise ValueError("invalid inverse scale grid/byte size")
        scales = np.frombuffer(read_bytes(s), dtype="<f4")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("invalid inverse scales (NaN/inf/nonpositive)")
        shapes[projection] = tuple(w.shape)
        read_total += w.nbytes + s.nbytes
    g, u, d = (shapes[p] for p in PROJECTIONS)
    if g != u or d != (g[1], g[0]):
        raise ValueError("expert gate/up/down dimensions disagree")
    if hidden_size is not None and g[1] != hidden_size:
        raise ValueError("checkpoint hidden-size mismatch")
    if intermediate_size is not None and g[0] != intermediate_size:
        raise ValueError("checkpoint intermediate-size mismatch")
    return {"layer": layer, "expert": expert, "hidden": g[1], "intermediate": g[0],
            "tensor_count": 6, "source_payload_bytes": read_total,
            "tensor_shapes": {key: list(value) for key, value in shapes.items()}}


def _lookup_quantize(x: np.ndarray) -> np.ndarray:
    """Encode E4M3FN nearest-even with finite 448 saturation."""
    z = np.asarray(x, dtype=np.float32)
    if not np.isfinite(z).all():
        raise ValueError("activation contains NaN/inf")
    positive_values = E4M3FN[:127]
    m = np.minimum(np.abs(z), np.float32(448.0))
    high = np.searchsorted(positive_values, m, side="left")
    low = np.maximum(high - 1, 0)
    d_low = m - positive_values[low]
    d_high = positive_values[high] - m
    choose_low = (d_low < d_high) | ((d_low == d_high) & ((low & 1) == 0))
    codes = np.where(choose_low, low, high).astype(np.uint8)
    return codes | np.where(np.signbit(z), 0x80, 0).astype(np.uint8)


def activation_blocks(x: np.ndarray, block_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Candidate dynamic activation quantization, per 128-wide input block.

    This candidate requires official pinned-kernel validation. All-zero scale=1.
    Weight tensors, scales and checkpoint files are NEVER modified.
    """
    if block_size != 128:
        raise ValueError("only 128-wide activation blocks are supported")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("finite 1-D activation required")
    bits = np.empty(x.shape, dtype=np.uint8)
    inv = np.empty((math.ceil(x.size / block_size),), dtype=np.float32)
    for j, start in enumerate(range(0, x.size, block_size)):
        chunk = x[start:start + block_size]
        maximum = np.max(np.abs(chunk), initial=0.0)
        inv[j] = np.float32(maximum / 448.0) if maximum > 0 else np.float32(1.0)
        bits[start:start + chunk.size] = _lookup_quantize(chunk / inv[j])
    return bits, inv


def activation_roundtrip(x: np.ndarray) -> np.ndarray:
    bits, scales = activation_blocks(x)
    out = np.empty(len(bits), dtype=np.float32)
    for j, start in enumerate(range(0, len(bits), 128)):
        out[start:start + 128] = E4M3FN[bits[start:start + 128]] * scales[j]
    return out


def _independent_fp8_decode(bits: np.ndarray) -> np.ndarray:
    """Independent arithmetic FP8 decoder, not reusing existing E4M3FN table."""
    q = np.asarray(bits, dtype=np.uint8)
    mag = q & 0x7F
    exponent = (mag >> 3).astype(np.int32)
    frac = (mag & 7).astype(np.float64)
    values = np.where(exponent == 0, frac * (2.0 ** -9),
                      (1.0 + frac / 8.0) * np.exp2(exponent - 7)).astype(np.float32)
    if np.any(mag == 0x7F):
        raise ValueError("source contains E4M3FN NaN")
    return np.where((q & 0x80) != 0, -values, values).astype(np.float32)


def oracle_matvec(weight, scale, x: np.ndarray) -> np.ndarray:
    """Independent dense NumPy oracle; intentionally unlike bounded streaming.

    Full matrix is F32 in transient memory: this oracle needs sufficient RAM.
    Never persist its temporary dequantized values as checkpoint weights.
    """
    rows, cols = weight.shape
    bits = np.frombuffer(read_bytes(weight), dtype=np.uint8).reshape(rows, cols)
    dequant = _independent_fp8_decode(bits)
    sg = np.frombuffer(read_bytes(scale), dtype="<f4").reshape(scale.shape)
    for r in range(sg.shape[0]):
        for c in range(sg.shape[1]):
            dequant[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] *= sg[r, c]
    return (dequant @ x).astype(np.float32)


def _silu(x):
    sig = np.empty_like(x)
    positive = x >= 0
    sig[positive] = 1 / (1 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    sig[~positive] = ex / (1 + ex)
    return x * sig


def _forward(index, layer, expert, x, *, mode, device, oracle=False):
    spec = names(layer, expert)
    def mv(which, v):
        w, s = (index[n] for n in spec[which])
        if mode == "dynamic-candidate":
            v = activation_roundtrip(v)
        return oracle_matvec(w, s, v) if oracle else block_fp8_matvec(w, s, v, device=device)
    gate, up = mv("gate_proj", x), mv("up_proj", x)
    product = (_silu(gate) * up).astype(np.float32)
    down = mv("down_proj", product)
    return {"gate": gate, "up": up, "swiglu": product, "output": down}


def _resource_peak():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        return None


def _gpu_stats(device: str):
    if device != "cuda":
        return None
    import torch
    torch.cuda.synchronize()
    return {"allocated_peak_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_peak_bytes": int(torch.cuda.max_memory_reserved()),
            "device": torch.cuda.get_device_name()}


def run(index: TensorIndex, layer: int, expert: int, *, mode="dequant-f32", device="cpu",
        seed: int = 42, cap_bytes=DEFAULT_READ_CAP, rtol=2e-5, atol=2e-4,
        expected_hidden=None, expected_intermediate=None) -> dict:
    if mode not in ("dequant-f32", "dynamic-candidate"):
        raise ValueError("unsupported activation mode")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    layout = validate_layout(index, layer, expert, hidden_size=expected_hidden,
                             intermediate_size=expected_intermediate)
    if layout["source_payload_bytes"] > cap_bytes:
        raise MemoryError("expert exceeds explicit source payload budget")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA hardware is unavailable")
        torch.cuda.reset_peak_memory_stats()
    x = np.random.default_rng(seed).normal(0, 0.1, layout["hidden"]).astype(np.float32)
    rss_before = _resource_peak()
    rss_samples = [rss_before] if rss_before is not None else []
    stop_sampler = Event()
    def _sample_rss():
        while not stop_sampler.wait(0.002):
            v = _resource_peak()
            if v is not None:
                rss_samples.append(v)
    sampler = Thread(target=_sample_rss, daemon=True)
    sampler.start()
    tracemalloc.start()
    try:
        with meter_reads() as candidate_reads:
            t0 = time.perf_counter()
            candidate = _forward(index, layer, expert, x, mode=mode, device=device)
            candidate_s = time.perf_counter() - t0
        with meter_reads() as reference_reads:
            t1 = time.perf_counter()
            oracle = _forward(index, layer, expert, x, mode=mode, device="cpu", oracle=True)
            oracle_s = time.perf_counter() - t1
    finally:
        stop_sampler.set()
        sampler.join()
        _, heap_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    rss_after = _resource_peak()
    if rss_after is not None:
        rss_samples.append(rss_after)
    comparison = {}
    pass_all = True
    for k in candidate:
        a, b = candidate[k], oracle[k]
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_abs = float(np.max(diff))
        max_rel = float(np.max(diff / np.maximum(np.abs(b.astype(np.float64)), 1e-6)))
        ok = bool(np.allclose(a, b, rtol=rtol, atol=atol))
        comparison[k] = {"max_abs": max_abs, "max_rel_floor_1e-6": max_rel, "within_tolerance": ok}
        pass_all &= ok
    return {"passed_internal_oracle": pass_all,
            "official_glm53_kernel_parity": "NOT_TESTED",
            "full_glm53_inference": False,
            "mode": mode, "device": device, "layout": layout,
            "source": {"repo": MODEL, "pinned_revision_required": REVISION,
                       "independent_source_attestation": "requires acquisition provenance"},
            "seed": seed, "tolerances": {"rtol": rtol, "atol": atol},
            "latency_seconds": {"candidate": candidate_s, "oracle": oracle_s},
            "memory_bytes": {"process_rss_before": rss_before,
                             "process_rss_after": rss_after,
                             "python_tracemalloc_peak": heap_peak,
                             "process_rss_peak": max(rss_samples) if rss_samples else None,
                             "peak_method": "2ms sampled process RSS; may miss sub-sample transient peaks"},
            "gpu": _gpu_stats(device),
            "read_bytes": {"candidate_logical": candidate_reads.bytes,
                           "reference_logical": reference_reads.bytes,
                           "candidate_read_calls": candidate_reads.calls,
                           "source_payload_one_forward": layout["source_payload_bytes"],
                           "measured_physical_disk_bytes": None,
                           "note": "Instrumented successful source read bytes; OS page cache/physical SSD bytes not tracked"},
            "numerical": comparison,
            "output_sha256": hashlib.sha256(candidate["output"].tobytes()).hexdigest(),
            "note": "Independent local mathematical oracle, NOT official GLM FP8 kernel equivalence"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="One original GLM FP8 expert, NEVER full model")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("dequant-f32", "dynamic-candidate"), default="dequant-f32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-mib", type=int, default=48)
    parser.add_argument("--evidence", type=Path, default=None)
    args = parser.parse_args(argv)
    index = TensorIndex.from_selected(args.model_dir,
                [n for pair in names(args.layer, args.expert).values() for n in pair])
    report = run(index, args.layer, args.expert, mode=args.mode, device=args.device,
                 seed=args.seed, cap_bytes=args.max_source_mib * 1024 * 1024)
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed_internal_oracle"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
