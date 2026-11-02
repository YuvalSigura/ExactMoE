"""Bounded original-byte FP8 expert forward, for one routed GLM expert.

Experimental *reference/fallback* computation, NOT an optimized native FP8
kernel or a full GLM-5.3 decoder. Reads full rows in bounded chunks; computes
with float32 values transiently reconstructed from ORIGINAL E4M3 weight bytes
and their inverse-scale tensor. Never modifies or rewrites checkpoint weights.

The original router must select experts elsewhere; this module never invents a
router decision. Numerical parity with GLM's production FP8 kernels is a
separate, still-open validation gate.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .storage import TensorIndex, TensorRecord, read_bytes

BLOCK = 128
MAX_STAGING_BYTES = 32 * 1024 * 1024


def e4m3fn_table() -> np.ndarray:
    """IEEE-like E4M3FN values: exponent bias 7; 0x7F/0xFF are NaN."""
    raw = np.arange(256, dtype=np.uint16)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    sign = np.where((raw & 128) != 0, -1.0, 1.0)
    subnormal = np.ldexp(mantissa.astype(np.float32) / 8.0, -6)
    normal = np.ldexp(1.0 + mantissa.astype(np.float32) / 8.0,
                      exponent.astype(np.int32) - 7)
    values = (sign * np.where(exponent == 0, subnormal, normal)).astype(np.float32)
    values[(exponent == 15) & (mantissa == 7)] = np.nan
    values.setflags(write=False)
    return values


E4M3FN = e4m3fn_table()


def _check_weight_and_scale(weight: TensorRecord, scale: TensorRecord,
                            block_size: int) -> tuple[int, int]:
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block size must be positive integer")
    if weight.dtype != "F8_E4M3" or len(weight.shape) != 2:
        raise ValueError(f"{weight.name}: only rank-2 native E4M3 FP8 weights supported")
    if scale.dtype != "F32" or len(scale.shape) != 2:
        raise ValueError(f"{scale.name}: inverse scales must be rank-2 F32")
    rows, cols = weight.shape
    if rows < 1 or cols < 1 or weight.nbytes != rows * cols:
        raise ValueError("invalid FP8 tensor byte length")
    if cols > MAX_STAGING_BYTES:
        raise ValueError("one FP8 matrix row exceeds the bounded staging budget")
    if scale.shape != (math.ceil(rows / block_size), math.ceil(cols / block_size)):
        raise ValueError(f"{scale.name}: blockwise scale shape mismatch")
    if scale.nbytes != 4 * math.prod(scale.shape):
        raise ValueError("invalid inverse-scale byte length")
    if weight.name + "_scale_inv" != scale.name:
        raise ValueError("weight and scale tensor names do not match")
    return rows, cols


def _matvec_cpu(weight: TensorRecord, scale: TensorRecord, x: np.ndarray,
                block_size: int) -> np.ndarray:
    rows, cols = _check_weight_and_scale(weight, scale, block_size)
    out = np.zeros(rows, dtype=np.float32)
    row_batch = max(1, min(block_size, MAX_STAGING_BYTES // cols))
    # Split at scale-row boundaries. The largest source read stays under 32 MiB.
    start = 0
    while start < rows:
        nrows = min(row_batch, rows - start, block_size - start % block_size)
        raw = read_bytes(weight, offset=start * cols, length=nrows * cols)
        bit_rows = np.frombuffer(raw, dtype=np.uint8).reshape(nrows, cols)
        scale_row_index = start // block_size
        ncol_blocks = scale.shape[1]
        scales = np.frombuffer(
            read_bytes(scale, offset=scale_row_index * ncol_blocks * 4,
                       length=ncol_blocks * 4), dtype="<f4"
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError(f"{scale.name}: invalid inverse scales")
        accum = np.zeros(nrows, dtype=np.float32)
        for col in range(0, cols, block_size):
            stop = min(cols, col + block_size)
            matrix = E4M3FN[bit_rows[:, col:stop]] * scales[col // block_size]
            if not np.isfinite(matrix).all():
                raise ValueError("FP8 checkpoint contains E4M3FN NaN")
            accum += matrix @ x[col:stop]
        out[start:start + nrows] = accum
        start += nrows
    return out


def _matvec_cuda(weight: TensorRecord, scale: TensorRecord, x: np.ndarray,
                 block_size: int) -> np.ndarray:
    """Correctness-oriented GPU fallback; float32 matvec, not native FP8 TensorCores.

    GPU receives only <=32 MiB of staged weights at a time; no retained kernels
    or full decoder model. Live VRAM may exceed the staged byte budget due to
    PyTorch CUDA allocator. Check actual device memory before model-level use.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch required for --device cuda") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    rows, cols = _check_weight_and_scale(weight, scale, block_size)
    x_gpu = torch.as_tensor(np.array(x, copy=True), device="cuda", dtype=torch.float32)
    out = np.empty(rows, dtype=np.float32)
    row_batch = max(1, min(block_size, MAX_STAGING_BYTES // cols))
    with torch.no_grad():
        start = 0
        while start < rows:
            nrows = min(row_batch, rows - start, block_size - start % block_size)
            bits = np.frombuffer(
                read_bytes(weight, offset=start * cols, length=nrows * cols),
                dtype=np.uint8,
            ).reshape(nrows, cols)
            ncol_blocks = scale.shape[1]
            scales = np.frombuffer(
                read_bytes(scale, offset=(start // block_size) * ncol_blocks * 4,
                           length=ncol_blocks * 4), dtype="<f4"
            )
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError("invalid inverse scales")
            accum = torch.zeros(nrows, dtype=torch.float32, device="cuda")
            for col in range(0, cols, block_size):
                stop = min(cols, col + block_size)
                data = E4M3FN[bits[:, col:stop]] * scales[col // block_size]
                if not np.isfinite(data).all():
                    raise ValueError("FP8 checkpoint contains E4M3FN NaN")
                a = torch.as_tensor(np.array(data, copy=True), device="cuda")
                accum.add_(torch.mv(a, x_gpu[col:stop]))
            out[start:start + nrows] = accum.cpu().numpy()
            start += nrows
    return out


def block_fp8_matvec(weight: TensorRecord, scale: TensorRecord, vector: Any,
                     *, block_size: int = BLOCK, device: str = "cpu") -> np.ndarray:
    """Compute W @ vector from immutable FP8 bytes, never materializing all W."""
    _, cols = _check_weight_and_scale(weight, scale, block_size)
    x = np.asarray(vector, dtype=np.float32)
    if x.ndim != 1 or x.shape[0] != cols:
        raise ValueError(f"expected vector of shape ({cols},)")
    if not np.isfinite(x).all():
        raise ValueError("input has non-finite values")
    if device == "cpu":
        return _matvec_cpu(weight, scale, x, block_size)
    if device == "cuda":
        return _matvec_cuda(weight, scale, x, block_size)
    raise ValueError("device must be cpu or cuda")


def forward_expert(index: TensorIndex, layer: int, expert: int, vector: Any,
                   *, device: str = "cpu", block_size: int = BLOCK) -> dict[str, Any]:
    """Three native named matrices (gate/up/down) + FP8 scales, one token.

    Does not include the shared expert, router probabilities, residual, attention
    or reference-logit comparison; deliberately not presented as GLM inference.
    """
    if layer < 0 or expert < 0:
        raise ValueError("layer/expert must be nonnegative")
    names = {
        part: f"model.layers.{layer}.mlp.experts.{expert}.{part}.weight"
        for part in ("gate_proj", "up_proj", "down_proj")
    }
    tensors = {
        part: (index[name], index[name + "_scale_inv"])
        for part, name in names.items()
    }
    input_vec = np.asarray(vector, dtype=np.float32)
    start = time.perf_counter()
    gate = block_fp8_matvec(*tensors["gate_proj"], input_vec,
                            block_size=block_size, device=device)
    up = block_fp8_matvec(*tensors["up_proj"], input_vec,
                          block_size=block_size, device=device)
    if gate.shape != up.shape:
        raise ValueError("gate and up projections disagree on shape")
    # Stable FP32 SiLU; a reference FP8 runtime may differ in accumulation.
    # Numerically stable FP32 sigmoid for SwiGLU.
    sigmoid = np.where(gate >= 0, 1 / (1 + np.exp(-np.maximum(gate, 0))),
                       np.exp(np.minimum(gate, 0)) / (1 + np.exp(np.minimum(gate, 0))))
    act = (gate * sigmoid * up).astype(np.float32)
    result = block_fp8_matvec(*tensors["down_proj"], act,
                              block_size=block_size, device=device)
    if result.shape != input_vec.shape:
        raise ValueError("down projection output dimension mismatch")
    if not np.isfinite(result).all():
        raise ValueError("non-finite output")
    duration = time.perf_counter() - start
    return {
        "passed": True,
        "scope": "one GLM routed expert only: no original router, shared expert or DSA",
        "layer": layer, "expert": expert, "device": device,
        "input_shape": list(input_vec.shape), "intermediate_size": len(gate),
        "output_shape": list(result.shape),
        "seconds": round(duration, 4),
        "source_weight_bytes": sum(t[0].nbytes for t in tensors.values()),
        "source_scale_bytes": sum(t[1].nbytes for t in tensors.values()),
        "output_preview": result[:8].tolist(),
        "checkpoint_data_unchanged": True,
        "numerical_reference_parity": "not yet independently verified",
    }
