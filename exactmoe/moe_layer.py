"""CPU reference GLM-style noaux_tc router and one sparse MoE FFN layer.

Reference semantics follow Hugging Face Transformers GlmMoeDsaTopkRouter:
router weights and logits in float32; sigmoid; correction bias for selection
ONLY; optional group filtering by top-2 sum; unbiased score normalization;
routed scaling; all original chosen experts plus always-on shared expert.

No DSA, residual, post-attention norm, embeddings, or decoder here. This is a
single hidden-state FFN stage, and not proof of full model fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .fp8_reference import block_fp8_matvec, expert_values
from .storage import TensorIndex, TensorRecord, read_bytes


@dataclass(frozen=True)
class RouteResult:
    expert_ids: tuple[int, ...]
    weights: tuple[float, ...]


def read_dense_vector_or_matrix(record: TensorRecord) -> np.ndarray:
    """Small control tensors only. BF16 reconstruction is exact in float32."""
    if record.nbytes > 32 * 1024 * 1024:
        raise MemoryError("control tensor too large for one-shot materialization")
    raw = read_bytes(record)
    if record.dtype == "F32":
        out = np.frombuffer(raw, "<f4").copy()
    elif record.dtype == "BF16":
        data = np.frombuffer(raw, "<u2").astype("<u4")
        out = (data << 16).view("<f4")
    else:
        raise ValueError(f"unsupported control tensor dtype {record.dtype}")
    expected = math.prod(record.shape)
    if len(out) != expected:
        raise ValueError("control tensor byte length mismatch")
    return out.reshape(record.shape)


def route_noaux_tc(hidden: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                   config: Mapping[str, Any]) -> RouteResult:
    x = np.asarray(hidden, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    b = np.asarray(bias, dtype=np.float32)
    if x.ndim != 1 or w.ndim != 2 or w.shape[1] != len(x):
        raise ValueError("invalid hidden/router weight shape")
    n = w.shape[0]
    if b.shape != (n,) or not np.isfinite(x).all() or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("router bias shape or non-finite values")
    if config.get("scoring_func", "sigmoid") != "sigmoid" or config.get("topk_method", "noaux_tc") != "noaux_tc":
        raise NotImplementedError("only GLM sigmoid/noaux_tc routing is supported")
    k = int(config.get("num_experts_per_tok", 8))
    groups = int(config.get("n_group", 1))
    keep_groups = int(config.get("topk_group", 1))
    if not 1 <= k <= n or not 1 <= keep_groups <= groups or n % groups or n // groups < 2:
        raise ValueError("unsupported router top-k or group geometry")
    logits = w @ x
    # Stable sigmoid: same mathematical function as PyTorch, with float32 dtype.
    scores = np.where(logits >= 0, 1 / (1 + np.exp(-np.maximum(logits, 0))),
                      np.exp(np.minimum(logits, 0)) / (1 + np.exp(np.minimum(logits, 0))))
    scores = scores.astype(np.float32)
    selected_scores = scores + b
    if groups > 1:
        grouped = selected_scores.reshape(groups, n // groups)
        top2 = np.sort(grouped, axis=1)[:, -2:].sum(axis=1)
        group_ids = np.argsort(-top2, kind="stable")[:keep_groups]
        group_mask = np.zeros(groups, dtype=bool)
        group_mask[group_ids] = True
        selected_scores = np.where(np.repeat(group_mask, n // groups), selected_scores, -np.inf)
    ids = np.argsort(-selected_scores, kind="stable")[:k]
    raw_weights = scores[ids].astype(np.float32)
    if config.get("norm_topk_prob", True):
        raw_weights /= raw_weights.sum(dtype=np.float32) + np.float32(1e-20)
    raw_weights *= np.float32(config.get("routed_scaling_factor", 1.0))
    if not np.isfinite(raw_weights).all():
        raise ValueError("router generated invalid weights")
    # PyTorch topk tie ordering is not guaranteed; this deterministic CPU path
    # picks lowest expert index for a tie. No tie => same selected *set*.
    return RouteResult(tuple(int(v) for v in ids), tuple(float(v) for v in raw_weights))


def _matvec_plain(record: TensorRecord, vector: np.ndarray,
                  chunk_bytes: int = 8 * 1024 * 1024) -> np.ndarray:
    if len(record.shape) != 2 or record.dtype not in ("BF16", "F32"):
        raise ValueError("expected rank-2 BF16/F32 dense linear")
    rows, cols = record.shape
    x = np.asarray(vector, dtype=np.float32)
    if x.shape != (cols,):
        raise ValueError("dense linear input dimension mismatch")
    itemsize = 2 if record.dtype == "BF16" else 4
    batch = max(1, chunk_bytes // (cols * itemsize))
    if cols * itemsize > chunk_bytes:
        raise ValueError("one dense matrix row exceeds budget")
    out = np.empty(rows, dtype=np.float32)
    start = 0
    while start < rows:
        nrows = min(rows - start, batch)
        raw = read_bytes(record, offset=start * cols * itemsize,
                         length=nrows * cols * itemsize)
        if record.dtype == "BF16":
            bits = np.frombuffer(raw, "<u2").astype("<u4")
            values = (bits << 16).view("<f4").reshape(nrows, cols)
        else:
            values = np.frombuffer(raw, "<f4").reshape(nrows, cols)
        out[start:start+nrows] = values @ x
        start += nrows
    return out


def _linear(index: TensorIndex, name: str, vector: np.ndarray, *, block_size: int,
            device: str) -> np.ndarray:
    record = index[name]
    if record.dtype == "F8_E4M3":
        return block_fp8_matvec(record, index[name + "_scale_inv"], vector,
                                block_size=block_size, device=device)
    return _matvec_plain(record, vector)


def forward_moe_ffn(index: TensorIndex, layer: int, hidden: np.ndarray,
                    config: Mapping[str, Any], *, device: str = "cpu",
                    block_size: int = 128) -> dict[str, Any]:
    """Run original router-selected experts AND shared expert for one FFN input.

    Does NOT perform original GLM residual or normalization; input must already
    be the correct hidden state from the preceding layer operators.
    """
    if layer < 0:
        raise ValueError("layer must be nonnegative")
    x = np.asarray(hidden, dtype=np.float32)
    prefix = f"model.layers.{layer}.mlp."
    t0 = time.perf_counter()
    weight = read_dense_vector_or_matrix(index[prefix + "gate.weight"])
    bias = read_dense_vector_or_matrix(index[prefix + "gate.e_score_correction_bias"])
    route = route_noaux_tc(x, weight, bias, config)
    result = np.zeros_like(x)
    for expert, routing_weight in zip(route.expert_ids, route.weights):
        output = expert_values(index, layer, expert, x, device=device,
                               block_size=block_size)
        result += np.float32(routing_weight) * output

    shared_count = int(config.get("n_shared_experts", 1))
    if shared_count != 1:
        raise NotImplementedError("only one always-on shared expert is supported")
    sp = prefix + "shared_experts."
    gate = _linear(index, sp + "gate_proj.weight", x, block_size=block_size, device=device)
    up = _linear(index, sp + "up_proj.weight", x, block_size=block_size, device=device)
    sigmoid = np.where(gate >= 0, 1 / (1 + np.exp(-np.maximum(gate, 0))),
                       np.exp(np.minimum(gate, 0)) / (1 + np.exp(np.minimum(gate, 0))))
    out_shared = _linear(index, sp + "down_proj.weight",
                         (gate * sigmoid * up).astype(np.float32),
                         block_size=block_size, device=device)
    result += out_shared
    if not np.isfinite(result).all() or result.shape != x.shape:
        raise ValueError("invalid one-layer MoE FFN output")
    return {
        "passed": True,
        "scope": "GLM sparse MoE FFN: router + selected experts + shared expert; NO DSA or residual",
        "layer": layer, "selected_experts": list(route.expert_ids),
        "routing_weights": list(route.weights), "device": device,
        "output_shape": list(result.shape), "output_preview": result[:8].tolist(),
        "seconds": round(time.perf_counter()-t0, 4),
        "reference_parity": "numerical oracle not yet run on full GLM source",
        "checkpoint_modified": False,
    }
