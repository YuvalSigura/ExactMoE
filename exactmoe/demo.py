"""Small executable correctness fixture for an unfused float32 SwiGLU MoE.

This is not a GLM adapter. Its routing rule is plain softmax top-k, while GLM
uses its own routing and normalization. The example is deliberately independent
of GLM so paging correctness can be established in isolation.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import Any

from .pager import ExpertPager
from .storage import TensorIndex


ROUTER_NAME = "layers.0.mlp.gate.weight"
PARTS = ("gate_proj", "up_proj", "down_proj")


def _deps():
    import numpy as np
    from safetensors.numpy import load_file, save_file
    return np, load_file, save_file


def make_fixture(directory: str | Path, experts: int = 8, hidden: int = 16,
                 intermediate: int = 32, seed: int = 7) -> Path:
    np, _, save_file = _deps()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    weights = {
        ROUTER_NAME: rng.normal(0, 0.2, (experts, hidden)).astype("float32"),
        "layers.0.mlp.experts.gate_proj.weight":
            rng.normal(0, 0.2, (experts, intermediate, hidden)).astype("float32"),
        "layers.0.mlp.experts.up_proj.weight":
            rng.normal(0, 0.2, (experts, intermediate, hidden)).astype("float32"),
        "layers.0.mlp.experts.down_proj.weight":
            rng.normal(0, 0.2, (experts, hidden, intermediate)).astype("float32"),
    }
    save_file(weights, str(root / "model.safetensors"))
    return root


def _forward_expert(np: Any, x: Any, gate: Any, up: Any, down: Any) -> Any:
    g = x @ gate.T
    activated = g / (1.0 + np.exp(-g))
    return ((activated * (x @ up.T)) @ down.T).astype("float32")


def _selection(np: Any, x: Any, router: Any, k: int):
    logits = router @ x
    # Stable tie-breaking by expert ID. Softmax over selected experts.
    selected = sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))[:k]
    selected_logits = logits[selected]
    p = np.exp(selected_logits - np.max(selected_logits))
    p = p / np.sum(p)
    return selected, p


def _as_f32(np: Any, raw: bytes, shape: tuple[int, ...], dtype: str):
    if dtype != "F32":
        raise NotImplementedError(
            f"demo supports only F32; {dtype} raw bytes stay preserved for a future kernel"
        )
    return np.frombuffer(raw, dtype="<f4").reshape(shape)


def forward_streaming(pager: ExpertPager, x: Any, top_k: int = 2):
    np, _, _ = _deps()
    index = pager.index
    router_record = index[ROUTER_NAME]
    router = _as_f32(np, pager.get(ROUTER_NAME), router_record.shape, router_record.dtype)
    selected, probs = _selection(np, x, router, top_k)
    # Selection comes from actual router, never from speculative predictions.
    for expert in selected:
        for part in PARTS:
            pager.prefetch(f"layers.0.mlp.experts.{part}.weight", expert)
    out = np.zeros_like(x)
    for expert, prob in zip(selected, probs):
        tensors = []
        for part in PARTS:
            name = f"layers.0.mlp.experts.{part}.weight"
            rec = index[name]
            raw = pager.get(name, expert)
            tensors.append(_as_f32(np, raw, rec.shape[1:], rec.dtype))
        out += prob * _forward_expert(np, x, *tensors)
    return out, selected


def forward_reference(directory: str | Path, x: Any, top_k: int = 2):
    np, load_file, _ = _deps()
    weights = load_file(str(Path(directory) / "model.safetensors"))
    selected, probs = _selection(np, x, weights[ROUTER_NAME], top_k)
    out = np.zeros_like(x)
    for expert, prob in zip(selected, probs):
        matrices = [
            weights[f"layers.0.mlp.experts.{part}.weight"][expert] for part in PARTS
        ]
        out += prob * _forward_expert(np, x, *matrices)
    return out, selected


def validate_fixture(directory: str | Path, cache_bytes: int = 16_384,
                     top_k: int = 2, seed: int = 11):
    np, _, _ = _deps()
    index = TensorIndex.from_directory(directory)
    hidden = index[ROUTER_NAME].shape[1]
    x = np.random.default_rng(seed).normal(size=hidden).astype("float32")
    expected, reference_ids = forward_reference(directory, x, top_k=top_k)
    with ExpertPager(index, cache_bytes=cache_bytes) as pager:
        actual, streaming_ids = forward_streaming(pager, x, top_k=top_k)
        stats = pager.cache.stats()
    max_error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert reference_ids == streaming_ids
    return {
        "passed": True,
        "max_abs_error": max_error,
        "selected_experts": streaming_ids,
        "cache": stats.__dict__,
    }


def run_demo(bench_steps: int = 0) -> dict[str, Any]:
    np, _, _ = _deps()
    with tempfile.TemporaryDirectory(prefix="exactmoe-") as d:
        make_fixture(d)
        outcome = validate_fixture(d)
        if bench_steps > 0:
            index = TensorIndex.from_directory(d)
            inputs = np.random.default_rng(23).normal(
                size=(bench_steps, 16)
            ).astype("float32")
            start = time.perf_counter()
            for x in inputs:
                forward_reference(d, x)
            ref_ms = (time.perf_counter() - start) * 1000 / bench_steps
            with ExpertPager(index, cache_bytes=16_384) as pager:
                start = time.perf_counter()
                for x in inputs:
                    forward_streaming(pager, x)
                streamed_ms = (time.perf_counter() - start) * 1000 / bench_steps
            outcome["benchmark"] = {
                "tokens": bench_steps,
                "reference_ms_per_step": round(ref_ms, 3),
                "paged_ms_per_step": round(streamed_ms, 3),
                "note": "Toy CPU timing only; not a GLM performance prediction.",
            }
        return outcome
