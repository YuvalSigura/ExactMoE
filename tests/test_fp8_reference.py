"""Unit fixtures for original-byte E4M3 block matvec and three-projection MoE.

These fixtures are independent safetensors files with HAND-CODED FP8 bytes,
not a claim of numerical parity against the full GLM-5.3 runtime.
"""
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from exactmoe.fp8_reference import E4M3FN, block_fp8_matvec, forward_expert
from exactmoe.storage import TensorIndex


def save_raw(path: Path, tensors: dict):
    header = {}
    buf = bytearray()
    for name, (dtype, shape, raw) in sorted(tensors.items()):
        header[name] = {
            "dtype": dtype, "shape": list(shape),
            "data_offsets": [len(buf), len(buf) + len(raw)],
        }
        buf.extend(raw)
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr + buf)


def make_pair(tmp_path, *, rows=131, cols=259, block=128):
    name = "model.layers.3.mlp.experts.0.gate_proj.weight"
    bits = np.random.default_rng(3).choice(
        np.array([0x00, 0x20, 0x30, 0x38, 0x40, 0xB8], dtype=np.uint8),
        size=(rows, cols),
    )
    scale_shape = (math.ceil(rows / block), math.ceil(cols / block))
    scales = np.random.default_rng(2).uniform(0.25, 1.2, scale_shape).astype("<f4")
    save_raw(tmp_path / "model.safetensors", {
        name: ("F8_E4M3", bits.shape, bits.tobytes()),
        name + "_scale_inv": ("F32", scale_shape, scales.tobytes()),
    })
    idx = TensorIndex.from_directory(tmp_path)
    return idx[name], idx[name + "_scale_inv"], bits, scales


def test_e4m3fn_hardcoded_values():
    assert E4M3FN[0x38] == 1
    assert E4M3FN[0x30] == 0.5
    assert E4M3FN[0x01] == 2 ** -9
    assert E4M3FN[0x7E] == 448
    assert E4M3FN[0xB8] == -1
    assert np.isnan(E4M3FN[0x7F])
    assert np.isnan(E4M3FN[0xFF])


def test_tiled_cpu_matvec_matches_independent_dense_reference(tmp_path):
    w, s, bits, scales = make_pair(tmp_path)
    x = np.random.default_rng(4).normal(0, 0.1, size=w.shape[1]).astype(np.float32)
    actual = block_fp8_matvec(w, s, x)
    dense = E4M3FN[bits].copy()
    for i in range(2):
        for j in range(3):
            dense[i*128:(i+1)*128,j*128:(j+1)*128] *= scales[i,j]
    expected = dense @ x
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_reject_wrong_shape_and_scales(tmp_path):
    w, s, _, _ = make_pair(tmp_path)
    with pytest.raises(ValueError, match="expected vector"):
        block_fp8_matvec(w, s, np.ones(11, dtype=np.float32))
    with pytest.raises(ValueError, match="device"):
        block_fp8_matvec(w, s, np.ones(259, dtype=np.float32), device="npu")
    with pytest.raises(ValueError, match="blockwise scale shape"):
        block_fp8_matvec(w, s, np.ones(259, dtype=np.float32), block_size=64)


def test_three_projection_native_named_expert(tmp_path):
    base = "model.layers.3.mlp.experts.0."
    tensors = {}
    shapes = {"gate_proj": (7, 5), "up_proj": (7, 5), "down_proj": (5, 7)}
    dense = {}
    for p, shape in shapes.items():
        raw = np.full(shape, 0x38, dtype=np.uint8)  # E4M3 = 1
        scales = np.array([[0.25]], dtype="<f4")
        name = base + p + ".weight"
        tensors[name] = ("F8_E4M3", shape, raw.tobytes())
        tensors[name + "_scale_inv"] = ("F32", (1, 1), scales.tobytes())
        dense[p] = np.ones(shape, dtype=np.float32) * 0.25
    save_raw(tmp_path / "model.safetensors", tensors)
    idx = TensorIndex.from_directory(tmp_path)
    x = np.array([0.5, -0.1, 0.3, 0.2, 0.6], dtype=np.float32)
    report = forward_expert(idx, 3, 0, x)
    gate = dense["gate_proj"] @ x
    up = dense["up_proj"] @ x
    expected = dense["down_proj"] @ (gate/(1+np.exp(-gate))*up)
    np.testing.assert_allclose(report["output_preview"], expected, rtol=1e-5, atol=1e-5)
    assert report["source_weight_bytes"] == 105
    assert report["source_scale_bytes"] == 12
    assert report["numerical_reference_parity"] == "not yet independently verified"
