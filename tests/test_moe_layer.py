"""Independent small-shape oracle for GLM noaux_tc choice and native FP8 MoE FFN."""
import json
import struct

import numpy as np
import pytest

from exactmoe.moe_layer import forward_moe_ffn, route_noaux_tc, read_dense_vector_or_matrix
from exactmoe.storage import TensorIndex


def write_raw(root, tensors):
    data = bytearray()
    header = {}
    for name, (dtype, shape, raw) in sorted(tensors.items()):
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [len(data), len(data)+len(raw)]}
        data += raw
    h = json.dumps(header).encode("utf-8")
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + data)


def bf16(array):
    f = np.asarray(array, dtype="<f4")
    return (f.view("<u4") >> 16).astype("<u2").tobytes()


def test_grouped_router_bias_only_affects_selection():
    cfg = {"num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
           "norm_topk_prob": True, "routed_scaling_factor": 2.5,
           "scoring_func": "sigmoid", "topk_method": "noaux_tc"}
    # Experts 0,1 in group 0, and 2,3 in group 1.
    w = np.array([[2, 0], [2, 0], [-2, 0], [-2, 0]], dtype=np.float32)
    bias = np.array([0, 0, 2, 2], dtype=np.float32)
    result = route_noaux_tc(np.array([1, 1], dtype=np.float32), w, bias, cfg)
    assert set(result.expert_ids) == {2, 3}
    assert sum(result.weights) == pytest.approx(2.5)
    with pytest.raises(NotImplementedError):
        route_noaux_tc(np.ones(2), w, bias, {**cfg, "scoring_func": "softmax"})


def test_tiny_original_byte_routed_and_shared_ffn(tmp_path):
    t = {}
    p = "model.layers.3.mlp."
    gate = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]], dtype=np.float32)
    t[p + "gate.weight"] = ("BF16", gate.shape, bf16(gate))
    bias = np.array([0, 2], dtype="<f4")
    t[p + "gate.e_score_correction_bias"] = ("F32", (2,), bias.tobytes())
    # Original named, immutable FP8 weight bytes and separate inverse scales.
    for expert in range(2):
        val = 0x38 if expert == 0 else 0x40
        for proj, shape in (("gate_proj", (3,4)), ("up_proj", (3,4)),
                            ("down_proj", (4,3))):
            name = p + f"experts.{expert}.{proj}.weight"
            t[name] = ("F8_E4M3", shape, bytes([val]) * np.prod(shape))
            t[name + "_scale_inv"] = ("F32", (1,1),
                                      np.array([[0.25]],dtype="<f4").tobytes())
    for proj, shape in (("gate_proj", (3,4)), ("up_proj", (3,4)),
                        ("down_proj", (4,3))):
        name = p + f"shared_experts.{proj}.weight"
        t[name] = ("F8_E4M3", shape, bytes([0x38]) * np.prod(shape))
        t[name + "_scale_inv"] = ("F32", (1,1),
                                  np.array([[0.125]],dtype="<f4").tobytes())
    write_raw(tmp_path, t)
    idx = TensorIndex.from_directory(tmp_path)
    np.testing.assert_array_equal(read_dense_vector_or_matrix(idx[p+"gate.weight"]), gate)
    x = np.array([.2,.4,-.1,.3],dtype=np.float32)
    cfg = {"num_experts_per_tok": 1, "n_group": 1, "topk_group": 1,
           "norm_topk_prob": True, "routed_scaling_factor": 2.5,
           "n_shared_experts": 1, "scoring_func": "sigmoid",
           "topk_method": "noaux_tc"}
    report = forward_moe_ffn(idx, 3, x, cfg)
    assert report["selected_experts"] == [1]
    # Independently evaluate dense arrays, not tiled reads.
    def ffn(magnitude):
        w_gate = np.ones((3,4), dtype=np.float32)*magnitude
        w_up = w_gate.copy()
        w_down = np.ones((4,3), dtype=np.float32)*magnitude
        g = w_gate@x
        return w_down @ (g/(1+np.exp(-g))*(w_up@x))
    expected = 2.5*ffn(.5) + ffn(.125)
    np.testing.assert_allclose(report["output_preview"], expected, rtol=1e-5, atol=1e-5)


def test_router_rejects_nonfinite_and_invalid_dimensions():
    weight = np.ones((4,2),dtype=np.float32)
    bias = np.zeros(4,dtype=np.float32)
    with pytest.raises(ValueError):
        route_noaux_tc(np.ones(3), weight, bias, {})
    with pytest.raises(ValueError):
        route_noaux_tc(np.ones(2), weight, np.full(4,np.nan), {})
