"""Tests for expert splitting by intermediate-dimension partition."""
import numpy as np
import pytest

from exactmoe.expert_splitter import split_expert, split_expert_summary


def test_split_shapes_and_coverage():
    rng = np.random.default_rng(1)
    intermediate, hidden, n = 1024, 256, 4
    gate = rng.normal(size=(intermediate, hidden)).astype(np.float32)
    up = rng.normal(size=(intermediate, hidden)).astype(np.float32)
    down = rng.normal(size=(hidden, intermediate)).astype(np.float32)

    result = split_expert(gate, up, down, n, name="test")
    assert result["status"] == "SPLIT_COMPLETE"
    assert result["n_splits"] == n
    assert len(result["sub_experts"]) == n

    # Reconstruct and verify exact coverage
    recon_gate = np.concatenate([se["gate"] for se in result["_arrays"]], axis=0)
    recon_down = np.concatenate([se["down"] for se in result["_arrays"]], axis=1)
    np.testing.assert_array_equal(recon_gate, gate)
    np.testing.assert_array_equal(recon_down, down)

    # Intermediate sizes sum to original
    total = sum(se["intermediate_size"] for se in result["sub_experts"])
    assert total == intermediate


def test_uneven_split():
    gate = np.ones((10, 4), dtype=np.float32)
    up = np.ones((10, 4), dtype=np.float32)
    down = np.ones((4, 10), dtype=np.float32)
    result = split_expert(gate, up, down, 3)
    sizes = [se["intermediate_size"] for se in result["sub_experts"]]
    assert sizes == [4, 3, 3]
    assert sum(sizes) == 10


def test_invalid_inputs():
    gate = np.zeros((8, 4))
    up = np.zeros((8, 4))
    down = np.zeros((4, 8))
    with pytest.raises(ValueError):
        split_expert(gate, up, down, 1)
    with pytest.raises(ValueError):
        split_expert(gate, up, np.zeros((5, 8)), 2)


def test_summary_is_json_safe():
    gate = np.zeros((16, 8), dtype=np.float32)
    up = np.zeros((16, 8), dtype=np.float32)
    down = np.zeros((8, 16), dtype=np.float32)
    summary = split_expert_summary(gate, up, down, 2)
    assert "_arrays" not in summary
    assert summary["sub_experts"][0]["gate_shape"] == [8, 8]
