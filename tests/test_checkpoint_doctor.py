import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import pytest

from exactmoe.checkpoint_doctor import checkpoint_status


def test_checkpoint_progress_and_corruption_are_reported(tmp_path: Path):
    save_file({"a": np.arange(6, dtype=np.float32)},
              str(tmp_path / "model-00001-of-00003.safetensors"))
    (tmp_path / "model-00002-of-00003.safetensors").write_bytes(b"truncated")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 72},
        "weight_map": {
            "a": "model-00001-of-00003.safetensors",
            "b": "model-00002-of-00003.safetensors",
            "c": "model-00003-of-00003.safetensors",
        },
    }))
    result = checkpoint_status(tmp_path)
    assert result["expected_shards"] == 3
    assert result["present_shards"] == 1
    assert result["missing_shards"] == 1
    assert result["invalid_or_partial_shards"] == 1
    assert result["validated_tensor_headers"] == 1
    assert result["status"] == "incomplete-or-unverified"
    # Do not automatically trust byte count when headers are not verified.
    skipped = checkpoint_status(tmp_path, verify_headers=False)
    assert skipped["present_shards"] == 2
    assert skipped["headers_checked"] is False
    assert skipped["status"] == "incomplete-or-unverified"


def test_fully_present_structural_report(tmp_path: Path):
    save_file({"weight": np.ones(3, dtype=np.float32)},
              str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-00001-of-00001.safetensors"},
    }))
    report = checkpoint_status(tmp_path)
    assert report["status"] == "verified-structure"
    assert report["expected_tensor_names"] == report["validated_tensor_headers"] == 1
    assert report["reported_source_weight_bytes"] is None


def test_reject_traversal_manifest(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../foreign.safetensors"}})
    )
    with pytest.raises(ValueError, match="unsafe"):
        checkpoint_status(tmp_path)
