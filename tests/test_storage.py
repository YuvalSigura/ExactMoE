from pathlib import Path
import json
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file

from exactmoe.storage import TensorIndex, read_bytes, read_first_axis_slice, scan_safetensors


def test_unsharded_scan_reads_exact_slice(tmp_path: Path):
    array = np.arange(24, dtype="float32").reshape(3, 2, 4)
    save_file({"layers.0.mlp.experts.up_proj.weight": array}, str(tmp_path / "model.safetensors"))
    ix = TensorIndex.from_directory(tmp_path)
    rec = ix["layers.0.mlp.experts.up_proj.weight"]
    assert rec.shape == (3, 2, 4)
    assert len(ix.packed_expert_candidates(expected_count=3)) == 1
    data = read_first_axis_slice(rec, 1)
    assert data == array[1].tobytes()
    assert read_bytes(rec) == array.tobytes()
    with pytest.raises(IndexError):
        read_first_axis_slice(rec, 3)
    with pytest.raises(ValueError):
        read_bytes(rec, offset=rec.nbytes + 1)


def test_sharded_manifest_and_mismatch(tmp_path: Path):
    save_file({"a": np.ones(2, dtype="float32")}, str(tmp_path / "a.safetensors"))
    save_file({"b": np.ones(3, dtype="float32")}, str(tmp_path / "b.safetensors"))
    manifest = tmp_path / "model.safetensors.index.json"
    manifest.write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
    assert len(TensorIndex.from_directory(tmp_path)) == 2
    manifest.write_text(json.dumps({"weight_map": {"a": "b.safetensors", "b": "a.safetensors"}}))
    with pytest.raises(ValueError, match="mismatch"):
        TensorIndex.from_directory(tmp_path)


def test_reject_path_traversal(tmp_path: Path):
    save_file({"a": np.ones(2, dtype="float32")}, str(tmp_path / "a.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../a.safetensors"}})
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        TensorIndex.from_directory(tmp_path)


def test_truncated_header_and_duplicates(tmp_path: Path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(struct.pack("<Q", 1000) + b"{}")
    with pytest.raises(ValueError, match="header exceeds file"):
        scan_safetensors(bad)
    duplicate = b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    bad.write_bytes(struct.pack("<Q", len(duplicate)) + duplicate + b"xxxx")
    with pytest.raises(ValueError, match="duplicate"):
        scan_safetensors(bad)


def test_fp8_opaque_bytes_preserved(tmp_path: Path):
    header = json.dumps({
        "layers.0.mlp.experts.gate_proj.weight":
        {"dtype": "F8_E4M3", "shape": [2, 4], "data_offsets": [0, 8]},
    }).encode()
    raw = bytes(range(8))
    path = tmp_path / "fp8.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
    rec = next(iter(scan_safetensors(path).values()))
    assert rec.dtype == "F8_E4M3"
    assert read_first_axis_slice(rec, 1) == raw[4:]
