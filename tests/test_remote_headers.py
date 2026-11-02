"""Remote header inspection uses synthetic responses and NEVER contacts a model hub."""
import json
import struct

import pytest

from exactmoe import remote_headers


class FakeResponse:
    def __init__(self, blob, start, end, status=206):
        self.status = status
        self.data = blob[start:end + 1]
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(blob)}"}
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def read(self, n):
        return self.data[:n]


def fake_urlopen(blob, status=206):
    def open_request(request, timeout=45):
        first, last = [int(s) for s in request.get_header("Range")[6:].split("-")]
        return FakeResponse(blob, first, last, status)
    return open_request


def payload(scale_shape=(2, 2)):
    header = {}
    offset = 0
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for suffix, shape, dtype, itemsize in (
            ("weight", (256, 256), "F8_E4M3", 1),
            ("weight_scale_inv", scale_shape, "F32", 4),
        ):
            name = f"model.layers.3.mlp.experts.0.{projection}.{suffix}"
            nbytes = shape[0] * shape[1] * itemsize
            header[name] = {
                "dtype": dtype, "shape": list(shape),
                "data_offsets": [offset, offset + nbytes]
            }
            offset += nbytes
    data = json.dumps(header).encode()
    return struct.pack("<Q", len(data)) + data + bytes(offset)


def call(monkeypatch, blob, **kwargs):
    monkeypatch.setattr(remote_headers, "urlopen", fake_urlopen(blob))
    return remote_headers.inspect_remote_header(
        "zai-org/GLM-5.3",
        "aca966e4e02791568aa6a4ced368624b3d897f42",
        "model-00002-of-00141.safetensors", **kwargs
    )


def test_remote_header_checks_fp8_scale_grid_without_loading_weights(monkeypatch):
    blob = payload()
    result = call(monkeypatch, blob)
    assert result["fp8_expert_weight_scale_pairs_in_shard"] == 3
    assert result["fp8_pair_shape_mismatches"] == 0
    assert result["tensor_count"] == 6
    assert result["downloaded_bytes"] < len(blob) // 10
    assert result["dtype_counts"] == {"F8_E4M3": 3, "F32": 3}


def test_bad_fp8_grid_detected(monkeypatch):
    result = call(monkeypatch, payload(scale_shape=(1, 2)))
    assert result["fp8_pair_shape_mismatches"] == 3


def test_refuses_http_200_even_if_bytes_are_accessible(monkeypatch):
    monkeypatch.setattr(remote_headers, "urlopen", fake_urlopen(payload(), status=200))
    with pytest.raises(RuntimeError, match="206"):
        remote_headers.inspect_remote_header(
            "zai-org/GLM-5.3",
            "aca966e4e02791568aa6a4ced368624b3d897f42",
            "model-00002-of-00141.safetensors"
        )


def test_invalid_shard_and_revision_do_not_access_network():
    with pytest.raises(ValueError):
        remote_headers.inspect_remote_header(
            "zai-org/GLM-5.3", "main", "../escape.safetensors"
        )


def test_reject_overstated_header_length(monkeypatch):
    bad = struct.pack("<Q", remote_headers.MAX_REMOTE_HEADER_BYTES + 1) + b"xx"
    monkeypatch.setattr(remote_headers, "urlopen", fake_urlopen(bad))
    with pytest.raises(ValueError, match="header length"):
        remote_headers.inspect_remote_header(
            "zai-org/GLM-5.3",
            "aca966e4e02791568aa6a4ced368624b3d897f42",
            "model-00002-of-00141.safetensors"
        )
