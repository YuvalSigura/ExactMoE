"""Synthetic provenance, FP8, malformed input, and eviction regressions.

The fixtures do NOT establish genuine GLM-5.3 checkpoint or kernel parity.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from exactmoe.expert_validation import (
    names, activation_blocks, activation_roundtrip, validate_layout, run, _lookup_quantize,
)
from exactmoe.expert_acquire import acquire_remote, validate_sample, _request
from exactmoe.storage import TensorIndex


def fake_remote_source(layer=3, expert=0, *, hidden=5, intermediate=7, broken=None):
    shape = {"gate_proj": (intermediate, hidden), "up_proj": (intermediate, hidden),
             "down_proj": (hidden, intermediate)}
    entries = {}
    payload = {}
    cursor = 0
    for i, (proj, (wn, sn)) in enumerate(names(layer, expert).items()):
        r, c = shape[proj]
        weight = np.full((r, c), 0x38 if i < 2 else 0xB8, dtype=np.uint8)
        scale = np.full((math.ceil(r / 128), math.ceil(c / 128)), 0.25 + i * 0.1, dtype="<f4")
        if broken == "scale-nan" and i == 0:
            scale[0, 0] = np.nan
        if broken == "bad-grid" and i == 0:
            scale = np.ones((2, 1), dtype="<f4")
        for n, dtype, arr in [(wn, "F8_E4M3", weight), (sn, "F32", scale)]:
            raw = arr.tobytes()
            payload[n] = raw
            entries[n] = {"dtype": dtype, "shape": list(arr.shape),
                          "data_offsets": [cursor, cursor + len(raw)]}
            cursor += len(raw)
    data = b"".join(payload.values())
    return entries, payload, data


def make_sample(tmp_path, broken=None, layer=3, expert=0, *, cap=48 * 1024 * 1024):
    entries, payload, data = fake_remote_source(layer, expert, broken=broken)
    shard = "model-00001-of-00141.safetensors"
    manifest = {"weight_map": {n: shard for n in entries}}
    config = {"hidden_size": 5, "moe_intermediate_size": 7}
    base = 100
    def metadata(name):
        return (config if name == "config.json" else manifest), 200
    def header(name):
        return entries, base, base + len(data), '"test-source-etag"', base
    def ranged(url, *, start, end):
        assert start >= base and end < base + len(data)
        return data[start - base:end - base + 1], base + len(data), '"test-source-etag"'
    out = tmp_path / "capture"
    return acquire_remote(out, layer, expert, cap_bytes=cap,
                          fetch_json=metadata, fetch_header=header, fetch_range=ranged), out


def _index(out, layer=3, expert=0):
    return TensorIndex.from_selected(
        out, [n for p in names(layer, expert).values() for n in p])


def test_exact_six_tensors_source_range_and_hashes(tmp_path):
    meta, out = make_sample(tmp_path)
    assert meta["revision"] == "aca966e4e02791568aa6a4ced368624b3d897f42"
    assert meta["whole_original_shard_sha256_verified"] is False
    assert validate_sample(out, 3, 0)["verified_sample_hashes"]
    assert len(meta["source"]) == 6
    assert sum(x["nbytes"] for x in meta["source"]) == meta["source_tensor_bytes"]
    report = run(_index(out), 3, 0)
    assert report["passed_internal_oracle"]
    assert report["read_bytes"]["candidate_logical"] == meta["source_tensor_bytes"]
    assert report["read_bytes"]["reference_logical"] == meta["source_tensor_bytes"]
    assert report["official_glm53_kernel_parity"] == "NOT_TESTED"
    assert report["full_glm53_inference"] is False


def test_candidate_dynamic_activation_and_independent_reference(tmp_path):
    _, out = make_sample(tmp_path)
    report = run(_index(out), 3, 0, mode="dynamic-candidate")
    assert report["passed_internal_oracle"]
    assert set(report["numerical"]) == {"gate", "up", "swiglu", "output"}
    assert report["mode"] == "dynamic-candidate"
    assert report["official_glm53_kernel_parity"] == "NOT_TESTED"


def test_activation_rounding_and_zero_blocks():
    assert _lookup_quantize(np.array([1., -1., 448., -448.], dtype=np.float32)).tolist() == [0x38, 0xB8, 0x7E, 0xFE]
    assert int(_lookup_quantize(np.array([1.0625]))[0]) == 0x38
    z = np.zeros(259, dtype=np.float32)
    z[258] = 4.
    bits, scales = activation_blocks(z)
    np.testing.assert_allclose(scales, np.array([1., 1., np.float32(4 / 448)]), rtol=0, atol=0)
    np.testing.assert_allclose(activation_roundtrip(z), z, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="128"):
        activation_blocks(z, block_size=64)
    with pytest.raises(ValueError, match="finite"):
        activation_blocks(np.array([np.nan]))


def test_reject_wrong_scales_and_shapes(tmp_path):
    with pytest.raises(ValueError, match="wrong tensor shape"):
        make_sample(tmp_path, broken="bad-grid")
    with pytest.raises(ValueError, match="inverse scales"):
        make_sample(tmp_path, broken="scale-nan")


def test_wrong_expert_revision_and_provenance_tampering(tmp_path):
    _, out = make_sample(tmp_path)
    meta_path = out / "sample-provenance.json"
    meta = json.loads(meta_path.read_text())
    meta["expert"] = 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="revision/expert"):
        validate_sample(out, 3, 0)
    meta["expert"] = 0
    meta["revision"] = "0" * 40
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="revision/expert"):
        validate_sample(out, 3, 0)
    meta["revision"] = "aca966e4e02791568aa6a4ced368624b3d897f42"
    meta_path.write_text(json.dumps(meta))
    path = out / "expert.safetensors"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="hash"):
        validate_sample(out, 3, 0)


def test_truncated_source_shard(tmp_path):
    _, out = make_sample(tmp_path)
    path = out / "expert.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises((ValueError, EOFError), match="offset|truncated|short"):
        validate_sample(out, 3, 0)


def test_bounded_cap_and_unsafe_identity(tmp_path):
    with pytest.raises(MemoryError, match="cap"):
        make_sample(tmp_path, cap=1)
    with pytest.raises(ValueError, match="sparse layer"):
        names(78, 0)
    with pytest.raises(ValueError, match="routed expert"):
        names(3, 256)


def test_independent_oracle_detects_corrupt_math(tmp_path, monkeypatch):
    _, out = make_sample(tmp_path)
    import exactmoe.expert_validation as ev
    original = ev.block_fp8_matvec
    def corrupted(*args, **kwargs):
        return original(*args, **kwargs) + 0.01
    monkeypatch.setattr(ev, "block_fp8_matvec", corrupted)
    result = ev.run(_index(out), 3, 0)
    assert not result["passed_internal_oracle"]
    assert not result["numerical"]["gate"]["within_tolerance"]


def test_existing_cache_eviction_preserves_bytes():
    from exactmoe.pager import ByteLRU
    cache = ByteLRU(4)
    cache.put(("expert-0", None), b"abcd")
    cache.put(("expert-1", None), b"xyz")
    assert cache.get(("expert-0", None)) is None
    assert cache.get(("expert-1", None)) == b"xyz"
    assert cache.stats().evictions == 1
    assert cache.stats().resident_bytes <= 4


def test_range_rejects_ignored_http_range_and_bad_content_range(monkeypatch):
    import exactmoe.expert_acquire as acquire
    class Response:
        def __init__(self, code, cr="", body=b"abcdefgh"):
            self.status = code
            self.headers = {"Content-Range": cr}
            self.body = body
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return self.body[:n]
    class Opener:
        response = None
        def open(self, request, timeout): return self.response
    fake = Opener()
    monkeypatch.setattr(acquire, "_OPENER", fake)
    url = "https://huggingface.co/zai-org/GLM-5.3/resolve/" + acquire.REVISION + "/model-00001-of-00141.safetensors"
    fake.response = Response(200, body=b"abcd")
    with pytest.raises(RuntimeError, match="206"):
        acquire._request(url, start=0, end=3)
    fake.response = Response(206, cr="bytes 0-4/100", body=b"abcd")
    with pytest.raises(ValueError, match="incorrect remote byte range"):
        acquire._request(url, start=0, end=3)
    fake.response = Response(206, cr="bytes 0-3/100", body=b"abc")
    with pytest.raises(ValueError, match="short/oversized"):
        acquire._request(url, start=0, end=3)

def test_malformed_fp8_weight_dtype_rejected(tmp_path):
    entries, payload, data = fake_remote_source()
    first = names(3, 0)["gate_proj"][0]
    entries[first]["dtype"] = "F32"
    shard = "model-00001-of-00141.safetensors"
    with pytest.raises(ValueError, match="inconsistent tensor payload|incorrect source dtype"):
        acquire_remote(tmp_path / "malformed", 3, 0,
            fetch_json=lambda filename: (
                ({"hidden_size": 5, "moe_intermediate_size": 7}
                 if filename == "config.json" else {"weight_map": {n: shard for n in entries}}), 200),
            fetch_header=lambda name: (entries, 100, 100 + len(data), '"etag"', 100),
            fetch_range=lambda url, *, start, end:
                (data[start - 100:end - 99], 100 + len(data), '"etag"'))


def test_remote_etag_change_is_rejected(tmp_path):
    entries, payload, data = fake_remote_source()
    shard = "model-00001-of-00141.safetensors"
    with pytest.raises(ValueError, match="source shard changed"):
        acquire_remote(tmp_path / "tampered", 3, 0,
            fetch_json=lambda filename: (
                ({"hidden_size": 5, "moe_intermediate_size": 7}
                 if filename == "config.json" else {"weight_map": {n: shard for n in entries}}), 200),
            fetch_header=lambda name: (entries, 100, 100 + len(data), '"etag-A"', 100),
            fetch_range=lambda url, *, start, end:
                (data[start - 100:end - 99], 100 + len(data), '"etag-B"'))


def test_pager_eviction_keeps_exact_source_bytes(tmp_path):
    from exactmoe.pager import ExpertPager
    from exactmoe.storage import read_bytes
    _, out = make_sample(tmp_path)
    index = _index(out)
    first = names(3, 0)["gate_proj"][0]
    expected = read_bytes(index[first])
    with ExpertPager(index, cache_bytes=40, workers=1) as pager:
        assert pager.get(first) == expected
        for pair in names(3, 0).values():
            for tensor in pair:
                pager.get(tensor)
        assert pager.get(first) == expected
        assert pager.cache.stats().evictions >= 1
        assert pager.cache.stats().resident_bytes <= 40
