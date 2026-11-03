"""CPU-only streamed-FP8 computation tests. All tensors synthetic, never GLM evidence."""
import hashlib
import json
import math
import struct

import numpy as np
import pytest

from exactmoe.expert_validation import oracle_matvec, _silu, names
from exactmoe.fp8_reference import block_fp8_matvec
from exactmoe.storage import TensorIndex
from exactmoe.streamed_fp8 import streamed_fp8_matvec, streamed_expert


def fixture(tmp_path, *, hidden=259, intermediate=131, invalid_weight=False, bad_scale=False):
    tensors = {}
    rng = np.random.default_rng(827)
    for part, (wn, sn) in names(3, 0).items():
        shape = ((intermediate, hidden) if part != "down_proj"
                 else (hidden, intermediate))
        bits = rng.choice(
            np.array([0x00, 0x20, 0x30, 0x38, 0x40, 0xB8], dtype=np.uint8),
            size=shape,
        )
        if invalid_weight and part == "gate_proj":
            bits[0, 0] = 0x7F  # E4M3FN NaN must fail closed.
        sg_shape = (math.ceil(shape[0] / 128), math.ceil(shape[1] / 128))
        scales = rng.uniform(0.2, 0.8, size=sg_shape).astype("<f4")
        if bad_scale and part == "gate_proj":
            scales[0, 0] = -1.0
        tensors[wn] = ("F8_E4M3", shape, bits.tobytes())
        tensors[sn] = ("F32", sg_shape, scales.tobytes())
    header = {}
    data = bytearray()
    checksums = {}
    for name, (dtype, shape, payload) in sorted(tensors.items()):
        checksums[name] = hashlib.sha256(payload).hexdigest()
        start = len(data)
        data.extend(payload)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [start, len(data)]}
    raw_header = json.dumps(header).encode()
    (tmp_path / "expert.safetensors").write_bytes(
        struct.pack("<Q", len(raw_header)) + raw_header + data
    )
    return TensorIndex.from_directory(tmp_path), checksums


def test_staged_matvec_matches_dense_oracle_across_all_scale_blocks(tmp_path):
    idx, checks = fixture(tmp_path)
    wn, sn = names(3, 0)["gate_proj"]
    x = np.random.default_rng(13).normal(0, 0.2, size=259).astype(np.float32)
    expected = oracle_matvec(idx[wn], idx[sn], x)
    y, report = streamed_fp8_matvec(
        idx[wn], idx[sn], x, max_chunk_bytes=259*8,
        expected_weight_sha256=checks[wn], expected_scale_sha256=checks[sn],
    )
    np.testing.assert_allclose(y, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(y, block_fp8_matvec(idx[wn], idx[sn], x),
                               atol=2e-5, rtol=2e-5)
    assert report["weight_chunks"] > 2
    assert report["bounded_row_group"] == 8
    assert report["expected_weight_hash_verified"]
    assert report["expected_scale_hash_verified"]
    assert report["peak_io_chunk_bytes"] <= 259 * 8
    assert report["logical_read_bytes"] == idx[wn].nbytes + idx[sn].nbytes
    assert report["physical_ssd_bytes_measured"] is False


def test_single_glm_named_six_tensor_expert_vs_separate_oracle(tmp_path):
    idx, digests = fixture(tmp_path, hidden=259, intermediate=131)
    x = np.random.default_rng(21).normal(0, 0.06, size=259).astype(np.float32)
    y, report = streamed_expert(idx, 3, 0, x,
                                expected_sha256_by_tensor=digests,
                                max_chunk_bytes=259*4)
    a = {}
    for part in ("gate_proj", "up_proj"):
        wn, sn = names(3, 0)[part]
        a[part] = oracle_matvec(idx[wn], idx[sn], x)
    wn, sn = names(3, 0)["down_proj"]
    reference = oracle_matvec(idx[wn], idx[sn],
                              _silu(a["gate_proj"]) * a["up_proj"])
    np.testing.assert_allclose(y, reference, atol=2e-4, rtol=2e-5)
    assert report["tensor_count"] == 6
    assert report["logical_bytes_streamed"] == report["source_payload_bytes_expected"]
    assert report["expected_local_hashes_verified"]
    assert not report["full_model_inference"]
    assert not report["external_origin_attested_by_this_routine"]
    assert report["gpu_native_fp8_kernel"] is False
    assert set(report["projection_receipts"]) == {"gate_proj", "up_proj", "down_proj"}


def test_corrupted_payload_hash_fails_after_tiling(tmp_path):
    idx, digest = fixture(tmp_path)
    wn, sn = names(3, 0)["gate_proj"]
    # Expected map reflects original; mutate source while preserving file length.
    file = tmp_path / "expert.safetensors"
    raw = bytearray(file.read_bytes())
    raw[-1] ^= 1
    file.write_bytes(raw)
    x = np.ones(259, np.float32)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        streamed_expert(idx, 3, 0, x,
                        expected_sha256_by_tensor=digest,
                        max_chunk_bytes=259*8)


def test_digests_six_must_be_all_or_none(tmp_path):
    idx, digests = fixture(tmp_path)
    x = np.ones(259, dtype=np.float32)
    wn = names(3, 0)["gate_proj"][0]
    with pytest.raises(ValueError, match="exactly all six"):
        streamed_expert(idx, 3, 0, x,
                        expected_sha256_by_tensor={wn: digests[wn]})
    digest_bad = dict(digests)
    digest_bad[wn] = "bad"
    with pytest.raises(ValueError, match="invalid digest"):
        streamed_expert(idx, 3, 0, x,
                        expected_sha256_by_tensor=digest_bad)


def test_unspecified_expected_hash_does_not_claim_origin(tmp_path):
    idx, _ = fixture(tmp_path, hidden=5, intermediate=7)
    x = np.ones(5, dtype=np.float32)
    result, rec = streamed_expert(idx, 3, 0, x, max_chunk_bytes=1024)
    assert result.shape == (5,)
    assert rec["expected_local_hashes_verified"] is False
    assert rec["external_origin_attested_by_this_routine"] is False


def test_invalid_scale_and_fp8_nan_fail(tmp_path):
    idx, _ = fixture(tmp_path / "badscale" if False else tmp_path, bad_scale=True)
    wn, sn = names(3, 0)["gate_proj"]
    with pytest.raises(ValueError, match="inverse scales"):
        streamed_fp8_matvec(idx[wn], idx[sn], np.ones(259))
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        idx, _ = fixture(Path(d), invalid_weight=True)
        wn, sn = names(3, 0)["gate_proj"]
        with pytest.raises(ValueError, match="E4M3FN NaN"):
            streamed_fp8_matvec(idx[wn], idx[sn], np.ones(259))


@pytest.mark.parametrize("budget", [0, 1, 258, 16 * 1024 * 1024 + 1, True])
def test_invalid_tiny_or_oversized_chunk_budget_fails(tmp_path, budget):
    idx, _ = fixture(tmp_path)
    wn, sn = names(3, 0)["gate_proj"]
    with pytest.raises(ValueError):
        streamed_fp8_matvec(idx[wn], idx[sn], np.ones(259),
                            max_chunk_bytes=budget)


def test_invalid_inputs_fail_closed(tmp_path):
    idx, _ = fixture(tmp_path)
    wn, sn = names(3, 0)["gate_proj"]
    with pytest.raises(ValueError, match="input vector"):
        streamed_fp8_matvec(idx[wn], idx[sn], np.array([np.nan]*259))
    with pytest.raises(ValueError, match="input vector"):
        streamed_fp8_matvec(idx[wn], idx[sn], np.ones(128))
    with pytest.raises(MemoryError, match="byte cap"):
        streamed_fp8_matvec(idx[wn], idx[sn], np.ones(259),
                            max_tensor_bytes=100)


def test_hash_of_result_stable_on_repeated_cpu_run(tmp_path):
    idx, digests = fixture(tmp_path, hidden=5, intermediate=7)
    x = np.array([1., .2, -.1, 2., .3], dtype=np.float32)
    a, ar = streamed_expert(idx, 3, 0, x, expected_sha256_by_tensor=digests)
    b, br = streamed_expert(idx, 3, 0, x, expected_sha256_by_tensor=digests)
    assert ar["output_sha256"] == br["output_sha256"]
    np.testing.assert_array_equal(a, b)
