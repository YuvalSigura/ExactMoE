"""CPU-only exact range streaming tests, using synthetic *bytes* not GLM weights."""
import hashlib

import pytest

from exactmoe.storage import TensorRecord
from exactmoe.streaming import MAX_CHUNK_BYTES, stream_record


@pytest.fixture
def record(tmp_path):
    target = tmp_path / "fake.safetensors"
    target.write_bytes(b"headerxx" + b"ABCDEFGHIJKL" + b"tail")
    return TensorRecord(name="synthetic.weight", path=target, dtype="F8_E4M3",
                        shape=(12,), start=8, nbytes=12)


def test_exact_chunk_bytes_and_offsets(record):
    delivered = []
    receipt = stream_record(record, lambda off, buf: delivered.append((off, buf)),
                            chunk_bytes=5)
    assert delivered == [(0, b"ABCDE"), (5, b"FGHIJ"), (10, b"KL")]
    assert receipt.nbytes == 12
    assert receipt.chunks == 3
    assert receipt.peak_payload_buffer_bytes == 5
    assert receipt.content_sha256 == hashlib.sha256(b"ABCDEFGHIJKL").hexdigest()
    assert receipt.source_stable_during_read
    assert not receipt.expected_hash_verified


def test_slice_does_not_widen_or_read_extra(record):
    chunks = []
    receipt = stream_record(record, lambda off, b: chunks.append((off, b)),
                            offset=3, length=6, chunk_bytes=2)
    assert chunks == [(3, b"DE"), (5, b"FG"), (7, b"HI")]
    assert receipt.nbytes == 6
    assert receipt.peak_payload_buffer_bytes == 2


def test_sha256_positive_and_negative(record):
    good = hashlib.sha256(b"ABCDEFGHIJKL").hexdigest()
    receipt = stream_record(record, lambda *_: None, expected_sha256=good, chunk_bytes=3)
    assert receipt.expected_hash_verified
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stream_record(record, lambda *_: None, expected_sha256="0"*64)
    with pytest.raises(ValueError, match="expected_sha256"):
        stream_record(record, lambda *_: None, expected_sha256="NOT-A-HASH")


def test_mutation_during_delivery_rejected(record):
    def mutate(_off, _buf):
        with record.path.open("r+b") as f:
            f.seek(record.start)
            f.write(b"X")
        # Set deterministic mtime to ensure stamp mismatch even if FS has coarse time
        import os
        os.utime(record.path, ns=(1_000_000_000, 1_000_000_000))
    with pytest.raises(OSError, match="source changed"):
        stream_record(record, mutate, chunk_bytes=12)


def test_short_source_fails(record):
    record.path.write_bytes(b"too short")
    with pytest.raises(ValueError, match="exceeds source file"):
        stream_record(record, lambda *_: None)


def test_sinks_cannot_claim_verification_before_return(record):
    # A throwing sink must never produce a successful receipt.
    def broken(_off, _buf):
        raise RuntimeError("consumer stopped")
    with pytest.raises(RuntimeError, match="consumer stopped"):
        stream_record(record, broken, chunk_bytes=4)


@pytest.mark.parametrize("kwargs", [
    {"chunk_bytes": 0},
    {"chunk_bytes": -1},
    {"chunk_bytes": MAX_CHUNK_BYTES + 1},
    {"chunk_bytes": True},
    {"max_total_bytes": 0},
    {"max_total_bytes": 2},
    {"offset": -1},
    {"offset": 13},
    {"offset": 11, "length": 2},
    {"length": -1},
    {"length": True},
])
def test_invalid_parameters_fail_closed(record, kwargs):
    with pytest.raises(ValueError):
        stream_record(record, lambda *_: None, **kwargs)


def test_full_default_limit_is_bounded(record):
    receipt = stream_record(record, lambda *_: None)
    assert receipt.peak_payload_buffer_bytes <= MAX_CHUNK_BYTES


def test_requires_record_and_sink(record):
    with pytest.raises(TypeError):
        stream_record({}, lambda *_: None)
    with pytest.raises(TypeError):
        stream_record(record, None)
