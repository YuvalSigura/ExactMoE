"""Bounded, source-format-agnostic exact tensor streaming for verified safetensors
TensorRecord ranges. Inspired by SSD-first design research; original code, no
vendored upstream C/CUDA. NOT a GLM inference kernel or physical SSD benchmark.

IMPORTANT: a consumer may see chunks before the *final* checksum verdict. It
must quarantine all derived artifacts until a successful StreamReceipt exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import os
from typing import Callable

from .storage import TensorRecord

MAX_CHUNK_BYTES = 16 * 1024 * 1024
DEFAULT_TOTAL_LIMIT = 128 * 1024 * 1024


@dataclass(frozen=True)
class StreamReceipt:
    tensor: str
    nbytes: int
    chunks: int
    content_sha256: str
    expected_hash_verified: bool
    source_stable_during_read: bool
    peak_payload_buffer_bytes: int
    note: str = "Logical successful reads. Physical SSD bytes and model output NOT measured."


def _sha256_digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _stamp(st: os.stat_result) -> tuple[int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def stream_record(
    record: TensorRecord,
    sink: Callable[[int, bytes], None],
    *,
    chunk_bytes: int = 2 * 1024 * 1024,
    max_total_bytes: int = DEFAULT_TOTAL_LIMIT,
    offset: int = 0,
    length: int | None = None,
    expected_sha256: str | None = None,
) -> StreamReceipt:
    """Read contiguous original bytes into a bounded single-chunk buffer.

    Source identity is compared before/after the read using the opened file
    descriptor. A supplied expected SHA256 is verified before returning. Do
    not expose or commit any output of sink until this function returns.
    """
    if not isinstance(record, TensorRecord):
        raise TypeError("record must be an indexed TensorRecord")
    if not callable(sink):
        raise TypeError("sink must be callable")
    for name, value in (("chunk_bytes", chunk_bytes), ("max_total_bytes", max_total_bytes),
                        ("offset", offset)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if chunk_bytes < 1 or chunk_bytes > MAX_CHUNK_BYTES:
        raise ValueError(f"chunk_bytes must be in 1..{MAX_CHUNK_BYTES}")
    if max_total_bytes < 1:
        raise ValueError("max_total_bytes must be positive")
    if length is None:
        length = record.nbytes - offset
    if type(length) is not int or length < 0:
        raise ValueError("length must be nonnegative")
    if offset > record.nbytes or length > record.nbytes - offset:
        raise ValueError("requested range exceeds indexed tensor")
    if length > max_total_bytes:
        raise ValueError("requested range exceeds explicit bounded max_total_bytes")
    if record.start < 0 or record.nbytes < 0 or length == 0:
        raise ValueError("invalid or empty tensor range")
    if expected_sha256 is not None and not _sha256_digest(expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase full SHA-256 hex digest")

    offset_abs = record.start + offset
    remaining = length
    digest = sha256()
    chunks = 0
    peak = 0
    with Path(record.path).open("rb", buffering=0) as file:
        before = _stamp(os.fstat(file.fileno()))
        if offset_abs + length > before[2]:
            raise ValueError("indexed tensor byte range exceeds source file")
        file.seek(offset_abs)
        while remaining:
            request = min(chunk_bytes, remaining)
            data = file.read(request)
            if len(data) != request:
                raise EOFError("short tensor read; source changed or truncated")
            digest.update(data)
            sink(offset + chunks * chunk_bytes, data)
            remaining -= request
            peak = max(peak, len(data))
            chunks += 1
        after = _stamp(os.fstat(file.fileno()))
        if before != after:
            raise OSError("safetensors source changed during streaming")
    final = digest.hexdigest()
    if expected_sha256 is not None and final != expected_sha256:
        raise ValueError("tensor SHA-256 mismatch; sink output MUST be discarded")
    return StreamReceipt(
        tensor=record.name, nbytes=length, chunks=chunks, content_sha256=final,
        expected_hash_verified=expected_sha256 is not None,
        source_stable_during_read=True,
        peak_payload_buffer_bytes=peak,
    )
