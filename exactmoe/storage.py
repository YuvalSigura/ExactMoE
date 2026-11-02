"""Read safetensors metadata and exact byte ranges, without loading model weights.

Safetensors stores a little-endian uint64 header size, a JSON header, then a
contiguous data section. All recorded tensor offsets are relative to that section.
No pickle, arbitrary Python execution, or tensor conversions are involved here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any, Iterable

from .glm_catalog import EXPERT_RE

MAX_HEADER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TensorRecord:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte position in safetensors file
    nbytes: int

    def first_axis_slice(self, index: int) -> tuple[int, int]:
        """Byte region for one contiguous first-axis row, without re-encoding."""
        if not self.shape or self.shape[0] <= 0:
            raise ValueError(f"{self.name}: first axis is empty or absent")
        if not 0 <= index < self.shape[0]:
            raise IndexError(f"{self.name}: expert index {index} out of range")
        if self.nbytes % self.shape[0]:
            raise ValueError(f"{self.name}: non-byte-aligned first-axis slice")
        width = self.nbytes // self.shape[0]
        return self.start + index * width, width


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key: {key}")
        result[key] = value
    return result


def scan_safetensors(path: str | Path) -> dict[str, TensorRecord]:
    """Scan header only. Handles FP8/BF16 without interpreting weight bytes."""
    path = Path(path).resolve(strict=True)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length < 2 or header_length > MAX_HEADER_BYTES:
            raise ValueError(f"{path}: invalid header length {header_length}")
        if 8 + header_length > file_size:
            raise ValueError(f"{path}: header exceeds file")
        header = json.loads(f.read(header_length), object_pairs_hook=_no_duplicate_keys)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    result: dict[str, TensorRecord] = {}
    start_of_data = 8 + header_length
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"invalid tensor header for {name}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(name, str)
            or not isinstance(dtype, str)
            or not isinstance(shape, list)
            or any(type(v) is not int or v < 0 for v in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(v) is not int for v in offsets)
        ):
            raise ValueError(f"invalid tensor definition: {name}")
        lo, hi = offsets
        if lo < 0 or hi < lo or start_of_data + hi > file_size:
            raise ValueError(f"{name}: invalid data offset range")
        result[name] = TensorRecord(
            name=name, path=path, dtype=dtype,
            shape=tuple(shape), start=start_of_data + lo, nbytes=hi - lo,
        )
    return result


def _safe_shard(root: Path, shard: str) -> Path:
    if not isinstance(shard, str) or not shard.endswith(".safetensors"):
        raise ValueError("shard must be a .safetensors file")
    path = (root / shard).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"shard escapes model directory: {shard}")
    if not path.is_file():
        raise ValueError(f"shard is not a regular file: {shard}")
    return path


class TensorIndex:
    def __init__(self, tensors: dict[str, TensorRecord], root: Path):
        self.tensors = tensors
        self.root = root

    @classmethod
    def from_directory(cls, directory: str | Path) -> "TensorIndex":
        root = Path(directory).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        manifest = root / "model.safetensors.index.json"
        mapped: dict[str, str] | None = None
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            mapped = data.get("weight_map")
            if not isinstance(mapped, dict) or not mapped:
                raise ValueError("manifest has no weight_map")
            shards = sorted(set(mapped.values()))
        else:
            shards = sorted(p.name for p in root.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors shards in {root}")
        tensors: dict[str, TensorRecord] = {}
        for shard in shards:
            for name, record in scan_safetensors(_safe_shard(root, shard)).items():
                if name in tensors:
                    raise ValueError(f"duplicate tensor across shards: {name}")
                tensors[name] = record
        if mapped is not None:
            if set(tensors) != set(mapped):
                raise ValueError("manifest tensor names do not match shard headers")
            for name, shard in mapped.items():
                if tensors[name].path != _safe_shard(root, shard):
                    raise ValueError(f"manifest shard mismatch: {name}")
        return cls(tensors, root)

    @classmethod
    def from_selected(cls, directory: str | Path, names: list[str]) -> "TensorIndex":
        """Read ONLY requested original tensor headers and their existing shards.

        This works with a partial model download once the *required* shard files
        are fully present, without checking or touching the other ~140 shards.
        It does not bypass manifest validation and never follows paths outside
        the model root.
        """
        root = Path(directory).resolve(strict=True)
        manifest_path = root / "model.safetensors.index.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("sharded index JSON is required for selective reads")
        if not names or len(names) != len(set(names)):
            raise ValueError("tensor name list must be nonempty and unique")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mapping = manifest.get("weight_map")
        if not isinstance(mapping, dict):
            raise ValueError("invalid model weight_map")
        missing = [name for name in names if name not in mapping]
        if missing:
            raise KeyError(f"missing names from official manifest: {missing[:5]}")
        expected = {name: _safe_shard(root, mapping[name]) for name in names}
        result: dict[str, TensorRecord] = {}
        for shard in sorted(set(expected.values())):
            scanned = scan_safetensors(shard)
            for name, path in expected.items():
                if path == shard:
                    rec = scanned.get(name)
                    if rec is None or rec.path != path:
                        raise ValueError(f"tensor {name} absent from mapped shard {shard.name}")
                    result[name] = rec
        if set(result) != set(names):
            raise ValueError("not all requested tensors could be validated")
        return cls(result, root)

    @classmethod
    def shard_plan(cls, directory: str | Path, names: list[str]) -> dict[str, list[str]]:
        """Report official shard filenames without requiring any weight shards."""
        root = Path(directory).resolve(strict=True)
        manifest_path = root / "model.safetensors.index.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        mapping = json.loads(manifest_path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(mapping, dict):
            raise ValueError("invalid model weight_map")
        plan: dict[str, list[str]] = {}
        for name in names:
            if name not in mapping:
                raise KeyError(name)
            shard = mapping[name]
            if not isinstance(shard, str) or not shard.endswith(".safetensors"):
                raise ValueError("invalid shard name")
            # Do not require shard to exist; reject path traversal lexically.
            if Path(shard).name != shard or "/" in shard or "\\" in shard:
                raise ValueError("unsafe shard name")
            plan.setdefault(shard, []).append(name)
        return dict(sorted(plan.items()))

    def __getitem__(self, name: str) -> TensorRecord:
        return self.tensors[name]

    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def total_weight_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def packed_expert_candidates(self, expected_count: int | None = None) -> list[TensorRecord]:
        """Candidate tensors, NOT an assertion that their first axis is experts."""
        return [
            t for t in self.tensors.values()
            if ".experts." in t.name and EXPERT_RE.fullmatch(t.name) is None and len(t.shape) >= 2
            and t.shape[0] > 1
            and (expected_count is None or t.shape[0] == expected_count)
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "tensor_count": len(self),
            "weight_bytes": self.total_weight_bytes,
            "shard_count": len({t.path for t in self.tensors.values()}),
            "dtypes": sorted(set(t.dtype for t in self.tensors.values())),
            "packed_expert_candidates": len(self.packed_expert_candidates()),
        }


def read_bytes(record: TensorRecord, offset: int = 0, length: int | None = None) -> bytes:
    """Read immutable bytes for one tensor or a bounded section of it."""
    if length is None:
        length = record.nbytes - offset
    if offset < 0 or length < 0 or offset + length > record.nbytes:
        raise ValueError("requested region is outside tensor boundaries")
    with record.path.open("rb", buffering=0) as f:
        f.seek(record.start + offset)
        data = f.read(length)
    if len(data) != length:
        raise EOFError(f"short read for {record.name}: got {len(data)} / {length}")
    return data


def read_first_axis_slice(record: TensorRecord, index: int) -> bytes:
    start, nbytes = record.first_axis_slice(index)
    return read_bytes(record, offset=start - record.start, length=nbytes)
