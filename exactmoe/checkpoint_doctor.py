"""Read-only progress/structural validator for partially downloaded checkpoints.

Checks official safetensors index, present shard paths and safetensors headers.
Does NOT read tensor payload bytes, download missing files or hash large data.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .storage import scan_safetensors


def checkpoint_status(directory: str | Path, *, verify_headers: bool = True,
                      sample_limit: int = 12) -> dict:
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest_file = root / "model.safetensors.index.json"
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    mapping = data.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("missing official shard map")
    by_shard: dict[str, set[str]] = defaultdict(set)
    for name, file in mapping.items():
        if not isinstance(name, str) or not isinstance(file, str):
            raise ValueError("invalid manifest entry")
        if (not file.endswith(".safetensors") or Path(file).name != file
                or "/" in file or "\\" in file):
            raise ValueError(f"unsafe shard path in manifest: {file}")
        by_shard[file].add(name)
    total_shards = len(by_shard)
    present = []
    missing = []
    damaged = []
    bytes_on_disk = 0
    checked_tensors = 0
    for name, names in sorted(by_shard.items()):
        path = (root / name).resolve(strict=False)
        if not path.is_relative_to(root):
            raise ValueError("shard escapes checkpoint directory")
        if not path.is_file():
            missing.append(name)
            continue
        size = path.stat().st_size
        bytes_on_disk += size
        if verify_headers:
            try:
                found = scan_safetensors(path)
                if set(found) != names:
                    raise ValueError("tensor names differ from official weight_map")
                checked_tensors += len(found)
            except (OSError, ValueError, EOFError, json.JSONDecodeError) as exc:
                damaged.append({"shard": name, "reason": str(exc)[:180]})
                continue
        present.append(name)

    metadata = data.get("metadata", {})
    source_bytes = metadata.get("total_size") if isinstance(metadata, dict) else None
    if type(source_bytes) is not int or source_bytes < 0:
        source_bytes = None
    # No completion claim from byte count alone: header validation is structural,
    # not source SHA-256 validation or numerical fidelity validation.
    return {
        "status": "verified-structure" if not missing and not damaged and verify_headers else "incomplete-or-unverified",
        "expected_shards": total_shards,
        "present_shards": len(present),
        "missing_shards": len(missing),
        "invalid_or_partial_shards": len(damaged),
        "headers_checked": verify_headers,
        "validated_tensor_headers": checked_tensors,
        "expected_tensor_names": len(mapping),
        "reported_source_weight_bytes": source_bytes,
        "existing_shard_file_bytes": bytes_on_disk,
        "missing_sample": missing[:sample_limit],
        "invalid_sample": damaged[:sample_limit],
        "note": (
            "Read-only structural audit. Header success cannot detect silent payload corruption; "
            "verify source hashes separately before declaring original model parity. "
            "Hugging Face partial .incomplete downloads are not counted as completed shards."
        ),
    }
