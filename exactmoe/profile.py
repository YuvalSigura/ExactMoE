"""Checkpoint reconnaissance without downloading any model weights.

Local mode reads only config.json and the shard index. Remote mode explicitly
downloads only those two small JSON metadata files from Hugging Face.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


CONFIG_FIELDS = (
    "model_type", "hidden_size", "num_hidden_layers", "n_routed_experts",
    "num_experts", "num_experts_per_tok", "first_k_dense_replace",
    "moe_intermediate_size", "intermediate_size", "index_topk",
    "index_n_heads", "max_position_embeddings", "vocab_size",
    "norm_topk_prob", "scoring_func", "n_group", "topk_group",
)


def _config_fields(obj: dict[str, Any]) -> dict[str, Any]:
    result = {key: obj[key] for key in CONFIG_FIELDS if key in obj}
    for child in ("text_config", "model_config"):
        nested = obj.get(child)
        if isinstance(nested, dict):
            result[child] = _config_fields(nested)
    return result


def profile_local(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve(strict=True)
    cfg = root / "config.json"
    manifest = root / "model.safetensors.index.json"
    if not cfg.is_file() or not manifest.is_file():
        raise FileNotFoundError(
            "Expected config.json and model.safetensors.index.json; no weight files needed"
        )
    config = json.loads(cfg.read_text(encoding="utf-8"))
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(obj, dict):
        raise ValueError("model metadata must be JSON objects")
    mapping = obj.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("missing weight_map")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("invalid weight_map entries")
    metadata = obj.get("metadata", {})
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    if not isinstance(total_size, int) or total_size < 0:
        total_size = None
    names = list(mapping)
    expert_names = [n for n in names if ".experts." in n]
    per_layer = Counter()
    for n in expert_names:
        chunks = n.split(".")
        for i in range(len(chunks) - 1):
            if chunks[i] == "layers" and chunks[i + 1].isdigit():
                per_layer[int(chunks[i + 1])] += 1
                break
    return {
        "model_directory": str(root),
        "config": _config_fields(config),
        "tensor_count": len(names),
        "shard_count": len(set(mapping.values())),
        "index_declared_weight_bytes": total_size,
        "expert_related_tensor_names": len(expert_names),
        "layers_with_expert_names": len(per_layer),
        "expert_related_name_samples": expert_names[:12],
        "note": (
            "Metadata only. Tensor names do not prove whether experts are individually "
            "addressable. The official checkpoint and FP8 scales require layout inspection."
        ),
    }


def profile_hf(repo_id: str, revision: str | None = None) -> dict[str, Any]:
    """Download ONLY two JSON files, never model weights."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError('Install optional package: pip install -e ".[research]"') from exc
    kwargs = {"repo_id": repo_id, "revision": revision}
    cfg = Path(hf_hub_download(filename="config.json", **kwargs))
    manifest = Path(hf_hub_download(filename="model.safetensors.index.json", **kwargs))
    if cfg.parent != manifest.parent:
        raise RuntimeError("downloaded metadata belong to different snapshot directories")
    result = profile_local(cfg.parent)
    result["hf_repo"] = repo_id
    result["requested_revision"] = revision
    return result
