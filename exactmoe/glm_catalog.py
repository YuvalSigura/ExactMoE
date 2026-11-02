"""Metadata-only inventory of individually named GLM MoE experts and FP8 scales.

No checkpoint weights are downloaded. This maps original tensor names to shards.
The authoritative model router must supply each expert ID.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Any, Mapping

EXPERT_RE = re.compile(
    r"^(?P<stem>.+\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+))\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale_inv)$"
)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class WeightLocation:
    tensor: str
    shard: str
    layer: int
    expert: int
    projection: str
    kind: str


class ExpertCatalog:
    def __init__(self, mapping: Mapping[str, str]):
        self.mapping = dict(mapping)
        self._catalog: dict[tuple[int, int], dict[tuple[str, str], WeightLocation]] = {}
        self.other_expert_names: list[str] = []
        for name, shard in mapping.items():
            if not isinstance(name, str) or not isinstance(shard, str):
                raise ValueError("weight_map must contain string names and shards")
            if ".experts." not in name:
                continue
            match = EXPERT_RE.fullmatch(name)
            if match is None:
                self.other_expert_names.append(name)
                continue
            layer, expert = int(match["layer"]), int(match["expert"])
            projection, kind = match["projection"], match["kind"]
            loc = WeightLocation(name, shard, layer, expert, projection, kind)
            self._catalog.setdefault((layer, expert), {})[(projection, kind)] = loc

    def locations(self, layer: int, expert: int, require_scales: bool) -> list[WeightLocation]:
        parts = self._catalog.get((layer, expert), {})
        kinds = ("weight", "weight_scale_inv") if require_scales else ("weight",)
        missing = [(p, k) for p in PROJECTIONS for k in kinds if (p, k) not in parts]
        if missing:
            raise KeyError(f"layer={layer}, expert={expert}, missing={missing}")
        return [parts[(p, k)] for p in PROJECTIONS for k in kinds]

    def audit(self, config: Mapping[str, Any]) -> dict[str, Any]:
        count, layers = config.get("n_routed_experts"), config.get("num_hidden_layers")
        if type(count) is not int or count <= 0:
            raise ValueError("n_routed_experts must be positive")
        if type(layers) is not int or layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        modes = config.get("mlp_layer_types")
        if modes is not None:
            if not isinstance(modes, list) or len(modes) != layers:
                raise ValueError("mlp_layer_types length mismatch")
            if any(x not in ("dense", "sparse") for x in modes):
                raise ValueError("unrecognized mlp_layer_types")
            sparse = [i for i, mode in enumerate(modes) if mode == "sparse"]
        else:
            dense = config.get("first_k_dense_replace", 0)
            if type(dense) is not int or not 0 <= dense <= layers:
                raise ValueError("invalid first_k_dense_replace")
            sparse = list(range(dense, layers))
        quant = config.get("quantization_config")
        fp8 = isinstance(quant, dict) and quant.get("quant_method") == "fp8"
        require_scales = fp8 or any(
            kind == "weight_scale_inv"
            for parts in self._catalog.values() for _, kind in parts
        )
        missing_weights = missing_scales = 0
        missing_samples: list[str] = []
        for layer in sparse:
            for expert in range(count):
                parts = self._catalog.get((layer, expert), {})
                for proj in PROJECTIONS:
                    for kind in (("weight", "weight_scale_inv") if require_scales else ("weight",)):
                        if (proj, kind) not in parts:
                            if kind == "weight":
                                missing_weights += 1
                            else:
                                missing_scales += 1
                            if len(missing_samples) < 8:
                                missing_samples.append(f"{layer}/{expert}/{proj}.{kind}")
        sparse_set = set(sparse)
        mtp_count = config.get("num_nextn_predict_layers", 0)
        if type(mtp_count) is not int or mtp_count < 0:
            raise ValueError("invalid num_nextn_predict_layers")
        mtp_layers = set(range(layers, layers + mtp_count))
        mtp_instances = sum(layer in mtp_layers for layer, _ in self._catalog)
        unexpected = [
            (layer, expert) for layer, expert in self._catalog
            if (layer not in sparse_set and layer not in mtp_layers) or not 0 <= expert < count
        ]
        shard_counts = Counter(loc.shard for parts in self._catalog.values() for loc in parts.values())
        nweights = sum(
            kind == "weight" for (layer, _), parts in self._catalog.items()
            if layer in sparse_set for _, kind in parts
        )
        nscales = sum(
            kind == "weight_scale_inv" for (layer, _), parts in self._catalog.items()
            if layer in sparse_set for _, kind in parts
        )
        expected = len(sparse) * count
        return {
            "expert_layout": "individually_named" if self._catalog else "unresolved",
            "sparse_layer_count": len(sparse),
            "expected_expert_instances": expected,
            "named_expert_instances": sum(layer in sparse_set for layer, _ in self._catalog),
            "mtp_named_expert_instances": mtp_instances,
            "expected_weight_tensors": expected * 3,
            "named_weight_tensors": nweights,
            "expected_scale_tensors": expected * 3 if require_scales else 0,
            "named_scale_tensors": nscales,
            "requires_scales": require_scales,
            "missing_weight_tensors": missing_weights,
            "missing_scale_tensors": missing_scales,
            "unexpected_expert_instances": len(unexpected),
            "unparsed_expert_names": len(self.other_expert_names),
            "unparsed_samples": self.other_expert_names[:5],
            "missing_samples": missing_samples,
            "expert_tensor_shards": len(shard_counts),
            "top_shards_by_tensor_count": shard_counts.most_common(5),
            "metadata_complete": bool(
                self._catalog and not missing_weights and not missing_scales
                and not unexpected and not self.other_expert_names
            ),
            "caveat": "Only metadata names and shards, not byte offsets, dtypes, numerical parity or runnable GLM.",
        }

    def plan(self, layer: int, expert_ids: list[int], require_scales: bool) -> dict[str, Any]:
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("expert IDs must be unique")
        by_shard: dict[str, list[str]] = defaultdict(list)
        for expert in expert_ids:
            if type(expert) is not int or expert < 0:
                raise ValueError("expert IDs must be nonnegative integers")
            for loc in self.locations(layer, expert, require_scales):
                by_shard[loc.shard].append(loc.tensor)
        return {
            "layer": layer,
            "selected_experts": expert_ids,
            "tensor_count": sum(map(len, by_shard.values())),
            "shard_count": len(by_shard),
            "by_shard": dict(sorted(by_shard.items())),
            "note": "No tensor bytes read, and expert IDs must come from official router.",
        }
