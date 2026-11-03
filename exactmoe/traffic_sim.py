"""Source-independent, read-only cache-traffic *simulation* for routed MoE traces.

Does not run GLM, predict router outputs, prefetch, access actual SSD bytes,
modify model tensors, or estimate CUDA latency. Never label simulated bytes
as physical I/O. Source_kind=captured is a declaration, not authentication.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

POLICIES = ("none", "lru", "lfu")
MAX_TRACE_BYTES = 32 * 1024 * 1024


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_trace(trace: Any) -> tuple[list[tuple[tuple[int, int, int], ...]], str]:
    """Validate provenance and freeze a per-token, per-layer original-router trace.

    Format: {"schema_version":1,"source_kind":"synthetic"|"captured",
    "checkpoint_sha256":"..." (required only if captured),
    "events":[{"token":0,"experts":[{"layer":3,"expert":42,"weight_bytes":123}]}]}.
    Each event is one target-model token position; IDs cannot be rescheduled.
    """
    if not isinstance(trace, dict) or type(trace.get("schema_version")) is not int or trace.get("schema_version") != 1:
        raise ValueError("unsupported trace schema (expected integer schema_version=1)")
    kind = trace.get("source_kind")
    if kind not in ("synthetic", "captured"):
        raise ValueError("source_kind must be synthetic or captured")
    if kind == "captured":
        sha = trace.get("checkpoint_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("captured trace requires a lowercase 64-character checkpoint_sha256")
    events = trace.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("events must be a nonempty list")
    if len(events) > 1_000_000:
        raise ValueError("too many token events")
    sizes: dict[tuple[int, int], int] = {}
    output: list[tuple[tuple[int, int, int], ...]] = []
    for ti, event in enumerate(events):
        if not isinstance(event, dict) or _integer(event.get("token"), "token") != ti:
            raise ValueError("tokens must be contiguous and ordered starting at 0")
        items = event.get("experts")
        if not isinstance(items, list) or not items or len(items) > 100_000:
            raise ValueError("each token must list at least one routed expert")
        group: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int]] = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("expert entry must be an object")
            layer = _integer(item.get("layer"), "layer")
            expert = _integer(item.get("expert"), "expert")
            size = _integer(item.get("weight_bytes"), "weight_bytes", 1)
            key = (layer, expert)
            if key in seen:
                raise ValueError("duplicate expert within token; deduplicate routing input first")
            seen.add(key)
            if key in sizes and sizes[key] != size:
                raise ValueError("conflicting byte sizes for same expert")
            sizes[key] = size
            group.append((layer, expert, size))
        output.append(tuple(group))
    return output, kind


def simulate(trace: dict[str, Any], cache_bytes: int, policy: str = "lru") -> dict[str, Any]:
    """Estimate logical expert-source reads under a *fixed* routed trace.

    'lfu' is history-only admission/eviction (no oracle and no future leak).
    Prefetch and CPU direct execution are *not* simulated here.
    """
    _integer(cache_bytes, "cache_bytes")
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    tokens, kind = validate_trace(trace)
    resident: OrderedDict[tuple[int, int], int] = OrderedDict()
    frequency: Counter[tuple[int, int]] = Counter()
    last_touch: dict[tuple[int, int], int] = {}
    hits = misses = evictions = bypass = read_bytes = demand_bytes = peak = 0
    resident_total = 0
    clock = 0
    for token in tokens:
        for layer, expert, size in token:
            clock += 1
            key = (layer, expert)
            frequency[key] += 1  # Only past and present demand is visible.
            last_touch[key] = clock
            demand_bytes += size
            if policy != "none" and key in resident:
                hits += 1
                resident.move_to_end(key)
                continue
            misses += 1
            read_bytes += size
            if policy == "none" or size > cache_bytes or cache_bytes == 0:
                bypass += 1
                continue
            if policy == "lfu" and resident and resident_total + size > cache_bytes:
                # The novel entry must be more valuable than the least used resident.
                # Equality retains the resident, avoiding churn on single-use entries.
                lowest = min(resident, key=lambda k: (frequency[k], last_touch[k]))
                if frequency[key] <= frequency[lowest]:
                    bypass += 1
                    continue
            while resident and resident_total + size > cache_bytes:
                if policy == "lru":
                    victim = next(iter(resident))
                else:
                    victim = min(resident, key=lambda k: (frequency[k], last_touch[k]))
                resident_total -= resident.pop(victim)
                evictions += 1
            resident[key] = size
            resident_total += size
            peak = max(peak, resident_total)
    canonical = json.dumps(trace, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "status": "SIMULATION_ONLY",
        "source_kind_declared": kind,
        "source_authenticated": False,
        "physical_ssd_bytes_measured": False,
        "router_modified": False,
        "cache_policy": policy,
        "cache_capacity_bytes": cache_bytes,
        "token_positions": len(tokens),
        "expert_demands": hits + misses,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": hits / (hits + misses),
        "cache_evictions": evictions,
        "cache_admission_bypasses": bypass,
        "peak_cached_bytes": peak,
        "uncached_demand_bytes": demand_bytes,
        "logical_source_read_bytes": read_bytes,
        "logical_read_bytes_per_token_position": read_bytes / len(tokens),
        "logical_bytes_saved_vs_no_cache": demand_bytes - read_bytes,
        "trace_sha256": hashlib.sha256(canonical).hexdigest(),
        "warning": "Model/router execution, accuracy, real disk reads, runtime latency and accepted speculative tokens NOT measured.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only expert cache traffic simulator; never runs GLM")
    parser.add_argument("--trace", required=True, type=Path, help="JSON recorded or synthetic routed-expert trace")
    parser.add_argument("--cache-mib", required=True, type=int, help="bounded resident expert cache in MiB")
    parser.add_argument("--policy", choices=POLICIES, default="lru")
    args = parser.parse_args(argv)
    try:
        if args.trace.stat().st_size > MAX_TRACE_BYTES:
            raise ValueError("trace too large")
        payload = json.loads(args.trace.read_text(encoding="utf-8"))
        result = simulate(payload, _integer(args.cache_mib, "cache-mib") * 1024 * 1024, args.policy)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
