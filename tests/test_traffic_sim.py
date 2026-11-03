"""Deterministic CPU-only cache experiments. Not evidence of GLM runtime speed."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from exactmoe.traffic_sim import simulate, validate_trace


def trace(*groups):
    return {
        "schema_version": 1,
        "source_kind": "synthetic",
        "events": [
            {"token": index, "experts": [
                {"layer": layer, "expert": expert, "weight_bytes": size}
                for layer, expert, size in group
            ]}
            for index, group in enumerate(groups)
        ],
    }


def test_repeated_hot_expert_reads_once():
    data = trace([(3, 42, 10)], [(3, 42, 10)], [(3, 42, 10)])
    out = simulate(data, 10)
    assert out["logical_source_read_bytes"] == 10
    assert out["uncached_demand_bytes"] == 30
    assert out["cache_hits"] == 2
    assert out["cache_hit_rate"] == 2 / 3
    assert out["peak_cached_bytes"] == 10
    assert out["router_modified"] is False


def test_zero_capacity_equal_no_cache():
    data = trace([(3, 42, 10)], [(3, 42, 10)])
    a = simulate(data, 0)
    b = simulate(data, 100, "none")
    assert a["logical_source_read_bytes"] == b["logical_source_read_bytes"] == 20
    assert a["cache_hits"] == b["cache_hits"] == 0


def test_lru_eviction_and_reload():
    data = trace([(1, 1, 10)], [(1, 2, 10)], [(1, 1, 10)])
    out = simulate(data, 10, "lru")
    assert out["cache_evictions"] == 2
    assert out["cache_misses"] == 3
    assert out["logical_source_read_bytes"] == 30
    assert out["peak_cached_bytes"] == 10


def test_oversize_bypass_does_not_evict_hot():
    data = trace([(1, 1, 10)], [(1, 2, 100)], [(1, 1, 10)])
    out = simulate(data, 10)
    assert out["cache_hits"] == 1
    assert out["logical_source_read_bytes"] == 110
    assert out["cache_admission_bypasses"] == 1
    assert out["cache_evictions"] == 0


def test_lfu_history_based_admission():
    data = trace([(1, 1, 10)], [(1, 1, 10)], [(1, 2, 10)], [(1, 1, 10)])
    lfu = simulate(data, 10, "lfu")
    lru = simulate(data, 10, "lru")
    assert lfu["logical_source_read_bytes"] == 20
    assert lru["logical_source_read_bytes"] == 30
    assert lfu["cache_admission_bypasses"] == 1


def test_frequencies_do_not_use_future():
    data = trace([(1, 1, 10)], [(1, 2, 10)], [(1, 2, 10)])
    first_two = simulate({**data, "events": data["events"][:2]}, 10, "lfu")
    assert first_two["logical_source_read_bytes"] == 20
    assert first_two["cache_admission_bypasses"] == 1
    # The second expert is frequent only in the future, not at its first access.


def test_same_expert_different_layer_is_different_weight():
    data = trace([(0, 42, 10)], [(1, 42, 10)])
    result = simulate(data, 10)
    assert result["cache_hits"] == 0
    assert result["logical_source_read_bytes"] == 20


def test_hash_is_canonical_for_key_order():
    data = trace([(2, 3, 10)])
    reordered = dict(reversed(list(data.items())))
    assert simulate(data, 10)["trace_sha256"] == simulate(reordered, 10)["trace_sha256"]


@pytest.mark.parametrize("bad", [
    {"schema_version": 1, "source_kind": "synthetic", "events": []},
    {"schema_version": 2, "source_kind": "synthetic", "events": []},
    {"schema_version": 1, "source_kind": "unknown", "events": []},
    {"schema_version": 1, "source_kind": "captured", "events": [{"token": 0, "experts": [{"layer": 1, "expert": 2, "weight_bytes": 1}]}]},
    trace([(1, 2, 10), (1, 2, 10)]),
    trace([(1, 2, 10)], [(1, 2, 11)]),
    {"schema_version": 1, "source_kind": "synthetic", "events": [{"token": 2, "experts": [{"layer": 1, "expert": 2, "weight_bytes": 1}]}]},
    trace([(1, 2, 0)]),
    trace([(1, -2, 10)]),
    trace([(1, 2, True)]),
])
def test_invalid_traces_fail_closed(bad):
    with pytest.raises(ValueError):
        validate_trace(bad)


def test_invalid_policy_and_budget():
    data = trace([(1, 2, 3)])
    with pytest.raises(ValueError):
        simulate(data, -1)
    with pytest.raises(ValueError):
        simulate(data, 1, "future_oracle")


def test_captured_trace_is_declaration_not_authentication():
    data = trace([(1, 2, 3)])
    data["source_kind"] = "captured"
    data["checkpoint_sha256"] = "0" * 64
    out = simulate(data, 3)
    assert out["source_kind_declared"] == "captured"
    assert out["source_authenticated"] is False
    assert out["status"] == "SIMULATION_ONLY"


def test_example_trace_can_run_as_cli(tmp_path):
    data = trace([(1, 2, 3)], [(1, 2, 3)])
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "exactmoe.traffic_sim", "--trace", str(path),
         "--cache-mib", "1", "--policy", "lru"],
        capture_output=True, text=True, timeout=15,
    )
    assert p.returncode == 0, p.stderr
    result = json.loads(p.stdout)
    assert result["logical_source_read_bytes"] == 3
    assert result["token_positions"] == 2
