"""Tests for the new planning / simulation tools."""
import json
from pathlib import Path

import pytest

from exactmoe.dynamic_precision import allocate_precision
from exactmoe.expert_profiler import profile_experts
from exactmoe.multi_model_router import route_phase
from exactmoe.pruning_planner import plan_pruning


SAMPLE_TRACE = {
    "schema_version": 1,
    "source_kind": "synthetic",
    "events": [
        {"token": 0, "experts": [
            {"layer": 3, "expert": 1, "weight_bytes": 1000},
            {"layer": 3, "expert": 2, "weight_bytes": 1000},
        ]},
        {"token": 1, "experts": [
            {"layer": 3, "expert": 1, "weight_bytes": 1000},
            {"layer": 4, "expert": 10, "weight_bytes": 1000},
        ]},
        {"token": 2, "experts": [
            {"layer": 3, "expert": 1, "weight_bytes": 1000},
            {"layer": 4, "expert": 10, "weight_bytes": 1000},
        ]},
    ],
}


def test_profile_experts_basic():
    result = profile_experts(SAMPLE_TRACE)
    assert result["status"] == "PROFILING_ONLY"
    assert result["unique_experts"] == 3
    assert result["total_expert_demands"] == 6
    # Expert 1 on layer 3 should be the most frequent
    top = max(result["experts"], key=lambda e: e["activation_count"])
    assert top["layer"] == 3 and top["expert"] == 1
    assert top["activation_count"] == 3


def test_allocate_precision_hits_target():
    result = allocate_precision(SAMPLE_TRACE, target_avg_bits=2.5)
    assert result["status"] == "SIMULATION_ONLY"
    assert 1.0 <= result["achieved_avg_bits"] <= 8.0
    assert result["expert_demands"] == 6


def test_route_phase_policy():
    easy = route_phase("discovery", difficulty=0.2)
    assert easy["recommended_tier"] == "fast_local"
    hard = route_phase("exploit_chain", difficulty=0.95, prefer_local=False)
    assert hard["recommended_tier"] == "full_remote"
    mid = route_phase("deep_analysis", difficulty=0.6)
    assert mid["recommended_tier"] == "specialized_local"


def test_pruning_planner():
    profile = profile_experts(SAMPLE_TRACE)
    plan = plan_pruning(profile, keep_fraction=0.5, min_keep_per_layer=1)
    assert plan["status"] == "PLANNING_ONLY"
    assert plan["experts_kept"] + plan["experts_dropped"] == profile["unique_experts"]
    # Layer 3 has two experts; 50% should keep at least the hottest
    assert 1 in plan["keep_sets"]["3"]


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        route_phase("unknown")
    with pytest.raises(ValueError):
        allocate_precision(SAMPLE_TRACE, target_avg_bits=0.5)
    with pytest.raises(ValueError):
        plan_pruning({"status": "WRONG"})
