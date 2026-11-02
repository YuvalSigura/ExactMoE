import json

from exactmoe.profile import profile_local


def test_profile_does_not_require_checkpoint_weights(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 78,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "text_config": {"hidden_size": 6144},
        "unused": "ignored",
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 756000000000},
        "weight_map": {
            "model.layers.3.mlp.experts.gate_up_proj": "model-00001.safetensors",
            "model.layers.4.mlp.experts.down_proj": "model-00002.safetensors",
            "model.embed_tokens.weight": "model-00001.safetensors",
        },
    }))
    p = profile_local(tmp_path)
    assert p["tensor_count"] == 3
    assert p["shard_count"] == 2
    assert p["config"]["n_routed_experts"] == 256
    assert p["config"]["text_config"]["hidden_size"] == 6144
    assert p["layers_with_expert_names"] == 2
    assert p["index_declared_weight_bytes"] == 756000000000
