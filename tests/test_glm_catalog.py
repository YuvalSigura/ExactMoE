import json

import pytest
from exactmoe.glm_catalog import ExpertCatalog
from exactmoe.profile import profile_local


def mapping(scales=True):
    out = {}
    for layer in (1, 2):
        for expert in range(3):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                out[f"{base}.weight"] = f"shard-{layer}.safetensors"
                if scales:
                    out[f"{base}.weight_scale_inv"] = f"shard-{layer}.safetensors"
    return out


def config():
    return {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 3,
        "mlp_layer_types": ["dense", "sparse", "sparse"],
        "first_k_dense_replace": 1,
        "n_routed_experts": 3,
        "num_experts_per_tok": 2,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }


def test_full_fp8_catalog_and_shard_plan():
    cat = ExpertCatalog(mapping())
    report = cat.audit(config())
    assert report["metadata_complete"]
    assert report["named_expert_instances"] == 6
    assert report["named_weight_tensors"] == 18
    assert report["named_scale_tensors"] == 18
    plan = cat.plan(2, [2, 0], True)
    assert plan["selected_experts"] == [2, 0]
    assert plan["tensor_count"] == 12
    assert plan["shard_count"] == 1
    with pytest.raises(ValueError):
        cat.plan(2, [2, 2], True)
    with pytest.raises(KeyError):
        cat.plan(0, [0], True)


def test_missing_fp8_scale_fails_closed():
    m = mapping()
    del m["model.layers.2.mlp.experts.1.gate_proj.weight_scale_inv"]
    cat = ExpertCatalog(m)
    assert cat.audit(config())["missing_scale_tensors"] == 1
    assert not cat.audit(config())["metadata_complete"]
    with pytest.raises(KeyError):
        cat.locations(2, 1, True)


def test_packed_name_is_not_named_expert():
    cat = ExpertCatalog({"model.layers.1.mlp.experts.gate_proj.weight": "x.safetensors"})
    report = cat.audit(config())
    assert report["expert_layout"] == "unresolved"
    assert report["unparsed_expert_names"] == 1
    assert not report["metadata_complete"]


def test_metadata_profile_does_not_require_shards(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(config()))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123}, "weight_map": mapping()})
    )
    result = profile_local(tmp_path)
    assert result["expert_layout"]["metadata_complete"]
    assert result["index_declared_weight_bytes"] == 123


def test_bf16_without_scales():
    cfg = config()
    cfg.pop("quantization_config")
    report = ExpertCatalog(mapping(scales=False)).audit(cfg)
    assert report["metadata_complete"]
    assert not report["requires_scales"]


def test_mtp_expert_layer_is_reported_separately():
    m = mapping()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for kind in ("weight", "weight_scale_inv"):
            m[f"model.layers.3.mlp.experts.0.{proj}.{kind}"] = "mtp.safetensors"
    cfg = config()
    cfg["num_nextn_predict_layers"] = 1
    result = ExpertCatalog(m).audit(cfg)
    assert result["metadata_complete"]
    assert result["named_expert_instances"] == 6
    assert result["mtp_named_expert_instances"] == 1
    assert result["unexpected_expert_instances"] == 0


def test_named_expert_matrix_does_not_appear_as_packed(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    from exactmoe.storage import TensorIndex

    tensor = np.arange(12, dtype=np.float32).reshape((3, 4))
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    save_file({name: tensor}, str(tmp_path / "model.safetensors"))
    ix = TensorIndex.from_directory(tmp_path)
    assert ix.packed_expert_candidates(expected_count=3) == []
