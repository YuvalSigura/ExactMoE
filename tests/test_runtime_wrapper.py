import json
from pathlib import Path

import numpy as np
import pytest

from exactmoe.runtime import behavior_repair as br
from exactmoe.runtime import launcher, role_profiles as rp, two_stage as ts
from exactmoe.runtime.gpu_lock import GpuLock, GpuLockHeld, read_holder
from exactmoe.runtime.variants import Variant, VariantRegistry, readiness, verification_status


def _variant(tmp_path: Path, **kw) -> Variant:
    base = dict(id="t", description="test", container=str(tmp_path / "ctr"), resident_bits=8, expert_bits=4,
                engine_dir=str(tmp_path / "eng"), python=str(tmp_path / "py.exe"))
    base.update(kw)
    return Variant(**base)


# --- GPU lock -------------------------------------------------------------------------------------------

def test_gpu_lock_is_exclusive_and_never_deletes_another_holder(tmp_path):
    path = tmp_path / "locks" / "GPU.lock"
    with GpuLock("a", "test", path=path) as lock_a:
        assert read_holder(path)["holder"] == "a"
        with pytest.raises(GpuLockHeld):
            GpuLock("b", "test", path=path).acquire()
        assert lock_a.held()
    assert not path.exists()
    path.write_text("CEM-Zero training", encoding="utf-8")       # someone else's lock, foreign format
    other = GpuLock("c", "test", path=path)
    with pytest.raises(GpuLockHeld) as exc:
        other.acquire()
    assert exc.value.holder == {"holder": "CEM-Zero training"}
    other.release()
    assert path.read_text(encoding="utf-8") == "CEM-Zero training"


# --- variants and launcher ------------------------------------------------------------------------------

def test_builtin_variant_records_how_it_differs(tmp_path):
    v = VariantRegistry().get("glm53-x4g64-r8")
    assert v.expert_bits == 4 and v.dsa_indexer and not v.mtp and not v.uses_gpu
    assert any("int4" in d for d in v.differs_from_original)


def test_registry_file_overrides_and_rejects_unknown_fields(tmp_path):
    f = tmp_path / "reg.json"
    f.write_text(json.dumps({"variants": [dict(_variant(tmp_path).to_dict(), id="glm53-x4g64-r8", port=9999)]}))
    assert VariantRegistry(f).get("glm53-x4g64-r8").port == 9999
    f.write_text(json.dumps({"variants": [dict(_variant(tmp_path).to_dict(), surprise=1)]}))
    with pytest.raises(ValueError):
        VariantRegistry(f)


def test_readiness_lists_what_is_missing(tmp_path):
    v = _variant(tmp_path)
    assert readiness(v)["ready"] is False
    ctr, eng = Path(v.container), Path(v.engine_dir)
    ctr.mkdir(); eng.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "chat_template.jinja", "out-00000.safetensors"):
        (ctr / name).write_text("x")
    for name in ("colibri.exe", "coli"):
        (eng / name).write_text("x")
    Path(v.python).write_text("x")
    state = readiness(v)
    assert state["missing"] == ["out-idx-*.safetensors (DSA indexer pass)"]
    (ctr / "out-idx-00000.safetensors").write_text("x")
    assert readiness(v)["ready"] is True


def test_serve_is_loopback_only_and_cpu_variant_hides_gpu(tmp_path):
    v = _variant(tmp_path)
    argv, env = launcher.serve_command(v)
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert env["CUDA_VISIBLE_DEVICES"] == "-1" and env["MTP"] == "0"
    for host in ("0.0.0.0", "192.168.1.5", "example.com"):
        with pytest.raises(launcher.NotLoopback):
            launcher.serve_command(v, host=host)
    with pytest.raises(ValueError):
        launcher.serve_command(_variant(tmp_path, extra_env={"COLI_API_KEY": "x"}))


def test_role_pin_disables_autopin(tmp_path):
    _, env = launcher.serve_command(_variant(tmp_path, pin_file="p.txt", pin_gb=40))
    assert env["PIN"] == "p.txt" and env["PIN_GB"] == "40" and env["AUTOPIN"] == "0"


def test_homelab_profile_matches_homelab_schema(tmp_path):
    prof = launcher.homelab_runtime_profile(_variant(tmp_path))
    assert prof["health"]["url"].startswith("http://127.0.0.1:")
    assert "NOT the exact model" in prof["notes"]
    try:
        from homelab_harness.runtime_profiles import RuntimeProfile
    except ImportError:
        pytest.skip("HOMELAB not installed in this environment; field names checked below")
    assert RuntimeProfile.from_dict(prof).id == prof["id"]


def test_homelab_profile_field_names_are_the_runtimeprofile_fields(tmp_path):
    expected = {"id", "name", "executable", "args", "cwd", "env", "enabled", "auto_start", "stop_on_exit",
                "show_console", "startup_timeout_seconds", "health", "notes", "model_profile_id"}
    assert set(launcher.homelab_runtime_profile(_variant(tmp_path))) == expected


def test_verification_needs_every_check(tmp_path):
    v = _variant(tmp_path)
    Path(v.container).mkdir()
    assert verification_status(v)["status"] == "unverified"
    for check in ("doctor", "plan", "first_token", "oracle_agreement"):
        launcher.record_evidence(v, check, True, {})
    assert verification_status(v)["status"] == "unverified"
    launcher.record_evidence(v, "homelab_smoke", True, {})
    assert verification_status(v)["status"] == "verified"
    launcher.record_evidence(v, "first_token", False, {})
    assert verification_status(v)["status"] == "unverified"


# --- role profiles --------------------------------------------------------------------------------------

def _usage(counts, layers=3, experts=4, engine=7):
    return rp.Usage(layers, experts, 1, engine, __import__("collections").Counter(counts))


def test_usage_roundtrip_and_validation(tmp_path):
    u = _usage({(0, 1): 5, (2, 3): 2, (3, 0): 1})        # layer 3 == n_layers is the MTP layer
    f = tmp_path / "u.txt"
    rp.write_usage(f, u)
    back = rp.read_usage(f)
    assert back.counts == u.counts and (back.n_layers, back.n_experts, back.engine_id) == (3, 4, 7)
    f.write_text("-1 3 4\n0 9 1\n")
    with pytest.raises(ValueError):
        rp.read_usage(f)
    f.write_bytes(b"\x01\x02\x03\x04rest")
    with pytest.raises(ValueError):
        rp.read_usage(f)


def test_delta_pin_and_coverage():
    before = _usage({(0, 0): 10, (1, 1): 4})
    after = _usage({(0, 0): 12, (1, 1): 4, (2, 2): 6})
    d = rp.usage_delta(before, after)
    assert dict(d.counts) == {(0, 0): 2, (2, 2): 6}
    assert rp.pin_set(d, budget_bytes=100, expert_bytes=100) == [(2, 2)]
    assert rp.coverage(d, [(2, 2)]) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        rp.usage_delta(after, before)


def test_merge_normalises_roles_and_report_shows_cross_hit():
    recon = _usage({(0, 0): 1000, (0, 1): 10})
    web = _usage({(1, 2): 3, (1, 3): 1})
    merged = rp.merge({"recon": recon, "web": web})
    top2 = rp.pin_set(merged, 200, 100)
    assert (1, 2) in top2 and (0, 0) in top2               # the quiet role is not crowded out
    rep = rp.role_report({"recon": recon, "web": web}, budget_bytes=100, expert_bytes=100)
    assert rep["roles"]["web"]["own_profile_hit"] == 0.75
    assert rep["roles"]["web"]["other_profiles_hit"]["recon"] == 0.0


# --- two-stage expert -----------------------------------------------------------------------------------

def _experts(seed=0, hidden=64, inter=128):
    rng = np.random.default_rng(seed)
    exact = ts.ExpertWeights(*(rng.normal(0, 0.1, s).astype(np.float32)
                               for s in ((inter, hidden), (inter, hidden), (hidden, inter))))
    q = lambda w: (np.round(w * 8) / 8).astype(np.float32)                         # coarse stand-in for int4
    return exact, ts.ExpertWeights(q(exact.gate), q(exact.up), q(exact.down))


def test_two_stage_limits_are_exact():
    exact, quant = _experts()
    x = np.random.default_rng(1).normal(0, 1, 64).astype(np.float32)
    full, hot = ts.two_stage_forward(quant, exact, x, 1.0)
    np.testing.assert_allclose(full, ts.expert_forward(exact, x), rtol=1e-4, atol=1e-5)
    assert hot.size == 128
    none, hot0 = ts.two_stage_forward(quant, exact, x, 0.0)
    np.testing.assert_allclose(none, ts.expert_forward(quant, x), rtol=1e-5, atol=1e-6)
    assert hot0.size == 0


def test_two_stage_error_falls_with_fraction_and_reads_only_hot_rows():
    exact, quant = _experts()
    x = np.random.default_rng(2).normal(0, 1, 64).astype(np.float32)
    ref = ts.expert_forward(exact, x)
    errs = [np.linalg.norm(ts.two_stage_forward(quant, exact, x, f)[0] - ref) for f in (0.0, 0.1, 0.25, 0.5)]
    assert errs == sorted(errs, reverse=True) and errs[-1] < errs[0]
    _, hot = ts.two_stage_forward(quant, exact, x, 0.25)
    poisoned = ts.ExpertWeights(exact.gate.copy(), exact.up.copy(), exact.down.copy())
    cold = np.setdiff1d(np.arange(128), hot)
    poisoned.gate[cold] = np.nan; poisoned.up[cold] = np.nan; poisoned.down[:, cold] = np.nan
    out, _ = ts.two_stage_forward(quant, poisoned, x, 0.25)
    assert np.isfinite(out).all()                          # cold FP8 rows are never touched
    assert ts.bytes_read_fraction(0.25) == pytest.approx(0.8125)


# --- behavioural repair protocol ------------------------------------------------------------------------

def test_split_is_deterministic_and_order_free():
    ids = [f"task-{i}" for i in range(200)]
    a, b = br.split_tasks(ids), br.split_tasks(list(reversed(ids)))
    assert a == b and not set(a["dev"]) & set(a["holdout"]) and 60 < len(a["holdout"]) < 140


def test_accept_requires_frozen_candidate_and_significant_gain():
    tasks = [f"t{i}" for i in range(100)]
    base = {t: i < 30 for i, t in enumerate(tasks)}
    good = {t: i < 60 for i, t in enumerate(tasks)}
    cand = {"edit": "remove direction", "layers": [15, 19]}
    h = br.freeze(cand)
    res = br.paired(base, good, tasks)
    assert (res.rescues, res.regressions) == (30, 0)
    assert br.accept(res, frozen_hash=h, candidate=cand)["accepted"] is True
    assert br.accept(res, frozen_hash=h, candidate={**cand, "layers": [15]})["accepted"] is False
    noisy = {t: (i < 31) if i != 0 else False for i, t in enumerate(tasks)}
    assert br.accept(br.paired(base, noisy, tasks), frozen_hash=h, candidate=cand)["accepted"] is False
    worse = {t: (30 <= i < 70) for i, t in enumerate(tasks)}
    assert br.accept(br.paired(base, worse, tasks), frozen_hash=h, candidate=cand)["accepted"] is False
    with pytest.raises(ValueError):
        br.paired(base, {}, tasks)


# --- score instead of generate --------------------------------------------------------------------------

def test_choose_ranks_options_by_their_own_tokens_only():
    from exactmoe.runtime import choose as ch
    prompt = "Decision: "
    table = {"Decision: continue": [-9.0, -9.0, -0.5], "Decision: stop": [-9.0, -9.0, -2.0],
             "Decision: delegate": [-9.0, -9.0, -1.0]}
    def fake_post(url, body, timeout):
        assert url.endswith("/v1/completions") and body["echo"] and body["max_tokens"] == 0
        vals = table[body["prompt"]]
        return {"choices": [{"logprobs": {"text_offset": [0, 9, 10], "token_logprobs": vals}}]}
    ranked = ch.choose("http://127.0.0.1:8410", "m", prompt, ["stop", "continue", "delegate"], post=fake_post)
    assert [s.option for s in ranked] == ["continue", "delegate", "stop"]
    assert ranked[0].tokens == 1 and ranked[0].logprob == -0.5        # prompt tokens are not counted
    assert 0.5 < ch.confidence(ranked) < 1.0
    with pytest.raises(ValueError):
        ch.choose("u", "m", prompt, ["a", "a"], post=fake_post)
    with pytest.raises(ValueError):
        ch.option_logprob({"choices": [{"logprobs": {"text_offset": [0], "token_logprobs": [-1.0]}}]}, 5)
