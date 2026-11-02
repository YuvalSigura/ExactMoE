from exactmoe.demo import make_fixture, validate_fixture, run_demo


def test_reference_and_paged_outputs_equal(tmp_path):
    make_fixture(tmp_path)
    report = validate_fixture(tmp_path, cache_bytes=8)
    assert report["passed"]
    assert report["max_abs_error"] <= 1e-6
    assert len(report["selected_experts"]) == 2


def test_benchmark_executes():
    result = run_demo(bench_steps=3)
    assert result["passed"]
    assert result["benchmark"]["tokens"] == 3
