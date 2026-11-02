from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import pytest

from exactmoe.pager import ByteLRU, ExpertPager
from exactmoe.storage import TensorIndex


def test_lru_budget_and_evictions():
    cache = ByteLRU(6)
    cache.put(("x", 0), b"aaa")
    cache.put(("x", 1), b"bbb")
    assert cache.get(("x", 0)) == b"aaa"
    cache.put(("x", 2), b"cccc")
    assert cache.get(("x", 1)) is None
    assert cache.stats().resident_bytes <= 6
    assert cache.stats().evictions >= 1
    cache.put(("huge", 0), b"x" * 7)
    assert cache.stats().resident_bytes <= 6


def test_prefetch_is_just_read_ahead(tmp_path: Path):
    arr = np.arange(36, dtype="float32").reshape(3, 3, 4)
    name = "layers.0.mlp.experts.up_proj.weight"
    save_file({name: arr}, str(tmp_path / "model.safetensors"))
    ix = TensorIndex.from_directory(tmp_path)
    with ExpertPager(ix, cache_bytes=48, max_prefetch=2) as pager:
        assert pager.prefetch(name, 2)
        assert pager.get(name, 2) == arr[2].tobytes()
        assert pager.get(name, 1) == arr[1].tobytes()
        assert pager.cache.stats().resident_bytes <= 48
        with pytest.raises(IndexError):
            pager.prefetch(name, 10)
    with pytest.raises(RuntimeError):
        pager.prefetch(name, 0)


def test_cache_invalid_capacity():
    with pytest.raises(ValueError):
        ByteLRU(-1)
