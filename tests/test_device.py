import numpy as np
import pytest
from safetensors.numpy import save_file

from exactmoe.pager import ExpertPager
from exactmoe.storage import TensorIndex


def test_uint8_staging_preserves_bits_and_budget(tmp_path):
    torch = pytest.importorskip("torch")
    from exactmoe.device import DeviceByteCache

    name = "layers.0.mlp.experts.gate_proj.weight"
    arr = np.arange(24, dtype=np.uint8).reshape(3, 8)
    save_file({name: arr}, str(tmp_path / "model.safetensors"))
    index = TensorIndex.from_directory(tmp_path)
    with ExpertPager(index, cache_bytes=64) as pager:
        device = DeviceByteCache(capacity_bytes=16, device="cpu")
        a = device.stage(pager, name, 0)
        assert a.dtype == torch.uint8
        assert bytes(a.numpy()) == bytes(arr[0])
        assert device.stage(pager, name, 0) is a
        device.stage(pager, name, 1)
        device.stage(pager, name, 2)
        assert device.stats().resident_bytes <= 16
        assert device.stats().cache_hits == 1
        assert device.stats().evictions >= 1


def test_reject_device_budget_and_oversized_tensor(tmp_path):
    pytest.importorskip("torch")
    from exactmoe.device import DeviceByteCache

    with pytest.raises(ValueError):
        DeviceByteCache(-1, "cpu")
    arr = np.ones((2, 16), dtype=np.uint8)
    name = "layers.0.mlp.experts.gate_proj.weight"
    save_file({name: arr}, str(tmp_path / "model.safetensors"))
    with ExpertPager(TensorIndex.from_directory(tmp_path), cache_bytes=0) as pager:
        with pytest.raises(MemoryError):
            DeviceByteCache(8, "cpu").stage(pager, name, 0)
