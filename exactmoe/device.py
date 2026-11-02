"""Optional PyTorch device staging of exact tensor bytes.

This is a staging primitive, NOT an FP8 compute kernel or a GLM runtime.
Returning uint8 rather than decoding to another dtype avoids accidental
numerical modification of FP8/BF16/other checkpoint formats.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from .pager import ExpertPager


@dataclass(frozen=True)
class DeviceStats:
    resident_bytes: int
    entries: int
    uploads: int
    cache_hits: int
    evictions: int


class DeviceByteCache:
    def __init__(self, capacity_bytes: int, device: str = "cuda"):
        if capacity_bytes < 0:
            raise ValueError("GPU cache budget must be nonnegative")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch required for device staging: pip install torch") from exc
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        self.capacity = capacity_bytes
        self._items: OrderedDict[tuple[str, int | None], object] = OrderedDict()
        self._resident = self._uploads = self._hits = self._evictions = 0
        self._lock = RLock()

    def stage(self, pager: ExpertPager, name: str, expert: int | None = None):
        """Transfer original bytes to device in uint8 storage, with bounded residency.

        Caller must release references to evicted tensors to reclaim device memory.
        PyTorch's caching allocator may reserve more VRAM than live tensor bytes.
        """
        key = (name, expert)
        with self._lock:
            if key in self._items:
                self._hits += 1
                self._items.move_to_end(key)
                return self._items[key]
        raw = pager.get(name, expert)
        if len(raw) > self.capacity:
            raise MemoryError(f"tensor slice {name} ({len(raw)} bytes) exceeds device budget")
        # Separate buffer copies bytes without interpreting their numeric meaning.
        host = self.torch.frombuffer(bytearray(raw), dtype=self.torch.uint8)
        tensor = host.to(self.device, non_blocking=False)
        with self._lock:
            if key in self._items:
                return self._items[key]
            while self._items and self._resident + len(raw) > self.capacity:
                _, old = self._items.popitem(last=False)
                self._resident -= old.numel()
                self._evictions += 1
            self._items[key] = tensor
            self._resident += tensor.numel()
            self._uploads += 1
        return tensor

    def stats(self) -> DeviceStats:
        with self._lock:
            return DeviceStats(
                resident_bytes=self._resident,
                entries=len(self._items),
                uploads=self._uploads,
                cache_hits=self._hits,
                evictions=self._evictions,
            )

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._resident = 0
