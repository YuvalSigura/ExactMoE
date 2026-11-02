"""Bounded byte-exact caching and asynchronous prefetch.

All cache keys identify exact immutable source byte ranges. Prefetch only changes
the time at which a range is read; it can never choose or modify routed experts.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import Callable

from .storage import TensorIndex, read_bytes, read_first_axis_slice


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    resident_bytes: int
    entries: int


class ByteLRU:
    def __init__(self, capacity_bytes: int):
        if capacity_bytes < 0:
            raise ValueError("cache capacity must be nonnegative")
        self.capacity = capacity_bytes
        self._items: OrderedDict[tuple[str, int | None], bytes] = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._evictions = 0
        self._lock = RLock()

    def get(self, key: tuple[str, int | None]) -> bytes | None:
        with self._lock:
            if key not in self._items:
                self._misses += 1
                return None
            self._hits += 1
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: tuple[str, int | None], value: bytes) -> None:
        with self._lock:
            if key in self._items:
                self._bytes -= len(self._items.pop(key))
            if len(value) > self.capacity:
                return
            while self._items and self._bytes + len(value) > self.capacity:
                _, removed = self._items.popitem(last=False)
                self._bytes -= len(removed)
                self._evictions += 1
            self._items[key] = value
            self._bytes += len(value)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits, misses=self._misses, evictions=self._evictions,
                resident_bytes=self._bytes, entries=len(self._items),
            )


class ExpertPager:
    """Page full tensors or packed expert rows from safetensors into bounded RAM.

    Does not perform CUDA transfer or evaluate the model. The caller supplies
    the selected expert IDs; this class does not infer them.
    """

    def __init__(self, index: TensorIndex, cache_bytes: int, workers: int = 2,
                 max_prefetch: int = 16):
        if workers < 1 or max_prefetch < 1:
            raise ValueError("workers and max_prefetch must be positive")
        self.index = index
        self.cache = ByteLRU(cache_bytes)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exactmoe-io")
        self._pending: dict[tuple[str, int | None], Future[bytes]] = {}
        self._pending_lock = RLock()
        self.max_prefetch = max_prefetch
        self._closed = False

    def _read(self, key: tuple[str, int | None]) -> bytes:
        name, expert = key
        record = self.index[name]
        return read_bytes(record) if expert is None else read_first_axis_slice(record, expert)

    def get(self, name: str, expert: int | None = None) -> bytes:
        key = (name, expert)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        with self._pending_lock:
            future = self._pending.pop(key, None)
        data = future.result() if future is not None else self._read(key)
        self.cache.put(key, data)
        return data

    def prefetch(self, name: str, expert: int | None = None) -> bool:
        key = (name, expert)
        with self._pending_lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            # Completed reads need not be retained indefinitely.
            for old, future in list(self._pending.items()):
                if future.done() and old != key:
                    try:
                        data = future.result()
                    except Exception:
                        del self._pending[old]
                        continue
                    self.cache.put(old, data)
                    del self._pending[old]
            if key in self._pending or len(self._pending) >= self.max_prefetch:
                return False
            # Never prefetch unspecified or invalid tensors.
            record = self.index[name]
            if expert is not None:
                record.first_axis_slice(expert)
            self._pending[key] = self._pool.submit(self._read, key)
            return True

    def close(self) -> None:
        with self._pending_lock:
            self._closed = True
        self._pool.shutdown(wait=True)
        with self._pending_lock:
            self._pending.clear()

    def __enter__(self) -> "ExpertPager":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
