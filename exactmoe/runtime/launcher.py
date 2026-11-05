"""Start, check and stop `coli serve` for a variant, on loopback only.

- The server binds 127.0.0.1. Any other host is refused, with no flag to override it.
- A variant that uses the GPU takes the shared GPU lock first and releases it when the server stops.
- `homelab_runtime_profile()` emits a record HOMELAB's RuntimeProfile.from_dict accepts, so HOMELAB's own
  RuntimeSupervisor can own the process instead of this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from .gpu_lock import GpuLock
from .variants import LOOPBACK, Variant, evidence_dir


class NotLoopback(ValueError):
    pass


def _check_host(host: str) -> str:
    if host not in LOOPBACK:
        raise NotLoopback(f"refusing to bind {host!r}: the GLM-5.3 runtime serves loopback only")
    return host


def base_url(variant: Variant, host: str = "127.0.0.1") -> str:
    return f"http://{_check_host(host)}:{variant.port}"


def serve_command(variant: Variant, host: str = "127.0.0.1") -> tuple[list[str], dict[str, str]]:
    """The exact argv and the extra environment for the engine."""
    _check_host(host)
    engine = Path(variant.engine_dir)
    argv = [variant.python, str(engine / "coli"), "serve",
            "--model", variant.container, "--host", host, "--port", str(variant.port),
            "--model-id", variant.id, "--ram", str(variant.ram_gb), "--ctx", str(variant.ctx),
            "--policy", "quality", "--gpu", "auto" if variant.uses_gpu else "none", *variant.extra_args]
    if "--host" in variant.extra_args or "--port" in variant.extra_args:
        raise ValueError("host and port come from the variant, not from extra_args")
    env = {"COLI_MODEL_ID": variant.id}
    if not variant.uses_gpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if not variant.mtp:
        env["MTP"] = "0"
    if variant.pin_file:
        env["PIN"] = variant.pin_file
        env["PIN_GB"] = str(variant.pin_gb or 20)
        env["AUTOPIN"] = "0"           # the role profile is authoritative; no drift from the live history
    for key, value in variant.extra_env.items():
        if key.upper() in {"COLI_API_KEY"}:
            raise ValueError("credentials are never stored in a variant")
        env[key] = str(value)
    return argv, env


def homelab_runtime_profile(variant: Variant, *, model_profile_id: str = "") -> dict[str, Any]:
    argv, env = serve_command(variant)
    return {
        "id": f"colibri-{variant.id}",
        "name": f"GLM-5.3 via Colibri ({variant.id})",
        "executable": argv[0],
        "args": argv[1:],
        "cwd": variant.engine_dir,
        "env": env,
        "enabled": True,
        "auto_start": False,
        "stop_on_exit": True,
        "show_console": False,
        "startup_timeout_seconds": 1800.0,     # first load of a ~420 GB container from NVMe
        "health": {"kind": "http", "url": f"{base_url(variant)}/v1/models", "timeout_seconds": 5.0},
        "notes": ("Variant lane, NOT the exact model: " + "; ".join(variant.differs_from_original)
                  + (". Takes the shared GPU lock." if variant.uses_gpu else ". CPU only.")),
        "model_profile_id": model_profile_id,
    }


def _get(url: str, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 - loopback URL built above
        return resp.status, resp.read()


def healthy(variant: Variant, *, timeout: float = 5.0, getter: Callable[[str, float], tuple[int, bytes]] = _get) -> bool:
    try:
        status, _ = getter(f"{base_url(variant)}/v1/models", timeout)
        return status < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


class Server:
    """One owned `coli serve` process. HOMELAB may own it instead via homelab_runtime_profile()."""

    def __init__(self, variant: Variant, *, log_path: Path | None = None, lock_path: Path | None = None):
        self.variant = variant
        self.log_path = log_path or (Path(variant.engine_dir) / f"serve-{variant.id}.log")
        self.lock = GpuLock(holder=f"exactmoe colibri {variant.id}", purpose="GLM-5.3 model server",
                            **({"path": lock_path} if lock_path else {})) if variant.uses_gpu else None
        self.process: subprocess.Popen | None = None

    def start(self, *, wait_seconds: float = 1800.0, poll: float = 5.0) -> None:
        if healthy(self.variant):
            raise RuntimeError(f"something already serves {base_url(self.variant)}; not starting a second server")
        argv, extra = serve_command(self.variant)
        if self.lock:
            self.lock.acquire()
        try:
            env = {**os.environ, **extra}
            log = open(self.log_path, "ab")
            self.process = subprocess.Popen(argv, cwd=self.variant.engine_dir, env=env, stdout=log,
                                            stderr=subprocess.STDOUT)
            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"coli serve exited with {self.process.returncode}; see {self.log_path}")
                if healthy(self.variant):
                    return
                time.sleep(poll)
            raise TimeoutError(f"coli serve not healthy after {wait_seconds:.0f}s; see {self.log_path}")
        except BaseException:
            self.stop()
            raise

    def stop(self, *, grace: float = 60.0) -> None:
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            self.process = None
            if self.lock:
                self.lock.release()


def chat(variant: Variant, messages: list[dict[str, str]], *, max_tokens: int = 64, temperature: float = 0.0,
         timeout: float = 3600.0) -> dict[str, Any]:
    body = json.dumps({"model": variant.id, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(f"{base_url(variant)}/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 - loopback URL
        return json.loads(resp.read())


def record_evidence(variant: Variant, check: str, ok: bool, details: dict[str, Any]) -> Path:
    root = evidence_dir(variant)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{check}.json"
    path.write_text(json.dumps({"check": check, "ok": ok, "variant": variant.id,
                                "utc": datetime.now(timezone.utc).isoformat(), "details": details}, indent=1),
                    encoding="utf-8")
    return path
