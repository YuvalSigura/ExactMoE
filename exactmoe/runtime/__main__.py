"""python -m exactmoe.runtime <command>

  variants                          list variants
  ready <variant>                   what is on disk (read-only)
  status <variant>                  which checks have evidence
  command <variant>                 the exact serve argv and env
  homelab-profile <variant>         a HOMELAB RuntimeProfile record (JSON)
  homelab-env <variant>             HOMELAB environment for this server (address, served model id, timeouts)
  serve <variant>                   start coli serve on loopback; Ctrl-C stops it and releases the GPU lock
  first-token <variant>             ask a running server one question; records the first_token evidence
  role-delta <before> <after> <out> one role's usage from two snapshots of <container>/.coli_usage
  role-report <budget_gb> name=file ...  RAM hit share per role: own, merged and other roles' pin sets
  role-merge <out> name=file ...    one pin file for a server that serves several roles
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from . import launcher, role_profiles
from .variants import VariantRegistry, readiness, verification_status


def _profiles(items: list[str]) -> dict[str, role_profiles.Usage]:
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"expected name=file, got {item!r}")
        out[name] = role_profiles.read_usage(Path(path))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m exactmoe.runtime", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=None, help="JSON file of extra variants")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("variants")
    for name in ("ready", "status", "command", "homelab-profile", "homelab-env", "serve", "first-token"):
        sub.add_parser(name).add_argument("variant")
    d = sub.add_parser("role-delta")
    d.add_argument("before", type=Path); d.add_argument("after", type=Path); d.add_argument("out", type=Path)
    r = sub.add_parser("role-report")
    r.add_argument("budget_gb", type=float); r.add_argument("profiles", nargs="+")
    m = sub.add_parser("role-merge")
    m.add_argument("out", type=Path); m.add_argument("profiles", nargs="+")
    a = ap.parse_args(argv)
    reg = VariantRegistry(a.registry)

    if a.cmd == "variants":
        for vid in reg.ids():
            v = reg.get(vid)
            print(f"{vid}: {v.description}")
        return 0
    if a.cmd in ("ready", "status", "command", "homelab-profile", "homelab-env", "serve", "first-token"):
        v = reg.get(a.variant)
    if a.cmd == "ready":
        print(json.dumps(readiness(v), indent=1)); return 0
    if a.cmd == "status":
        print(json.dumps(verification_status(v), indent=1)); return 0
    if a.cmd == "command":
        argv_, env = launcher.serve_command(v)
        print(json.dumps({"argv": argv_, "env": env, "cwd": v.engine_dir}, indent=1)); return 0
    if a.cmd == "homelab-profile":
        print(json.dumps(launcher.homelab_runtime_profile(v), indent=1)); return 0
    if a.cmd == "homelab-env":
        for key, value in launcher.homelab_env(v).items():
            print(f"{key}={value}")
        return 0
    if a.cmd == "serve":
        state = readiness(v)
        if not state["ready"]:
            print(json.dumps(state, indent=1)); return 2
        server = launcher.Server(v)
        server.start()
        print(f"serving {v.id} at {launcher.base_url(v)} (Ctrl-C stops it)", flush=True)
        try:
            while server.process and server.process.poll() is None:
                time.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
        return 0
    if a.cmd == "first-token":
        if not launcher.healthy(v):
            print(f"no healthy server at {launcher.base_url(v)}"); return 2
        t0 = time.monotonic()
        reply = launcher.chat(v, [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
                              max_tokens=32)
        text = (reply.get("choices") or [{}])[0].get("message", {}).get("content", "")
        ok = "391" in text
        path = launcher.record_evidence(v, "first_token", ok, {"reply": text, "seconds": round(time.monotonic() - t0, 1),
                                                                "usage": reply.get("usage")})
        print(json.dumps({"ok": ok, "reply": text, "evidence": str(path)}, indent=1)); return 0 if ok else 1
    if a.cmd == "role-delta":
        delta = role_profiles.usage_delta(role_profiles.read_usage(a.before), role_profiles.read_usage(a.after))
        role_profiles.write_usage(a.out, delta)
        print(f"{a.out}: {len(delta.counts)} experts, {delta.total} selections"); return 0
    if a.cmd == "role-report":
        rep = role_profiles.role_report(_profiles(a.profiles), int(a.budget_gb * 1024 ** 3))
        print(json.dumps(rep, indent=1)); return 0
    if a.cmd == "role-merge":
        merged = role_profiles.merge(_profiles(a.profiles))
        role_profiles.write_usage(a.out, merged)
        print(f"{a.out}: {len(merged.counts)} experts"); return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
