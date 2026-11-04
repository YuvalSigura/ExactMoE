"""Role profiles: which experts each HOMELAB role uses, and which to keep resident in RAM.

This is the safe form of "split the model into several models". Every role still runs the whole model:
an expert that is not pinned is read from disk, never dropped. Roles differ only in what sits in RAM,
so output is unchanged and only speed differs.

Input is Colibri's text usage format (`STATS=<file>`, or the live `<container>/.coli_usage`):
    -1 <n_layers> <n_experts>        dimensions
    -2 <format_version> <engine_id>  identity
    <layer> <expert> <count>         one row per used expert
A role's usage is the difference of the live usage file before and after that role's requests.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

INT4_G64_EXPERT_BYTES = 37_757_952 * 9 // 16      # 36 MiB FP8 expert -> int4 + one fp32 scale per 64


@dataclass
class Usage:
    n_layers: int
    n_experts: int
    version: int = 1
    engine_id: int = 0
    counts: Counter = field(default_factory=Counter)   # (layer, expert) -> selections

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def read_usage(path: Path) -> Usage:
    raw = path.read_bytes()
    if raw[:4] and not raw[:1].isdigit() and raw[:1] not in b"-\r\n \t":
        raise ValueError(f"{path}: binary usage history (another engine); only the text format is read")
    dims = ident = None
    counts: Counter = Counter()
    for line_no, line in enumerate(raw.decode("ascii").split("\n"), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_no}: expected 3 fields, got {line!r}")
        a, b, c = (int(p) for p in parts)
        if a == -1:
            dims = (b, c)
        elif a == -2:
            ident = (b, c)
        elif a >= 0:
            if c < 0:
                raise ValueError(f"{path}:{line_no}: negative count")
            counts[(a, b)] += c
    if dims is None:
        raise ValueError(f"{path}: no dimensions record (-1 <layers> <experts>)")
    for (layer, expert) in counts:
        if not (0 <= layer <= dims[0] and 0 <= expert < dims[1]):   # layer == n_layers is the MTP layer
            raise ValueError(f"{path}: row ({layer}, {expert}) outside {dims}")
    version, engine_id = ident if ident else (1, 0)
    return Usage(dims[0], dims[1], version, engine_id, counts)


def write_usage(path: Path, usage: Usage) -> None:
    lines = [f"-1 {usage.n_layers} {usage.n_experts}", f"-2 {usage.version} {usage.engine_id}"]
    lines += [f"{l} {e} {n}" for (l, e), n in sorted(usage.counts.items()) if n > 0]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _same_shape(a: Usage, b: Usage) -> None:
    if (a.n_layers, a.n_experts, a.engine_id) != (b.n_layers, b.n_experts, b.engine_id):
        raise ValueError("usage files come from different models or engines")


def usage_delta(before: Usage, after: Usage) -> Usage:
    """What one role added to the live usage history between two snapshots."""
    _same_shape(before, after)
    delta = Counter()
    for key, n in after.counts.items():
        d = n - before.counts.get(key, 0)
        if d < 0:
            raise ValueError("the usage history shrank between snapshots (rewritten or reset); recollect")
        if d:
            delta[key] = d
    return Usage(after.n_layers, after.n_experts, after.version, after.engine_id, delta)


def pin_set(usage: Usage, budget_bytes: int, expert_bytes: int = INT4_G64_EXPERT_BYTES) -> list[tuple[int, int]]:
    k = max(0, budget_bytes // expert_bytes)
    ranked = sorted(usage.counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [key for key, _ in ranked[:k]]


def coverage(usage: Usage, pinned: list[tuple[int, int]] | set[tuple[int, int]]) -> float:
    """Share of this usage's expert selections that the pinned set serves from RAM."""
    if not usage.total:
        return 0.0
    pinned = set(pinned)
    return sum(n for key, n in usage.counts.items() if key in pinned) / usage.total


def merge(profiles: dict[str, Usage], weights: dict[str, float] | None = None, scale: int = 1_000_000) -> Usage:
    """One pin profile for a server that serves several roles. Each role is normalised to its share first,
    so a talkative role does not crowd out the others."""
    if not profiles:
        raise ValueError("no profiles to merge")
    first = next(iter(profiles.values()))
    weights = weights or {name: 1.0 for name in profiles}
    wsum = sum(weights[name] for name in profiles)
    acc: Counter = Counter()
    for name, usage in profiles.items():
        _same_shape(first, usage)
        if not usage.total:
            continue
        for key, n in usage.counts.items():
            acc[key] += n / usage.total * weights[name] / wsum
    counts = Counter({key: max(1, round(v * scale)) for key, v in acc.items()})
    return Usage(first.n_layers, first.n_experts, first.version, first.engine_id, counts)


def role_report(profiles: dict[str, Usage], budget_bytes: int,
                expert_bytes: int = INT4_G64_EXPERT_BYTES) -> dict[str, object]:
    """For each role: RAM hit share with its own pin set, with the merged set, and with every other role's set.
    If the off-diagonal numbers are close to the diagonal, separate role profiles are not worth a restart."""
    own = {name: pin_set(u, budget_bytes, expert_bytes) for name, u in profiles.items()}
    merged = pin_set(merge(profiles), budget_bytes, expert_bytes)
    report: dict[str, object] = {"budget_bytes": budget_bytes, "experts_pinned": len(merged), "roles": {}}
    for name, usage in profiles.items():
        report["roles"][name] = {
            "selections": usage.total,
            "own_profile_hit": round(coverage(usage, own[name]), 4),
            "merged_profile_hit": round(coverage(usage, merged), 4),
            "other_profiles_hit": {o: round(coverage(usage, own[o]), 4) for o in profiles if o != name},
        }
    return report
