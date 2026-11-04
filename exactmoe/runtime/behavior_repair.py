"""Protocol for behavioural repair of a converted variant (in the style of MIT's "Weight-Space Programming").

Lesson from their logs: the big gain did not come from shrinking numeric error. It came from finding a
specific behavioural failure (Qwen's repetition loops), locating the internal direction that drives it,
and making a small targeted edit, judged only by measurement. After conversion, GLM-5.3 may have its
own failures of that kind, which a numeric error metric does not show: loops, broken tool-call JSON,
premature stops. Repair targets those.

This module fixes the rules, not the edit:
- DEV and HOLDOUT tasks are split once, deterministically, before any edit is tried. HOLDOUT is only
  for confirming a frozen candidate.
- Each candidate is scored as a paired comparison (same tasks, base vs edited) on DEV.
- Before HOLDOUT, one candidate is frozen together with its hash.
- The candidate is accepted only if the HOLDOUT paired gain exceeds `min_se` standard errors, with no
  more regressions than allowed.
- The reference behaviour comes from the exact lane (original FP8) wherever it can be run. Localising
  a failure compares int4 against FP8 layer by layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence


def split_tasks(task_ids: Sequence[str], holdout_fraction: float = 0.5, salt: str = "exactmoe-repair-v1") -> dict[str, list[str]]:
    """Deterministic split by hash: the same task always lands on the same side, whatever the order."""
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task ids must be unique")
    dev, hold = [], []
    for tid in task_ids:
        h = int(hashlib.sha256(f"{salt}:{tid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (hold if h < holdout_fraction else dev).append(tid)
    return {"dev": sorted(dev), "holdout": sorted(hold)}


@dataclass(frozen=True)
class PairedResult:
    n: int
    base_successes: int
    edited_successes: int
    rescues: int          # base failed, edited succeeded
    regressions: int      # base succeeded, edited failed

    @property
    def gain(self) -> float:
        return (self.edited_successes - self.base_successes) / self.n if self.n else 0.0

    @property
    def se(self) -> float:
        """Standard error of the paired difference."""
        if self.n < 2:
            return math.inf
        d_mean = self.gain
        sq = (self.rescues * (1 - d_mean) ** 2 + self.regressions * (-1 - d_mean) ** 2
              + (self.n - self.rescues - self.regressions) * d_mean ** 2)
        return math.sqrt(sq / (self.n - 1) / self.n)


def paired(base: dict[str, bool], edited: dict[str, bool], tasks: Sequence[str]) -> PairedResult:
    missing = [t for t in tasks if t not in base or t not in edited]
    if missing:
        raise ValueError(f"{len(missing)} tasks lack a result on one side, e.g. {missing[0]!r}")
    b = [bool(base[t]) for t in tasks]
    e = [bool(edited[t]) for t in tasks]
    return PairedResult(len(tasks), sum(b), sum(e),
                        sum(1 for x, y in zip(b, e) if not x and y),
                        sum(1 for x, y in zip(b, e) if x and not y))


def freeze(candidate: dict) -> str:
    """Hash of a candidate edit description; recorded before HOLDOUT is run."""
    return hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()


def accept(holdout: PairedResult, *, frozen_hash: str, candidate: dict, min_se: float = 1.5,
           max_regression_share: float = 0.02) -> dict[str, object]:
    if freeze(candidate) != frozen_hash:
        return {"accepted": False, "reason": "candidate changed after it was frozen"}
    if holdout.n == 0:
        return {"accepted": False, "reason": "empty holdout"}
    if holdout.regressions / holdout.n > max_regression_share:
        return {"accepted": False, "reason": f"{holdout.regressions} regressions on {holdout.n} holdout tasks"}
    if not holdout.gain > min_se * holdout.se:
        return {"accepted": False, "reason": f"gain {holdout.gain:.4f} not above {min_se} SE ({holdout.se:.4f})"}
    return {"accepted": True, "gain": holdout.gain, "se": holdout.se,
            "rescues": holdout.rescues, "regressions": holdout.regressions}
