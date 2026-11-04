"""Two-stage expert evaluation: int4 first, then the strongest neurons again from the original FP8.

This builds on splitting an expert into neuron groups:
  1. Compute the expert with int4 weights. This also gives approximate activations h~ for all neurons.
  2. Rank neurons by |h~_i| * ||D~[:, i]||, which estimates each neuron's share of the output.
  3. For the top fraction, read the FP8 rows of gate/up and the FP8 column of down. Recompute those
     neurons exactly and swap their contribution in.
Nothing is predicted ahead of time: the selection comes from the int4 pass itself.

Measured on GLM-5.3 layer 3, 8 router-selected experts, proxy N(0, 0.1) inputs, holdout seeds:
  int4 g64 15.0% -> top 10%: 9.5% (0.66 of FP8 bytes read) -> top 25%: 6.9% (0.81)

This is the numpy reference. An engine port must agree with it to float tolerance.
Down columns are strided in the original layout. A fast port needs a transposed FP8 copy of down_proj
(about 1/3 of the original expert bytes) so each neuron's column is contiguous.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def silu(z: np.ndarray) -> np.ndarray:
    return z / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class ExpertWeights:
    gate: np.ndarray   # (inter, hidden)
    up: np.ndarray     # (inter, hidden)
    down: np.ndarray   # (hidden, inter)


def expert_forward(w: ExpertWeights, x: np.ndarray) -> np.ndarray:
    return w.down @ (silu(w.gate @ x) * (w.up @ x))


def select_hot(h_approx: np.ndarray, down_col_norms: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    k = int(round(fraction * h_approx.shape[0]))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    score = np.abs(h_approx) * down_col_norms
    return np.argpartition(-score, k - 1)[:k]


def two_stage_forward(quant: ExpertWeights, exact: ExpertWeights, x: np.ndarray, fraction: float,
                      down_col_norms: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (output, hot neuron indices). `exact` rows/columns are only touched at the hot indices,
    which is what lets an engine read just those bytes."""
    h = silu(quant.gate @ x) * (quant.up @ x)
    norms = down_col_norms if down_col_norms is not None else np.linalg.norm(quant.down, axis=0)
    hot = select_hot(h, norms, fraction)
    out = quant.down @ h
    if hot.size:
        he = silu(exact.gate[hot] @ x) * (exact.up[hot] @ x)
        out = out - quant.down[:, hot] @ h[hot] + exact.down[:, hot] @ he
    return out, hot


def bytes_read_fraction(fraction: float, int4_ratio: float = 0.5625) -> float:
    """Bytes read per expert relative to the FP8 expert: all int4 bytes plus `fraction` of the FP8 rows."""
    return int4_ratio + fraction
