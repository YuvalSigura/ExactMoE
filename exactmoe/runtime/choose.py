"""Score instead of generate: pick among known options by the model's own likelihood.

Many HOMELAB decisions are choices (decision kind, role, a task from a list). Generating the answer costs
about one second per token on this machine. Reading tokens (prefill) is far cheaper, and a repeated prompt
prefix is reused from the KV cache. So each option is scored as a continuation of the same prompt: one
/v1/completions request with echo=true, max_tokens=1 (the server's minimum; that token is ignored) and logprobs, summing the log-probabilities of the
option's own tokens. The highest total wins (or the highest mean per token, for options of very different
length). The model's ranking is returned in full, so a caller can also see how sure it was.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Callable, Sequence
import urllib.request


@dataclass(frozen=True)
class Scored:
    option: str
    logprob: float          # sum over the option's tokens
    tokens: int
    @property
    def mean(self) -> float:
        return self.logprob / self.tokens if self.tokens else -math.inf


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 - loopback base URL from launcher
        return json.loads(resp.read())


def option_logprob(reply: dict, prompt_chars: int, end_chars: int | None = None) -> tuple[float, int]:
    """Sum the echoed token logprobs whose text starts inside [prompt_chars, end_chars): the option itself,
    not the prompt and not the one token the server insists on generating after it."""
    lp = (reply.get("choices") or [{}])[0].get("logprobs") or {}
    offsets, values = lp.get("text_offset") or [], lp.get("token_logprobs") or []
    total, n = 0.0, 0
    for off, v in zip(offsets, values):
        if off >= prompt_chars and (end_chars is None or off < end_chars) and v is not None:
            total += float(v); n += 1
    if n == 0:
        raise ValueError("the reply carries no logprobs for the option's tokens (echo/logprobs unsupported?)")
    return total, n


def choose(base_url: str, model: str, prompt: str, options: Sequence[str], *, by_mean: bool = False,
           timeout: float = 3600.0, post: Callable[[str, dict, float], dict] = _post) -> list[Scored]:
    if not options:
        raise ValueError("no options")
    if len(set(options)) != len(options):
        raise ValueError("options must be distinct")
    out = []
    for opt in options:
        body = {"model": model, "prompt": prompt + opt, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0}
        lp, n = option_logprob(post(f"{base_url}/v1/completions", body, timeout), len(prompt), len(prompt) + len(opt))
        out.append(Scored(opt, lp, n))
    return sorted(out, key=(lambda s: s.mean) if by_mean else (lambda s: s.logprob), reverse=True)


def confidence(ranked: Sequence[Scored]) -> float:
    """Probability share of the winner among the options (softmax over total logprobs)."""
    m = max(s.logprob for s in ranked)
    w = [math.exp(s.logprob - m) for s in ranked]
    return w[0] / sum(w)
