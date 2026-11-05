"""Named GLM-5.3 runtime variants: which converted model, which engine, which settings.

Each variant records what it is and how it differs from the original FP8 model. It also records which
checks it has passed. A variant is "unverified" until evidence files exist for it. The registry never
upgrades that status on its own.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any

MODELS_ROOT = Path(os.environ.get("GLM53_MODELS_ROOT", r"C:\AI-Models"))
COLIBRI_DIR = Path(os.environ.get("COLIBRI_DIR", r"C:\colibri-review\src\c"))
COLIBRI_PYTHON = Path(os.environ.get("COLIBRI_PYTHON", r"C:\colibri-review\oracle-venv\Scripts\python.exe"))
META_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja")
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
CHECKS = ("doctor", "plan", "first_token", "oracle_agreement", "homelab_smoke")


@dataclass
class Variant:
    id: str
    description: str
    container: str                     # Colibri converted-model folder (out-*.safetensors + meta files)
    resident_bits: int                 # dense/attention/shared expert storage: 8 = int8, 16 = FP8 passthrough
    expert_bits: int                   # routed experts: 4 = int4 group-scaled
    group_size: int = 64
    dsa_indexer: bool = True           # converted with --indexer: sparse attention beyond index_topk tokens
    mtp: bool = False                  # MTP head converted (speculative decoding)
    uses_gpu: bool = False
    engine_dir: str = str(COLIBRI_DIR)
    python: str = str(COLIBRI_PYTHON)
    ram_gb: int = 100                  # RAM budget given to the engine (machine has 128 GB)
    ctx: int = 32768
    port: int = 8410
    pin_file: str = ""                 # Colibri usage/stats file of a role profile (see role_profiles.py)
    pin_gb: float = 0.0
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)   # appended to `coli serve`
    differs_from_original: list[str] = field(default_factory=list)

    def container_path(self) -> Path:
        return Path(self.container)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Variant":
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"variant {raw.get('id')!r} has unknown fields: {sorted(unknown)}")
        return cls(**raw)


def builtin_variants() -> list[Variant]:
    return [
        Variant(
            id="glm53-x4g64-r8",
            description="GLM-5.3 full, routed experts int4 gs64, resident int8, DSA indexer kept, no MTP",
            container=str(MODELS_ROOT / "GLM-5.3-colibri-x4g64-r8"),
            resident_bits=8, expert_bits=4, group_size=64, dsa_indexer=True, mtp=False,
            differs_from_original=[
                "routed experts int4 gs64: measured 14-15% MoE-output error on layer-3 experts (proxy input)",
                "resident tensors int8 instead of FP8",
                "no MTP head (no speculative decoding)",
            ],
        ),
        Variant(
            id="glm53-x4g64-r8-cuda",
            description="Same container as glm53-x4g64-r8, with coli's VRAM plan applied on the RTX 5060 Ti (--auto-tier)",
            container=str(MODELS_ROOT / "GLM-5.3-colibri-x4g64-r8"),
            resident_bits=8, expert_bits=4, group_size=64, dsa_indexer=True, mtp=False, uses_gpu=True,
            extra_args=["--auto-tier"],
            differs_from_original=[
                "routed experts int4 gs64: measured 14-15% MoE-output error on layer-3 experts (proxy input)",
                "resident tensors int8 instead of FP8",
                "no MTP head (no speculative decoding)",
                "GPU kernels: not bit-identical to the CPU path (oracle 17/17 on the tiny model)",
            ],
        ),
        Variant(
            id="glm53-x4g64-r8-fast",
            description=("Same container; HOMELAB Colibri branch (homelab/prefill-gpu-staging): prompt experts, "
                         "attention projections and the MLA attention core on the GPU as cuBLAS GEMMs"),
            container=str(MODELS_ROOT / "GLM-5.3-colibri-x4g64-r8"),
            resident_bits=8, expert_bits=4, group_size=64, dsa_indexer=True, mtp=False, uses_gpu=True,
            extra_env={
                "CUDA_DENSE": "0",              # the whole dense set (17.6 GB) does not fit in 16 GB
                "CUDA_DENSE_MATCH": "self_attn.kv_b_proj,self_attn.o_proj,self_attn.q_b_proj,"
                                    "self_attn.q_a_proj,self_attn.kv_a_proj",
                "CUDA_EXPERT_GB": "0",          # VRAM goes to attention; the expert tier gave no decode gain here
                "COLI_CUDA_GEMM_MIN": "32",
                "COLI_PFGPU": "4",
                "COLI_CUDA_MLA_PREFILL": "1",
                "DIRECT": "1",
                "PILOT": "1", "PILOT_REAL": "1",   # cross-layer prefetch: +7% decode measured
                "OMP_NUM_THREADS": "14",            # +10% over the default 8 P-core threads
            },
            ram_gb=112,
            differs_from_original=[
                "routed experts: GPTQ int4 (Bitdeer, being migrated in place) or RTN int4 g64 where not yet migrated",
                "resident tensors int8 instead of FP8",
                "no MTP head (no speculative decoding)",
                "GPU GEMMs: TF32 for experts and projections, FP32 attention; tiny-model logits match the CPU path",
            ],
        ),
        Variant(
            id="glm53-x4g64-r8-turbo",
            description="fast + cache-aware routing (CACHE_ROUTE J=2 M=16): 1.10 tok/s measured vs 0.75; for routine roles",
            container=str(MODELS_ROOT / "GLM-5.3-colibri-x4g64-r8"),
            resident_bits=8, expert_bits=4, group_size=64, dsa_indexer=True, mtp=False, uses_gpu=True,
            extra_args=["--no-think"],
            extra_env={
                "CUDA_DENSE": "0",
                "CUDA_DENSE_MATCH": "self_attn.kv_b_proj,self_attn.o_proj,self_attn.q_b_proj,"
                                    "self_attn.q_a_proj,self_attn.kv_a_proj",
                "CUDA_EXPERT_GB": "0", "COLI_CUDA_GEMM_MIN": "32", "COLI_PFGPU": "4", "COLI_CUDA_MLA_PREFILL": "1",
                "DIRECT": "1", "PILOT": "1", "PILOT_REAL": "1", "OMP_NUM_THREADS": "14",
                "CACHE_ROUTE": "1", "ROUTE_J": "2", "ROUTE_M": "16",
            },
            ram_gb=112,
            differs_from_original=[
                "CHANGES THE MODEL'S ROUTING: the top-2 experts are always the router's; the other 6 slots prefer "
                "experts already in RAM within the router's top-16 (about 16% of slots substituted). LL on 431 "
                "tokens showed no loss, BUT the routed-MoE output against FP8 (layer 3, real inputs) is 36% off "
                "(p90 68%): substitution is costly; not for proof decisions. COLI_SKIP_LANE is the measured "
                "better trade (same savings, ~3x less error) and should replace it once tested in the engine",
                "routed experts GPTQ/RTN int4, resident int8, no MTP, GPU GEMMs as in -fast",
            ],
        ),
        Variant(
            id="glm53-x4g64-r8-lane",
            description=("fast + the fast lane (COLI_SKIP_LANE=0.10): in decode, a selected expert that is not in RAM "
                         "and carries < 10% of the token's routing mass is skipped, no substitute, no renormalisation"),
            container=str(MODELS_ROOT / "GLM-5.3-colibri-x4g64-r8"),
            resident_bits=8, expert_bits=4, group_size=64, dsa_indexer=True, mtp=False, uses_gpu=True,
            extra_args=["--no-think"],
            extra_env={
                "CUDA_DENSE": "0",
                "CUDA_DENSE_MATCH": "self_attn.kv_b_proj,self_attn.o_proj,self_attn.q_b_proj,"
                                    "self_attn.q_a_proj,self_attn.kv_a_proj",
                "CUDA_EXPERT_GB": "0", "COLI_CUDA_GEMM_MIN": "32", "COLI_PFGPU": "4", "COLI_CUDA_MLA_PREFILL": "1",
                "DIRECT": "1", "PILOT": "1", "PILOT_REAL": "1", "OMP_NUM_THREADS": "14",
                "COLI_SKIP_LANE": "0.10",
            },
            ram_gb=112,
            differs_from_original=[
                "CHANGES THE MODEL'S ROUTING in decode: non-resident experts under 10% of the routing mass are "
                "skipped. Offline (layer 3, real inputs, LRU 54/layer): 24% of disk loads avoided at 10.9% routed-"
                "output error, vs CACHE_ROUTE's 36% avoided at 35.8%. Engine speed not yet measured",
                "routed experts GPTQ int4 (own / Bitdeer per shard), resident int8, no MTP, GPU GEMMs as in -fast",
            ],
        ),
    ]


class VariantRegistry:
    """Built-in variants plus an optional JSON file of operator-defined ones (same id overrides)."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._variants: dict[str, Variant] = {v.id: v for v in builtin_variants()}
        if path is not None and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw.get("variants", []):
                v = Variant.from_dict(item)
                self._variants[v.id] = v

    def ids(self) -> list[str]:
        return sorted(self._variants)

    def get(self, variant_id: str) -> Variant:
        try:
            return self._variants[variant_id]
        except KeyError:
            raise KeyError(f"unknown variant {variant_id!r}; known: {self.ids()}") from None


def readiness(variant: Variant) -> dict[str, Any]:
    """What is present on disk for this variant. Read-only; it does not open weight data."""
    root = variant.container_path()
    engine = Path(variant.engine_dir)
    missing: list[str] = []
    if not root.is_dir():
        return {"variant": variant.id, "ready": False, "missing": [f"container folder {root}"]}
    shards = sorted(root.glob("out-[0-9]*.safetensors"))
    idx = sorted(root.glob("out-idx-*.safetensors"))
    for name in META_FILES:
        if not (root / name).is_file():
            missing.append(name)
    if not shards:
        missing.append("out-*.safetensors")
    if variant.dsa_indexer and not idx:
        missing.append("out-idx-*.safetensors (DSA indexer pass)")
    if not (engine / "colibri.exe").is_file() and not (engine / "colibri").is_file():
        missing.append(f"engine binary in {engine}")
    if not (engine / "coli").is_file():
        missing.append(f"coli launcher in {engine}")
    if not Path(variant.python).is_file():
        missing.append(f"python {variant.python}")
    if variant.pin_file and not Path(variant.pin_file).is_file():
        missing.append(f"pin file {variant.pin_file}")
    return {"variant": variant.id, "ready": not missing, "missing": missing,
            "shards": len(shards), "indexer_shards": len(idx),
            "shard_bytes": sum(p.stat().st_size for p in shards + idx)}


def evidence_dir(variant: Variant) -> Path:
    return variant.container_path() / ".exactmoe-evidence" / variant.id   # per variant: two variants can share a container


def verification_status(variant: Variant) -> dict[str, Any]:
    """Which checks have an evidence record. Status is 'unverified' until every check has one with ok=true."""
    found: dict[str, Any] = {}
    root = evidence_dir(variant)
    for check in CHECKS:
        path = root / f"{check}.json"
        if path.is_file():
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                found[check] = bool(rec.get("ok"))
            except ValueError:
                found[check] = False
        else:
            found[check] = None
    status = "verified" if all(found[c] is True for c in CHECKS) else "unverified"
    return {"variant": variant.id, "status": status, "checks": found}
