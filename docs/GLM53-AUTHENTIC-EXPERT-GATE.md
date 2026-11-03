# GLM-5.3 authentic expert gate — engineering status

**Pinned publisher revision:** `zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42`  
**Status, 2026-11-03:** Acquisition + validation code present on main. Authentic source payload has **not** been downloaded in the cloud environment. Official kernel parity remains NOT_TESTED. Full-model inference is not claimed.

Operator commands: [docs/GLM53-AUTHENTIC-EXPERT-OPERATOR.md](GLM53-AUTHENTIC-EXPERT-OPERATOR.md)

## Current gates

| Gate | Status | Evidence |
|------|--------|----------|
| Bounded acquisition code (6 tensors, 48 MiB cap, HTTP 206 only) | Implemented | `exactmoe/expert_acquire.py` |
| Local provenance + SHA-256 validation | Implemented | `validate_sample` |
| Independent CPU mathematical oracle | Implemented | `expert_validation.py` |
| Streamed FP8 CPU path | Implemented | `streamed_fp8.py` |
| Authentic original tensors acquired | **Blocked** — requires operator machine | — |
| Official GLM production-kernel parity | NOT_TESTED | — |
| RTX 5060 Ti VRAM measurement | Not run | — |
| Full GLM-5.3 inference | Not started | — |

## Exact Windows commands (operator)

```powershell
py -3 -m pip install -e ".[dev]"
py -3 -m pip install psutil
py -3 -m exactmoe.expert_acquire --out X:\GLM53-one-expert --layer 3 --expert 0 --max-source-mib 48
py -3 -m exactmoe.expert_validation X:\GLM53-one-expert --layer 3 --expert 0 --device cpu --mode dequant-f32 --evidence evidence\expert-cpu.json
py -3 -m exactmoe.streamed_fp8 X:\GLM53-one-expert --layer 3 --expert 0 --check-local-oracle
```

Success for this gate is `passed_internal_oracle: true` on the acquired sample, with provenance showing the pinned revision and the six tensor hashes.  
`official_glm53_kernel_parity` must remain `NOT_TESTED` and `full_glm53_inference` must remain false.

## Remaining blockers

- No authentic GLM-5.3 expert payload is present in the cloud environment.
- Official production FP8 kernel reference outputs for the fixed seed are unavailable.
- CUDA VRAM measurement requires the operator RTX 5060 Ti.
- Router + shared expert + remaining layers are explicitly out of scope until this gate passes.

No pruning, quantization of the reference checkpoint, second-level routers, or invented traces are part of this gate.
