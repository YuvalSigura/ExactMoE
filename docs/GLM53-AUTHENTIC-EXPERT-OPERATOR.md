# GLM-5.3 Authentic FP8 Expert Gate — Operator Commands

**Pinned revision:** `zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42`  
**Target:** Layer 3, Expert 0  
**Source payload cap:** 48 MiB  
**Status:** Acquisition and validation code exist. Authentic source bytes have **not** been downloaded in this environment. Full-model inference is **not** claimed.

This document gives the exact Windows commands to acquire and execute **one** original routed expert. The download runs on the operator machine. The code never falls back to a full shard or full checkpoint download.

## Prerequisites

```powershell
git clone https://github.com/YuvalSigura/ExactMoE.git
cd ExactMoE
py -3 -m pip install -e ".[dev]"
py -3 -m pip install psutil
```

Optional CUDA path requires a separately installed CUDA-enabled PyTorch that sees the RTX 5060 Ti.

## Stage 1 — Bounded acquisition (operator machine)

Creates a compact local sample containing the six original tensors (3 FP8 weights + 3 inverse scales) plus provenance. Does **not** download whole shards.

```powershell
py -3 -m exactmoe.expert_acquire --out X:\GLM53-one-expert --layer 3 --expert 0 --max-source-mib 48
```

Expected source payload for the pinned config: **37,757,952 bytes** (37,748,736 FP8 + 9,216 scales).

If this fails with HTTP 200, missing Content-Range, or redirect outside Hugging Face HTTPS hosts, the downloader aborts. Do not remove the range checks.

## Stage 2 — Numerical execution on the acquired sample

```powershell
py -3 -m exactmoe.expert_validation X:\GLM53-one-expert --layer 3 --expert 0 --device cpu --mode dequant-f32 --evidence evidence\expert-cpu.json
py -3 -m exactmoe.streamed_fp8 X:\GLM53-one-expert --layer 3 --expert 0 --check-local-oracle
```

Optional CUDA:

```powershell
py -3 -m exactmoe.expert_validation X:\GLM53-one-expert --layer 3 --expert 0 --device cuda --mode dequant-f32 --evidence evidence\expert-cuda.json
```

Success criteria for this gate:
- `passed_internal_oracle` is true
- `official_glm53_kernel_parity` remains `NOT_TESTED`
- `full_glm53_inference` remains false
- Provenance reports the pinned revision and the six tensor SHA-256 values
- Evidence JSON records latency, process RSS peak, and logical source bytes

A synthetic fixture pass is **not** an authentic-expert pass.

## What is still blocked

- Authentic source bytes have not been acquired in the cloud environment.
- Official production FP8 kernel outputs for the same fixed input are not available.
- RTX 5060 Ti VRAM measurement requires the operator machine.
- Router + shared expert + remaining 77 layers + DSA/MLA + tokenizer are out of scope until this gate passes.

## Not authorized in this gate

No pruning, no 1-bit / 1.58-bit quantization of the reference checkpoint, no second-level sub-expert router, no bit-plane approximation presented as exact inference, no invented router traces, no full checkpoint download.
