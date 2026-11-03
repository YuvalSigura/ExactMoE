# ExactMoE

> **Direction update · 2026-11-03 · DRAFT / NOT A FULL-MODEL RUNNER**
>
> **Reuse proven execution systems; preserve the original GLM-5.3 model; verify original bytes and mathematics.**
> Our plan is no longer to build every inference subsystem from scratch. It is to **adapt carefully selected components** from Strata, Project Maya and kimi-k3-in-c, alongside independent GLM-5.3 reference kernels, into a hardware-aware, SSD-first runner for the **full** pinned GLM-5.3 on a Core Ultra 7 265KF + 128GB RAM + RTX 5060 Ti 16GB. A new feature branch and PR are **not evidence** of working inference.

## New upstream-first implementation plan

**[Full architecture, source-by-source transfer map, concrete acceptance gates and known blockers](docs/UPSTREAM_ADAPTATION_PLAN.md)** · **[Machine-readable source manifest, original upstream SHAs, licenses and exact donor paths](docs/UPSTREAM_SOURCE_MANIFEST.json)**

| Repository / pinned revision | Components to reuse or study | Required changes for our full original GLM-5.3 |
| --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata/tree/ffa80d73133ea5c548aefe57889a16e8085a938f) (MIT) | `src/core/expert_cache.cpp`, `src/core/expert_source.cpp`, `src/program/generate.cpp`, `src/split/src/engine.cu`, `serve/server.py` | GPU cache, CPU fallback and batch scheduling are candidate **design/kernel donors**; add original safetensors FP8/scale handling and full-GLM-specific compute. Strata's RAM-held expert pools alone cannot hold ~704 GiB here. |
| [Project Maya](https://github.com/mw00/project-maya/tree/10e69a0eb11541004abb24f9fa47147e9002326f) (MIT) | `src/core/glm_model.cu`, `glm_window.cu`, `glm_prefill.cu`, `glm_mtp.cu`, `expert_source.cpp`, `serve/server.py` | Its working GLM specialization is **GLM-5.3-Flash**. Full GLM-5.3 has **78 layers vs 45**, **256 experts vs 288** and a different attention stack; replace/port affected layers, router geometry and original-block-FP8 source. Never rebrand Flash as the full model. |
| [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/81bb6c23ccee2154bc35cad922588ab66494a9a9) (Apache-2.0) | `src/io/k3_st.c`, `k3_load.c`, `k3_trunk.c`, `src/cache/k3_cache.c`, `tools/sim_cache.py` | Borrow **bounded SSD/trunk streaming, coalesced reads, and trace-based cache evaluation patterns**. Kimi's 2.78T model uses **MXFP4**, not GLM's FP8, and its C99 kernels are not drop-in GLM kernels. Preserve Apache NOTICE if any actual code is vendored. |

**External code policy:** these are reviewed *sources to selectively port*, **not** code copied wholesale into this branch. Each concrete port requires the source commit, exact attribution/license, dependency/license review, minimal isolated changes, reference parity and regression tests. Model weights/drafters are separately licensed; never automatically download them or treat DFlash2 as unconditionally MIT.

**Two distinct lanes:**
1. `Reference / Original FP8`: unchanged publisher `zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42`, official router and all model layers; correctness first, then out-of-core performance, then loopback-only API + tool calling for HOMELAB. **This is the primary gate.**
2. `Performance / Variant` (optional): model quantization or architecture adaptations only under **separate names and benchmark suites**, never a replacement for original-model evidence. GLM-5.3-Flash can be tested as an optional practical *different model*, not a milestone for full GLM-5.3.

### First new CPU-only measurement: Expert Traffic Simulator

The first implemented piece is an original, provenance-aware **logical I/O simulator**. It replays a fixed per-token original-router expert trace, comparing no cache, byte-budgeted LRU and history-only LFU. It **does not** fetch weights, execute GLM, measure physical SSD traffic or predict runtime speed. The included trace is *synthetic* (not real GLM outputs).

```powershell
python -m pip install -e ".[dev]"
python -m exactmoe.traffic_sim --trace docs/experiments/traffic_synthetic.json --cache-mib 2 --policy none
python -m exactmoe.traffic_sim --trace docs/experiments/traffic_synthetic.json --cache-mib 2 --policy lru
python -m exactmoe.traffic_sim --trace docs/experiments/traffic_synthetic.json --cache-mib 2 --policy lfu
python -m pytest -q tests/test_traffic_sim.py
```

### Bounded original-byte streaming — added in this branch

`exactmoe.streaming.stream_record` reads a single indexed safetensors tensor/range in small byte-exact chunks, preserving original bytes and refusing out-of-range or oversized requests. It records a SHA-256 digest, optionally enforces an expected digest, and checks whether source file metadata changed while reading. **A sink must quarantine all received bytes until a successful receipt is returned; this is not a complete cryptographic file-authenticity service.** Inspired by Kimi's SSD-first approach, but newly written for ExactMoE's existing `TensorRecord` and independent of Kimi's MXFP4 kernels.

```powershell
python -m pytest -q tests/test_streaming.py
```

This is genuine bounded file I/O on a named tensor range, **not** original GLM-5.3 execution, CUDA staging, or a physical SSD throughput benchmark.

**Milestone sequence:** authenticate one actual original expert → independent FP8 numeric oracle → correct router/shared MoE stage → all 78 layers + DSA/MLA + full token → measure native SSD/RAM/VRAM traffic → proven performance optimization → 32-token acceptance → local streaming/tool-capable API. Full details and per-stage negative tests are in the [adaptation plan](docs/UPSTREAM_ADAPTATION_PLAN.md). **Current status: simulated/coded components exist; authentic real-expert parity and full model generation are not yet proven.**

---

## Previous engineering README and reproducibility instructions

## Complete plan to first real local GLM-5.3 answer

Read [GLM53 end-to-end execution plan](docs/GLM53-END-TO-END-PLAN.md) and [all candidate hardware/runtime architectures](docs/GLM53-ALL-ARCHITECTURES.md) before claiming model support. It specifies 12 dependency-ordered engineering gates, real FP8/DSA correctness, bounded CPU/GPU/NVMe paging, reference parity, a 32-token **full model** acceptance test, and the later local HTTP/UI release.

**Status:** Python paging/control-plane demos and pinned model metadata have been validated. The full model cannot yet produce a token here; the proposed `exactmoe serve` command in the plan is NOT implemented.

**Current development gate:** [bounded authentic FP8 expert acquisition, independent numerical oracle and evidence report](docs/GLM53-AUTHENTIC-EXPERT-GATE.md). Code and synthetic checks exist on the development branch; no original publisher expert payload, production-kernel parity or full-model generation has been validated in this environment.


**Preserve the weights. Virtualize the memory. Verify the math.**

[![CI](https://github.com/YuvalSigura/ExactMoE/actions/workflows/ci.yml/badge.svg)](https://github.com/YuvalSigura/ExactMoE/actions/workflows/ci.yml)

ExactMoE investigates how to run frontier-scale Mixture-of-Experts models on consumer machines by moving unchanged model weight bytes between SSD, RAM and GPU. The long-term research target is the **full GLM-5.3 (~744B total parameters)**, not a distilled or smaller substitute.

> **Honest status (v0.1):** This repository currently contains a functional **safetensors metadata scanner, byte-preserving tensor/slice reader, bounded RAM cache, asynchronous prefetch, and runnable correctness-checked miniature MoE.** It is **not yet a GLM-5.3 inference runtime**. It does not implement complete GLM inference; it provides an experimental GLM-style routed FP8 FFN reference (see next section), but not DSA attention, full model numerical parity or native fast FP8 CUDA kernels.

## Experimental real FP8 routed expert and one MoE FFN stage

The repo now has a **single-expert** and **single sparse-FFN-stage** executor,
both using the *original local safetensors FP8 weights and scales* in bounded
CPU/GPU float32 tile computation:

```powershell
py -3 -m pip install -e ".[dev]"
exactmoe expert-plan "X:\Models\GLM-5.3" --layer 3 --expert 0
exactmoe expert-smoke "X:\Models\GLM-5.3" --layer 3 --expert 0 --device cpu
exactmoe layer-smoke "X:\Models\GLM-5.3" --layer 3 --device cpu
```

These commands require the **original sharded index and only the necessary completed shard files**. `expert-plan` reads the manifest and lists needed filenames without loading any model weights; `expert-smoke` loads only the referenced expert shards. `layer-smoke` first reads the router and then only selected expert/shared-expert shards.
They do not generate text, run DSA or validate production numeric parity.
Optional CUDA device is `--device cuda` with a separate CUDA-capable PyTorch
installation. Full details: [FP8+MoE stage](docs/FP8-EXPERT-EXECUTION.md).

### Track the original 141-shard download safely

While your pinned GLM-5.3 FP8 checkpoint is downloading:

```powershell
git pull --ff-only origin main
exactmoe checkpoint-status "X:\Models\GLM-5.3"
```

This read-only command checks the safetensors index, enumerates missing
shards, and validates *present* shard headers and tensor names against the
official manifest. It does **not** download weights, inspect weight payload
bytes, validate source cryptographic hashes or claim successful inference.
Incomplete Hugging Face `.incomplete` transfer files will not be considered
valid finished shards.

## Why

A sparse MoE does not evaluate every expert for each token, but all potentially selected experts still need to be accessible. The research question is whether we can move the weight residency requirement to NVMe and accept slower generation **without modifying the original checkpoint or model semantics**.

Preserving weight bytes alone is **not** proof of identical generated answers. Numeric kernels, precision, routing, tokenizer, RNG and sampling need to be validated against a reference implementation.

## Quick start (Python 3.10+)

```bash
git clone https://github.com/YuvalSigura/ExactMoE.git
cd ExactMoE
python -m pip install -e ".[dev]"
exactmoe demo
exactmoe bench --steps 100
python -m pytest -q
```

### Profile the full GLM checkpoint without downloading 756GB of weights

```bash
python -m pip install -e ".[research]"
exactmoe profile --hf zai-org/GLM-5.3
# For reproducibility, use --revision <commit-sha> when known
```

This downloads **only config.json and model.safetensors.index.json**, and reports
model configuration, shard count, reported weight size and expert-related tensor names.
The GLM profile now also audits individually named expert tensors and corresponding inverse-scale entries, producing a metadata-only shard inventory. A passing audit is NOT proof of valid tensor offsets, numerical parity or working inference.

### Inspect any local safetensors checkpoint (metadata only)

```bash
exactmoe inspect /path/to/your/model --show-candidates 8 --expected-experts 256
```

On Windows PowerShell:

```powershell
exactmoe inspect "D:\\Models\\GLM-5.3" --show-candidates 8 --expected-experts 256
```

This **does not download** weights and **does not perform inference**. A full GLM-5.3 checkpoint requires hundreds of GB of free disk space and may have different tensor layouts. The indexer reads only safetensors headers, not entire checkpoint weights. Untrusted file paths in shard manifests are rejected.

## Architecture

```text
Original safetensors shards (NVMe)
        | read original byte ranges
        v
  TensorIndex + ExpertPager
        |     Bounded byte LRU (RAM)
        |     Async prefetch (never changes router decisions)
        v
Future: GPU staging / native FP8 expert kernels
        |
Future: faithful GLM-5.3 routing + DSA + KV + sampling
        |
Reference parity tests
```

Optional GPU staging: `exactmoe.device.DeviceByteCache` accepts exact source bytes
from the pager and uploads them as `torch.uint8` tensors under a configurable
resident-byte budget. On your RTX 5060 Ti it can demonstrate byte-for-byte
transfer, **not** correct FP8 matrix multiplication or complete GLM inference.
Install a CUDA-enabled PyTorch build separately if using this feature.

Current demonstration: synthetic F32 top-k SwiGLU expert FFNs are evaluated using **the same selected expert IDs** in an in-memory dense reference and in the byte-streamed implementation. The demo verifies parity on deterministic inputs and exercises eviction/prefetch behavior. This is a **control-plane proof of concept**, not a quality benchmark for the frontier model.

## Roadmap and science

See [ARCHITECTURE](docs/ARCHITECTURE.md), [GLM integration plan](docs/GLM-5.3.md), and [ROADMAP](docs/ROADMAP.md).

### Research invariants

1. Never silently quantize, prune, merge or change checkpoint tensors.
2. The reference router must determine each expert choice; prefetch errors may cost time but not accuracy.
3. Treat architecture coverage and numerical agreement as separate gates.
4. Publish reproducible latency, I/O, memory, and parity metrics before claiming model support.

### References

- [GLM-5 source](https://github.com/zai-org/glm-5)
- [GLM-5.3 checkpoint](https://huggingface.co/zai-org/GLM-5.3)
- [HF GLM-MoE-DSA architecture](https://huggingface.co/docs/transformers/en/model_doc/glm_moe_dsa)
- [safetensors format](https://github.com/huggingface/safetensors)
- [vLLM GLM-5.3 reference recipe](https://recipes.vllm.ai/zai-org/GLM-5.3)

MIT license for ExactMoE original code. Model weights retain their own license and are not redistributed here.

### Reproducible GLM metadata gate

    exactmoe profile --hf zai-org/GLM-5.3 --revision aca966e4e02791568aa6a4ced368624b3d897f42

Check the expert_layout.metadata_complete field: it must be true before attempting FP8 expert paging. This checks metadata only, not checkpoint data or output quality.

### Inspect real GLM-5.3 FP8 tensor shapes WITHOUT downloading weight shards

Safetensors files put the full tensor **header** at the start of each file.
The `remote-header` command makes two bounded HTTP Range requests to the
pinned model revision, verifies byte-range responses (HTTP 206 only), and parses
the true FP8 expert matrix and inverse-scale tensor shapes. It **never reads
the weight payload**. The server must support HTTP byte ranges.

```powershell
exactmoe remote-header --hf zai-org/GLM-5.3 --revision aca966e4e02791568aa6a4ced368624b3d897f42 --shard model-00002-of-00141.safetensors --samples 5 | Out-File -Encoding utf8 glm53-shard02-header.json
```

A response with HTTP 200 is **rejected before any bytes are read** to avoid
accidentally downloading an entire multi-gigabyte shard. Header length is capped
at 16 MiB. The JSON report records the actual number of bytes transferred and
FP8 128×128 block scale shape checks. This is metadata inspection ONLY;
no GPU matrix multiplication, routing or model parity has been verified.

No model tensors, checkpoints or local hardware reports should be committed.

### Intel Core Ultra 7 265KF: hybrid CPU + NPU + NVIDIA GPU path

For the 20-core Arrow Lake CPU (8 P-cores + 12 E-cores), Intel AI Boost NPU
(13 INT8 TOPS peak), 128 GiB RAM, and RTX 5060 Ti 16GB:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\profile-ai-hardware.ps1
# Optionally install OpenVINO and repeat with -CheckOpenVINO to enumerate Intel NPU
```

This read-only profiler creates a **local** hardware report under
`artifacts/hardware/ai-platform.json`. Hardware capability is not runtime
model support. See [Core Ultra 265KF design](docs/CORE-ULTRA-265KF.md) for
a proposed work partition and strict same-quality validation criteria.
