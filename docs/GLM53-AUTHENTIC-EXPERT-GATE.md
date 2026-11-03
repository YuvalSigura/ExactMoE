# GLM-5.3 authentic expert gate — engineering status

**Branch:** `feat/glm53-authentic-fp8-evidence-gate`  
**Pinned publisher revision:** `zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42`  
**Status, 2026-11-02:** implementation + synthetic checks only; **authentic source payload and official kernel parity blocked**. No full-model inference.

## Scope and reuse
This gate adds `expert_acquire.py` and `expert_validation.py` without replacing existing `fp8_reference.py`, `moe_layer.py`, router, pager, CLI or completed tests. It validates one original individually named routed expert: gate, up, down and three corresponding inverse scales. For the pinned configuration, weight shapes are [2048,6144], [2048,6144], [6144,2048]; inverse-scale F32 grids are [16,48], [16,48], [48,16]. Expected payload is **37,748,736 original FP8 bytes plus 9,216 scale bytes = 37,757,952 bytes**. This is a computed expectation, not evidence of acquisition.

The bounded extractor uses immutable revision HTTPS URLs, manifest/config + shard headers, absolute safetensors byte offsets and strict 206 Content-Range verification. It caps each range to 16 MiB, all six weight/scale payloads to 48 MiB (configurable downward), and metadata to fixed limits. It rejects HTTP 200 instead of inadvertently downloading whole shards. A compact, separate safetensors sample preserves **tensor payload bytes exactly**, while its new container header/offsets are different. Its provenance JSON includes original source shard and absolute ranges, per-tensor SHA-256, ETags, shapes, dtypes and pinned revision. Pinned HTTPS provenance is **not equivalent to validating original complete-shard LFS SHA-256**; that remains unverified.

Validation rejects identity/revision mismatch, wrong dtype, malformed scales, malformed shapes, truncated local data, and tampered extracted bytes. The existing bounded FP8 CPU / optional PyTorch CUDA F32 staged matvec is reused. An independent dense F32 oracle uses separately coded exponent/mantissa decoding and SwiGLU. The optional `dynamic-candidate` mode quantizes **temporary activations** per 128-element input block, including nearest-even E4M3FN and maxabs/448 scaling, with a separately implemented exhaustive-encoding oracle. This is a **candidate activation policy** requiring official kernel confirmation, not claimed production equivalence.

Read counters report logical successful source read bytes, not physical NVMe bytes. Timings use a high-resolution clock. Memory output reports 2ms-sampled process RSS, tracemalloc peak and optional PyTorch CUDA allocator peaks; these are not exact whole-machine RAM or WDDM VRAM maxima.

## Commands

```powershell
py -3 -m pip install -e ".[dev]"
py -3 -m pip install psutil
py -3 -m exactmoe.expert_acquire --out X:\GLM53-one-expert --layer 3 --expert 0 --max-source-mib 48
py -3 -m exactmoe.expert_validation X:\GLM53-one-expert --layer 3 --expert 0 --device cpu --mode dequant-f32 --evidence evidence\expert-cpu.json
py -3 -m exactmoe.expert_validation X:\GLM53-one-expert --layer 3 --expert 0 --device cpu --mode dynamic-candidate --evidence evidence\expert-dynamic.json
py -3 -m exactmoe.expert_validation X:\GLM53-one-expert --layer 3 --expert 0 --device cuda --mode dequant-f32 --evidence evidence\expert-cuda.json
py -3 -m pytest -q
```

Acquisition requires publisher support for bounded HTTP 206 responses; no whole-shard fallback is initiated. CUDA command requires a separately verified CUDA-enabled PyTorch installation. The CPU mathematical oracle transiently reconstructs one matrix in F32; budget is bounded by the 48 MiB source cap but process RSS is measured, not artificially claimed to remain under a fixed constant.

## Local synthetic experiment (not authentic model evidence)

Performed 2026-11-02 with injected six-tensor synthetic source and contract-equivalent copies of existing storage/FP8 code, because direct GitHub checkout was blocked in the isolated execution container:

| Metric | Dequant F32 | Dynamic candidate |
| --- | ---: | ---: |
| Source tensor payload | 117 B | 117 B |
| Candidate logical read bytes | 117 B | 117 B |
| Candidate wall time | 0.000457 s | 0.000679 s |
| Sampled process RSS peak | 107,655,168 B | 107,794,432 B |
| Output max absolute error vs independent local oracle | 0 | 0 |
| Synthetic numerical check | PASS | PASS |
| Genuine original checkpoint tested | NO | NO |
| Official GLM kernel parity | NOT TESTED | NOT TESTED |
| Actual CUDA peak VRAM | NOT AVAILABLE | NOT AVAILABLE |

These short synthetic timings and memory peaks **must not** be extrapolated to an original GLM expert. The tests include corrupted math, wrong scales, tensor identity, truncated payloads, strict HTTP range behavior, and cache eviction. All new local synthetic fixtures passed in the isolated contract-equivalent harness; a full native repository regression run is still required.

## Blocking evidence and next acceptance gate

This environment did not have a genuine GLM-5.3 original expert payload or a supported NVIDIA CUDA runtime. No source checkpoint shard was downloaded. The publisher's pinned production FP8 kernel outputs and original hidden-state fixture were not available, so neither official activation policy parity nor genuine-expert execution has passed. An actual RTX 5060 Ti/SM120 VRAM measurement requires that workstation.

To clear the gate: acquire the exact bounded six original tensors, validate their provenance and local SHA-256, run the CPU experiment with reported peak RSS/read bytes/timing, capture trusted original-production per-projection outputs for the same fixed activation, calibrate numerical tolerances, and run CUDA if compatible hardware is present. Only after those checks can a **single authentic expert** be called independently validated.

**Never report a GLM-5.3 generated token, 78-layer decoder, DSA, router end-to-end parity, OpenAI API or agent tool calling as a success from this milestone.**
