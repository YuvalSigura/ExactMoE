# ExactMoE: End-to-End Plan for Full GLM-5.3 on a Consumer PC

**Status:** Research and engineering execution plan, NOT a declaration of successful GLM inference.
**Owner:** ExactMoE project.
**Target checkpoint:** zai-org/GLM-5.3, pinned revision aca966e4e02791568aa6a4ced368624b3d897f42, **native FP8**.
**Local target:** Intel Core Ultra 7 265KF (8P + 12E; AVX2, no AVX-512), RTX 5060 Ti 16 GiB-class VRAM, 128 GiB DDR5 RAM, NVMe SSD.
**Mandatory constraint:** Do not delete/merge/retrain experts, change router/top-k, substitute another model, or further quantize original checkpoint weights.
**Architecture options:** see [All candidate execution paths](GLM53-ALL-ARCHITECTURES.md), including reuse-first AVX2 FP8 KTransformers, InferSSD and a custom paging backend. Correctness and end-to-end validation decide the winner.
**Explicit non-promise:** Feasibility and real-world performance on this specific PC have NOT been demonstrated. Native FP8 checkpoint semantics and numerical reference agreement are measurable goals; bitwise GPU-independent equality is not guaranteed.

## Definition of DONE — two different success bars

**D1: Correct first answer (mandatory).** From a clean local installation, run the exact pinned complete GLM-5.3 checkpoint and issue a local prompt, produce at least 32 autoregressive tokens without a remote inference API, and terminate normally. The entire main decoder must execute; not a small replacement, sampled experts only, or fabricated completion. Keep a per-layer trace and show it respects model routing, original FP8 weights and scales. A single-token smoke test precedes this.

**D2: Repeatable usable system (release).** Provide a local OpenAI-compatible HTTP endpoint, tokenizer/chat template, streaming text, correct stop tokens, bounded RAM/VRAM/disk usage, recovery from interrupted requests, documented install/start/stop instructions, and a reproducible correctness/performance report. Usability speed is evaluated from measurements, not pre-promised.

**Quality contract:** Three levels of comparison are explicitly distinct:
- **Checkpoint identity:** all original source bytes/FP8 scales are preserved, pinned and verified where accessed.
- **Semantic fidelity:** original dense layers, MoE routing (including correction bias and shared experts), DSA attention/indexer, normalization, embeddings, and sampling code paths execute.
- **Numerical fidelity:** operator and final-logit parity with an independent reference within calibrated dtype/kernel tolerances; top-k expert IDs match. Greedy completions should match on a fixed regression set when numerical margins allow. Random sampling is distributional, not bit-for-bit across backends.

## Evidence already collected

User's pinned official model JSON profile (2026-11-01, recorded privately; do not publish local paths or raw hardware reports):
- 78 decoder layers: 3 dense and 75 sparse MoE; 256 routed experts per sparse layer; top-8.
- 19,200 named main-layer expert instances; 57,600 expert matrix weights and 57,600 FP8 inverse-scale tensors; no missing names in that metadata map.
- Additional MTP layer with 256 expert entries, which is OPTIONAL for baseline greedy inference; preserve MTP data, do not drop it from the source checkpoint.
- Official manifest reports 141 safetensors shards, 118,629 tensor entries and **755,617,140,416 bytes = 703.72 GiB** model tensor data.
- Local GPU detected as NVIDIA GeForce RTX 5060 Ti 16,311 MiB; current local C: free space reported ~337.6 GiB, insufficient for the full unmodified checkpoint.
- Metadata completeness does **not** establish actual file headers, tensor byte shapes or valid numerical execution.

Approximate physical I/O per **single generated token** for cache-cold main MoE experts:
- One expert's 3 FP8 matrices at hidden=6144 and MoE intermediate=2048: 3 x 6144 x 2048 = **37,748,736 weight bytes (~36 MiB)**; scale overhead is additional.
- Eight routed experts x 75 main sparse layers = 600 expert activations per token; potentially ~22.65 GB of expert weight reads per cold token, *before* dense/attention/embedding weights and scales.
- An ideal 10 GB/s sequential storage interface alone has a theoretical 2.265 second/token transfer floor for those cold expert bytes. Real scattered I/O, PCIe transfers, compute, operating-system caches and prefill often make it much slower; cache hits may reduce SSD bytes. **Do not advertise an expected token/s figure** before benchmarking.
- Other (non-main-expert) checkpoint tensors are too large to assume always resident in 16 GiB VRAM: page dense layers, DSA projections, shared experts and the final head as well.
- Plan dedicated storage with **at least 850 GiB free; preferably >=1 TiB free** for source checkpoint, indexes, temporary working data and logs. Re-check capacity immediately before any full download.

## System architecture

NVMe immutable checkpoint shards -> validated tensor range index -> async bounded read queue -> RAM byte cache -> bounded CPU staging buffers -> GPU tensors / mathematically faithful kernels -> original model layers -> KV/DSA state -> output head -> token sampler -> local API.

A speculative predictor (CPU/NPU) MAY hint which pages to fetch, but must NEVER select experts in place of the official model router. No predicting or skipping layer computation for correctness. NPU optimization is strictly optional and comes **after** D1/D2 correctness.

### Explicit memory budget mechanism (initial, tunable)

- RAM: start with a **64 GiB maximum** weight byte cache (not a target to fill), plus separately bounded I/O staging. Leave ample free RAM for Windows/WSL, page tables, pinned buffers, runtime, and other applications. Alert on paging to the Windows system pagefile.
- GPU: start with a **conservative <=8 GiB weight/staging budget**, leaving the rest for kernels, attention/KV state and WDDM/display headroom. Actual peak GPU memory must be sampled via CUDA / nvidia-smi, not computed from cached tensor bytes alone.
- Never assume whole model, all dense weights, all 256 experts of a layer, or 1M-token KV state fit into host RAM or VRAM.
- Every read/upload must be bounded and use backpressure; return explicit failure if limits cannot be met. Eviction must not invalidate in-flight CUDA work. No silent reduction of context or model precision.

## Dependency-ordered execution gates

### P0 — Lock architecture, repository baseline and test matrix
**Implement:** Pin GLM/HF model revision and upstream Transformers/vLLM formula commit(s), CUDA/PyTorch versions, Python environment, Windows/NVIDIA driver, license terms. Record config and exact tensor manifest hashes. Separate reference implementation from candidate backend. Maintain Linux+Windows CI and security baseline.

**Exit:** immutable model revision documented; reproducible dependency manifest; CI green; status accurately says which operations are implemented. Current Python toy paging tests already exist, but this does not pass later GLM gates.

### P1 — Hardware and platform feasibility on the actual PC
**Implement:** Capture 265KF feature flags, physical P/E topology, RAM configured MT/s, usable pinned memory, PCIe topology, SSD read bandwidth under random/small-range and sequential loads, GPU SM120 feature support and native FP8 kernel compatibility. Compare native Windows vs WSL2 Linux for CUDA/toolchain reliability. Intel NPU enumeration is optional. No overclocking or driver modification by scripts.

**Exit:** real hardware JSON, bandwidth and latency report, conservative live-memory budgets; chosen development OS and verified CUDA build can launch a tiny kernel on GPU. KTransformers' recent releases document AVX2-only native FP8 MoE kernels; test this backend on the 265KF before writing new FP8 CPU kernels, but do not mistake tested GLM-5 or GLM-5.3-Flash support for full GLM-5.3 compatibility.

### P2 — Complete native checkpoint layout, byte integrity and disk store
**Implement:** Read safetensors headers for representative, then all 141 pinned shards (remote byte-range metadata wherever supported, no full-weight download yet). Cross-check every tensor shape, dtype, byte range, shard mapping and FP8 inverse-scale shape, including shared experts, dense weights, DSA indexer and MTP tensors. Respect Hugging Face signed/redirected download URLs; fail closed if Range is ignored. Add per-tensor metadata store and pinned revision validation. Make ranged reads use strict byte caps, expected file identity and hashes (where available).

**Exit:** machine-readable table mapping all tensor names -> exact ranges, dtypes, shapes and scale companions; no unclassified tensors for baseline decoder. A complete manifest of *names alone* is insufficient.

### P3 — Real original expert byte reading on local SSD
**Implement:** Acquire a small bounded sample of authentic weight payload from one specific expert (gate/up/down and all scales), ideally by HTTP Range with explicit byte cap; if this is unsupported, obtain one required whole shard with explicit capacity check and download consent. Store original bytes locally; test exact hashes and zero-copy/streaming reads. Never persist a dequantized replacement as the canonical checkpoint.

**Exit:** one real original expert and all scales can be reconstructed from untouched bytes, with shape/dtype validation. No numerical claim yet.

### P4 — Faithful block-FP8 expert math (CPU reference first)
**Implement:** Implement E4M3 decode and 128x128 inverse-scale reconstruction, using dynamic activation scaling and matmul accumulation as specified by the official checkpoint/runtime. Implement gate/up/down, SwiGLU and shared expert FFN. Reference route can be slow CPU F32 math using original FP8 byte values; any conversion is transient compute, not a new checkpoint. Build NVIDIA SM120 implementation using CUDA/Triton/cuBLASLt as supported, with slow-but-correct fallback when required.

**Exit:** isolated real expert forward pass; compare intermediate activations and outputs to an independent official FP8 reference and publish tolerance, max/relative error, dtype policy, RAM/VRAM and execution time. Failing kernels cannot be used in end-to-end decode.

### P5 — Router and ONE complete native GLM sparse MoE layer
**Implement:** Original sigmoid scoring, float32 router behavior, correction bias, grouped selection (where configured), 8-of-256 top-k, normalization, routed scale factor, shared expert, and correct residual/norm handling from pinned GLM implementation. Route IDs only from exact original logits. Page required experts and scales according to output of that router. Include dense-first-layer case.

**Exit:** multiple fixed hidden-state inputs yield the same chosen expert IDs and numerically close full sparse-layer outputs as reference, even with 0% RAM cache hits and forced evictions.

### P6 — Attention/DSA and complete decoder component coverage
**Implement:** Match GLM5's actual MLA/DSA sparse indexer, layer-specific indexer reuse/skip pattern, RoPE, pre-attention norms, attention masks, KV state and layer ordering. Port reference semantics from pinned Transformers/vLLM before replacing kernels. Verify early dense layers, each sparse layer, embeddings and final norm/logit head. Treat MTP as *disabled for initial generation*; keep all checkpoint data unchanged for future enablement.

**Exit:** every model operation for a single-token forward is accounted for and individually testable. Independent per-layer traces pass tolerated parity; no shortcut to a generic attention implementation.

### P7 — Memory-bounded 78-layer end-to-end executor
**Implement:** Combine storage, original router, FP8 compute, dense/attention paging, KV state, and final head with deterministic scheduling. Never load entire model to RAM. Support per-layer resource caps, CUDA event lifetime, bounded pinned transfer queues, cancellation and crash-safe checkpoints of the disk index.

**Exit:** first real full-model forward pass produces finite logits for an actual token on RTX 5060 Ti + Core Ultra 265KF + 128 GiB RAM, staying within verified resource budgets.

### P8 — Tokenizer, prefill and first local completion
**Implement:** Official pinned tokenizer, chat template, thinking/tool format and special stop tokens; greedy decoding; cache updates and DSA context correctness. Start with one short system/user prompt, context 128–256 tokens, max 1 generated token; then 4, 16, 32, 128. Add prompt prefill by exact layer-wise execution. Expand to 2,048 and later 8,192 context only after small tests pass.

**Exit (D1):** full 32+ token locally generated answer (not hosted API), with source checkpoint pinned, correct token boundaries, no OOM, measured wall-time, and a complete generation trace.

### P9 — Prove original-model equivalence and quality
**Implement:** Independently obtain official reference outputs from the same pinned checkpoint (trusted vendor implementation on capable local/leased hardware, with costs explicitly approved if applicable). Save tensor-level and final-logit fixtures, compare exact router choices and logit numerical errors. Run fixed greedy prompts plus standard coding and authorized defensive cyber evaluations; report divergence. Validate stop/restart, disk-cache eviction, cross-OS behavior and randomized prompts. Correctness gates must use reference, not just two functions sharing the same bug.

**Exit:** explicit reproducible numerical tolerance table and benchmark reports; no evidence of a systematic quality regression. If reference is unavailable, label results *unverified equivalence*, not *same quality*.

### P10 — Make it less slow without changing the model
**Implement:** measure time spent in SSD read, RAM cache, PCIe transfer, CUDA compute, DSA attention, prefill and sampling. Optimize scheduling, page grouping, reuse and prefetch; use CPU P/E topology hints only after actual measurement, not assumed logical core IDs. NPU-aided prefetch is optional: predictions can alter page-loading order only, never router output. Consider speculative decoding only with a mathematically valid acceptance algorithm whose output distribution matches target-model sampling.

**Exit:** measured end-to-end latency improvement over P8 baseline, while P9 parity suite remains unchanged and green. Publish realistic cold/warm-cache tok/s and time-to-first-token with context lengths and hardware details.

### P11 — Local product/UX and release validation
**Implement:** CLI start/stop, localhost-only HTTP API (OpenAI compatible), streaming tokens, basic chat UI, structured logs, hardware metrics, configuration validation, safe update/rollback and concise installation instructions. Protect endpoint exposure; do not expose model server publicly by default. Provide automated 1-token smoke and 32-token acceptance scripts.

**Exit (D2):** on a clean user machine, repeat the exact startup procedure, connect a local client, generate the complete answer, verify no hidden cloud inference, save benchmark/parity report and document all limitations.

## Implementation sequence and review strategy

- **Critical path:** P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P11.
- **Parallel preparation:** API shell, local telemetry and CI fixtures can be built before P8; avoid reporting them as inference progress.
- **Optimization path:** P10 follows correctness. Do not train an NPU predictor before the original router runs faithfully.
- **Small model fixtures:** permitted ONLY to test infrastructure and prevent regressions. Never substitute a small model as evidence for the D1 success bar.
- **Go/no-go between gates:** each gate needs an automated test, a hardware run (where applicable), saved evidence and a reviewed PR. No silent bypass.
- **Reference priority:** GLM5 source/Transformers architecture and vLLM operator behavior; reuse reference formulas with appropriate license compliance instead of guessing DSA kernels.

## Alternative paths and selection criteria

Do not assume a custom PyTorch/Triton runtime is the best initial architecture. First investigate whether current KTransformers AVX2 FP8 kernels plus SGLang's GLM-5.3 DSA code can be used with native bytes and disk-backed experts; in parallel audit InferSSD's SSD cache/range reader and GGUF constraints. Use original FP8 checkpoint only for the final acceptance. An AVX2 compatibility mode is not evidence that full GLM-5.3 weights can fit 128GB RAM. See [all pathways](GLM53-ALL-ARCHITECTURES.md) for decision matrix, kill criteria and optional lossless disk repacking, reversible compression, GPU tiling, KV spill, predictive prefetch and exact speculative decoding.

## Failure responses (no false completion)

| Problem | Required engineering response |
|---|---|
| Official pinned shard does not support Range | Use official download tool for ONE shard, after user approves storage/network expense; validate exact bytes |
| RTX 5060 Ti lacks a compatible native FP8 blockwise backend | Implement slower verified FP32-on-GPU path using source FP8 data/scales, then port accelerated kernel; no hidden re-quantization |
| Non-MoE layers exceed 16 GiB VRAM | Page non-MoE tensors by layer and chunk the largest matmuls; do not load all attentions/embeddings permanently |
| Host pagefile thrashing | Shrink real RAM cache and staging queue; measure hard faults, not just Python process RSS |
| Slow random SSD I/O | Batch reads by shard, prefetch safely, experiment with cache policy; always verify parity |
| DSA/reference disagreement | Stop, pin upstream operator and config versions, inspect layer traces; do not approximate sparse attention |
| Time to first token is enormous | Report measured limitation; optimize prefill and I/O, never imply verified interactive speed |
| Numerical mismatch / router expert divergence | Halt P8 acceptance until fixed or justified by reference tolerance policy |
| Model file storage < required free space | Do not download complete checkpoint; finish header and one-expert gates first |
| NPU not detected | Leave NPU optimization disabled; CPU+GPU execution must still function |

## Proposed eventual command — NOT YET IMPLEMENTED

    exactmoe serve --model-dir D:/models/GLM-5.3 --device cuda --ctx-size 256 --ram-cache-gib 64 --gpu-budget-gib 8 --host 127.0.0.1 --port 8000

Illustrative local acceptance request after the server exists:

    curl.exe http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{ ... official GLM chat prompt ... }"

The command-line server does not exist at present; this syntax is an API DESIGN TARGET, not instructions to run now.

## Immediate next three concrete PRs

1. **P1 hardware feasibility bench and available-runtime probe:** add read-only effective RAM speed, optional SM120 CUDA kernel smoke, SSD random-read benchmark on a bounded synthetic file with cleanup, and report generator; run on user's PC.
2. **P2 layout audit:** bulk inspect pinned shard headers (metadata only), cross-validate scale shape, tensor byte offsets, official manifest and model-level tensor coverage; handle HTTP Range failures cleanly. Do not store private user report data in Git.
3. **P3+P4 minimum genuine expert:** strict capped acquisition of a real expert's unchanged FP8 bytes and scale tensors; independent F32 reference op, then compatible GPU matmul, validated by separate oracle fixtures.

**Storage purchase/download milestone:** only after P3/P4 validate real expert computation and P1 demonstrates a feasible RAM/VRAM paging strategy. Before P7, acquire the full ~703.7 GiB checkpoint with >=850 GiB available free; prefer >=1 TiB and preserve pinned hashes.

## References / dependency evidence

- Model/checkpoint: https://huggingface.co/zai-org/GLM-5.3
- Publisher source: https://github.com/zai-org/glm-5
- Reference architecture: https://huggingface.co/docs/transformers/en/model_doc/glm_moe_dsa
- vLLM production reference: https://recipes.vllm.ai/zai-org/GLM-5.3
- NVIDIA GPU capability: https://developer.nvidia.com/cuda/gpus
- Intel CPU/NPU specifications: https://www.intel.com/content/www/us/en/products/sku/241062/intel-core-ultra-7-processor-265kf-30m-cache-up-to-5-50-ghz/specifications.html
- Format: https://github.com/huggingface/safetensors
- KTransformers CPU FP8 limitation: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md

Do not compare model parameter counts inferred from the FP8 shard byte size: HF model card may count 753B parameters while upstream calls the family 744B-A40B. The exact pinned, verified on-disk tensor byte count is the capacity planning authority.
