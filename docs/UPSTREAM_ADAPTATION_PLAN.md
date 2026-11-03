# ExactMoE v0.2 — upstream-first GLM-5.3 full runtime adaptation plan

**Status (2026-11-03): IMPLEMENTATION IN PROGRESS; NOT FULL-MODEL INFERENCE.** This plan was written after reading actual source/documentation from three independent projects. Upstream commit pins, exact source paths, and adoption decisions are maintained in [UPSTREAM_SOURCE_MANIFEST.json](UPSTREAM_SOURCE_MANIFEST.json). A path listed as a **candidate** does not imply it builds, fits, or computes the full publisher GLM-5.3. No upstream source code has been vendored into this branch. Attribution/legal and numerical gates are mandatory before copying any source. **Do not merge this branch until reviewed.**

## 0. Product definition, constraints and milestone meanings

**Ultimate product:** the full original `zai-org/GLM-5.3` checkpoint at pinned revision `aca966e4e02791568aa6a4ced368624b3d897f42`, executed locally for a loopback-only, tool-capable HOMELAB security research API using the operator's Core Ultra 7 265KF / 128GB RAM / NVIDIA RTX 5060 Ti 16GB / NVMe.

**Invariant reference lane:** preserve publisher FP8 weight bytes and scales, expert set and top-k, *all* decoder/attention operations and tokenizer semantics. Matching bytes != bitwise numerical parity across different kernels. Exact model identity, output distributions and reference numeric tolerances must be measured separately. No silently truncated experts, switched architectures or compressed substitute advertised as "original GLM-5.3."

**Optional performance lane:** quantized, repacked or changed-routing experiments may be done as explicit **different model variants**, with frozen quality/behavior benchmarks including security domain, tool calling and long contexts, compared against the full reference lane. **Do not weaken the original-FP8 acceptance gate or silently promote a variant.**

**Practical intermediate product:** separately offer GLM-5.3-Flash via an external supported runtime **only with a conspicuous distinct model ID and measured capability**. The 45-layer, 288-expert Flash is NOT the 78-layer, 256-expert full GLM-5.3. No model download/installer run is authorized merely by this document. No remote/cloud model provider is an acceptable silent fallback for the final local product.

The full checkpoint is approximately 703.7 GiB of original FP8 checkpoint data, much larger than 128 GiB RAM; a viable runtime must support genuinely **SSD-resident expert data** and a bounded live working set. Merely offloading a model whose experts *all remain resident in RAM* is insufficient on this machine.

## 1. Three distinct upstreams: what to use, not a combined wholesale fork

| Source (pin) | Verified source modules to study | Transferable function | Non-transferable / changes required |
| --- | --- | --- | --- |
| **Strata** `Niko1221/Strata@ffa80d73133ea5c548aefe57889a16e8085a938f` (MIT) | `include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`, `include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp`, `src/program/generate.cpp`, `src/split/src/moe.cuh`, `src/split/src/engine.cu`, `serve/server.py` | GPU hot-expert cache, host/GPU division, async scheduling, token verification/MTP mechanisms and API interface *concepts* | Current supported model packs and quantized/packed kernels cannot be treated as native original GLM-5.3 FP8. All-cards split optimized/tested for specific Linux hardware and model families, not our single RTX 5060 Ti Windows machine. Its RAM-resident expert model does not alone solve ~704GiB >128GiB. |
| **Project Maya** `mw00/project-maya@10e69a0eb11541004abb24f9fa47147e9002326f` (MIT) | `src/core/glm_model.cu`, `glm_window.cu`, `glm_prefill.cu`, `glm_rows.cu`, `glm_mtp.cu`, `src/core/expert_source.cpp`, `src/core/expert_cache.cpp`, `serve/server.py`, `docs/GLM5-FLASH.md` | GLM-family adaptation patterns, dtype-aware expert serving, GPU graph, active-expert scheduling, tokenizers and tool-capable server; possible foundation for a **separate Flash control** | Maya targets **GLM-5.3-Flash** (45 layers, KDA + sparse MLA, 288 experts, GGUF/Maya quants). Full model uses different graph/operators, 78 layers, 256 experts, original block-FP8 safetensors. Cannot just change config dimensions. Maya has diverged from Strata; replace neither repository's shared `generate.cpp` / `server.py` wholesale. |
| **kimi-k3-in-c** `FareedKhan-dev/kimi-k3-in-c@81bb6c23ccee2154bc35cad922588ab66494a9a9` (Apache-2.0, preserve NOTICE) | `src/io/k3_st.c`, `src/io/k3_load.c`, `src/io/k3_trunk.c`, `src/cache/k3_cache.c`, `tools/sim_cache.py`, `docs/ARCHITECTURE.md`, `docs/data/expert-cache-capacity.txt` | **SSD-first original checkpoint reader**, coalesced range reads per expert, capped trunk residency/ring buffer, explicit cache cost/trace methodology, fail-closed metadata parsing and weightless parity gates | Kimi uses 93 layers, 896 experts/top-16 and native **MXFP4** experts with KDA/Gated MLA, *not* GLM-5.3 blockwise FP8 E4M3 and GLM DSA. Its 8.24GB RSS / ~26.5s per token measurements are author-published for Kimi and cannot predict GLM's speed. Do NOT transplant `k3_ops.c` as a GLM kernel. |

Additional **official/reference donors** (must independently pin and audit source before reuse): GLM-5.3 publisher reference, Hugging Face `GlmMoeDsa`, SGLang/vLLM GLM-5.3 implementations and KTransformers AVX2 FP8 expert kernels. All require version, ABI, model-config and license checks. Reference implementations can be too memory-hungry to run end-to-end on the target machine; use selective oracle comparisons.

### License / provenance

* Strata and Maya repositories state **MIT**. Retain the complete upstream MIT copyright notices in any vendored substantial source, with paths and source commits recorded in `third_party/NOTICE.md`. Check nested third-party files individually, not merely repository root LICENSE.
* Kimi C repository states **Apache-2.0** and contains a `NOTICE` file. If copying a Kimi file, preserve LICENSE, NOTICE and the attribution and modified-file notices required under Apache-2.0. Check `third_party/` separately.
* **No weights** are imported or redistributed; model licenses, GGUF rights and third-party drafters require independent review. In particular do not assume that DFlash2 weights have the same MIT license as engine code.
* This branch currently implements **original Python simulation code**, **not** a copied/vendorized C/CUDA port. Explicit code adoption must go through a smaller, separately testable PR.

## 2. Target architecture: modules, custody, and memory ownership

```text
HOMELAB / local consumers
       |
  loopback-only OpenAI-compatible ExactMoE server [NOT IMPLEMENTED]
       | request identity, tool-call protocol, tokenizer, deterministic error states
  GLM-5.3 FULL model adapter [NOT IMPLEMENTED]
       | tokenizer / 78-layer decoder / DSA + MLA / router and MoE / final norm + head
       | independent per-operator reference parity
  Expert execution dispatcher [FUTURE]
       +------ native original FP8 CPU AVX2 expert kernels [FUTURE]
       +------ bounded CUDA staging and FP8 expert GPU kernels [FUTURE]
       | select CPU vs GPU on measured cost, not heuristic claims
  SSD-first weight system
       +------ original safetensors + manifest and SHA provenance [PARTIALLY WORKING]
       +------ immutable byte-range source + coalesced expert reads [PARTIAL]
       +------ bounded byte RAM LRU/prefetch [WORKING PROTOTYPE]
       +------ model-aware residency / pinned dense trunk / spill [FUTURE]
       +------ GPU VRAM expert cache [PARTIAL byte staging, no full kernel]
       +------ trace simulator for policy choices [NEW; CPU ONLY]
       |
 NVMe source checkpoint (~703.7 GiB) / 128GiB RAM / RTX 5060Ti 16GiB VRAM
```

**Memory budget:** reserve Windows, drivers, I/O buffers, attention/DSA state, pinned host allocations, output head, CUDA buffers and minimum KV. Explicit accounting for every allocation is mandatory. Keep paging *online* and bounded. Source tensors are immutable; no byte-offset guesswork, symlink/path escape or unchecked network response.

**Physical I/O:** captured source reads, OS file cache hits, disk throughput and latency, PCIe upload bytes, CUDA execution and prefetch savings must be instrumented separately. A Python `read()` byte counter is NOT physical NVMe traffic. Benchmark SSD cold/warm with controlled filesystem cache when feasible.

### Accuracy gate: what "Exact" means

1. Check original source tensor IDs, FP8 dtype, exact inverse-scale mapping, 128×128 block shape and verifiable hashes.
2. Independently compute one gate/up/down FP8 expert on *identical reference activations* with separate implementation.
3. Extend to exact router IDs, unbiased route aggregation and shared experts; compare each intermediate.
4. Match one full 78-layer decode pass including real DSA/attention and token IDs, using native semantics and calibrated tolerances.
5. Require bounded peak memory and report actual throughput; avoid claiming bit-exactness if CPU/GPU floating-point reduction order differs.
6. Then validate 32-token streaming answer and official tool-call protocol.

An isolated expert or synthetic FP8 test is a checkpoint toward the above, not a release.

## 3. Ordered plan — use what works first

| Gate | Work | Concrete admission criteria | What is not proven by gate |
| --- | --- | --- | --- |
| **U0 (NOW)** | Pin upstream commits, license and source-module map. Add CPU-only trace simulator with tests. | Source SHA manifest complete; CLI runs without models; CI supports Windows/Linux; outputs marked SIMULATION_ONLY. | No model speed, GLM architecture or RAM advantage. |
| **U1** | Add local **read-only** independent source probes for Strata/Maya/Kimi layout; compare bounded SSD-first loader semantics and failure behavior to `TensorIndex`/`ExpertPager`. | Source/payload byte hashes and negative tests; exact metadata; no full download. | No CUDA inference. |
| **U2** | Evaluate native kernel donor fit and API dependency graph on pinned commits; build a standalone *C++/CUDA source spike* in a separate directory, not wholesale vendor. | Can compile on Windows or justified WSL2 with sm_120, test CUDA available on user's RTX, no license ambiguity, no overwritten upstream sources. | Compilation alone is not GLM output. |
| **U3** | Complete PR #18's actual six-tensor original expert acquisition, independently verified FP8 + activation scaling, same-input oracle parity. | Real publisher expert ranges and auditable numerical report from operator workstation. | No full layer or token. |
| **U4** | Compose one full original GLM-5.3 MoE FFN, using pinned router/shared expert; CPU native and GPU optional. | Exact expert IDs + verified route weighted sum + baseline timing/IO under capped memory. | No DSA/decoder. |
| **U5** | Implement *full* DSA/MLA, tokenizer, KV and 78-layer decode, borrowing tested kernels where architecture matches. | Independent numerical oracle on all relevant intermediate operators, 1-token true full model output, bounded memory, original weights. | A usable speed is not guaranteed. |
| **U6** | Measured optimizations: cache admission policies, SSD read coalescing, trunk streaming, CPU/GPU overlap, bounded prefetch, optional exact MTP/speculation. | Improvement in actual **physical bytes/token** and/or wall-clock on frozen real traces, no router/quality change. | Synthetic traces don't qualify. |
| **U7** | Loopback server + streaming + tool calls for HOMELAB; monitor; resilience and privacy. | 32-token true original answer; end-to-end tool-call tests; proper local model identity; restart and cancellation tests. | No scientific claim about cybersecurity competence until separate eval. |
| **U8** | Optional Flash product control and explicit lossy quantization experiments. | Separate model IDs, no silent fallback, reference benchmarks and behavior regression guardrails. | Not original FP8 GLM-5.3. |

**STOP conditions:** no reliable source reference/authoritative config; mismatch in source offsets/inverse scales/router IDs; failing independent parity; memory exceed; hidden provider/model fallback; unlicensed dependency; code claims supported but only mock-data tests; or physical I/O lower bound that makes selected performance target impossible.

## 4. Kimi's strongest lesson: policy economics, not a miracle RAM threshold

The Kimi README reports 2.78T parameters, ~1.56TB checkpoint, ~8.24GB peak RSS and ~26.5s/token for a measured low-RAM run. These are **author-reported**, on different model, MXFP4 native format and hardware; not speed estimates for GLM.

Their expert-cache replay `docs/data/expert-cache-capacity.txt` records 100,096 expert requests on ~68 token positions and reports 36.24% LRU hit at 8–64GB, 49.19% at 128GB. Its own caveat says this trace **re-prefills the whole prefix each step**, inflating reuse; do not extrapolate its hit rates to normal incremental autoregressive decode. The same file compares offline Belady with **future knowledge** (an oracle, impossible as a production online policy) and a PIN+LRU experimental upper bound.

The README phrase "128GB+ keeps the entire model in RAM" must NOT be repeated literally: the architecture table says ~1.45TB routed expert weights stay on disk, and only the ~109GB dense trunk plus smaller parts can fit when RAM is sufficiently large. This correction matters to ExactMoE's 128GB machine budget.

**Consequence for ExactMoE:** get actual original-router expert traces, measure miss demand and revisit cache policy *before* trying to copy or split experts. A small bounded working set improves *memory fit*, not necessarily physical bytes per token; at high unique-expert churn, unavoidable misses dominate. Consider cost of FIFO/LRU/LFU, speculative prefetch waste and SSD throughput separately.

## 5. First working artifact on this branch: Expert Traffic Simulator

`exactmoe/traffic_sim.py` supplies **source-independent CPU-only** replay of original-router `(layer, expert, weight_bytes)` demand. Choices: no cache, byte-capped LRU and history-only LFU. No future IDs, Belady lookahead, model output or changed router decisions. Results explicitly report `SIMULATION_ONLY`, logical source bytes (not physical SSD), and an unverified source-kind declaration. The companion tests ensure bounded resident memory and fail-closed trace validation.

Example:

```powershell
python -m exactmoe.traffic_sim --trace docs/experiments/traffic_synthetic.json --cache-mib 512 --policy lru
python -m exactmoe.traffic_sim --trace docs/experiments/traffic_synthetic.json --cache-mib 512 --policy lfu
python -m pytest -q tests/test_traffic_sim.py
```

The included fixture is artificial and cannot prove GLM routing locality. To migrate to real captures in a separate PR, record per-token, per-layer routed expert IDs and exact tensor byte sizes from the **unchanged official router**. Include checkpoint model/revision SHA, process configuration, token IDs, request workload and lineage, with a signed evidence digest. Reject partial/unknown provenance and do not expose prompts, secrets or model weights.

## 6. What not to do

* Do not vendor the whole Strata repository (thousands of files, multiple architectures/backends) or wholesale-merge Maya's divergent `generate.cpp` or `serve/server.py`. Select coherent components, copy only after license audit, and compile/verify in isolation.
* Do not transplant Kimi's native MXFP4 expert matmul or claim it natively computes GLM FP8 E4M3. Only transplant SSD-first data-flow and verifiable I/O *ideas* unless an explicit compatible kernel port is proven.
* Do not label GLM-5.3-Flash, Kimi K3, Qwen3.8, or a quantized/pruned GLM "full original GLM-5.3".
* Do not equate CI PASS with hardware acceptance, numerical parity, 8GB RSS, 20+ tok/s, or actual source SSD throughput.
* Do not assume a different CPU/GPU multi-card benchmark generalizes to one RTX 5060 Ti 16GB.
* Do not run auto-downloading vendor installers or copy model weights into git.
* Do not change CEM, RDL, or the HOMELAB authority state.

## 7. Handoff for the next implementing engineer

1. Start from feature head `feat/strata-maya-runtime-adaptation-20261103`; inspect its relationship to draft PR #18. It is based on **PR #18's head**, not an unreviewed merge to `main`. Preserve PR #18 provenance and never claim upstream tests covered it.
2. Verify the three pinned upstream commits and their present licenses, inspect exact donor paths, and record any updated/renamed modules.
3. Execute the new simulator and all regression tests in Windows/Linux CI. Fix detected failures before touching native engine code.
4. Run bounded **authentic FP8** expert acquisition from PR #18 on authorized machine without a full checkpoint download, confirm independent oracle before code port.
5. Implement **one** isolated SSD-first native expert read/coalescing and instrument physical bytes + memory, with tests; carry its provenance and licensing in a separate small review PR. Avoid copy/pasting thousands of lines.
6. Do not make full-inference claims until all `U5`–`U7` gates pass. Report exact SHA, fixtures, CI links, true hardware runs, bad news and next blocker.

Original full model reference: https://huggingface.co/zai-org/GLM-5.3
Upstream engines: https://github.com/Niko1221/Strata ; https://github.com/mw00/project-maya ; https://github.com/FareedKhan-dev/kimi-k3-in-c
