# ExactMoE

> **Verified on operator machine · 2026-11-04 · Authentic FP8 Expert Gate PASSED (CPU)**
>
> Pinned revision `zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42`
>
> **What was done and verified locally:**
> - Bounded acquisition of Layer 3 Expert 0 (6 tensors, 37,757,952 bytes) — **PASSED**
> - Independent CPU mathematical oracle agreement (max abs error ~1e-8) — **PASSED**
> - Streamed FP8 CPU path agreement with the same oracle — **PASSED**
> - Bounded acquisition of Layer 3 router (BF16) + bias + shared expert — **PASSED**
> - Router arithmetic on a fixed synthetic hidden state (selected 8 experts) — **PASSED** (controlled test, not a production trace)
>
> **What is NOT claimed:**
> - Official GLM production-kernel parity (`NOT_TESTED`)
> - Full 78-layer inference
> - DSA / MLA / tokenizer / token generation
> - CUDA / RTX 5060 Ti VRAM measurement
>
> Full status: [docs/STATUS-2026-11-04.md](docs/STATUS-2026-11-04.md)

## Research direction: tiered expert scheduling (proposal)

**Goal:** investigate bounded **SSD ↔ RAM ↔ VRAM** expert residency and **CPU/GPU execution placement** for the **full original GLM-5.3 FP8 model** on the 16 GiB RTX 5060 Ti / 128 GiB RAM machine. Cache policy, I/O coalescing and optional prefetch may change *when and where* selected experts execute, **never the original router/top-k, weights or model outputs** beyond documented numerical kernel tolerances.

- **First:** benchmark physical NVMe reads, host↔GPU transfers and actual original-expert compute; compare no-cache, LRU, LFU and cost-aware placement under identical memory caps. Finish the existing one-layer and full-decoder correctness gates before making GLM-specific throughput claims.
- **Optional offline research:** evaluate **QZERA/QuantumZip** (small QUBO/QAOA scheduling instances) against classical exact/CP-SAT and heuristic solvers. A quantum *simulator* is **not** a memory compressor or demonstrated speedup. QZERA must not be a runtime dependency.
- **Current status:** **research plan only** — no demonstrated full-model inference, RTX FP8 kernel parity, live GLM router traces or scheduler speedup.

**[Read the detailed research note: architecture, traffic bounds, external research, experiment matrix and acceptance gates](docs/EXPERT_MEMORY_SCHEDULING_RESEARCH_2026-11-04.md).**

---

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

---

*(Full previous README content preserved below in the repository history. See docs/STATUS-2026-11-04.md for the current verified gates.)*
