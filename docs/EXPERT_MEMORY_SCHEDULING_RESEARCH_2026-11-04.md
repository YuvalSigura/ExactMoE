# ExactMoE — Tiered Expert Memory & Scheduling Research

**Research note · 2026-11-04 · PROPOSED / NOT IMPLEMENTED / NOT A FULL-MODEL INFERENCE CLAIM**

**Target:** the **full, unchanged** publisher FP8 `zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42` on the operator workstation (Intel Core Ultra 7 265KF, 128 GiB-class system RAM, RTX 5060 Ti 16 GiB, NVMe). This note extends the [upstream-first plan](UPSTREAM_ADAPTATION_PLAN.md); it does **not** replace the existing correctness gates or create a separate competing runtime.

## 1. Research question and invariants

**Question:** Can a measured, bounded three-tier SSD / RAM / VRAM system, coupled with CPU/GPU execution placement, reduce physical bytes fetched and end-to-end token latency for a frontier MoE model that cannot fit in host memory?

- **Reference lane:** preserve original checkpoint bytes and FP8 scales, all routed and shared experts, the official router/top-k, DSA/MLA, decoder and tokenizer. Cache admission, eviction, overlap and placement may change **where** and **when** exact computation executes, not **which** experts execute. Numerical comparison uses stated tolerances, not an unsupported promise of bitwise parity across implementations.
- **Variant lane:** bitplane storage, additional quantization, expert pruning or changed routing are separately identified model variants, each with accuracy and behavior benchmarks; never silently substitute for the reference.
- Explicit bounded memory accounting: weights, host staging, pinned buffers, OS/process overhead, CUDA allocations, attention/KV/DSA buffers and OS filesystem cache. Fail closed on source errors, insufficient capacity or unknown version.

## 2. Current evidence versus gaps

Read [2026-11-04 status](STATUS-2026-11-04.md) before using this plan. On the operator machine, bounded acquisition of **one original Layer 3 expert** (six tensors, **37,757,952 bytes**), independent CPU arithmetic comparison and streamed FP8 CPU validation passed. A Layer 3 router/shared-expert acquisition and **synthetic-input** top-8 router test also passed.

**Not established:** true all-selected-experts Layer 3 aggregation; parity against official production kernels; native FP8 CUDA expert compute on the RTX 5060 Ti; original-router traces for full real-token decoding; complete 78-layer inference; generated tokens; any measured SSD/RAM/VRAM scheduler speedup.

The full original FP8 checkpoint is approximately **703.7 GiB** per the existing adaptation plan. A traditional RAM-resident expert pool is insufficient on this workstation.

## 3. Back-of-envelope traffic bound — not a speed prediction

For GLM-5.3's 75 MoE layers and eight routed experts per layer, a single token triggers **600 routed-expert invocations** (separate from the shared experts, attention and dense components). If other experts have the same serialized size as the measured Layer 3 expert, the **illustrative** payload is:

```text
600 × 37,757,952 bytes = 22,654,771,200 bytes ≈ 22.65 GB ≈ 21.10 GiB per token
```

This is an extrapolation from one actual expert, **not** measured disk traffic, nor a statement that every expert is fetched from SSD. With an illustrative *ideal* SSD sequential read speed of 10 GB/s, a 0%-reuse all-SSD read requirement alone would take ≥2.27 seconds per token; 50% avoided SSD bytes would still take ≥1.13 seconds of raw transfer service. The real critical path depends on access fragmentation, actual physical bandwidth, OS caching, PCIe copies, CPU/GPU compute, queueing, concurrency, and overlap. Read counters at the Python interface are **not** physical NVMe bytes.

For this specific throughput hypothesis, reducing the illustrative required SSD transfer below 10 GB per token (a necessary, never sufficient condition for 1 token/s at 10 GB/s) would require avoiding more than ~55.9% of the extrapolated payload. Verify actual sizes and the attainable bandwidth before treating this bound as operational.

## 4. Candidate architecture (research-only)

```text
GLM-5.3 full decoder (original token/router semantics)
        |
        +--> Original router expert IDs (ground truth for this layer)
        |
        v
ExactMoE Expert Dispatcher & Timing Instrumentation
        +--> CPU vs GPU execution: measured cost (including copies)
        +--> cache admission / eviction (byte-capped)
        +--> bounded asynchronous read/coalescing and prefetch
        +--> cancellation, SHA/source provenance, back-pressure
        |
        +-- VRAM: hot expert cache + pinned required attention/compute state
        +-- RAM: warm expert cache, CPU native FP8 compute, staged I/O
        +-- SSD: original immutable safetensors bytes + verified offsets/scales

OFFLINE research lane:
  frozen traces + hardware calibration --> LRU/LFU/cost-aware/CP-SAT/Belady baselines
                                      \--> optional QuantumZip/QAOA experiments
                     accepted policy --> compact deterministic online scheduler
```

Design rules:

1. **Never substitute predicted router IDs for real IDs.** Predictions may only speculatively fetch data; wrong guesses must not change model output.
2. **CPU compute is an option, not a fallback assumed to be slow.** If an expert is in RAM, compare `CPU_compute` against `RAM_to_GPU_copy + GPU_compute + synchronization` on identical activations, scales and output tolerances.
3. **Priority:** prove correct single-expert and complete single-layer FP8 execution first; optimize after measurements. Keep scheduling logic outside numerical kernels.
4. **Use immutable source ranges and exact tensor metadata.** Coalesce reads when physically beneficial; do not silently repackage/quantize reference weights.
5. **Prediction is conditional.** With a small GPU cache, incorrect prefetches may evict useful experts. Compare no-prefetch and request scheduling against any predictor.
6. **Multi-request optimizations** are separate from single-user token latency; evaluate throughput and tail latency independently.

## 5. What QZERA / QuantumZip can and cannot contribute

The operator's private [Quantum repository](https://github.com/YuvalSigura/Quantum) includes a [versioned capability matrix](https://github.com/YuvalSigura/Quantum/blob/main/docs/QZERA-CAPABILITY-MATRIX.md), with **QAOA MaxCut reference with gradients**, experimental tiered state storage, buffer ownership and native CUDA experiments. Its [2026-10-26 hardware report](https://github.com/YuvalSigura/Quantum/blob/main/docs/benchmarking/QZERA-WINDOWS-C0-C1-GPU-QUALIFICATION-2026-10-26.md) records narrow GPU/CPU quantum-kernel results. **None of these establishes GLM FP8 CUDA compute or working MoE scheduling.**

**Candidate reuse (after independent isolation/testing):**
- Memory contracts: buffer leases, bounded pinned-host transfers, GPU arena lifecycle, cancellation and integrity receipts.
- Controlled offline quantum optimizer: express an intentionally **small**, predeclared cache-placement or scheduling subproblem as QUBO/Ising, with correct capacity penalties and validation of feasibility.
- Research harness: compare QuantumZip/QAOA with exact enumeration for small problems, OR-Tools CP-SAT, greedy, LRU, LFU and cost-aware scheduling **on the same frozen workload**, including circuit construction/optimization/simulation wall time.

**No quantum speedup is assumed.** QuantumZip is a classical simulator; QAOA is not an inference accelerator or lossless weight compressor, and its optimizer overhead may exceed the decision window. Run quantum experiments **offline** unless a measured, independently validated benefit and bounded on-line latency are shown. High-entanglement simulation may use substantial classical memory.

**CEM-Zero** may later propose policies, but any novel policy must pass leakage-resistant held-out tests and classical baselines. Do not make ExactMoE correctness, install, or runtime depend on QuantumZip or CEM-Zero. Treat Intel NPU predictors or DreamOS-inspired scheduling as optional candidates, not established capabilities.

## 6. Experiment design and gates

| Gate | Experiment | Evidence required | Stop / caveat |
| --- | --- | --- | --- |
| **M0** | Baseline NVMe sequential vs scattered 4–64 MiB reads, physical I/O, CPU FP8, CUDA availability and PCIe staging | Real operator-machine p50/p95, throughput, peak RSS/VRAM, compiler/driver/SHA; cold and warm cache controls | No synthetic logical-byte count presented as disk bytes |
| **M1** | Original pinned model Layer 3 eight selected experts + shared path | Independent router IDs, exact routed weighted sum, source hashes and numeric tolerances | Synthetic inputs do not qualify as production traces |
| **M2** | Byte-capped SSD/RAM/VRAM policy simulator | No cache, LRU, LFU, cost-aware, optional predictive prefetch; byte-weighted misses on identical frozen traces | Synthetic traces prove only simulator correctness |
| **M3** | Native placement microbenchmark | Same expert inputs/outputs: CPU RAM-resident vs H2D+GPU; pinned/ordinary memory, overlap, p50/p95 | Never benchmark different precision as if equivalent |
| **M4** | Offline policy optimization / QAOA study | Exact small-instance optimum, CP-SAT and heuristics; feasibility, objective gap, end-to-end wall time, seeds and held-out traces | Negative quantum result is valid; do not claim advantage from simulator-only runs |
| **M5** | Original model full decoder and live trace capture | Independent DSA/MLA/token/78-layer parity; actual per-token/per-layer router IDs; full-model token | Required before claims about GLM-specific cache hit rates |
| **M6** | Production-path ablation | Measured physical bytes/token, p50/p95 time/token, prefill vs decode, RAM/VRAM peak, cold/warm, concurrency cases, output parity | No promotion unless real end-to-end improvement and unchanged semantics |

The key metrics are **physical SSD bytes/token**, actual device I/O throughput and latency, PCIe transferred bytes, useful vs wasted prefetch bytes, expert cache hit rate **weighted by bytes and reuse cost**, CPU/GPU compute time, time/token, and peak memory. Log source revision, workload/replay lineage, seeds and system settings. Do not log sensitive prompt contents or publish proprietary weights.

### Scope and order

1. Finish the existing [U3/U4/U5 correctness path](UPSTREAM_ADAPTATION_PLAN.md) and get hardware calibration.
2. Build cost-aware scheduling against simple deterministic baselines before developing an ML predictor.
3. Introduce offline optimization only after traces and costs are credible.
4. Re-evaluate the decision based on **end-to-end measured** improvements, not simulated cache hits alone.

## 7. Related work and differentiation

The ideas of offloading MoE experts, cache replacement and three-tier memory are established prior work. ExactMoE must **not** claim those concepts as novel:

- [SSD-LLaMA (2026)](https://arxiv.org/abs/2609.18110): SSD-native expert delivery, SSD/RAM/VRAM hierarchy and heterogeneous execution. Its RTX 5090 result is not a performance prediction for the target RTX 5060 Ti.
- [FlashMoE (2026)](https://arxiv.org/abs/2601.17063): learned recency/frequency-aware SSD expert cache replacement.
- [Bevy (2026)](https://arxiv.org/abs/2611.04794): multi-request runahead scheduling; warns that inaccurate prefetch can hurt under limited GPU cache. Applicable to multi-request settings, not proof for single-stream decoding.
- [PagedMoE (2026)](https://arxiv.org/abs/2611.03173): cross-precision bitplane caching; treat as an explicitly **lossy or representation-modifying variant**, not original FP8 reference.
- [NVIDIA-NeMo GLM-5 model notes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/glm/glm5/README.md): 78 layers (first three dense), 256 routed experts with top-8; refer to the pinned target config for exact loader assumptions.

**Possible research contribution to test, not an existing claim:** hardware-calibrated *joint* cache-placement and CPU/GPU execution decisions, with safe prefetch/avoid-prefetch admission and rigorous end-to-end evidence for a single consumer GPU with **less host memory than the full original FP8 checkpoint**.

---

**Status:** documentation and hypotheses only. No changes to runtime, dependencies, CI, pretrained weights or performance claims are authorized by this note.
