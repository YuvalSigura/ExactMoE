# GLM-5.3 on this PC: night report (2026-11-04/05)

Machine: Core Ultra 7 265KF (8P+12E, AVX2), 128 GB RAM, RTX 5060 Ti 16 GB, one NVMe.

Model: GLM-5.3 full (753B) as a Colibri container (`C:\AI-Models\GLM-5.3-colibri-x4g64-r8`). Routed experts are int4, resident tensors int8, and the DSA indexer is kept. The original FP8 checkpoint is untouched (`C:\AI-Models\GLM-5.3-full`).

All numbers below were measured on this machine. "Quality" means one of two things:
- log-likelihood (LL) on 3 texts, 431 continuation tokens;
- or, where stated, expert-output error against the original FP8 experts, on 798 **real** layer-3 inputs.

## 1. Where it stands

| | Start of evening | Now | How |
|---|---|---|---|
| Prefill, new 835-token prompt | 479 s (1.7 tok/s) | **103 s (8.1 tok/s)** | GPU paths written tonight (section 3) |
| Prefill, repeated prompt | — | 2,300 tokens in 0.3 s | KV prefix resume, already in Colibri |
| Decode | 0.62 tok/s | **0.79 tok/s**; 1.05 with memory-preferring routing | threads, prefetch, routing |
| Expert quality (real inputs) | 13.1% error (RTN int4) | **7.9% (GPTQ), migration running** | Bitdeer GPTQ repack |

## 2. Quality findings (real layer-3 inputs, against FP8)

| Expert representation | Error | Bytes vs 4-bit |
|---|---|---|
| RTN int4 g64 (container at the start of the night) | 13.1% | 1.0 |
| **GPTQ int4 g128** (Bitdeer; also my own GPTQ: 7.57%) | **7.9%** | 1.0 |
| GPTQ + two-stage hot neurons from FP8, 10% | 5.3% | +10% reads |
| GPTQ + two-stage hot neurons from FP8, 25% | 3.1% | +25% reads |
| My own GPTQ 3-bit (gate/up) | **10.5%** | 0.83 |
| RTN 3-bit / 2-bit | 31% / 82% | 0.75 / 0.5 |
| GPTQ top-2-bit plane ("frozen tunnel" as plain bit-plane) | 53% | 0.5 |
| Gate-first (keep 50% of neurons) | 13.1% | 0.67 |

Notes:
- **Synthetic input reversed the ranking.** On proxy input, GPTQ looked *worse* than RTN. Judge quantizers only on real inputs.
- **Experts cannot be merged or substituted.** After neuron alignment, gate rows match at 0.056 against a random baseline of 0.046. Output cosine is 0.06, and substitution error is 99.5%.
- **The FP8 dense overlay is built and correct**: it is the original values, 16 GB, in `C:\AI-Models\GLM-5.3-overlay-fp8dense`, enabled with `COLI_MODEL_DIRS`. But LL on 431 tokens cannot resolve its effect: −847.7 against −840.0 for int8, which is inside the spread between numeric paths. **Not enabled by default.**

## 3. Engine work

All of it is on the local Colibri branch `homelab/prefill-gpu-staging` (`C:\colibri-review\src`). Every item is gated by an environment variable and off by default.

| Change | Effect | Check |
|---|---|---|
| `COLI_PFGPU=<rows>`: prompt experts not on the VRAM tier run on the GPU through a staging slot | prompt experts on the GPU | CPU/GPU LL within spread |
| `coli_cuda_expert_mlp_gemm`: decode int4 once, then TF32 cuBLAS GEMMs | GPU expert time 132 s → 34 s | tiny oracle 17/17 |
| `coli_cuda_matmul_gemm` + `COLI_CUDA_GEMM_MIN` + `CUDA_DENSE_MATCH` (attention projections on the GPU) | projections 73 s → 7 s | LL within spread |
| `COLI_CUDA_MLA_PREFILL`: FP32 batched-GEMM MLA core with causal/DSA mask | attention core 61–73 s → 2.7 s | tiny oracle logits identical to the CPU path |
| fp8 (fmt=8) in the GEMM path | needed by the FP8 overlay | matches the CPU fp8 path |
| `COLI_DUMP_MOE`: real calibration inputs | enabled section 2 | — |

### Engine bugs and bad defaults found

| Finding | Status |
|---|---|
| `DIRECT=0` by default on Windows: 2.9 GB/s against 5.7 GB/s with DIRECT=1, and it fills the page cache | the fast variant sets `DIRECT=1` |
| `IDOT=1` default (int8 activations on the CPU) lowers LL slightly | documented |
| 8 OMP threads by default; 14 is about 10% faster | the fast variant sets it |
| The VRAM expert tier gave no decode gain here | VRAM goes to attention |
| Colibri's `COLI_CUDA_ATTN` is slower than the CPU for prefill (94 s vs 61 s) | replaced by the MLA kernel |
| The original GPU kernel recomputes each expert per token (no GEMM) | replaced |
| **Mixed group sizes across experts crash the engine** (slots reused with the first expert's geometry) | GPTQ stored as g64 with each scale duplicated; lossless |

## 4. Routing and speed sweep (decode tok/s, total LL over 431 tokens)

| Arm | LL | tok/s | RAM hit |
|---|---|---|---|
| Baseline (PILOT on, 14 threads) | −840.0 | 0.79 | 72% |
| TOPK=6 | −843.1 | 0.93 | 76% |
| TOPK=4 | −902.8 ✗ | 1.14 | 85% |
| TOPP=0.9 | −828.2 | 0.89 | 76% |
| CACHE_ROUTE J=4 M=12 | −840.1 | 0.93 | 84% |
| **CACHE_ROUTE J=2 M=16** | **−822.8** | **1.05** | **88%** |

CACHE_ROUTE has not yet been validated in true token-by-token generation: SCORE reads the text in one batch, and `COLI_PREFILL_CHUNK=1` did not change that.

## 5. Ideas analysed but not pursued, with the reason

- **Self-speculation with a spine-only draft.** Its fixed dense cost is about 0.5 s per draft token, so it gives about 0.55 tok/s at 60% acceptance. It pays only above roughly 90% acceptance.
- **Global "hot experts".** 80% of 7 million selections need 8,231 experts (162 GB at int4), so usage is spread wide.
- **Expert merging.** The experts are orthogonal (section 2).

## 6. Running now / next

1. **GPTQ migration, in place, hottest experts first.**
   - Log: `C:\AI-Models\GLM-5.3-gptq-migration2.log`. State: `<container>\.gptq-migration.json`.
   - The container is valid after every shard; the stale KV was moved to `.coli_kv.rtn-weights-backup`.
2. **After the migration:**
   - re-score;
   - validate CACHE_ROUTE in real generation;
   - test grammar-forced drafts with the HOMELAB CEO decision schema (`response_format: json_schema`);
   - test the `glm53-x4g64-r8-fast` serve variant end to end.
3. **Fast mode container:** my own 3-bit GPTQ for all experts (~300 GB). It needs all-layer calibration dumps and a CUDA PyTorch for speed.
4. **For HOMELAB:** short JSON answers. At about 1 s per output token, every token not written is the cheapest speed-up.

## 7. Generation and server tests (01:00–01:30, 28 of 140 shards on GPTQ)

| Test | Result |
|---|---|
| Decode, x86-64-v3 build | 0.75 tok/s (hit 66%) |
| Decode, `-march=native` build (AVX-VNNI) | 0.76 tok/s: **no gain**, the v3 build stays |
| **CACHE_ROUTE J=2 M=16, real token-by-token generation** | **1.10 tok/s (+47%)**, RAM hit 89%. The output is coherent and correct (the five-item list) |
| **Server, HOMELAB-style CEO decision** (system + user chat, `--no-think`, 98 prompt tokens) | **valid JSON on the first try**, a sensible decision (delegate to `web_tester`, safe and non-destructive confirmation), **~183 s for ~100 output tokens** |
| Same with `response_format: json_schema` (grammar-forced drafts) | 183 s: **no gain**. The output is mostly free-text values, so there is little to force |
| `runtime.choose` (score instead of generate, one request per option) | **282 s for 4 options: slower than generating.** Every short prefill still touches hundreds of experts on disk. It would need one batched forward over all options with a shared prefix (engine work). Its pick also disagreed with generation, because the scoring prompt was not in chat format |

What this means for HOMELAB:
- **Fast mode:** CACHE_ROUTE.
- **Answers:** short JSON values, because answer length is the main lever.
- **Thinking:** keep `--no-think` for routine roles.

## 8. Morning: own GPTQ on the GPU (2026-11-05, 08:00–)

- **The Bitdeer migration was stopped** at 121/140 shards (HuggingFace throughput fell to about 8 MB/s). It was stopped between shards, so the container is valid.
- **The engine now dumps calibration inputs for every layer:** `COLI_DUMP_MOE=all` writes `<COLI_DUMP_MOE_FILE>.<layer>` (built as `colibri_next.exe`).
  - The first dump is 3,476 tokens × 75 layers.
  - Windows shows 0 bytes for these files while the engine has them open; the real size is correct.
- **GPU GPTQ (torch cu128 on the 5060 Ti).** The recipe is act-order plus per-group MSE clip search, int4 g64.
  - It runs batched: the gate and up rows of 16 experts share one Hessian in a single column loop, and the down matrices go through a batched loop. That is **0.58 s per expert, against 12 s unbatched**, so about 3 hours for all 19,200 experts.
  - A batch of 32 experts spills out of 16 GB of VRAM into shared memory and slows to a crawl.
- **Honest quality, layer 3, 32 hottest experts, 1,565 real routed rows:**

  | Calibration | Own GPTQ | Container (Bitdeer) |
  |---|---|---|
  | Calibration text contains the evaluation text (contaminated) | 3.66% | 8.27% |
  | **Evaluation text excluded from calibration** | **7.86%** | **8.27%** |
  | Earlier: same-domain split (600 calibration rows / 198 held out, same document) | 6.4% | 7.6% |

  **GPTQ fits its calibration domain.** The gain is small on text from another domain and clear on text from the same domain. So the build calibrates on **HOMELAB-domain text**: about 12,000 tokens of harness prompts, code and architecture documents, plus the mixed set.
- **Tried and dropped:**
  - Fitting `down_proj` to the quantized gate/up path (error compensation): 31% error. A 2,048-wide least-squares fit from 600 rows overfits.
  - Hessian weighting by routed tokens: no gain.

## 9. Crash, lean quantizer, and the fast lane (2026-11-05, 09:00–10:10)

### Crash: `HYPERVISOR_ERROR (0x20001)`, about 08:50

| Check | Result |
|---|---|
| Hardware errors (WHEA), last 14 days | none |
| NVIDIA driver GPU errors (event 153) | two, at 08:39 and 08:49, both during my GPU runs |
| GPU errors before today | none |

**Probable cause:** VRAM oversubscription.
- The 32-expert run, and the build with 14,000 calibration rows, went past 16 GB.
- The Windows driver then pages GPU memory into system RAM ("sysmem fallback"), and with VBS/Hyper-V active a GPU fault in that state can bring the system down.

**Nothing was damaged:**
- the build had written nothing;
- all 160 shards were verified;
- no HOMELAB runtime file was zero-filled.

**Countermeasures:**
- a hard cap on PyTorch VRAM (`set_per_process_memory_fraction(0.6)`): the first overload became a clean out-of-memory error with no GPU fault;
- the lean quantizer below;
- a monitor that watches for event 153.

### Lean GPTQ (`gptq_lean.py`)
- Returns only the codes and scales.
- Keeps one permuted copy of the weights.
- Factorizes the gate/up Hessian once per layer instead of once per batch.
- Accumulates the `down_proj` Hessians over calibration chunks.
- Splits the update into row chunks.

**Effect:** peak memory is **4.3 GB instead of 8.5 GB**. The gate/up codes are bit-identical to before, and down agrees on 99.8% of codes. The full build runs at 90 s per shard.

### Own GPTQ against Bitdeer, on text the calibration has not seen

| Calibration | Error |
|---|---|
| Mixed (excluding the evaluation text) | 7.86% |
| **HOMELAB domain plus mixed, 13,147 rows** | **6.57%** |
| Bitdeer | 8.27% |

**Conclusion:** the domain and the amount of calibration data decide the result.

### Expert pruning by usage (router bias set to −1e4): ruled out

| Experts pruned | Share of usage | Routed-output error |
|---|---|---|
| 10% | 4.5% | 14% |
| 20% | 10.7% | 28% |
| 30% | 17.6% | 44% |

The experts are close to orthogonal, so substituting one expert for another breaks the output.

### Fast lane against substitution

Layer 3, 400 real tokens in text order, a simulated LRU cache of 54 experts per layer. Error is measured on the routed output against FP8.

| Policy | Disk loads avoided | Error, mean / p90 |
|---|---|---|
| CACHE_ROUTE J=2 M=16 (the turbo variant) | 36% | **35.8% / 68%** |
| Skip non-resident experts below 10% of the routing mass | 24% | 10.9% / 25% |
| Skip non-resident experts below 12% of the routing mass | 66% | 32% / 56% |
| Skip, then renormalize | — | always worse |

**Turbo's substitution damages quality much more than LL on 431 tokens showed.** It must not be used for anything that matters.

The fast lane is in the engine as `COLI_SKIP_LANE=<t>`:
- it applies in decode only;
- the strongest expert is always kept;
- there is no renormalization;
- it reports the share of skipped slots.

It is built but not yet run (the GPTQ build holds the container and the GPU).

## 10. Afternoon: full own-GPTQ container, first HOMELAB decision, MTP measured (2026-11-05)

- **Own GPTQ in every routed expert:**
  - all 140 shards, 19,286 experts;
  - out-00000 has 20 experts;
  - calibration: mixed plus HOMELAB-domain text, 14,147 rows;
  - held-out layer 3: 6.57% against Bitdeer's 8.27%.
- **Engine:** `colibri_next.exe` is installed as `colibri.exe`; `colibri_prev.exe` is kept.
- **HOMELAB catalog:** the container is registered with a full-content digest (`890a2814…`, 426.7 GB).
- **The served `/props` endpoint** reports the model and a 32,768-token window. The first answer was correct (391) in 34 s.
- **First CEO decision through HOMELAB's own `CeoMissionCognition`:**
  - valid JSON on the first attempt, and a sensible decision (delegate recon);
  - but **1,213 s**: 2,581 prompt tokens (about 215 s to the first token) plus **583 output tokens at about 0.58 tok/s**.
  - **Answer length is the main cost.**
- **MTP head** (layer 78, int8, 9.7 GB overlay `C:\AI-Models\GLM-5.3-overlay-mtp`, converted from the local FP8):

  | Arm | tokens/forward | acceptance | decode tok/s | expert hit |
  |---|---|---|---|---|
  | no speculation | 1.00 | — | **0.72** | 67% |
  | MTP draft=1 | **1.82** | **82%** | 0.66 | 60% |
  | MTP draft=2 | 1.85 | 76% | 0.67 | 61% |

  The drafts are good, but **a 2-token forward costs 2.77 s against 1.40 s.** Consecutive tokens route to nearly disjoint experts, so verifying 2 tokens reads almost twice the experts from disk, and decode here is limited by expert I/O, not compute.
  - **MTP does not speed up this machine.** It would only pay with the experts resident in fast memory.
  - **Speculation also changes greedy text** at near-ties (batched verification numerics differ). The output is equivalent, not bitwise identical.
- **What follows, for speed:**
  - fewer output tokens (a short CEO answer format; each token saved is about 1.4 s);
  - a higher RAM hit rate without substitution (`-lane`, to be measured in the engine).

### Fast lane measured in the engine (same prompt, temperature 0, 160 tokens max, no speculation)

| `COLI_SKIP_LANE` | decode tok/s | expert hit | decode slots skipped | answer |
|---|---|---|---|---|
| off | 0.72 | 67% | 0 | valid JSON, delegates the cross-account check |
| 0.10 | **0.92 (+28%)** | 77% | 19% | the same decision and plan, reworded |
| 0.12 | **1.19 (+65%)** | 85% | 30% | the same decision and plan, reworded |

- At 0.12 the lane is faster than CACHE_ROUTE turbo (1.10), with about a third of its offline routed-output error.
- This is one prompt. Gate 2's decision agreement (fast against lane) is the real quality test.

### Lane combined with MTP (`colibri_next.exe`, with the lane also applied to verification passes of ≤4 positions)

| Arm | tok/s | tok/forward | acceptance | hit |
|---|---|---|---|---|
| lane 0.12 alone | **1.20** | 1.00 | — | 85% |
| lane + MTP draft=1 (lane only on single-token passes, the first build) | 0.66 | 1.85 | 85% | 61% |
| lane + MTP draft=1 (lane on verification too) | 1.04 | 1.75 | 75% | 82% |
| lane + MTP, draft positions at t=0.30 (`COLI_SKIP_LANE_DRAFT`) | 1.19 | 1.62 | 62% | 87% |
| lane + MTP, draft positions resident-only (t=1.0) | 1.05 | 1.65 | 65% | 86% |

- **Result: MTP adds no speed on this machine even with the lane.** Consecutive tokens use nearly disjoint experts, so each verified token costs its own expert reads.
- Skipping harder on the draft positions saves those reads but lowers acceptance by about the same amount.
- **MTP becomes useful only when most experts are resident** (for example after domain pruning). The head is kept for that experiment.
- `COLI_SKIP_LANE` now also applies to MTP verification passes (≤4 positions). Caveat: a ≤4-token prefill delta after a KV-prefix resume is treated the same way.
