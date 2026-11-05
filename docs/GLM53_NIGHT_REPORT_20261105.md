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
