# All feasible architecture paths — original GLM-5.3, no additional weight quantization

**Scope:** preserve the *pinned* original publisher FP8 checkpoint and original router, produce all layers' actual inference locally on the user's 265KF + RTX 5060 Ti 16GB + 128GB RAM. Original FP8 isn't an extra lossy transformation we choose; BF16 is a different source checkpoint and ~2x storage.

This is an architectural *decision matrix*. None of these paths proves full GLM-5.3 operation on the target machine without end-to-end and reference testing.

## Competing viable architectures

| Path | How it works | Primary benefit | Blocking risk | Priority / decision |
|---|---|---|---|---|
| A — KTransformers AVX2 + SGLang with original FP8 | Reuse GLM routing/DSA CUDA frontend and AVX2-native expert kernels; modify its expert loader to page from NVMe rather than requiring >=checkpoint-size RAM | Most existing GLM native kernel/operator work may be reusable | Native *full GLM-5.3* support on 265KF/16GB still unverified; baseline KTransformers keeps model weights in RAM; larger DSA/caches may not fit GPU | **Probe first**, isolate one original expert, add NVMe source |
| B — InferSSD / llama.cpp backend adaptation | Reuse its SSD expert tier/cache and CUDA execution; add support for GLM-MoE-DSA, pinned original FP8 safetensors or byte-preserving compatible container | Tested SSD-first execution architecture and bounded cache | Current tested models are GGUF and quantized, Windows in progress; direct native publisher FP8 and entire GLM 5.3 decoder unverified | **Probe second**; reuse paging design more than assumption of model support |
| C — Custom ExactMoE scheduler with existing PyTorch/Triton/vLLM kernels | Implement original shard reader and page/stage weights; port original GLM operations incrementally; use reference runtime's mathematics | Maximum checkpoint fidelity and resource accounting | Massive attention/DSA/FP8 kernel engineering and benchmarking cost | **Fallback / integration path** if A/B unsupported |
| D — GPU out-of-core per-expert and per-tile | Load one selected expert at a time (or pieces of each matrix), original blockwise FP8+scales, compute on GPU; spill buffers between operators | 16GB GPU can be sufficient in principle for bounded *working set* | Transfer and kernel launch overhead; GPU/CPU resources and DSA context | **Must-have execution technique** independent of A/B/C |
| E — CPU AVX2 native FP8 fallback | Decode only selected experts/blocks and compute reference/expert FFNs on host using original bytes | Correctness oracle, works if native GPU path blocked | Extremely slow, extra RAM traffic; still must implement 78-layer decoder | **Mandatory fallback** on one expert; optional hybrid later |
| F — Immutable NVMe store + 128GB adaptive RAM cache | SSD retains all files; cache hot experts by (layer,expert) and scales; asynchronously prefetch selected weights | Dramatic reduction in repeated SSD reads if locality exists | Many unique selections and 75 layers; misses and file fragmentation dominate | **Mandatory**; benchmark actual hit-rate |
| G — Lossless expert-level compression (Zstd/LZ4) | Compress *original bytes reversibly* on disk, decompress to exact bytes before use | Potentially saves storage and I/O | FP8 entropy may make compression weak; CPU decompression overhead | **Experiment only** after baseline, never assume savings |
| H — Packing / interleaving experts by shard | Repackage into an indexed, checkpoint-fingerprinted container of exact original FP8 bytes/scales, aligned by expert/layer | Fewer seeks, more contiguous NVMe access | Temporary additional storage; must prove byte identity | **Optimization** after correct baseline |
| I — CPU/P-core/E-core task graph | P-cores for control and latency-sensitive staging; E-cores for bounded I/O / cache, after testing topology | Overlap reads and GPU compute | Windows topology/affinity and memory contention | **Later measured optimization**; do not infer CPU index |
| J — NPU predictor for prefetch-only | Small auxiliary INT8 model predicts likely future resident pages | Potentially overlap next reads earlier | 13 TOPS NPU not an FP8 GLM accelerator; wrong predictions waste reads | **Optional** after router parity and prefetch benchmarks |
| K — Exact speculative decoding / MTP | Predict token candidates with a separate small drafter, verify with original target model and mathematically correct accept/reject | May amortize huge target-model weight passes over accepted tokens | More complicated prefill, high target batch cost, MTP validation | **After exact base inference** |
| L — KV/indexer caching and host spill | Allocate short context first; chunk DSA sparse-attention cache; spill state to RAM where exact | Enables longer context despite VRAM cap | More host transfer; must match reference attention | **Mandatory** before realistic context |
| M — Additional consumer GPU or machine nodes | Distribute experts/layers across GPU VRAM/machines (NVLink not assumed); preserve same original tensors | Larger working set, less swapping if enough GPU memory | PCIe/network speed, purchase cost, operator implementation | **Hardware upgrade option**, not baseline requirement |
| N — Cold remote byte-range weight source | Fetch original expert bytes on demand from pinned HF model via HTTP ranges, verify bytes and hashes | Could demonstrate actual expert without storing full checkpoint | Network bandwidth/latency, 22GB+ cold expert reads per token, remote reliability | **ONLY for small M3/P3 experiments**, not realistic full local inference |

## Shortlisted recommendation

1. **A / reuse KTransformers FP8 AVX2 + existing GLM code**, paired with **F/D** streaming and bounded GPU math. Test the released AVX2 path on an original one-expert tensor, never assume its documented GLM-5 or GLM-5.3-Flash support equals GLM-5.3 full compatibility.
2. In parallel, examine **B / InferSSD** as a paging engine and/or standalone benchmark for an unmodified GGUF control checkpoint. It is NOT itself a validated native publisher-GLM-5.3 runner.
3. Maintain **C / ExactMoE custom kernel integration** for any missing GLM-5.3 original operator. Avoid reimplementing robust CUDA math that can legally and correctly be integrated.
4. Add **G/H/I/J/K** only when correctness is proven and telemetry identifies a bottleneck.

## Why other tempting methods are NOT equivalent to this goal

- Q4/Q5/GGUF quantization, NVFP4/W4A8 conversion: reduce weight precision and can change logits; optionally test *separately* only if user changes quality constraint.
- Pruning experts, altering active top-k, routing shortcuts and mixture-of-depth early exits: alter computation, unacceptable for original-model semantic parity.
- LoRA/QLoRA/fine-tuning: customizing weights is a different goal; it does not make untouched full checkpoint fit in RAM.
- Distillation/multiple smaller models: cannot claim identical base-model behavior.
- Weight deduplication is only lossless for **bit-identical** blocks; expect little on unrelated FP8 weights unless measured.
- Dynamic re-fetch of a massive model from the internet per token fails the user's local independence/performance objective.
- Small Intel NPU cannot host full GLM expert arithmetic; 13 TOPS is a peak INT8 marketing bound, not FP8 native transformer throughput.
- DirectStorage/GPUDirect Storage is an optional **measured I/O backend**, not guaranteed on consumer RTX/Windows. NVIDIA's GPUDirect Storage ecosystem is Linux-focused; Windows is not generally supported for GDS. Do not claim it eliminates all PCIe and SSD constraints.
- Extra context tricks like KV quantization may change attention numerical behavior and therefore aren't enabled in the strict parity baseline.

## Hard engineering and safety gates

**A. Pin reference**: confirm revision, hash, config, tokenizer, license, cache semantics and versioned GLM-MoE-DSA implementation. Numeric parity must compare independent runs of the same checkpoint, not self-tests against a copied bug.

**B. Probe code reuse**: build/inspect KTransformers AVX2 FP8 and GLM-5.3 full architecture support; probe InferSSD cache logic + build environment. Evaluate Windows native vs WSL2 Linux based on working CUDA and I/O, not preference.

**C. Prove real original FP8 expert**: obtain 3 source matrix ranges + inverse scales (verified intact), run reference and CUDA/AVX2 candidate on frozen activations. Do **not** download ~756GB yet.

**D. Original routing + exact full layer**: produce matching grouped top-k experts and full selected expert residual with original shared expert and scales; measure data transfer bottleneck.

**E. DSA/attention + 78-layer full pass**: every tensor accessed, bounded resident memory, dynamic SSD cache. Use single token and 128–256 token context initially; don't target maximum long context.

**F. Full checkpoint acquisition**: once C/D/E are technically justified and full data needed, ensure >=850GiB **free**, preferably >=1TiB, download original pinned shards with resume/hash. User may remove irrelevant *local model files only*, preserving repo/code, virtual environments if useful, cache and reference reports. The exact directory matters; no blanket recursive deletion.

**G. 32-token acceptance**: generate an actual local answer from full GLM-5.3 and validate output vs reference tolerance/quality, respecting the same original FP8 weights.

**H. Product**: add OpenAI-compatible localhost endpoint/chat, observability, benchmark data and restart consistency. Never label test fixture output as GLM inference.

## Benchmark design

Record per-run: fixed prompt and token IDs, native checkpoint SHA, selected expert IDs/layer, SSD GiB/token, cold/warm RAM cache hit %, GPU staging bytes, kernel time, RAM peak, VRAM peak, median and 95th percentile seconds/token, prefill seconds, output checksums and maximum logit error vs reference. Declare CPU affinity policy and OS.

At 75 sparse layers and 8 experts/layer, each cold token may read about **22.65GB** of routed expert FP8 weights (plus scales, shared and dense/attention weights), assuming expert hidden=6144 and intermediate=2048. This alone imposes a lower bound of >2 seconds/token on a 10 GB/s idealized read path. This is NOT a measured inference speed or a guaranteed upper bound.

## Sources used to define strategy

- Intel 265KF official features: https://www.intel.com/content/www/us/en/products/sku/241062/intel-core-ultra-7-processor-265kf-30m-cache-up-to-5-50-ghz/specifications.html
- KTransformers releases AVX2 FP8: https://github.com/kvcache-ai/ktransformers/releases
- KTransformers AVX2 guide: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md
- KTransformers current GLM compatibility notes: https://ktransformers.net/en/docs/supported-models/glm
- InferSSD architecture: https://github.com/wici-ai/inferssd/blob/main/docs/architecture.md
- InferSSD release limitations: https://github.com/wici-ai/inferssd/blob/main/CHANGELOG.md
- GLM original checkpoint: https://huggingface.co/zai-org/GLM-5.3
- NVIDIA GPUDirect Storage support: https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/features/gpudirect.html

**No decision to remove models, purchase hardware, download 756GB, or use cloud GPU is made by this document.**
