# GLM-5.3 original FP8 streamed expert — source-backed CPU spike

**2026-11-03. Status: ENGINEERING PROTOTYPE; NOT AN INFERENCE ENGINE; NOT OFFICIAL GLM PARITY.**

This branch is stacked on `feat/strata-maya-runtime-adaptation-20261103` (itself based on PR #18). It provides **one additional compute path**, not merely a metadata scan:
- `exactmoe/streamed_fp8.py` evaluates the three gate/up/down E4M3 FP8 matrix-vector products and SwiGLU for **one existing GLM-5.3 routed expert**, using true indexed original safetensors byte ranges and inverse-scale blocks.
- Each weight matrix is streamed **once through an opened file descriptor** (via `stream_record`) in row-aligned groups; each scale tensor is read and validated under a 2 MiB cap, with positive finite inverse scales. With 2 MiB default chunk budget, the streamed FP8 **raw matrix** is never all resident at once.
- A caller-supplied six-tensor SHA-256 map is all-or-nothing. The routine computes and verifies full weights + scale digests on the bytes consumed. The digest source is not by itself an independent publisher signature. Any failed hash or source change refuses to return an output.
- Mathematical CPU path reconstructs transient FP32 blocks from original E4M3 codes and 128x128 inverse scales, performs F32 matvec and returns a private F32 output only after all six tensors have completed validation.
- No original file modification, full-shard download, model pruning, expert switching or precision conversion of checkpoint data. This is **not** native FP8 TensorCore compute and its rounding differs potentially from production.

## Upstream borrowing

The **design** uses the SSD-first, coalesced expert read lesson from [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/81bb6c23ccee2154bc35cad922588ab66494a9a9), with the staged expert CPU/GPU/cache ideas mapped from [Strata](https://github.com/Niko1221/Strata/tree/ffa80d73133ea5c548aefe57889a16e8085a938f) and [Project Maya](https://github.com/mw00/project-maya/tree/10e69a0eb11541004abb24f9fa47147e9002326f). **No external engine source is copied into this implementation**. Native Kimi MXFP4 kernels are not GLM FP8 kernels. This CPU spike complements, rather than replaces, PR #18's existing FP8 expert and independently written local mathematical oracle.

## Run without external data (CPU fixtures)

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_streamed_fp8.py
python -m pytest -q
```

`tests/test_streamed_fp8.py` builds synthetic six-tensor safetensors files with original tensor naming conventions, covers nonuniform 128x128 scale grids, partial last K/row blocks, row chunk boundaries, invalid FP8 NaN, invalid scales, incomplete commitment, corrupt hashes, too-small budgets, and compares against separately written existing independent F32 oracle. **Synthetic PASS does not establish original GLM correctness or throughput.**

## Operator-controlled authentic one-expert measurement (NOT performed in this GitHub environment)

The operator must **explicitly authorize** fetching the pinned original six tensors from Hugging Face (the prior PR #18 tool), into a chosen, local untracked directory with adequate free space. No full 141-shard checkpoint download is needed for this gate:

```powershell
# Run ONLY on your own workstation after confirming model storage / source trust.
python -m exactmoe.expert_acquire --out "D:\\AI\\glm53-expert3-0" --layer 3 --expert 0 --max-source-mib 48
python -m exactmoe.streamed_fp8 "D:\\AI\\glm53-expert3-0" --layer 3 --expert 0 --chunk-kib 512 --check-local-oracle
```

The CLI first verifies sample identity/revision, per-tensor SHA, shapes and source scale validity via the existing `validate_sample`, executes the streamed algorithm, compares with `expert_validation.oracle_matvec` on matching random inputs if requested, and revalidates the sample immediately before emitting a report. This intentionally **does not claim** a publisher whole-shard hash, official FP8 dynamic activation parity or exact model inference. CPU memory used by the optional full-F32 oracle is **not bounded by the streamed execution's chunk budget**; run the oracle only with enough free RAM. Those commands are examples, **not evidence they were executed**.

Full publisher model must be independently reverified against `zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42`. The local sample provenance's stored digests reflect the HTTPS range bytes obtained, **not** a signature independent of the downloader. Source files and evidence never belong in Git.

## Explicit limits and next gate

- This one-expert path excludes the official router's eight selected experts, shared FFN, all 78 layers, MLA/DSA attention, tokenizer, MTP, KV and tool-calling endpoint.
- A `read()` byte count is **logical source** traffic, not physical NVMe I/O; the OS page cache may serve requests. RSS/VRAM and actual disk throughput must be measured separately.
- A source `fstat` stamp before/after a stream is not a cryptographic authenticity proof of the entire shard, and weight values may change on disk between distinct tensor reads. Pre/post six-tensor verification and immutable-source discipline are needed for operator measurements.
- Inference output from FP32 decoded matvec must match the independent oracle within precommitted tolerances; it need **not** be bit-identical to native dynamic FP8 TensorCore output.
- Full GLM-5.3 goal is preserved; GLM-5.3-Flash, Kimi K3, Qwen and optional quantized variants are **distinct model IDs**.

**Next:** run operator-side genuine FP8 sample verification; obtain reference activations and real production FP8 kernel outputs, build a small compatible AVX2/CUDA compute backend only after numeric acceptance, then expand to the pinned original router/shared-expert MoE FFN. Merge only after independent code review and successful CI; this branch is not accepted simply because synthetic Python checks pass.
