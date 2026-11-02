# ExactMoE

**Preserve the weights. Virtualize the memory. Verify the math.**

[![CI](https://github.com/YuvalSigura/ExactMoE/actions/workflows/ci.yml/badge.svg)](https://github.com/YuvalSigura/ExactMoE/actions/workflows/ci.yml)

ExactMoE investigates how to run frontier-scale Mixture-of-Experts models on consumer machines by moving unchanged model weight bytes between SSD, RAM and GPU. The long-term research target is the **full GLM-5.3 (~744B total parameters)**, not a distilled or smaller substitute.

> **Honest status (v0.1):** This repository currently contains a functional **safetensors metadata scanner, byte-preserving tensor/slice reader, bounded RAM cache, asynchronous prefetch, and runnable correctness-checked miniature MoE.** It is **not yet a GLM-5.3 inference runtime**. It does not implement GLM routing, DSA attention, FP8 matmul or CUDA paging.

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
