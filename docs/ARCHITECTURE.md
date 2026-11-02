# Architecture and correctness contract

## Goal

Implement a disk-backed inference runtime for large MoE checkpoints that preserves every original weight bit and the full model computation. Slower inference is acceptable. **This is a target, not a demonstrated GLM capability.**

## Delivered in v0.1

- `TensorIndex`: scans safetensors file headers and optional sharded `model.safetensors.index.json`. Supports opaque FP8/BF16 tensor metadata without conversion.
- `read_first_axis_slice`: reads exact contiguous byte ranges of packed tensors in row-major safetensors layout.
- `ExpertPager`: bounded byte-cache and asynchronous prefetch. Selection is external and authoritative.
- `demo`: complete synthetic one-layer float32 MoE test demonstrating equal selected experts and output to an in-memory reference.

## Planned execution for full model

1. Load the public checkpoint's real config/tokenizer and map tensors including per-expert FP8 scales.
2. Compute embeddings and each layer exactly as the official architecture defines.
3. Run the **original** router, including score functions, corrections, grouping and normalization.
4. Resolve selected experts; read each original weight and scale slice from NVMe through RAM to GPU staging buffers.
5. Execute native supported FP8 operations with documented accumulation behavior; do not dequantize to reduced precision.
6. Process attention/DSA and maintain KV cache with faithful positional encodings and masks.
7. Validate per-layer outputs and final logits against a known reference before introducing optimization.
8. Only then introduce speculative prefetch or decoding; these must not change target distribution.

## Risks and required gates

| Gate | Required evidence | Current status |
|---|---|---|
| G0 scanner | Header integrity, shard mappings, bounds, raw bytes | Implemented/tests |
| G1 pager | Cache budget, eviction, concurrent reads, Windows | Implemented/tests; CI pending |
| G2 miniature inference | Reference equality on deterministic F32 fixture | Implemented/tests |
| G3 GLM layout | Actual 5.3 tensor names, expert packing, FP8 scales | Not implemented |
| G4 GLM kernels | Router, DSA, MTP optional, FP8, attention and KV | Not implemented |
| G5 reference parity | Layer/logit comparison on official checkpoint | Not validated |
| G6 usable hardware | Peak RAM/VRAM, disk read, latency, sustained workload | Not measured |

## Numerical identity is nuanced

- **Weight preservation** means raw bytes read from the source checkpoint are unchanged.
- **Model-semantic parity** means identical formulas, routing decisions and components.
- **Numerical parity** requires specified tolerances versus a reference implementation and careful control of accumulation and kernel order.
- **Bitwise parity** additionally requires matched kernels/platform, determinism and sampling. It is not promised.
- A claim of \"same quality\" requires validated model-level evaluations; a synthetic test cannot establish this.

## Memory and performance

For a checkpoint around 756GB on disk, a 128GB host cannot hold all weights in RAM. Disk-backed reads can make execution possible in principle but with significant latency. A 16GB GPU must receive bounded working sets. Real throughput depends on per-token selected expert reuse, transfer overhead, SSD access patterns and CPU/GPU compute. No GLM token/s performance figure is implied by this repository.
