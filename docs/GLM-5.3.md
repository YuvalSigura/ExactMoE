# GLM-5.3: integration requirements (not yet implemented)

**Target:** full `zai-org/GLM-5.3` native FP8 checkpoint, approx. 744B parameters (about 39–40B active). The official model files are approximately 756GB, subject to revisions. Do not clone/download them until sufficient SSD capacity is available.

## Prerequisites

- Model architecture and exact config for the pinned checkpoint revision.
- Shard manifest and tensor layout verified on actual files.
- Official tokenizer, chat template, proper RoPE, attention masks and cache semantics.
- GLM-MoE-DSA architecture, including 78 layers, initial dense layers, sparse experts, shared experts, DSA attention/indexer and all exact routing rules.
- FP8 formats **and scaling tensors**: streaming quantized bytes alone does not implement a correct matmul.
- A suitable GPU kernel and fallback for unsupported FP8 operations on RTX 5060 Ti; measure device capabilities rather than assuming parity.
- Correct reference logits from a sufficiently capable official deployment, with fixed prompt, tokenizer, version and deterministic decoding.

## Practical milestones

1. `inspect` the pinned checkpoint's local shards and compare indexed bytes with `model.safetensors.index.json`.
2. Document the real tensor names and decide whether experts are packed along axis 0, stored separately or use another layout. The baseline only slices axis 0; do not assume it fits the checkpoint.
3. Build a faithful **single GLM MoE layer** reference, using the same scores, routing, shared experts and scales.
4. Page exact per-expert bytes into allocated CPU/GPU buffers, avoid full-tensor materialization, validate against layer reference.
5. Integrate DSA and the complete decoder, KV-cache and tokenizer.
6. Compare per-layer outputs, expert selections and logits against the unmodified model. Set numeric error budgets by dtype and kernel.
7. Benchmark 512/2048/8192 prompt lengths and sustained decode on 16GB VRAM + 128GB RAM.

## Strict non-goals

- No expert count changes, pruning, merges, retraining, distillation or hidden quantization.
- No claiming full GLM inference from a successful F32 toy demo.
- No reliance on external proprietary API to produce the local model's answers.

## Primary upstream references

- https://github.com/zai-org/glm-5
- https://huggingface.co/zai-org/GLM-5.3
- https://huggingface.co/docs/transformers/en/model_doc/glm_moe_dsa
- https://recipes.vllm.ai/zai-org/GLM-5.3
