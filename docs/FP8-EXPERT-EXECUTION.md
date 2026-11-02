# P4/P5 experimental original-FP8 expert and routed MoE FFN executor

**Status:** implemented a single routed expert and sparse MoE FFN computation
from local unmodified GLM-5.3 FP8 checkpoint bytes, with tiny independent
numerical test fixtures. It has NOT generated a token from GLM-5.3. Full DSA,
RMSNorm/residual ordering, embeddings, attention, KV cache, logits and sampling
remain absent. Full native FP8 kernel parity is **NOT proven**.

## What runs

- Native safetensors rank-2 FP8 E4M3FN weight bytes are sliced by **output rows**
  directly from the pinned, original local checkpoint. There is no secondary
  source weight conversion, quantization or serialization.
- Each `weight_scale_inv` tensor is validated as F32 with block grid
  `ceil(out/128) x ceil(in/128)`, and applied to reconstructed FP8 data.
- **CPU**: NumPy float32 per-block matvec without materializing an entire
  expert matrix. **CUDA**: optional PyTorch float32 GEMV over staged <=32MiB
  tiles; this is a *GPU float32 reference/fallback*, not native Tensor Core
  FP8 or the same arithmetic sequence as official vLLM.
- One expert: gate, up and down projections plus SwiGLU.
- One MoE FFN stage: sigmoid/noaux_tc router in float32; add expert correction
  bias **for selection only**; group scoring by top-2; selected top-k experts;
  use *unbiased* scores for normalized mixture weights; shared expert always-on.
- Strict low memory budget for individual matrix row blocks and small control
  tensors. Actual allocated CUDA/OS memory remains to be measured on hardware.

## Commands after checkpoint download is complete

Open PowerShell in the ExactMoE repo, then:

```powershell
git pull --ff-only origin main
py -3 -m pip install -e ".[dev]"

# Replace X: with the drive containing the complete official 141-shard checkout.
py -3 -m exactmoe.cli expert-plan "X:\Models\GLM-5.3" --layer 3 --expert 0
py -3 -m exactmoe.cli expert-smoke "X:\Models\GLM-5.3" --layer 3 --expert 0 --device cpu
py -3 -m exactmoe.cli layer-smoke "X:\Models\GLM-5.3" --layer 3 --device cpu

# Optional: CUDA-enabled PyTorch installed separately and a working NVIDIA GPU:
py -3 -m exactmoe.cli layer-smoke "X:\Models\GLM-5.3" --layer 3 --device cuda
```

For CUDA, first confirm `py -3 -c "import torch; print(torch.cuda.is_available())"`
is True. **Do not download model weights again for these commands.** They use
the existing local checkpoint. The manifest and the shard files containing the *requested* tensors must exist. The new selective reader does not require unrelated shards. Incomplete required shards fail closed; files outside the official model root are rejected.

The CLI uses a seeded synthetic *hidden-state vector* to exercise one FFN.
This vector is NOT a token embedding from a prior GLM layer. The result
proves the data/compute path on authentic bytes once run, **NOT full GLM
model output quality, official layer parity or a meaningful text completion**.

## Validation contract

| Evidence | Status |
| --- | --- |
| E4M3FN hand-coded numerical values and NaNs | Unit tests |
| 128x128/partial-block FP8 scale layout | Unit tests |
| Streaming GEMV vs independent dense FP8 numerical oracle | Unit tests |
| Sigmoid+biased selection+unbiased weighted top-k routing | Unit tests |
| Gate/up/down and shared-expert numerical toy oracle | Unit tests |
| Windows and Ubuntu Python CI | Automated |
| Actual pinned GLM-5.3 model bytes on 265KF/5060 Ti | Awaiting user's completed download |
| Router choices vs production GLM source hidden states | Not yet tested |
| Dynamic activation FP8 scaling and production FP8 kernel parity | Not implemented |
| Exact DSA/IndexShare, all 78 layers, complete logits | Not implemented |
| First original GLM token and 32-token completion | Not implemented |

**Important numerical distinction:** Computing the mathematical weights
`E4M3(bytes) * inverse_block_scale` and accumulating float32 is NOT
equivalent to reproducing dynamic FP8 activation quantization, the exact
official matmul accumulation order, or an arbitrary official deployment's
logits. This implementation is a transparent, bounded *reference/fallback*,
not yet production-parity inference. Target output quality must be measured
against an independent pinned reference.

Official model/router reference:
- https://huggingface.co/zai-org/GLM-5.3
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
- https://huggingface.co/docs/transformers/main/model_doc/glm_moe_dsa

## Next engineering gate

1. Run this on one real GLM-5.3 layer and log expert selection and output.
2. Acquire independent layer activations and FFN outputs from official reference
   (same pinned checkpoint, router, token positions, dtype and kernel policy).
3. Implement native GLM activation FP8 semantics/scale quantization and ensure
   match of output tolerance, not just weights/bytes.
4. Integrate RMSNorm/residual/DSA and all layer tensor kinds into the executor.
5. Run a real one-token decoder pass, then 32 tokens with official tokenizer,
   and only then add local serving API and performance optimizations.

### Begin experiments before all 141 shards finish downloading

Run `exactmoe expert-plan <model-dir> --layer 3 --expert 0` after the JSON index arrives. It lists exact **original** shard filenames for all six tensors (3 matrices and 3 scale grids). Once these shard files complete, run `expert-smoke`; it scans only those shard headers. For `layer-smoke`, the CLI first loads router weights and computes the original selected IDs for a deterministic vector, then requires only the selected expert/shared expert shards. Shards are not downloaded by ExactMoE.
