# ExactMoE — read-only first-run procedure for GLM-5.3

## A. Identify your actual checkpoint location first

Use the same `--local-dir` as the **already-running** Hugging Face transfer.
Do not start a second download to another directory. The official pinned
revision is `aca966e4e02791568aa6a4ced368624b3d897f42`.

On Windows PowerShell:

```powershell
Set-Location (git rev-parse --show-toplevel)
git pull --ff-only origin main
py -3 -m pip install -e ".[dev]"
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\first-run-glm53.ps1 -ModelDir "C:\Models\GLM-5.3"
```

Change the model path to match your existing download. The script does **not**
install packages, download shards, delete anything, modify drivers, or alter
model files. It validates the expected official FP8 config, disk free space,
NVIDIA detection, shard header integrity and the exact six-tensor shard
location for expert layer 3 / ID 0. Index-only reporting uses no checkpoint
weight payload I/O.

## B. Confirm storage requirements

Official indexed source weight bytes: **755,617,140,416 bytes = 703.72 GiB**.
A previously observed Windows C: drive had ~337.6 GiB free before any cleanup
or ongoing download. The original download needs the unmodified files plus
filesystem and HF temporary metadata. Aim for >=850 GiB available *before*
starting a fresh full download; if partially downloaded, account for existing
completed and incomplete shards before assessing remaining capacity. Prefer
~1 TiB available for safe working headroom.

The source checkpoint will NOT fit entirely in 128 GiB RAM or 16 GiB VRAM.
No existing software is yet validated for full original GLM-5.3 on this exact
hardware. No full-inference or token-per-second claim is implied.

## C. Minimal checkpoint and targeted-first-expert commands

If and ONLY if the JSON metadata is missing, obtain just the two small files:

```powershell
hf download zai-org/GLM-5.3 config.json model.safetensors.index.json `
  --revision aca966e4e02791568aa6a4ced368624b3d897f42 `
  --local-dir "C:\Models\GLM-5.3"
```

After `expert-plan` reports which exact shard filename(s) contain layer-3
expert-0 gate, up, down weights and scales, use that list to decide when they
have finished downloading. If downloading a single shard separately, the
canonical syntax is:

```powershell
hf download zai-org/GLM-5.3 model-00002-of-00141.safetensors `
  --revision aca966e4e02791568aa6a4ced368624b3d897f42 `
  --local-dir "C:\Models\GLM-5.3"
```

**The shard filename above is an illustrative candidate only**; use the
actual filenames printed by `expert-plan`. Do NOT launch a simultaneous
second transfer for the same file. Only start it if the full-model transfer
is paused or not running and enough disk capacity remains.

Run original-byte expert smoke with the local model directory:

```powershell
py -3 -m exactmoe.cli expert-smoke "C:\Models\GLM-5.3" --layer 3 --expert 0 --device cpu
```

Optionally test CUDA only if a CUDA-capable PyTorch installation exists and
`torch.cuda.is_available()` returns True. Downloading the source model and
having a working CUDA runtime are independent.

## D. When checkpoint download is complete

```powershell
py -3 -m exactmoe.cli checkpoint-status "C:\Models\GLM-5.3"
```

Target: 141 present shards, 0 missing, 0 invalid, and 118,629 tensor headers.
**Structural integrity does not prove payload hashes**, original checkpoint
byte identity or any inference output. HF revision is pinned via the download
command, but the preflight cannot independently prove the local files' commit
from file names alone.

## E. Linux/WSL2 for high-performance integration

KTransformers AVX2's documented minimums include Linux, NVIDIA GPU >=24GiB
and RAM as large as the model checkpoint. This PC has 16GiB VRAM/128GiB RAM.
It is NOT a turnkey path. Prefer WSL2 Ubuntu for later Linux-only CUDA
kernel experiments if development reaches that gate. GPU driver belongs
on **Windows**, not inside WSL; see NVIDIA's CUDA-on-WSL guidance.

Avoid duplicate full checkpoints inside both Windows and WSL. Access the
single Windows-hosted checkpoint from WSL for initial tests, and measure
real throughput before choosing a permanent model filesystem.

## F. Current correctness limits

The experimental CPU and CUDA FP32 tiled paths can compute genuine E4M3
source expert weights with original F32 block inverse scales. Dynamic FP8
activation quantization, complete DSA, all 78 original decoder layers,
tokenizer and independent full-model logits parity are still missing.
The goal is one original GLM-generated token and then >=32 verified tokens,
not a synthetic demo labelled as final inference.

## Sources

- https://huggingface.co/docs/huggingface_hub/main/package_reference/cli
- https://huggingface.co/docs/huggingface_hub/main/guides/download
- https://huggingface.co/zai-org/GLM-5.3
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md
- https://github.com/kvcache-ai/ktransformers/issues/2244
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
