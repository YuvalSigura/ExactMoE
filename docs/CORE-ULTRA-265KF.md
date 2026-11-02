# ExactMoE: Intel Core Ultra 7 265KF hardware plan

## Confirmed CPU capabilities (Intel Product Specifications)

- Arrow Lake-S, **20 physical cores = 8 P-cores + 12 E-cores**, 20 threads, up to 5.5 GHz.
- **Intel AI Boost NPU**, rated **13 TOPS INT8 peak**; this NPU is present on the KF model even though it has **no Intel integrated GPU**.
- **AVX2**, Intel Deep Learning Boost, 30 MB L3, 36 MB L2. **No official AVX-512 or AMX** on 265KF. Do not compile mandatory AVX-512/AMX-only kernels.
- Two DDR5 memory channels; 128 GiB system RAM is usable as an expert *cache*, not sufficient for all original GLM-5.3 weights.
- CPU supports PCIe 5.0/4.0 and 24 CPU PCIe lanes, but actual storage/GPU topology depends on motherboard slot wiring.

Sources:
- https://www.intel.com/content/www/us/en/products/sku/241062/intel-core-ultra-7-processor-265kf-30m-cache-up-to-5-50-ghz/specifications.html
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md

## Architecture design: keep the target model authoritative

```text
NVMe: unchanged FP8 expert weights and inverse-scale tensors
             |
             v
CPU (8P + 12E): SSD I/O, expert index, RAM cache, page scheduling
             |                 |
             |             optional Intel NPU
             |             *prefetch predictor only*
             |                 |  may predict, NEVER change routing
             v
128 GiB DDR5 RAM: variable-size expert cache + staging buffers
             |
             v
RTX 5060 Ti 16GB: exact GLM forward kernels (future)
             |
             v
Original router and unchanged weight data determine output
```

### Recommended task partition (HYPOTHESES until measured)

| Hardware | Candidate work | Quality contract |
|---|---|---|
| P-cores (8) | Time-critical routing/controller, host tensor prep, CUDA launch; potential CPU fallback | Original operations and numerically validated kernels only |
| E-cores (12) | Concurrent bounded NVMe reads, decomposed prefetch, metadata & eviction workers | May affect *latency* and caching, not output |
| Intel NPU (13 peak INT8 TOPS) | Small trained cache/next-expert predictor, or other independent auxiliary model | Prediction MUST NOT override original router/expert selection |
| 128GiB RAM | LRU / cost-aware expert cache, pinned staging buffers | Exact checkpoint FP8 bytes/scales preserved |
| RTX 5060 Ti 16GB | Real FP8 expert compute + attention where supported; verify kernels | Must pass per-layer/TP logits parity |
| NVMe | Cold-store checkpoint and byte-range loads | Verify checksum, offsets and revision |

The NPU's 13 TOPS is INT8 peak, not FP8 transformer throughput. The NPU cannot be assumed to accelerate native GLM-5.3 FP8 expert matmuls, DSA or arbitrary CUDA kernels. OpenVINO support is conditional on the driver, compatible graph operators and device availability. The 265KF has no iGPU, so Intel GPU inference is unavailable on the processor itself.

### Why these assignments?

- P/E groups have different performance/energy characteristics. Windows Thread Director scheduling is a reasonable baseline. **Do not force affinity by logical CPU index**: topology order is not guaranteed. Use OS CPU Sets or verified affinity information only after profiling.
- At 2 DDR5 channels and 5600 MT/s, *theoretical* peak bandwidth is ~89.6 GB/s (2 x 64-bit channels x 5600 MT/s); effective sustained bandwidth can be far lower. Distinguish from NVMe read speed and PCIe transfer throughput. Verify actual configured memory speed.
- Per-token expert cache misses can dominate generation. Prefetch can be *speculative* while output remains exact, provided the actual router output is always authoritative.
- CPU direct compute of many large FFNs may be very slow, but may serve as a verified fallback if a single weight tensor exceeds GPU staging limits.
- OpenVINO `AUTO` does **not** necessarily choose NPU; select and test `NPU` explicitly.

### Before making optimization claims

1. Run the read-only profiler below and inspect whether Windows/OpenVINO actually enumerates NPU.
2. Benchmark baseline system memory copy and sustained NVMe reads separately on the target machine, including CPU utilization and thermal throttling.
3. Benchmark cache hit rates, SSD bytes/token, PCIe transfer time, P-core/E-core scheduling and CUDA kernel latency on representative tensor slices.
4. Compare model-selected expert IDs and reference layer output before/after any scheduling change.
5. Enable predictive NPU prefetch **only if** it improves end-to-end latency and does not change output.

## Read-only Windows audit

Run from the repository root:

```powershell
git pull --ff-only origin main
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\profile-ai-hardware.ps1
```

Report output is under `artifacts/hardware/ai-platform.json`. It intentionally does not include user names, account IDs, hostnames, device serials or file contents; the report stays local unless explicitly shared.

Optional Intel NPU discovery:

```powershell
py -3 -m pip install openvino
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\profile-ai-hardware.ps1 -CheckOpenVINO
```

If `NPU` is not available in OpenVINO, first inspect **Device Manager > Neural processors**. Install/update [Intel's official Windows NPU driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) only if needed and compatible, then rerun. Merely installing Python packages does not install hardware drivers.

### Validation scope

The PowerShell collector is read-only and tested for syntax in CI; GitHub Actions runners are **not the user's Core Ultra 7 265KF**, so no benchmark or inference performance is assumed. Changing BIOS power/voltage, overclocking or disabling E-cores is NOT recommended for initial experiments.
