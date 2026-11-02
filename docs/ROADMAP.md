# Engineering roadmap

- [x] **M0:** Source repository, package, tests, CI and documented scope.
- [x] **M1:** Non-materializing metadata scanner for safetensors shards.
- [x] **M2:** Cross-platform bounded RAM byte-LRU and expert slice paging.
- [x] **M3:** Executable mini-MoE demonstrating reference output parity.
- [ ] **M4:** Inventory real GLM-5.3 checkpoint tensor names/shape and FP8 scale layout.
- [ ] **M5:** Faithful per-layer GLM routed expert computation with reference parity.
- [ ] **M6:** GPU staging allocator and exact native FP8 kernels; measurable memory budgets.
- [ ] **M7:** GLM-5.3 full attention/DSA, KV-cache, tokenizer, autoregressive decode.
- [ ] **M8:** Model-level numerical and cyber/coding benchmark validation.
- [ ] **M9:** Predictive prefetch, cache policy tuning, safe speculative decoding.

A checked box denotes only implemented code that must pass CI. It does not imply performance or capability on GLM-5.3.
