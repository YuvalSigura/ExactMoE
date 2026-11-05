import runpy, numpy as np
S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
c = runpy.run_path(S + r"\combo_experiment.py", run_name="lib2")
experts, colibri_int4, moe, exact, route, x = c["experts"], c["colibri_int4"], c["moe"], c["exact"], c["route"], c["x"]
ref = moe(exact)
for rank in (0, 16, 64, 256):
    w = {}; extra = 0; base = 0
    for eid, m in experts.items():
        mats = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            W = m[name][0]; Wq, _ = colibri_int4(W, 64); base += W.size
            if rank:
                U, s, Vt = np.linalg.svd(W - Wq, full_matrices=False)
                Wq = Wq + (U[:, :rank] * s[:rank]) @ Vt[:rank]
                extra += rank * (W.shape[0] + W.shape[1]) * 2           # bf16 factors
            mats.append(Wq)
        w[eid] = mats
    y = moe(w)
    print(f"int4 g64 + rank-{rank:3d} error repair: stored={0.5625 + extra/base:.3f} of FP8  err={np.linalg.norm(y-ref)/np.linalg.norm(ref):.4f}")
