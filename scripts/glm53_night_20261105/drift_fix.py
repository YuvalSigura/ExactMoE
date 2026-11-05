"""MIT-style 'direction' correction applied to quantization drift: learn the mean int4-vs-FP8 MoE output error on
calibration inputs, subtract it on holdout. Proxy gaussian inputs, real layer-3 experts, fixed routing."""
import runpy, io, contextlib, numpy as np
S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
with contextlib.redirect_stdout(io.StringIO()):
    r = runpy.run_path(S + r"\repair_ab.py", run_name="lib3")
moe, exact, quant = r["moe"], r["exact"], r["quant"]
cal = [np.random.default_rng(100 + s).normal(0, 0.1, 6144).astype(np.float32) for s in range(32)]
hold = [np.random.default_rng(200 + s).normal(0, 0.1, 6144).astype(np.float32) for s in range(8)]
E = np.array([moe(quant, x) - moe(exact, x) for x in cal])
bias = E.mean(0)
U, s, Vt = np.linalg.svd(E - bias, full_matrices=False)
print("share of calibration error energy in the mean:", float(np.sum(bias**2) * len(cal) / np.sum(E**2)))
print("share in top-1 / top-4 principal directions:", float(s[0]**2 / np.sum(s**2)), float(np.sum(s[:4]**2) / np.sum(s**2)))
for name, f in (("int4", lambda y: y), ("int4 - mean drift", lambda y: y - bias)):
    errs = [np.linalg.norm(f(moe(quant, x)) - moe(exact, x)) / np.linalg.norm(moe(exact, x)) for x in hold]
    print(f"{name:20s} holdout err={np.mean(errs):.4f}")
