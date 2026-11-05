"""lean vs reference batched GPTQ on 8 layer-3 experts: codes equality + held-out error + peak VRAM/time."""
import io, contextlib, runpy, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
torch.cuda.set_per_process_memory_fraction(0.6, 0)
from gptq_gpu import quantize_layer_batch, silu
from gptq_lean import factor, quantize_layer_lean, dequant
from exactmoe.storage import TensorIndex
O = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(O + r"\colibri_int4_quality.py", run_name="lib"); sp = runpy.run_path(O + r"\split_expert_experiment.py", run_name="lib_split")
full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
Xe = np.fromfile(O + r"\l3_in.f32", np.float32).reshape(-1, 6144)
X = torch.from_numpy(np.concatenate([np.fromfile(r"C:\AI-Models\calib\l.3", np.float32).reshape(-1, 6144)[1000:],
                                     np.fromfile(r"C:\AI-Models\calib\h.3", np.float32).reshape(-1, 6144)])).cuda()
Hx = X.T @ X
routes = [[int(v) for v in q["route_noaux_tc"](x, q["router_w"], q["router_b"], q["config"]).expert_ids] for x in Xe]
B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
chosen = [e for e, _ in Counter(e for r in routes for e in r).most_common(B)]
def fp8(e, m):
    p = f"model.layers.3.mlp.experts.{e}.{m}"
    return torch.from_numpy(np.ascontiguousarray(sp["dequant"](full[p + ".weight"], full[p + ".weight_scale_inv"]), dtype=np.float32))
Gs, Us, Ds = (torch.stack([fp8(e, m) for e in chosen]).cuda() for m in ("gate_proj", "up_proj", "down_proj"))
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
fac = factor(Hx.unsqueeze(0)); (gc, gsc), (uc, usc), (dc, dsc) = quantize_layer_lean(Gs, Us, Ds, X, fac)
torch.cuda.synchronize(); print(f"lean: {time.time()-t:.1f}s for {B}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
lean = (dequant(gc, gsc), dequant(uc, usc), dequant(dc, dsc))
if len(sys.argv) > 2:
    torch.cuda.reset_peak_memory_stats(); t = time.time()
    (G0, U0, D0), packed = quantize_layer_batch(Gs, Us, Ds, X, Hx); torch.cuda.synchronize()
    print(f"ref:  {time.time()-t:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    print("code agreement gate/up/down:", [round(float((a == b).float().mean()), 5) for a, b in ((gc, packed[0][0]), (uc, packed[1][0]), (dc, packed[2][0]))])
Xt = torch.from_numpy(Xe).cuda(); errs = []
for b, e in enumerate(chosen):
    rows = [i for i, r in enumerate(routes) if e in r]; xh = Xt[rows]
    f = lambda g, u, d: (silu(xh @ g.T) * (xh @ u.T)) @ d.T
    ref = f(Gs[b], Us[b], Ds[b]); errs += (torch.linalg.norm(f(lean[0][b], lean[1][b], lean[2][b]) - ref, dim=1) / torch.linalg.norm(ref, dim=1)).cpu().tolist()
print("lean held-out err", round(float(np.mean(errs)), 4), "rows", len(errs))
