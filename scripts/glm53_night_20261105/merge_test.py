"""Can experts be merged 'when needed'? Two measures on layer-3 experts (original FP8), 798 real inputs.
A) permutation-aligned weight similarity: for every gate row of expert A, the best-matching gate row of B
   (max |cosine|); a random-pair baseline shows what 'unrelated' looks like.
B) functional similarity: cosine between the outputs of A and B on the same real inputs, and the error of
   replacing A by the best scalar multiple of B (what a substitution or a merge would cost)."""
import io, contextlib, json, runpy, sys
from pathlib import Path
import numpy as np
S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from exactmoe.storage import TensorIndex  # noqa: E402
X = np.fromfile(S + r"\l3_in.f32", dtype=np.float32).reshape(-1, 6144)
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(S + r"\colibri_int4_quality.py", run_name="lib")
    sp = runpy.run_path(S + r"\split_expert_experiment.py", run_name="lib_split")
silu = q["silu"]
full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
ids = [222, 191, 59, 127, 177, 100, 62, 17, 135, 179, 26, 111]
W = {}
for e in ids:
    p = f"model.layers.3.mlp.experts.{e}."
    W[e] = [sp["dequant"](full[p + m + ".weight"], full[p + m + ".weight_scale_inv"]) for m in ("gate_proj", "up_proj", "down_proj")]
Xs = X[::4]                                                  # 200 real inputs
outs = {e: np.array([W[e][2] @ (silu(W[e][0] @ x) * (W[e][1] @ x)) for x in Xs]) for e in ids}
def rownorm(a): return a / np.linalg.norm(a, axis=1, keepdims=True)
gate = {e: rownorm(W[e][0]) for e in ids}
rng = np.random.default_rng(0); rand = rownorm(rng.normal(size=(2048, 6144)).astype(np.float32))
aligned, functional, subst = [], [], []
for i, a in enumerate(ids):
    for b in ids[i + 1:]:
        aligned.append(float(np.mean(np.max(np.abs(gate[a] @ gate[b].T), axis=1))))
        A, B = outs[a].reshape(-1), outs[b].reshape(-1)
        functional.append(float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B))))
        c = float(A @ B / (B @ B)); subst.append(float(np.linalg.norm(A - c * B) / np.linalg.norm(A)))
base = float(np.mean(np.max(np.abs(gate[ids[0]] @ rand.T), axis=1)))
self_check = float(np.mean(np.max(np.abs(gate[ids[0]] @ gate[ids[0]][np.random.default_rng(1).permutation(2048)].T), axis=1)))
out = {"aligned_gate_row_match": {"pairs_mean": round(float(np.mean(aligned)), 4), "pairs_max": round(float(np.max(aligned)), 4),
                                  "random_baseline": round(base, 4), "same_expert_permuted": round(self_check, 4)},
       "output_cosine": {"mean": round(float(np.mean(functional)), 4), "max": round(float(np.max(functional)), 4)},
       "substitution_error_best_scalar": {"min": round(float(np.min(subst)), 4), "mean": round(float(np.mean(subst)), 4)}}
print(json.dumps(out, indent=1))
