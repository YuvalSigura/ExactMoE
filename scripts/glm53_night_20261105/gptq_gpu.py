"""GPU GPTQ for GLM-5.3 routed experts (torch, fp32 on the RTX 5060 Ti). Library + a layer-3 test.

quantize_expert(G, U, D, X, w) -> int4 g64 codes/scales for gate, up, down, in Colibri fmt4 layout
(byte = (lo+8) | (hi+8)<<4, low nibble = even column; qs f32 [O, I/64]).
Recipe (measured on CPU in gptq_plus.py: 6.46% vs Bitdeer 7.65%): act-order + per-group MSE clip search.
New arms tested here:
  rw    Hessian over all calibration tokens, plus the tokens the router sends to this expert weighted by w
  comp  down_proj is fitted to the QUANTIZED gate/up path: target D h_fp, input h_q (error compensation),
        i.e. W' = D Hfq Hqq^-1, then GPTQ of W' with Hessian Hqq
"""
import sys, time
import numpy as np
import torch

DEV = "cuda"

def gptq_t(W, H, bits=4, gs=64, mse=True, block=128, damp=0.01):
    """W [O,I] fp32 cuda, H [I,I]. Returns (dequant W, codes int8 [O,I], scale [O,I/gs]) in original column order."""
    W = W.clone().float(); O, I = W.shape; qmax = (1 << (bits - 1)) - 1; ng = I // gs
    H = H.clone().float(); dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    Wg = W.view(O, ng, gs); amax = Wg.abs().amax(dim=2)
    if mse:
        d = torch.diag(H).view(1, ng, gs)
        best, best_err = amax.clone(), torch.full_like(amax, float("inf"))
        for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
            s = torch.clamp(amax * c / qmax, min=1e-12).unsqueeze(2)
            e = (((torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s - Wg) ** 2) * d).sum(dim=2)
            better = e < best_err; best_err = torch.where(better, e, best_err); best = torch.where(better, amax * c, best)
        amax = best
    scale = torch.clamp(amax / qmax, min=1e-12)                        # [O, ng]
    perm = torch.argsort(torch.diag(H), descending=True); inv = torch.argsort(perm)
    Wp = W[:, perm].contiguous(); Hp = H[perm][:, perm]
    Hp += damp * torch.mean(torch.diag(Hp)) * torch.eye(I, device=W.device)
    L = torch.linalg.cholesky(Hp); Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Sp = scale[:, perm // gs]                                          # per-column scale, permuted order
    Q = torch.zeros_like(Wp); C = torch.zeros((O, I), dtype=torch.int8, device=W.device)
    for i1 in range(0, I, block):
        i2 = min(i1 + block, I); W1 = Wp[:, i1:i2].clone(); Err = torch.zeros_like(W1); Hb = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            s = Sp[:, i1 + j]; w = W1[:, j]
            code = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
            C[:, i1 + j] = code.to(torch.int8); q = code * s; Q[:, i1 + j] = q
            err = (w - q) / Hb[j, j]
            W1[:, j:] -= err.unsqueeze(1) * Hb[j, j:].unsqueeze(0); Err[:, j] = err
        Wp[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, inv], C[:, inv], scale

def gptq_rows_shared(W, H, bits=4, gs=64, block=128, damp=0.01):
    """gptq_t for many row-blocks sharing one Hessian (all gate+up rows of one layer): same math, one column loop."""
    return gptq_t(W, H, bits, gs, True, block, damp)

def gptq_batched(Ws, Hs, bits=4, gs=64, block=128, damp=0.01):
    """GPTQ for B matrices with their own Hessians in one column loop. Ws [B,O,I], Hs [B,I,I] (act-order per matrix)."""
    W = Ws.clone().float(); B, O, I = W.shape; qmax = (1 << (bits - 1)) - 1; ng = I // gs
    H = Hs.clone().float(); dg = torch.diagonal(H, dim1=1, dim2=2)
    dead = dg == 0
    H = H + torch.diag_embed(dead.float()); W = W * (~dead).unsqueeze(1).float()
    dg = torch.diagonal(H, dim1=1, dim2=2)
    Wg = W.view(B, O, ng, gs); amax = Wg.abs().amax(dim=3); d = dg.view(B, 1, ng, gs)
    best, best_err = amax.clone(), torch.full_like(amax, float("inf"))
    for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        s = torch.clamp(amax * c / qmax, min=1e-12).unsqueeze(3)
        e = (((torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s - Wg) ** 2) * d).sum(dim=3)
        better = e < best_err; best_err = torch.where(better, e, best_err); best = torch.where(better, amax * c, best)
    scale = torch.clamp(best / qmax, min=1e-12)                               # [B,O,ng]
    perm = torch.argsort(dg, dim=1, descending=True); inv = torch.argsort(perm, dim=1)
    Wp = torch.gather(W, 2, perm.unsqueeze(1).expand(B, O, I)).contiguous()
    Hp = torch.gather(torch.gather(H, 1, perm.unsqueeze(2).expand(B, I, I)), 2, perm.unsqueeze(1).expand(B, I, I))
    Hp = Hp + damp * dg.mean(dim=1).view(B, 1, 1) * torch.eye(I, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hp)), upper=True)
    Sp = torch.gather(scale, 2, (perm // gs).unsqueeze(1).expand(B, O, I))
    Q = torch.zeros_like(Wp); C = torch.zeros((B, O, I), dtype=torch.int8, device=W.device)
    for i1 in range(0, I, block):
        i2 = min(i1 + block, I); W1 = Wp[:, :, i1:i2].clone(); Err = torch.zeros_like(W1); Hb = Hinv[:, i1:i2, i1:i2]
        for j in range(i2 - i1):
            s = Sp[:, :, i1 + j]; w = W1[:, :, j]
            code = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
            C[:, :, i1 + j] = code.to(torch.int8); q = code * s; Q[:, :, i1 + j] = q
            err = (w - q) / Hb[:, j, j].unsqueeze(1)
            W1[:, :, j:] -= err.unsqueeze(2) * Hb[:, j, j:].unsqueeze(1); Err[:, :, j] = err
        Wp[:, :, i2:] -= torch.bmm(Err, Hinv[:, i1:i2, i2:])
    ix = inv.unsqueeze(1).expand(B, O, I)
    return torch.gather(Q, 2, ix), torch.gather(C, 2, ix), scale

def quantize_layer_batch(Gs, Us, Ds, X, Hx, bits=4):
    """Gs,Us [B,F,Hd], Ds [B,Hd,F]. gate+up of all B experts share Hx -> one stacked GPTQ; downs batched."""
    B, F, Hd = Gs.shape
    GU = torch.cat([Gs, Us], dim=1).reshape(1, 2 * B * F, Hd)
    Q, C, Sc = gptq_batched(GU, Hx.unsqueeze(0), bits)
    Q = Q.view(B, 2, F, Hd); C = C.view(B, 2, F, Hd); Sc = Sc.view(B, 2, F, Hd // 64)
    hf = silu(torch.einsum("nh,bfh->bnf", X, Gs)) * torch.einsum("nh,bfh->bnf", X, Us)   # [B,n,F]
    Hd_ = torch.bmm(hf.transpose(1, 2), hf)
    Dq, Dc, Ds_ = gptq_batched(Ds, Hd_, bits)
    return (Q[:, 0], Q[:, 1], Dq), ((C[:, 0], Sc[:, 0]), (C[:, 1], Sc[:, 1]), (Dc, Ds_))

def silu(x): return x * torch.sigmoid(x)

def quantize_expert(G, U, D, X, routed_idx=None, w=0.0, comp=False, bits=4, Hx=None):
    """G,U [F,Hd], D [Hd,F] fp32 cuda; X [n,Hd] calibration (all tokens); routed_idx: token rows routed here."""
    if Hx is None: Hx = X.T @ X
    Hg = Hx
    if routed_idx is not None and w > 0 and len(routed_idx):
        Xr = X[routed_idx]; Hg = Hx / X.shape[0] + w * (Xr.T @ Xr) / len(routed_idx)
    Gq, Gc, Gs = gptq_t(G, Hg, bits); Uq, Uc, Us = gptq_t(U, Hg, bits)
    Xd = X if (routed_idx is None or w == 0 or not len(routed_idx)) else torch.cat([X, X[routed_idx].repeat(max(1, int(w)), 1)])
    hf = silu(Xd @ G.T) * (Xd @ U.T)
    if comp:
        hq = silu(Xd @ Gq.T) * (Xd @ Uq.T); Hqq = hq.T @ hq; Hfq = hf.T @ hq
        Hd = Hqq + 0.01 * torch.mean(torch.diag(Hqq)) * torch.eye(Hqq.shape[0], device=DEV)
        Dt = torch.linalg.solve(Hd, (D @ Hfq).T).T                       # D Hfq Hqq^-1
        Dq, Dc, Ds = gptq_t(Dt, Hqq, bits)
    else:
        Dq, Dc, Ds = gptq_t(D, hf.T @ hf, bits)
    return (Gq, Uq, Dq), ((Gc, Gs), (Uc, Us), (Dc, Ds))

def pack_fmt4(codes, scale):
    """codes int8 [O,I] in [-8,7] -> U8 [O*I/2] (low nibble = even column), scale [O,ng] -> f32 flat."""
    c = (codes.to(torch.int16) + 8).to(torch.uint8)
    b = c[:, 0::2] | (c[:, 1::2] << 4)
    return b.contiguous().cpu().numpy().reshape(-1), scale.float().contiguous().cpu().numpy().reshape(-1)

if __name__ == "__main__":
    import io, contextlib, runpy
    from collections import Counter
    from pathlib import Path
    S = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
    sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
    from exactmoe.storage import TensorIndex
    Xn = np.fromfile(S + r"\l3_in.f32", dtype=np.float32).reshape(-1, 6144)
    with contextlib.redirect_stdout(io.StringIO()):
        q = runpy.run_path(S + r"\colibri_int4_quality.py", run_name="lib")
        sp = runpy.run_path(S + r"\split_expert_experiment.py", run_name="lib_split")
    full = TensorIndex.from_directory(Path(r"C:\AI-Models\GLM-5.3-full"))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    routes = [[int(v) for v in q["route_noaux_tc"](x, q["router_w"], q["router_b"], q["config"]).expert_ids] for x in Xn]
    CALn, cal_r, hold_r = 600, routes[:600], routes[600:]
    X = torch.from_numpy(Xn).to(DEV); CAL, HOLD = X[:CALn], X[CALn:]
    Hx = CAL.T @ CAL
    chosen = [e for e, _ in Counter(e for r in hold_r for e in r).most_common(N)]
    def fwd(g, u, d, x): return (silu(x @ g.T) * (x @ u.T)) @ d.T
    t0 = time.time(); mats = []
    for e in chosen:
        p = f"model.layers.3.mlp.experts.{e}."
        mats.append([torch.from_numpy(np.ascontiguousarray(sp["dequant"](full[p + m + ".weight"], full[p + m + ".weight_scale_inv"]), dtype=np.float32)) for m in ("gate_proj", "up_proj", "down_proj")])
    Gs, Us, Ds = (torch.stack([m[i] for m in mats]).to(DEV) for i in range(3)); t1 = time.time()
    (Gq, Uq, Dq), packed = quantize_layer_batch(Gs, Us, Ds, CAL, Hx); torch.cuda.synchronize(); t2 = time.time()
    print(f"load {t1-t0:.0f}s, batched GPTQ of {len(chosen)} experts {t2-t1:.1f}s ({(t2-t1)/len(chosen):.2f}s/expert)")
    errs = []
    for b, e in enumerate(chosen):
        rows = [i for i, r in enumerate(hold_r) if e in r]; xh = HOLD[rows]; ref = fwd(Gs[b], Us[b], Ds[b], xh)
        errs += (torch.linalg.norm(fwd(Gq[b], Uq[b], Dq[b], xh) - ref, dim=1) / torch.linalg.norm(ref, dim=1)).cpu().numpy().tolist()
    w8, qs = pack_fmt4(*packed[0]); print("batched ao_mse err", round(float(np.mean(errs)), 4), "n", len(errs), "packed", w8.shape, qs.shape)
