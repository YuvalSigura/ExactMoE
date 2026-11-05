"""Memory-lean batched GPTQ (same math as gptq_gpu.gptq_batched / quantize_layer_batch).

* returns int8 codes + scales only (no fp32 dequantized copy); dequant() rebuilds W when a check needs it
* one permuted working copy of W (no extra clones); the per-Hessian factorization (act-order permutation,
  MSE-clip needs the diagonal only, upper Cholesky of H^-1) is computed once and reused -> gate/up of all
  256 experts of a layer share one factorization
* down_proj Hessians accumulate over calibration rows in chunks (the calibration never expands to [B, n, F])
Peak VRAM for 16 experts at hidden 6144 / ffn 2048: about 4-5 GB.
"""
import torch

QMAX = {4: 7, 3: 3}


def factor(H, damp=0.01):
    """H [B,I,I] -> (perm [B,I], Hinv_upper [B,I,I] in permuted order, diag [B,I] original order)."""
    H = H.float(); B, I, _ = H.shape
    dg = torch.diagonal(H, dim1=1, dim2=2).clone()
    dead = dg == 0
    if dead.any():
        H = H + torch.diag_embed(dead.float()); dg = torch.where(dead, torch.ones_like(dg), dg)
    perm = torch.argsort(dg, dim=1, descending=True)
    Hp = torch.gather(torch.gather(H, 1, perm.unsqueeze(2).expand(B, I, I)), 2, perm.unsqueeze(1).expand(B, I, I))
    Hp.diagonal(dim1=1, dim2=2).add_(damp * dg.mean(dim=1, keepdim=True))
    L = torch.linalg.cholesky(Hp); del Hp
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True); del L
    return perm, Hinv, dg, dead


def gptq_lean(W, fac, bits=4, gs=64, block=128):
    """W [B,O,I] (consumed: modified in place). fac from factor() with batch 1 (shared) or B."""
    perm, Hinv, dg, dead = fac
    B, O, I = W.shape; qmax = QMAX[bits]; ng = I // gs
    Bf = perm.shape[0]
    if dead.any():
        W.mul_((~dead).unsqueeze(1).float())
    # MSE clip search (weighted by the Hessian diagonal), in row chunks to bound temporaries
    scale = torch.empty((B, O, ng), device=W.device)
    d = dg.view(Bf, 1, ng, gs)
    for o0 in range(0, O, 4096):
        Wg = W[:, o0:o0 + 4096].view(B, -1, ng, gs); amax = Wg.abs().amax(dim=3)
        best, best_err = amax.clone(), torch.full_like(amax, float("inf"))
        for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
            s = torch.clamp(amax * c / qmax, min=1e-12).unsqueeze(3)
            e = (((torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s - Wg) ** 2) * d).sum(dim=3)
            better = e < best_err; best_err = torch.where(better, e, best_err); best = torch.where(better, amax * c, best)
        scale[:, o0:o0 + 4096] = torch.clamp(best / qmax, min=1e-12)
    pidx = perm.unsqueeze(1).expand(B, O, I) if Bf == B else perm.expand(B, I).unsqueeze(1).expand(B, O, I)
    W = torch.gather(W, 2, pidx)                                      # the one permuted working copy
    Hv = Hinv if Bf == B else Hinv.expand(B, I, I)
    C = torch.empty((B, O, I), dtype=torch.int8, device=W.device)
    gcol = (perm // gs)                                                # [Bf, I] group of each permuted column
    for i1 in range(0, I, block):
        i2 = min(i1 + block, I); W1 = W[:, :, i1:i2]; Hb = Hv[:, i1:i2, i1:i2]
        Err = torch.empty_like(W1)
        gb = gcol[:, i1:i2]
        Sb = torch.gather(scale, 2, (gb if Bf == B else gb.expand(B, -1)).unsqueeze(1).expand(B, O, i2 - i1))
        for j in range(i2 - i1):
            s = Sb[:, :, j]
            w = W1[:, :, j]
            code = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
            C[:, :, i1 + j] = code.to(torch.int8)
            err = (w - code * s) / Hb[:, j, j].unsqueeze(1)
            W1[:, :, j:] -= err.unsqueeze(2) * Hb[:, j, j:].unsqueeze(1); Err[:, :, j] = err
        if i2 < I:
            Hr = Hv[:, i1:i2, i2:]
            for o0 in range(0, O, 8192):              # row chunks: the update temporary stays small
                W[:, o0:o0 + 8192, i2:] -= torch.bmm(Err[:, o0:o0 + 8192], Hr)
    del W
    inv = torch.argsort(perm, dim=1)
    iidx = inv.unsqueeze(1).expand(B, O, I) if Bf == B else inv.expand(B, I).unsqueeze(1).expand(B, O, I)
    return torch.gather(C, 2, iidx), scale


def dequant(C, scale, gs=64):
    B, O, I = C.shape
    return (C.float().view(B, O, I // gs, gs) * scale.unsqueeze(3)).view(B, O, I)


def down_hessians(X, Gs, Us, chunk=2048):
    B, F, _ = Gs.shape
    H = torch.zeros((B, F, F), device=X.device)
    for r0 in range(0, X.shape[0], chunk):
        x = X[r0:r0 + chunk]
        h = torch.einsum("nh,bfh->bnf", x, Gs); h = h * torch.sigmoid(h) * torch.einsum("nh,bfh->bnf", x, Us)
        H.baddbmm_(h.transpose(1, 2), h); del h
    return H


def quantize_layer_lean(Gs, Us, Ds, X, gu_factor, bits=4):
    """gu_factor = factor(Hx[None]) for this layer (computed once). Returns ((Gc,Gs),(Uc,Us),(Dc,Ds))."""
    B, F, Hd = Gs.shape
    Hd_ = down_hessians(X, Gs, Us)
    Dc, Dsc = gptq_lean(Ds.clone(), factor(Hd_), bits); del Hd_
    GU = torch.cat([Gs, Us], dim=1).reshape(1, 2 * B * F, Hd)
    C, Sc = gptq_lean(GU, gu_factor, bits); del GU
    C = C.view(B, 2, F, Hd); Sc = Sc.view(B, 2, F, Hd // 64)
    return (C[:, 0], Sc[:, 0]), (C[:, 1], Sc[:, 1]), (Dc, Dsc)
