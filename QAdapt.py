"""
QAdapt: Information-Theoretic Adaptive Quantization for Hypergraph Neural Networks
==================================================================================
"""

import os, sys, time, math, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

BIT_CHOICES = [4, 8, 16]


# =============================================================================
# SECTION 0 - QUANTISATION PRIMITIVES (STE + learnable LSQ step size)
# =============================================================================

class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x
    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


def grad_scale(x, s):
    return _GradScale.apply(x, s)


def round_ste(x):
    """Round with straight-through estimator (dQ/dx = 1 on the active range)."""
    return (x.round() - x).detach() + x


def lsq_quantize(x: torch.Tensor, step: torch.Tensor, b: int) -> torch.Tensor:
    """
    Learned step-size quantisation (LSQ). The step-size gradient is scaled by
    1 / sqrt(|x| * Qmax) as in the paper's 'Learnable step sizes' paragraph.
    Symmetric signed grid with b bits.
    """
    qmax = 2 ** (b - 1) - 1
    qmin = -(2 ** (b - 1))
    gscale = 1.0 / math.sqrt(max(x.numel(), 1) * qmax + 1e-9)
    s = grad_scale(step.abs() + 1e-8, gscale)
    xq = round_ste(torch.clamp(x / s, qmin, qmax))
    return xq * s


# =============================================================================
# SECTION 1 - SPECTRAL / STRUCTURE UTILITIES
# =============================================================================

def compute_laplacian_eigenpairs(H: np.ndarray, W_e: np.ndarray,
                                 K: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Top-K eigenpairs of the normalised hypergraph Laplacian L = I - A_sym."""
    n, m = H.shape
    D_v = np.maximum((H * W_e[None, :]).sum(axis=1), 1e-8)
    D_e = np.maximum(H.sum(axis=0), 1e-8)
    D_v_invsqrt = sparse.diags(1.0 / np.sqrt(D_v))
    D_e_inv = sparse.diags(1.0 / D_e)
    W_e_diag = sparse.diags(W_e.astype(float))
    H_sp = sparse.csr_matrix(H.astype(float))
    A_sym = D_v_invsqrt @ H_sp @ W_e_diag @ D_e_inv @ H_sp.T @ D_v_invsqrt
    L = sparse.eye(n) - A_sym
    K_actual = max(1, min(K, n - 2))
    try:
        vals, vecs = eigsh(L, k=K_actual, which='SM', tol=1e-6)
    except Exception:
        Ld = L.toarray()
        vals, vecs = np.linalg.eigh(Ld)
        vals, vecs = vals[:K_actual], vecs[:, :K_actual]
    norms = np.linalg.norm(vecs, axis=0, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    order = np.argsort(np.abs(vals))
    return vals[order].astype(np.float32), vecs[:, order].astype(np.float32)


def build_cooccurrence_structure(H_np: np.ndarray):
    """
    Build the sparse co-occurrence support and all per-hyperedge index tables used by
    the O(|E| dbar_e^2) attention. Returns a dict of numpy/torch structures.
    """
    n, m = H_np.shape
    H_sp = sparse.csr_matrix(H_np)
    C = (H_sp @ H_sp.T).tocoo()
    mask = C.row != C.col
    rows = C.row[mask].astype(np.int64)
    cols = C.col[mask].astype(np.int64)
    E = len(rows)

    # directed-edge lookup (i,j) -> position
    edge_lookup = {}
    for pos in range(E):
        edge_lookup[(int(rows[pos]), int(cols[pos]))] = pos

    # reverse-edge index for symmetrisation S = 1/2 (M + M^T)
    rev = np.empty(E, dtype=np.int64)
    for pos in range(E):
        rev[pos] = edge_lookup.get((int(cols[pos]), int(rows[pos])), pos)

    # per-hyperedge member lists and the flat ordered-pair edge positions
    members = []
    pair_src = []   # flat member-local source index per hyperedge (list of np arrays)
    pair_dst = []
    pair_pos = []   # flat global edge position for each ordered member pair
    for e in range(m):
        ids = np.where(H_np[:, e] > 0)[0].astype(np.int64)
        members.append(ids)
        k = len(ids)
        if k < 2:
            pair_src.append(np.zeros(0, np.int64))
            pair_dst.append(np.zeros(0, np.int64))
            pair_pos.append(np.zeros(0, np.int64))
            continue
        si, di = np.meshgrid(np.arange(k), np.arange(k), indexing='ij')
        si = si.reshape(-1); di = di.reshape(-1)
        keep = si != di
        si, di = si[keep], di[keep]
        gi, gj = ids[si], ids[di]
        pos = np.array([edge_lookup[(int(a), int(b))] for a, b in zip(gi, gj)],
                       dtype=np.int64)
        pair_src.append(si); pair_dst.append(di); pair_pos.append(pos)

    return {
        'n': n, 'm': m, 'E': E,
        'rows': rows, 'cols': cols, 'rev': rev,
        'members': members, 'pair_src': pair_src, 'pair_dst': pair_dst,
        'pair_pos': pair_pos,
    }


def segment_softmax(logits: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Row-wise (per-source) softmax over a flat edge list. seg = source node id."""
    m = torch.full((n,), -1e30, device=logits.device)
    m = m.scatter_reduce(0, seg, logits, reduce='amax', include_self=True)
    z = torch.exp(logits - m[seg])
    denom = torch.zeros(n, device=logits.device).scatter_add(0, seg, z)
    return z / (denom[seg] + 1e-12)


# =============================================================================
# SECTION 2 - CRITIC + DEBIASED MOMENTUM-QUEUE MI ESTIMATOR (Step 1)
# =============================================================================

class CriticNetwork(nn.Module):
    """f_theta: [x_i ; h_e^ctx] in R^{2d} -> R   (2d->128->64->1)."""
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, x_i, h_e):
        return self.net(torch.cat([x_i, h_e], dim=-1)).squeeze(-1)


class InformationDensityEstimator(nn.Module):
    """
    rho_{i,e} = IC(x_i, h_e) * SW(i,e),  IC = exp(I_hat) > 0.

    I_hat uses the debiased, momentum-queue clipped-SMILE estimator (Eq. SMILE):
        I_hat(x_i; h_e) = f_theta(x_i,h_e)
                          - log( (1/M) sum_q clip(exp(f_theta'(x_i,h_q)), e^-C, e^C) )
    with an EMA encoder f_theta' (m=0.999) and a FIFO queue of M cached contexts.
    A plain InfoNCE term is returned separately as L_MI to train the online critic.
    SW = (sum_k softplus(a_k) phi_k(i)^2) 1_e(i)  >= 0   (non-negative spectral energy).
    """
    def __init__(self, d: int, K: int = 32, N_neg: int = 64,
                 queue_size: int = 256, momentum: float = 0.999, smile_C: float = 5.0):
        super().__init__()
        self.d, self.K, self.N_neg = d, K, N_neg
        self.M, self.mom, self.C = queue_size, momentum, smile_C
        self.W_ctx = nn.Linear(d, d, bias=False)
        self.critic = CriticNetwork(d)
        self.critic_ema = deepcopy(self.critic)
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)
        self.alpha_tilde = nn.Parameter(torch.zeros(K))   # gates a_k = softplus(.)
        self.register_buffer('queue', torch.zeros(queue_size, d))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_full', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _ema_update(self):
        for pe, p in zip(self.critic_ema.parameters(), self.critic.parameters()):
            pe.mul_(self.mom).add_(p.detach(), alpha=1 - self.mom)

    @torch.no_grad()
    def _enqueue(self, ctx: torch.Tensor):
        b = ctx.shape[0]
        ptr = int(self.queue_ptr.item())
        if b >= self.M:
            self.queue.copy_(ctx[-self.M:].detach())
            self.queue_ptr[0] = 0
            self.queue_full[0] = 1
            return
        end = ptr + b
        if end <= self.M:
            self.queue[ptr:end] = ctx.detach()
        else:
            first = self.M - ptr
            self.queue[ptr:] = ctx[:first].detach()
            self.queue[:end - self.M] = ctx[first:].detach()
        self.queue_ptr[0] = end % self.M
        if end >= self.M:
            self.queue_full[0] = 1

    def hyperedge_context(self, X, members):
        m = len(members)
        ctx = torch.zeros(m, self.d, device=X.device)
        Wx = self.W_ctx(X)
        for e, ids in enumerate(members):
            if len(ids):
                ctx[e] = Wx[ids].mean(dim=0)
        return ctx

    def forward(self, X, members, eigvecs):
        """Returns rho_ie (sparse dict: per-hyperedge value tensors), ctx, L_MI."""
        device = X.device
        ctx = self.hyperedge_context(X, members)                      # (m,d)
        a = F.softplus(self.alpha_tilde)                              # (K,) >=0
        node_energy = (eigvecs ** 2 * a.unsqueeze(0)).sum(-1)        # (n,) >=0  phi^2

        # queue sample for SMILE denominator
        if int(self.queue_full.item()) == 1:
            qbank = self.queue
        else:
            ptr = int(self.queue_ptr.item())
            qbank = self.queue[:max(ptr, 1)]
        qbank = qbank.detach()

        rho_ie = []                 # list aligned with members: value per member
        mi_pos, mi_neg_ls = [], []  # for InfoNCE L_MI
        all_ctx_for_queue = []
        for e, ids in enumerate(members):
            if len(ids) == 0:
                rho_ie.append(torch.zeros(0, device=device)); continue
            x_pos = X[ids]
            h_e = ctx[e].unsqueeze(0).expand(len(ids), -1)
            pos = self.critic(x_pos, h_e)                             # (k,)
            # ---- SMILE clipped momentum denominator (debiased) ----
            qs = min(self.M, qbank.shape[0])
            sel = qbank[torch.randint(0, qbank.shape[0], (qs,), device=device)]
            xe = x_pos.unsqueeze(1).expand(-1, qs, -1).reshape(-1, self.d)
            he = sel.unsqueeze(0).expand(len(ids), -1, -1).reshape(-1, self.d)
            with torch.no_grad():
                sc = self.critic_ema(xe, he).reshape(len(ids), qs)
            clip = torch.clamp(sc.exp(), math.exp(-self.C), math.exp(self.C))
            log_denom = torch.log(clip.mean(dim=1) + 1e-12)
            I_hat = pos - log_denom                                   # (k,)  debiased
            rho = torch.exp(I_hat) * node_energy[ids]                 # IC * SW  >=0
            rho_ie.append(rho)
            # ---- plain InfoNCE pieces for L_MI (train online critic) ----
            nneg = min(self.N_neg, ctx.shape[0] - 1)
            if nneg > 0:
                negc = ctx[torch.randperm(ctx.shape[0], device=device)[:nneg]].detach()
                xe2 = x_pos.unsqueeze(1).expand(-1, nneg, -1).reshape(-1, self.d)
                he2 = negc.unsqueeze(0).expand(len(ids), -1, -1).reshape(-1, self.d)
                negs = self.critic(xe2, he2).reshape(len(ids), nneg)
                mi_pos.append(pos)
                mi_neg_ls.append(torch.logsumexp(
                    torch.cat([pos.unsqueeze(1), negs], dim=1), dim=1) - math.log(nneg + 1))
            all_ctx_for_queue.append(ctx[e].unsqueeze(0))

        if mi_pos:
            L_MI = -(torch.cat(mi_pos) - torch.cat(mi_neg_ls)).mean()
        else:
            L_MI = torch.zeros((), device=device)

        if self.training:
            self._ema_update()
            if all_ctx_for_queue:
                self._enqueue(torch.cat(all_ctx_for_queue, dim=0))
        return rho_ie, ctx, L_MI


# =============================================================================
# SECTION 3 - QAdapt CONVOLUTION LAYER (sparse, factored, paper-aligned)
# =============================================================================

class QAdaptConv(nn.Module):
    def __init__(self, in_f, out_f, K=32, N_neg=64, dropout=0.5, omega_min=1e-3):
        super().__init__()
        self.in_f, self.out_f, self.K, self.omega_min = in_f, out_f, K, omega_min
        self.density = InformationDensityEstimator(in_f, K, N_neg)
        self.P_e = nn.Linear(in_f, in_f, bias=False)     # W_P (intra-hyperedge)
        self.W_Q = nn.Linear(in_f, in_f, bias=False)     # co-occurrence query
        self.W_K = nn.Linear(in_f, in_f, bias=False)     # co-occurrence key
        self.zeta_tilde = nn.Parameter(torch.tensor(0.0))     # zeta = softplus(.) >=0
        self.omega_tilde = nn.Parameter(torch.zeros(K))       # omega = softplus(.) >=0
        self.gamma_tilde = nn.Parameter(torch.tensor(0.0))    # gamma = sigmoid(.)
        self.gamma_struct = nn.Parameter(torch.ones(K) / K)   # Struct(i,j) weights
        self.Theta = nn.Linear(in_f, out_f, bias=True)
        self.W_res = nn.Linear(in_f, in_f, bias=False)
        self.dropout = nn.Dropout(dropout)
        # allocator over per-nonzero features [Sens ; rho_ij ; Struct] -> {4,8,16}
        self.allocator = nn.Sequential(
            nn.Linear(3, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, len(BIT_CHOICES)))
        self.lsq_S = nn.Parameter(torch.ones(len(BIT_CHOICES)))   # step sizes (edges)
        self.lsq_P = nn.Parameter(torch.ones(len(BIT_CHOICES)))   # step sizes (Phi cols)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.struct = None  # filled by set_structure

    def set_structure(self, struct, eigvecs, eigvals, degrees, device):
        self.struct = struct
        self.eigvecs = torch.as_tensor(eigvecs, dtype=torch.float32, device=device)
        self.eigvals = torch.as_tensor(eigvals, dtype=torch.float32, device=device)
        self.degrees = torch.as_tensor(degrees, dtype=torch.float32, device=device)
        s = struct
        self.e_rows = torch.as_tensor(s['rows'], device=device)
        self.e_cols = torch.as_tensor(s['cols'], device=device)
        self.e_rev = torch.as_tensor(s['rev'], device=device)
        self.pair_pos = [torch.as_tensor(p, device=device) for p in s['pair_pos']]
        self.pair_src = [torch.as_tensor(p, device=device) for p in s['pair_src']]
        self.pair_dst = [torch.as_tensor(p, device=device) for p in s['pair_dst']]

    def tau(self):
        return max(0.1, 2.0 * (0.95 ** (self.step.item() / 100.0)))

    def gumbel_softmax(self, logits, hard):
        g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y = F.softmax((logits + g) / self.tau(), dim=-1)
        if hard:
            idx = y.argmax(-1, keepdim=True)
            yh = torch.zeros_like(y).scatter_(-1, idx, 1.0)
            return yh - y.detach() + y
        return y

    # ---- sparse operator application (never densifies N x N) ----
    def _spmm(self, Sval, X):
        """S @ X on the directed co-occurrence edge list."""
        n = X.shape[0]
        out = torch.zeros_like(X)
        msg = Sval.unsqueeze(1) * X[self.e_cols]
        return out.index_add(0, self.e_rows, msg)

    def _core_B(self, Sval, Phi):
        """K x K core  B = Phi^T S Phi."""
        SPhi = torch.zeros_like(Phi)
        msg = Sval.unsqueeze(1) * Phi[self.e_cols]
        SPhi = SPhi.index_add(0, self.e_rows, msg)
        return Phi.t() @ SPhi

    def _apply_T(self, Sval, X, Phi, omega, gamma):
        """T X = (1-gamma) S X + gamma Phi Om^1/2 (B (Om^1/2 (Phi^T X))) + X W_res."""
        B = self._core_B(Sval, Phi)                      # (K,K)
        om_half = torch.sqrt(omega + 1e-12)
        glob = Phi @ (om_half.unsqueeze(1) *
                      (B @ (om_half.unsqueeze(1) * (Phi.t() @ X))))
        loc = self._spmm(Sval, X)
        return (1 - gamma) * loc + gamma * glob + self.W_res(X)

    def _attention(self, X, rho_ie):
        """Build the symmetrised sparse operator S_val and the pair density rho_ij."""
        s = self.struct
        n, E = s['n'], s['E']
        device = X.device
        eps = 1e-8
        zeta = F.softplus(self.zeta_tilde)

        Xp = self.P_e(X)
        A_hyper = torch.zeros(E, device=device)
        rho_pair = torch.zeros(E, device=device)
        # ---- intra-hyperedge attention (per-hyperedge k x k softmax) + rho_ij ----
        for e, ids in enumerate(s['members']):
            k = len(ids)
            if k < 2:
                continue
            xi = Xp[ids]
            scores = (xi @ xi.t()) / math.sqrt(self.in_f)             # (k,k)
            r = rho_ie[e].clamp(min=eps)
            scores = scores + zeta * torch.log(r).unsqueeze(1)        # +zeta log rho_{i,e}
            scores = scores - torch.eye(k, device=device) * 1e9       # mask self
            attn = F.softmax(scores, dim=-1)
            src, dst, pos = self.pair_src[e], self.pair_dst[e], self.pair_pos[e]
            A_hyper = A_hyper.index_add(0, pos, attn[src, dst])
            half = 0.5 * (r[src] + r[dst])
            rho_pair = rho_pair.index_add(0, pos, half)

        # ---- sparse co-occurrence (node-level) attention ----
        Q = self.W_Q(X); Kk = self.W_K(X)
        sc_node = (Q[self.e_rows] * Kk[self.e_cols]).sum(-1) / math.sqrt(self.in_f)
        sc_node = sc_node + zeta * torch.log(rho_pair.clamp(min=eps))
        A_node = segment_softmax(sc_node, self.e_rows, n)

        M = A_hyper + A_node
        S_val = 0.5 * (M + M[self.e_rev])                            # S = 1/2 (M+M^T)
        return S_val, rho_pair

    def _struct_term(self, Phi):
        """Struct(i,j) = sum_k g_k phi_k(i) phi_k(j) on edges."""
        return (self.gamma_struct.unsqueeze(0) *
                Phi[self.e_rows] * Phi[self.e_cols]).sum(-1)

    def _quantize_phi(self, Phi, omega):
        """Spectral-basis quantisation: per-column bits by omega, prune omega<omega_min."""
        # bit per column: larger omega -> more bits
        with torch.no_grad():
            w = omega.detach()
            th1, th2 = torch.quantile(w, 0.5), torch.quantile(w, 0.85)
            col_bits = torch.where(w >= th2, 16, torch.where(w >= th1, 8, 4))
            keep = (w >= self.omega_min).float()
        Phi_q = torch.zeros_like(Phi)
        for bi, b in enumerate(BIT_CHOICES):
            colmask = (col_bits == b).float().unsqueeze(0)
            Phi_q = Phi_q + colmask * lsq_quantize(Phi, self.lsq_P[bi], b)
        Phi_q = Phi_q * keep.unsqueeze(0)
        mean_bits_phi = (col_bits.float() * keep).sum() / (keep.sum() + 1e-8)
        return Phi_q, mean_bits_phi

    def forward(self, X, quantize=True):
        device = X.device
        Phi = self.eigvecs
        omega = F.softplus(self.omega_tilde)
        gamma = torch.sigmoid(self.gamma_tilde)

        # Step 1 (shared by teacher & student)
        rho_ie, _, L_MI = self.density(X, self.struct['members'], Phi)
        # Step 2: attention -> sparse operator S and pair density
        S_val, rho_edge = self._attention(X, rho_ie)

        if not quantize:
            # ---- full-precision T-backbone (teacher) ----
            TX = self._apply_T(S_val, X, Phi, omega, gamma)
            out = self.dropout(self.Theta(TX))
            return {'output': out, 'S_val': S_val, 'rho_edge': rho_edge, 'L_MI': L_MI}

        # Step 3: LABEL-FREE embedding-Jacobian (Hutchinson) saliency on S_val.
        # enable_grad so it also works when the outer call is under torch.no_grad().
        with torch.enable_grad():
            Sdet = S_val.detach().clone().requires_grad_(True)
            TXs = self._apply_T(Sdet, X.detach(), Phi.detach(),
                                omega.detach(), gamma.detach())
            v = torch.randn_like(TXs)
            g = torch.autograd.grad((v * TXs).sum(), Sdet)[0]
        sens = g.detach() ** 2
        # allocator features: [Sens ; rho_ij ; Struct]
        struct_e = self._struct_term(Phi).detach()
        feat = torch.stack([sens, rho_edge.detach(), struct_e], dim=-1)
        feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
        logits = self.allocator(feat)
        if self.training and self.step.item() < 200:
            beta = self.gumbel_softmax(logits, hard=False)          # soft (E,3)
        else:
            idx = logits.argmax(-1, keepdim=True)                   # hard / eval
            beta = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        exp_bits = (beta * torch.tensor([float(b) for b in BIT_CHOICES],
                                        device=device)).sum(-1)

        # quantise sparse operator entries (per-nonzero mixed precision),
        # then re-symmetrise so S_q stays a symmetric operator.
        S_q = torch.zeros_like(S_val)
        for bi, b in enumerate(BIT_CHOICES):
            S_q = S_q + beta[:, bi] * lsq_quantize(S_val, self.lsq_S[bi], b)
        S_q = 0.5 * (S_q + S_q[self.e_rev])

        # quantise spectral basis Phi
        Phi_q, phi_bits = self._quantize_phi(Phi, omega)

        # cores for spectral-preservation loss (K x K)
        B = self._core_B(S_val, Phi)
        B_q = self._core_B(S_q, Phi_q)

        # ---- quantised student forward ----
        TXq = self._apply_T(S_q, X, Phi_q, omega, gamma)
        out = self.dropout(self.Theta(TXq))

        self.step += 1
        return {
            'output': out, 'S_val': S_val, 'S_q': S_q, 'rho_edge': rho_edge,
            'exp_bits': exp_bits, 'phi_bits': phi_bits, 'B': B, 'B_q': B_q,
            'L_MI': L_MI,
        }


# =============================================================================
# SECTION 4 - FULL QAdapt NETWORK (+ self-distillation)
# =============================================================================

class QAdaptNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5,
                 K=32, N_neg=64, lambda1=0.1, lambda2=0.05, lambda3=0.01, lambda4=0.1,
                 b_target_S=5.94, b_target_P=8.0, kd_alpha_f=1.0, kd_alpha_l=1.0,
                 kd_tau=2.0):
        super().__init__()
        self.l1, self.l2, self.l3, self.l4 = lambda1, lambda2, lambda3, lambda4
        self.bS, self.bP = b_target_S, b_target_P
        self.kd_af, self.kd_al, self.kd_tau = kd_alpha_f, kd_alpha_l, kd_tau
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList([
            QAdaptConv(hidden_dim, hidden_dim, K=K, N_neg=N_neg, dropout=dropout)
            for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def set_structure(self, struct, eigvals, eigvecs, degrees, device):
        for layer in self.layers:
            layer.set_structure(struct, eigvecs, eigvals, degrees, device)

    def _run(self, X, quantize):
        X = self.input_proj(X)
        louts, feats = [], []
        for layer in self.layers:
            res = layer(X, quantize=quantize)
            X = F.relu(res['output'])
            louts.append(res); feats.append(X)
        return self.output_layer(X), louts, feats

    def forward(self, X, H_np=None, W_e=None):
        # student (quantised)
        s_logits, s_louts, s_feats = self._run(X, quantize=True)
        # teacher (full precision) - detached self-distillation target
        with torch.no_grad():
            t_logits, _, t_feats = self._run(X, quantize=False)
        return {'logits': s_logits, 'layer_outputs': s_louts, 'feats': s_feats,
                'teacher_logits': t_logits.detach(),
                'teacher_feats': [f.detach() for f in t_feats]}

    def compute_loss(self, out, labels, mask, task):
        logits = out['logits']
        if task == 'classification':
            L_task = F.cross_entropy(logits[mask], labels[mask])
        else:
            L_task = F.mse_loss(logits[mask].squeeze(-1), labels[mask].float())

        louts = out['layer_outputs']
        # l1: rho-weighted quantisation error  sum rho_ij (S-S~)^2
        L1 = torch.stack([(lo['rho_edge'] * (lo['S_val'] - lo['S_q']) ** 2).mean()
                          for lo in louts]).mean()
        # l2: spectral preservation  ||Lambda - Lambda~||^2  on K x K cores
        L2 = torch.stack([(torch.linalg.eigvalsh(lo['B']) -
                           torch.linalg.eigvalsh(lo['B_q'])).pow(2).mean()
                          for lo in louts]).mean()
        # l3: budget on S and Phi
        L3 = torch.stack([(lo['exp_bits'].mean() - self.bS) ** 2 +
                          (lo['phi_bits'] - self.bP) ** 2 for lo in louts]).mean()
        # l4: MI contrastive term
        L4 = torch.stack([lo['L_MI'] for lo in louts]).mean()
        # KD: self-distillation from the full-precision backbone
        tau = self.kd_tau
        L_feat = torch.stack([F.mse_loss(sf, tf) for sf, tf in
                              zip(out['feats'], out['teacher_feats'])]).mean()
        if task == 'classification':
            L_kl = F.kl_div(F.log_softmax(logits / tau, -1),
                            F.softmax(out['teacher_logits'] / tau, -1),
                            reduction='batchmean') * tau * tau
        else:
            L_kl = F.mse_loss(logits, out['teacher_logits'])
        L_KD = self.kd_af * L_feat + self.kd_al * L_kl

        total = (L_task + L_KD + self.l1 * L1 + self.l2 * L2 +
                 self.l3 * L3 + self.l4 * L4)
        return {'total': total, 'task': L_task, 'kd': L_KD,
                'l1': L1, 'l2': L2, 'l3': L3, 'mi': L4}


# =============================================================================
# SECTION 5 - BASELINE HGNN (sparse)
# =============================================================================

class BaselineHGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, H_np, W_e):
        D_v = np.maximum((H_np * W_e[None, :]).sum(axis=1), 1e-8)
        D_e = np.maximum(H_np.sum(axis=0), 1e-8)
        Dv_isq = torch.FloatTensor(1.0 / np.sqrt(D_v)).to(X.device).unsqueeze(1)
        De_inv = torch.FloatTensor(1.0 / D_e).to(X.device)
        W_et = torch.FloatTensor(W_e).to(X.device)
        H_t = torch.FloatTensor(H_np).to(X.device)
        h1 = Dv_isq * X
        h2 = H_t.t() @ h1
        h3 = (W_et * De_inv).unsqueeze(1) * h2
        h4 = Dv_isq * (H_t @ h3)
        h = self.encoder(h4)
        return {'logits': self.output_layer(h), 'layer_outputs': []}


# =============================================================================
# SECTION 6 - METRICS
# =============================================================================

def information_retention_score(S_val, S_q) -> float:
    denom = torch.norm(S_val.detach(), p='fro').item() ** 2
    if denom < 1e-12:
        return 1.0
    num = torch.norm(S_val.detach() - S_q.detach(), p='fro').item() ** 2
    return float(np.exp(-num / denom))


def spectral_preservation_score(B, B_q) -> float:
    try:
        lo = torch.linalg.eigvalsh(B.detach()).cpu().numpy()
        lq = torch.linalg.eigvalsh(B_q.detach()).cpu().numpy()
        lo = np.sort(np.abs(lo))[::-1]; lq = np.sort(np.abs(lq))[::-1]
        d = np.linalg.norm(lo)
        if d < 1e-12:
            return 1.0
        return float(np.clip(1.0 - np.linalg.norm(lo - lq) / d, 0, 1))
    except Exception:
        return 0.0


def measure_inference_time(model, X, H_np, W_e, n_runs=20, warmup=5) -> float:
    model.eval(); device = next(model.parameters()).device; X = X.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(X, H_np, W_e)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(X, H_np, W_e)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def evaluate(model, X, H_np, W_e, labels, mask, task='classification', measure_time=True):
    model.eval(); device = next(model.parameters()).device
    X = X.to(device); labels = labels.to(device)
    with torch.no_grad():
        out = model(X, H_np, W_e)
        logits = out['logits']; louts = out['layer_outputs']
        ir, sp, bits = [], [], []
        for lo in louts:
            if 'S_q' in lo:
                ir.append(information_retention_score(lo['S_val'], lo['S_q']))
                sp.append(spectral_preservation_score(lo['B'], lo['B_q']))
                bits.append(lo['exp_bits'].mean().item())
        comp = 32.0 / np.mean(bits) if bits else 1.0
        avg_ir = float(np.mean(ir)) if ir else -1.0
        avg_sp = float(np.mean(sp)) if sp else -1.0
        if task == 'classification':
            probs = F.softmax(logits[mask], dim=-1).cpu().numpy()
            preds = probs.argmax(1); true = labels[mask].cpu().numpy()
            res = {'acc': accuracy_score(true, preds),
                   'f1': f1_score(true, preds, average='macro', zero_division=0)}
            try:
                res['auc'] = (roc_auc_score(true, probs, multi_class='ovr', average='macro')
                              if probs.shape[1] > 2 else roc_auc_score(true, probs[:, 1]))
            except ValueError:
                res['auc'] = 0.0
        else:
            preds = logits[mask].squeeze(-1).cpu().numpy()
            true = labels[mask].float().cpu().numpy()
            res = {'mae': mean_absolute_error(true, preds),
                   'rmse': float(np.sqrt(mean_squared_error(true, preds))),
                   'r2': r2_score(true, preds)}
    res.update({'comp_ratio': comp, 'info_retain': avg_ir, 'spec_pres': avg_sp})
    res['time_ms'] = measure_inference_time(model, X, H_np, W_e) if measure_time else 0.0
    return res


def statistical_summary(qa, bl, task):
    metrics = (['acc', 'f1', 'auc'] if task == 'classification' else ['mae', 'rmse', 'r2'])
    metrics += ['comp_ratio', 'info_retain', 'spec_pres', 'time_ms']
    summary = {}
    for m in metrics:
        q = np.array([r[m] for r in qa if m in r])
        b = np.array([r[m] for r in bl if m in r])
        if not len(q):
            continue
        t, p = ttest_rel(q, b) if (len(q) > 1 and len(b) == len(q)) else (0.0, 1.0)
        diffs = q - b if len(b) == len(q) else q - q.mean()
        cohen = diffs.mean() / (diffs.std() + 1e-10) if len(diffs) > 1 else 0.0
        summary[m] = {'qadapt_mean': float(q.mean()), 'qadapt_std': float(q.std()),
                      'baseline_mean': float(b.mean()) if len(b) else 0.0,
                      'baseline_std': float(b.std()) if len(b) else 0.0,
                      't_stat': float(t), 'p_value': float(p),
                      'cohen_d': float(cohen), 'significant': bool(p < 0.01)}
    return summary


def print_summary_table(summary, task):
    print("\n" + "=" * 100)
    print("  RESULTS SUMMARY  |  5-Fold CV  |  p < 0.01")
    print("=" * 100)
    lm = {'acc': 'Accuracy', 'f1': 'F1 (macro)', 'auc': 'AUC (macro)', 'mae': 'MAE',
          'rmse': 'RMSE', 'r2': 'R2', 'comp_ratio': 'Comp. Ratio',
          'info_retain': 'Info Retain', 'spec_pres': 'Spec Pres', 'time_ms': 'Time (ms)'}
    lower = {'mae', 'rmse', 'time_ms'}
    print(f"{'Metric':<14}{'QAdapt':>22}{'Baseline':>22}{'Impr':>12}{'p':>9}{'Sig':>5}")
    print("-" * 100)
    for k, lab in lm.items():
        if k not in summary:
            continue
        s = summary[k]
        imp = (s['baseline_mean'] - s['qadapt_mean']) if k in lower \
            else (s['qadapt_mean'] - s['baseline_mean'])
        sig = "***" if s['p_value'] < 0.001 else "**" if s['p_value'] < 0.01 else \
              "*" if s['p_value'] < 0.05 else ""
        print(f"{lab:<14}{s['qadapt_mean']:>11.4f}+/-{s['qadapt_std']:.4f} "
              f"{s['baseline_mean']:>11.4f}+/-{s['baseline_std']:.4f} "
              f"{imp:>+11.4f}{s['p_value']:>9.4f}{sig:>5}")
    print("=" * 100)


# =============================================================================
# SECTION 7 - DATA LOADING (unchanged from the original runner)
# =============================================================================

def _read_file(folder, candidates, cols, max_rows=None):
    for fname in candidates:
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(path, usecols=lambda c: c in cols, nrows=max_rows)
            else:
                df = pd.read_excel(path, usecols=lambda c: c in cols, nrows=max_rows)
            df = df[[c for c in cols if c in df.columns]]
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            print(f"  Loaded {len(df):>6} rows  <-  {fname}")
            return df
        except Exception as e:
            print(f"  [warn] Could not read {fname}: {e}")
    return pd.DataFrame(columns=cols)


def _build_hypergraph(hyperedges, entity_types, feature_dim, seed=42):
    all_ents = sorted({e for edge in hyperedges for e in edge})
    e2idx = {e: i for i, e in enumerate(all_ents)}
    n, m = len(all_ents), len(hyperedges)
    H = np.zeros((n, m), dtype=np.float32)
    for eid, members in enumerate(hyperedges):
        for ent in members:
            if ent in e2idx:
                H[e2idx[ent], eid] = 1.0
    rng = np.random.default_rng(seed)
    unique_types = list(set(entity_types.values()))
    type_embeds = {t: rng.standard_normal(feature_dim) for t in unique_types}
    X = rng.standard_normal((n, feature_dim)).astype(np.float32)
    for i, ent in enumerate(all_ents):
        t = entity_types.get(ent, 'unknown')
        X[i] += 0.5 * type_embeds.get(t, np.zeros(feature_dim))
    return H, X, all_ents, e2idx


def _safe_int(val):
    try:
        return int(float(val))
    except Exception:
        return str(val)


def load_imdb(folder, max_rows=None, feature_dim=64):
    df_um = _read_file(folder, ['user_movies.xlsx', 'user_movies.csv'],
                       ['userID', 'movieID', 'rating'], max_rows)
    df_md = _read_file(folder, ['movie_directors.xlsx', 'movie_directors.csv'],
                       ['movieID', 'directorID'], max_rows)
    df_ma = _read_file(folder, ['movie_actors.xlsx', 'movie_actors.csv'],
                       ['movieID', 'actorID'], max_rows)
    df_mg = _read_file(folder, ['movie_genres.xlsx', 'movie_genres.csv'],
                       ['movieID', 'genreID'], max_rows)
    etype, hedges, genre_map = {}, [], {}
    def add(a, at, b, bt):
        etype[a] = at; etype[b] = bt; hedges.append([a, b])
    for _, r in df_um.iterrows():
        add(f"user_{_safe_int(r['userID'])}", "user", f"movie_{_safe_int(r['movieID'])}", "movie")
    for _, r in df_md.iterrows():
        add(f"movie_{_safe_int(r['movieID'])}", "movie", f"director_{_safe_int(r['directorID'])}", "director")
    for _, r in df_ma.iterrows():
        add(f"movie_{_safe_int(r['movieID'])}", "movie", f"actor_{_safe_int(r['actorID'])}", "actor")
    uniq_g = sorted(df_mg['genreID'].dropna().unique().tolist())
    g2idx = {g: i for i, g in enumerate(uniq_g)}
    for _, r in df_mg.iterrows():
        mid, gid = _safe_int(r['movieID']), _safe_int(r['genreID'])
        add(f"movie_{mid}", "movie", f"genre_{gid}", "genre")
        if mid not in genre_map and gid in g2idx:
            genre_map[mid] = g2idx[gid]
    if not hedges:
        return make_synthetic(task='classification')
    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim)
    n = len(ents); labels = np.full(n, -1, dtype=np.int64)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'movie':
            try:
                mid = _safe_int(ent.split('_')[1])
                if mid in genre_map:
                    labels[i] = genre_map[mid]
            except Exception:
                pass
    nc = len(uniq_g) if uniq_g else 1
    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges, {nc} classes [clf]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, nc


def load_dblp(folder, max_rows=None, feature_dim=64):
    df_pa = _read_file(folder, ['paper_author.xlsx', 'paper_author.csv'], ['paperID', 'authorID'], max_rows)
    df_pc = _read_file(folder, ['paper_conf.xlsx', 'paper_conf.csv'], ['paperID', 'confID'], max_rows)
    etype, hedges, label_map = {}, [], {}
    for _, r in df_pa.iterrows():
        p, a = f"paper_{_safe_int(r['paperID'])}", f"author_{_safe_int(r['authorID'])}"
        etype[p] = 'paper'; etype[a] = 'author'; hedges.append([p, a])
    uniq_c = sorted(df_pc['confID'].dropna().unique().tolist()); c2idx = {c: i for i, c in enumerate(uniq_c)}
    for _, r in df_pc.iterrows():
        pid, cid = _safe_int(r['paperID']), _safe_int(r['confID'])
        p, c = f"paper_{pid}", f"conf_{cid}"; etype[p] = 'paper'; etype[c] = 'conf'; hedges.append([p, c])
        if pid not in label_map and cid in c2idx:
            label_map[pid] = c2idx[cid]
    if not hedges:
        return make_synthetic(task='classification')
    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim); n = len(ents)
    labels = np.full(n, -1, dtype=np.int64)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'paper':
            try:
                pid = _safe_int(ent.split('_')[1])
                if pid in label_map:
                    labels[i] = label_map[pid]
            except Exception:
                pass
    nc = max(len(uniq_c), 1)
    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges, {nc} classes [clf]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, nc


def load_acm(folder, max_rows=None, feature_dim=64):
    df_pa = _read_file(folder, ['paper_author.xlsx', 'paper_author.csv'], ['paperID', 'authorID'], max_rows)
    df_ps = _read_file(folder, ['paper_subject.xlsx', 'paper_subject.csv'], ['paperID', 'subjectID'], max_rows)
    etype, hedges, label_map = {}, [], {}
    for _, r in df_pa.iterrows():
        p, a = f"paper_{_safe_int(r['paperID'])}", f"author_{_safe_int(r['authorID'])}"
        etype[p] = 'paper'; etype[a] = 'author'; hedges.append([p, a])
    uniq_s = sorted(df_ps['subjectID'].dropna().unique().tolist()); s2idx = {s: i for i, s in enumerate(uniq_s)}
    for _, r in df_ps.iterrows():
        pid, sid = _safe_int(r['paperID']), _safe_int(r['subjectID'])
        p, s = f"paper_{pid}", f"subject_{sid}"; etype[p] = 'paper'; etype[s] = 'subject'; hedges.append([p, s])
        if pid not in label_map and sid in s2idx:
            label_map[pid] = s2idx[sid]
    if not hedges:
        return make_synthetic(task='classification')
    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim); n = len(ents)
    labels = np.full(n, -1, dtype=np.int64)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'paper':
            try:
                pid = _safe_int(ent.split('_')[1])
                if pid in label_map:
                    labels[i] = label_map[pid]
            except Exception:
                pass
    nc = max(len(uniq_s), 1)
    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges, {nc} classes [clf]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, nc


def load_amazon(folder, max_rows=None, feature_dim=64):
    df = _read_file(folder, ['user_product.xlsx', 'user_product.csv', 'ratings.xlsx', 'ratings.csv'],
                    ['userID', 'productID', 'rating'], max_rows)
    if df.empty:
        return make_synthetic(task='regression')
    etype, hedges, rating_map = {}, [], {}
    for _, r in df.iterrows():
        u, p = f"user_{_safe_int(r['userID'])}", f"product_{_safe_int(r['productID'])}"
        etype[u] = 'user'; etype[p] = 'product'; hedges.append([u, p])
        pid = _safe_int(r['productID'])
        if pid not in rating_map:
            rating_map[pid] = float(r.get('rating', 0.0))
    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim); n = len(ents)
    labels = np.zeros(n, dtype=np.float32)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'product':
            try:
                pid = _safe_int(ent.split('_')[1]); labels[i] = rating_map.get(pid, 0.0)
            except Exception:
                pass
    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges [reg]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, 1


def load_yelp(folder, max_rows=None, feature_dim=64):
    df = _read_file(folder, ['user_business.xlsx', 'user_business.csv', 'reviews.xlsx', 'reviews.csv'],
                    ['userID', 'businessID', 'rating'], max_rows)
    if df.empty:
        return make_synthetic(task='regression')
    etype, hedges, rating_map = {}, [], {}
    for _, r in df.iterrows():
        u, b = f"user_{_safe_int(r['userID'])}", f"business_{_safe_int(r['businessID'])}"
        etype[u] = 'user'; etype[b] = 'business'; hedges.append([u, b])
        bid = _safe_int(r['businessID'])
        if bid not in rating_map:
            rating_map[bid] = float(r.get('rating', 0.0))
    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim); n = len(ents)
    labels = np.zeros(n, dtype=np.float32)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'business':
            try:
                bid = _safe_int(ent.split('_')[1]); labels[i] = rating_map.get(bid, 0.0)
            except Exception:
                pass
    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges [reg]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, 1


def make_synthetic(n_nodes=500, n_he=200, feat_dim=64, n_classes=5,
                   task='classification', seed=42):
    rng = np.random.default_rng(seed)
    H = np.zeros((n_nodes, n_he), dtype=np.float32)
    for e, s in enumerate(rng.integers(2, 10, size=n_he)):
        H[rng.choice(n_nodes, s, replace=False), e] = 1.0
    X = rng.standard_normal((n_nodes, feat_dim)).astype(np.float32)
    W_e = np.ones(n_he, dtype=np.float32)
    if task == 'classification':
        lbl = rng.integers(0, n_classes, n_nodes).astype(np.int64); nc = n_classes
    else:
        lbl = rng.standard_normal(n_nodes).astype(np.float32); nc = 1
    print(f"  Synthetic: {n_nodes} nodes, {n_he} hyperedges [{task}]")
    return H, X, W_e, lbl, nc


LOADERS = {'imdb': load_imdb, 'dblp': load_dblp, 'acm': load_acm,
           'amazon': load_amazon, 'yelp': load_yelp}


def load_dataset(name, data_path=None, max_rows=None, feature_dim=64):
    loader = LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset '{name}'.")
    if data_path and os.path.isdir(data_path):
        return loader(data_path, max_rows=max_rows, feature_dim=feature_dim)
    return make_synthetic(task=DATASET_CONFIGS[name]['task'], feat_dim=feature_dim)


# =============================================================================
# SECTION 8 - TRAINING
# =============================================================================

def train_model(model, X, H_np, W_e, labels, train_mask, val_mask, test_mask,
                task='classification', num_epochs=200, lr=0.005, weight_decay=5e-4,
                patience=30, model_type='qadapt', verbose=True, mi_interval=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device); X = X.to(device)
    labels_t = (torch.LongTensor(labels) if task == 'classification'
                else torch.FloatTensor(labels)).to(device)

    if model_type == 'qadapt':
        if verbose:
            print("    Precomputing eigenpairs + co-occurrence structure...")
        K = max(1, min(32, H_np.shape[0] - 2))
        ev, evec = compute_laplacian_eigenpairs(H_np, W_e, K=K)
        degs = H_np.sum(axis=1).astype(np.float32)
        struct = build_cooccurrence_structure(H_np)
        model.set_structure(struct, ev, evec, degs, device)

    mi_params = [p for n, p in model.named_parameters() if 'density.critic' in n and 'ema' not in n]
    main_params = [p for n, p in model.named_parameters()
                   if not ('density.critic' in n and 'ema' not in n)]
    opt_main = torch.optim.Adam(main_params, lr=lr, weight_decay=weight_decay)
    opt_mi = torch.optim.Adam(mi_params, lr=lr * 0.5) if mi_params else None

    best_val, best_state, pc = -np.inf, None, 0
    for epoch in range(num_epochs):
        model.train(); opt_main.zero_grad()
        if opt_mi:
            opt_mi.zero_grad()
        out = model(X, H_np, W_e)
        if model_type == 'qadapt':
            losses = model.compute_loss(out, labels_t, train_mask, task)
            loss = losses['total']
        else:
            loss = (F.cross_entropy(out['logits'][train_mask], labels_t[train_mask])
                    if task == 'classification'
                    else F.mse_loss(out['logits'][train_mask].squeeze(-1), labels_t[train_mask]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_main.step()
        if opt_mi and epoch % mi_interval == 0:
            opt_mi.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                vl = model(X, H_np, W_e)['logits']
                vm = ((vl[val_mask].argmax(1) == labels_t[val_mask]).float().mean().item()
                      if task == 'classification'
                      else -F.mse_loss(vl[val_mask].squeeze(-1), labels_t[val_mask]).item())
            if vm > best_val:
                best_val, best_state, pc = vm, deepcopy(model.state_dict()), 0
            else:
                pc += 1
            if verbose and epoch % 20 == 0:
                if model_type == 'qadapt':
                    print(f"    Ep{epoch:03d} loss={loss.item():.4f} "
                          f"(task={losses['task'].item():.3f} kd={losses['kd'].item():.3f} "
                          f"l1={losses['l1'].item():.3f} l2={losses['l2'].item():.3f} "
                          f"mi={losses['mi'].item():.3f}) val={vm:.4f}")
                else:
                    print(f"    Ep{epoch:03d} loss={loss.item():.4f} val={vm:.4f}")
            if pc >= patience:
                if verbose:
                    print(f"    Early stop @ {epoch}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return evaluate(model, X, H_np, W_e, labels_t, test_mask, task=task)


def run_five_fold_cv(H, X, W_e, labels, valid_idx, task, feat_dim, hidden_dim,
                     output_dim, cfg, n_splits=5):
    X_t = torch.FloatTensor(X)
    if task == 'classification':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(valid_idx, labels[valid_idx]))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(valid_idx))
    qa, bl = [], []
    for fold, (tr, te) in enumerate(splits):
        print(f"\n  Fold {fold + 1}/{n_splits}")
        trval = valid_idx[tr]; test_ids = valid_idx[te]
        vs = max(1, len(trval) // 5)
        val_ids, train_ids = trval[:vs], trval[vs:]
        def mk(ids):
            m = torch.zeros(len(labels), dtype=torch.bool); m[ids] = True; return m
        qm = QAdaptNet(feat_dim, hidden_dim, output_dim, num_layers=cfg['num_layers'],
                       dropout=cfg['dropout'], K=cfg['K'], N_neg=cfg['N_neg'],
                       lambda1=cfg['lambda1'], lambda2=cfg['lambda2'])
        qa.append(train_model(qm, X_t, H, W_e, labels, mk(train_ids), mk(val_ids),
                              mk(test_ids), task=task, num_epochs=cfg['num_epochs'],
                              model_type='qadapt'))
        bm = BaselineHGNN(feat_dim, hidden_dim, output_dim,
                          num_layers=cfg['num_layers'], dropout=cfg['dropout'])
        bl.append(train_model(bm, X_t, H, W_e, labels, mk(train_ids), mk(val_ids),
                              mk(test_ids), task=task, num_epochs=cfg['num_epochs'],
                              model_type='baseline'))
    return qa, bl


# =============================================================================
# SECTION 9 - CONFIGS + MAIN
# =============================================================================

DATASET_CONFIGS = {
    'imdb':   {'task': 'classification', 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.5,
               'K': 32, 'N_neg': 64, 'num_epochs': 200, 'lambda1': 0.1, 'lambda2': 0.05},
    'dblp':   {'task': 'classification', 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5,
               'K': 32, 'N_neg': 64, 'num_epochs': 200, 'lambda1': 0.1, 'lambda2': 0.05},
    'acm':    {'task': 'classification', 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3,
               'K': 32, 'N_neg': 64, 'num_epochs': 200, 'lambda1': 0.1, 'lambda2': 0.05},
    'amazon': {'task': 'regression', 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3,
               'K': 32, 'N_neg': 64, 'num_epochs': 200, 'lambda1': 0.1, 'lambda2': 0.05},
    'yelp':   {'task': 'regression', 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3,
               'K': 32, 'N_neg': 64, 'num_epochs': 200, 'lambda1': 0.1, 'lambda2': 0.05},
}


def run_dataset(name, args):
    print(f"\n{'=' * 70}\n  Dataset: {name.upper()}\n{'=' * 70}")
    cfg = deepcopy(DATASET_CONFIGS[name]); task = cfg['task']
    if args.num_epochs:
        cfg['num_epochs'] = args.num_epochs
    folder = None
    if args.data_path:
        for c in [args.data_path, os.path.join(args.data_path, name.upper()),
                  os.path.join(args.data_path, name), os.path.join(args.data_path, name.capitalize())]:
            if os.path.isdir(c):
                folder = c; break
    H, X, W_e, labels, nc = load_dataset(name, data_path=folder,
                                         max_rows=args.max_rows, feature_dim=args.feature_dim)
    feat_dim = X.shape[1]; output_dim = nc if task == 'classification' else 1
    valid = (np.where(labels >= 0)[0] if task == 'classification' else np.arange(len(labels)))
    print(f"\n  Running 5-fold CV ...")
    qa, bl = run_five_fold_cv(H, X, W_e, labels, valid, task, feat_dim,
                              cfg['hidden_dim'], output_dim, cfg)
    summary = statistical_summary(qa, bl, task)
    print_summary_table(summary, task)
    return summary


def main():
    p = argparse.ArgumentParser(description='QAdapt (paper-aligned)')
    p.add_argument('--dataset', type=str, default='imdb',
                   choices=['imdb', 'dblp', 'acm', 'amazon', 'yelp', 'all'])
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--max_rows', type=int, default=None)
    p.add_argument('--feature_dim', type=int, default=64)
    p.add_argument('--num_epochs', type=int, default=None, help='override config epochs')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    datasets = ['imdb', 'dblp', 'acm', 'amazon', 'yelp'] if args.dataset == 'all' else [args.dataset]
    print(f"\n{'=' * 70}\n  QAdapt (paper-aligned)\n  Datasets: {datasets}\n{'=' * 70}")
    alls = {}
    for ds in datasets:
        try:
            alls[ds] = run_dataset(ds, args)
        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
