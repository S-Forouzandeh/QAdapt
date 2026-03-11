"""
QAdapt: Information-Theoretic Mixed-Precision Quantization for Hypergraph Neural Networks
==========================================================================================
Single-file implementation — all equations from the paper in one place.
Supports BOTH classification (IMDB, DBLP, ACM) and regression (Amazon, Yelp).

USAGE:
------
# Option A — provide your own data folder:
    python qadapt.py --dataset imdb   --data_path /your/path/to/IMDB
    python qadapt.py --dataset dblp   --data_path /your/path/to/DBLP
    python qadapt.py --dataset acm    --data_path /your/path/to/ACM
    python qadapt.py --dataset amazon --data_path /your/path/to/Amazon
    python qadapt.py --dataset yelp   --data_path /your/path/to/Yelp
    python qadapt.py --dataset all    --data_path /your/root/data/folder

# Option B — run without any data (auto-generates synthetic hypergraph):
    python qadapt.py --dataset imdb
    python qadapt.py --dataset amazon
    python qadapt.py --dataset all

# Limit rows for quick debug:
    python qadapt.py --dataset imdb --data_path /your/path --max_rows 200

DATA FOLDER EXPECTED FORMAT (per dataset):
------------------------------------------
  IMDB   (classification): user_movies.xlsx, movie_directors.xlsx,
                            movie_actors.xlsx, movie_genres.xlsx
  DBLP   (classification): paper_author.xlsx / paper_author.csv,
                            paper_conf.xlsx   / paper_conf.csv
  ACM    (classification): paper_author.xlsx / paper_author.csv,
                            paper_subject.xlsx / paper_subject.csv
  Amazon (regression)    : user_product.xlsx / user_product.csv
                           (must contain: userID, productID, rating)
  Yelp   (regression)    : user_business.xlsx / user_business.csv
                           (must contain: userID, businessID, rating)

  If a file is missing, synthetic data is used automatically — no crash.
  CSV and XLSX are both supported."""

# PAPER EQUATIONS IMPLEMENTED:
#   hgnn_conv   : X^(l+1) = sigma(D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X^(l) Theta)
#   info_density: rho_{i,e} = IC(x_i, h_e) * SW(i,e)
#   InfoNCE     : hat_I(x_i; h_e) = log exp(f(xi,he)) / mean_j exp(f(xi,he_neg))
#   SW          : SW(i,e) = sum_k alpha_k phi_k(i) * indicator_e(i)
#   A_hyper     : softmax((P_e x_i)^T (P_e x_j)/sqrt(d) + alpha*log(rho+eps))
#   A_node      : softmax((W x_i)^T (W x_j)/sqrt(d) + alpha*log(rho_bar+eps))
#   SpectralFus : A_final = Phi diag(omega) Phi^T (A_hyper + A_node)
#   Fisher      : S_{ij} = EMA((dL/dA_{ij})^2, beta=0.99)
#   Structure   : Structure(i,j) = sum_k gamma_k phi_k(i) phi_k(j)
#   BitWidth    : MLP_alloc([S_Fisher; rho; Structure; phi_local; s_global])
#   Gumbel      : tau(t) = max(0.1, 2.0 * 0.95^(t/100)), hard after epoch 200
#   Q_adaptive  : sum_{b in {4,8,16}} beta^(b) * Q(A; b, s^(b))
#   Loss        : L = L_task + lambda1*L_compression + lambda2*L_spectral

# METRICS (Table 1 of paper):
#   Classification : Accuracy, F1 (macro), AUC (macro OvR)
#   Regression     : MAE, RMSE, R2
#   Efficiency     : Inference time (ms/batch), Compression ratio vs FP16
#   Theory         : Information Retention score, Spectral Preservation score
#   Statistics     : 5-fold CV, paired t-test (p<0.01), Cohen d

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import time
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from copy import deepcopy
from collections import defaultdict
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# SECTION 1 — SPECTRAL UTILITIES
# =============================================================================

def compute_laplacian_eigenpairs(H: np.ndarray, W_e: np.ndarray,
                                  K: int = 32, tol: float = 1e-6
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-K smallest eigenpairs of the normalised hypergraph Laplacian.
        L = I - D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}
    Uses ARPACK (eigsh) with tolerance 1e-6.
    Eigenvectors are L2-normalised and sorted by eigenvalue magnitude.
    Returns: eigvals (K,), eigvecs (n, K)
    """
    n, m = H.shape
    D_v = np.maximum((H * W_e[None, :]).sum(axis=1), 1e-8)   # (n,)
    D_e = np.maximum(H.sum(axis=0), 1e-8)                    # (m,)

    D_v_invsqrt = sparse.diags(1.0 / np.sqrt(D_v))
    D_e_inv     = sparse.diags(1.0 / D_e)
    W_e_diag    = sparse.diags(W_e.astype(float))
    H_sp        = sparse.csr_matrix(H.astype(float))

    # Symmetric adjacency A = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}
    A_sym = D_v_invsqrt @ H_sp @ W_e_diag @ D_e_inv @ H_sp.T @ D_v_invsqrt
    L     = sparse.eye(n) - A_sym

    K_actual = min(K, n - 2)
    try:
        vals, vecs = eigsh(L, k=K_actual, which='SM', tol=tol)
    except Exception:
        Ld = L.toarray()
        vals_all, vecs_all = np.linalg.eigh(Ld)
        vals, vecs = vals_all[:K_actual], vecs_all[:, :K_actual]

    # L2-normalise each eigenvector
    norms = np.linalg.norm(vecs, axis=0, keepdims=True)
    vecs  = vecs / np.maximum(norms, 1e-8)

    # Sort by eigenvalue magnitude
    order = np.argsort(np.abs(vals))
    return vals[order].astype(np.float32), vecs[:, order].astype(np.float32)


# =============================================================================
# SECTION 2 — CRITIC NETWORK  (f_theta for InfoNCE)
# =============================================================================

class CriticNetwork(nn.Module):
    """
    f_theta: R^d x R^d -> R
    Architecture (from paper Preliminaries):
        [x_i ; h_e^ctx] in R^{2d}
        -> Linear(2d, 128) -> ReLU
        -> Linear(128, 64) -> ReLU
        -> Linear(64, 1)
    Updated every 5 main-model iterations (mi_update_interval).
    """
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, 128), nn.ReLU(),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_i: torch.Tensor, h_e: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_i, h_e], dim=-1))


# =============================================================================
# SECTION 3 — INFORMATION DENSITY ESTIMATION  (Step 1)
# =============================================================================

class InformationDensityEstimator(nn.Module):
    """
    Computes rho_{i,e} = IC(x_i, h_e) * SW(i,e)

    IC  — mutual information via InfoNCE (N=64 negatives per positive pair)
    SW  — spectral structural weight using Laplacian eigenvectors

    Shared W_ctx: O(d^2) parameters, independent of |E|.
    """
    def __init__(self, d: int, K: int = 32, N_neg: int = 64):
        super().__init__()
        self.d     = d
        self.K     = K
        self.N_neg = N_neg

        # Shared context projection W_ctx (paper: scalable parameterisation)
        self.W_ctx = nn.Linear(d, d, bias=False)

        # Critic network f_theta
        self.critic = CriticNetwork(d)

        # Learnable spectral coefficients {alpha_k}, shared across all hyperedges
        self.alpha = nn.Parameter(torch.ones(K) / K)

    # ------------------------------------------------------------------
    def compute_hyperedge_context(self, X: torch.Tensor,
                                   H_np: np.ndarray) -> torch.Tensor:
        """
        h_e^ctx = MeanPool({W_ctx x_j : j in V_e})     shape: (m, d)
        """
        n, d = X.shape
        m    = H_np.shape[1]
        ctx  = torch.zeros(m, d, device=X.device)
        for e in range(m):
            node_ids = np.where(H_np[:, e] > 0)[0]
            if len(node_ids) == 0:
                continue
            ctx[e] = self.W_ctx(X[node_ids]).mean(dim=0)
        return ctx                                      # (m, d)

    # ------------------------------------------------------------------
    def information_content(self, X: torch.Tensor,
                             ctx: torch.Tensor,
                             H_np: np.ndarray) -> torch.Tensor:
        """
        hat_I(x_i; h_e^ctx) via InfoNCE with N=64 negatives:
            log( exp(f(x_i, h_e)) / (1/N) sum_j exp(f(x_i, h_ej')) )
        Returns rho_IC of shape (n, m).
        """
        n, m   = H_np.shape
        rho_IC = torch.zeros(n, m, device=X.device)

        for e in range(m):
            node_ids = np.where(H_np[:, e] > 0)[0]
            if len(node_ids) == 0:
                continue

            x_pos = X[node_ids]                                     # (|V_e|, d)
            h_pos = ctx[e].unsqueeze(0).expand(len(node_ids), -1)   # (|V_e|, d)

            # Sample N negatives from E \ {e}
            neg_pool = [j for j in range(m) if j != e]
            neg_ids  = np.random.choice(neg_pool,
                                        min(self.N_neg, len(neg_pool)),
                                        replace=False)
            # Positive scores
            pos_score = self.critic(x_pos, h_pos).squeeze(-1)       # (|V_e|,)

            # Negative scores
            neg_ctx  = ctx[neg_ids]                                  # (N, d)
            x_exp    = x_pos.unsqueeze(1).expand(-1, len(neg_ids), -1)
            neg_exp  = neg_ctx.unsqueeze(0).expand(len(node_ids), -1, -1)
            neg_sc   = self.critic(x_exp, neg_exp).squeeze(-1)      # (|V_e|, N)

            # InfoNCE estimate
            log_denom         = torch.logsumexp(neg_sc, dim=-1) - math.log(len(neg_ids))
            ic                = pos_score - log_denom                # (|V_e|,)
            rho_IC[node_ids, e] = ic.detach()

        return rho_IC

    # ------------------------------------------------------------------
    def structural_weight(self, H_np: np.ndarray,
                          eigvecs: torch.Tensor) -> torch.Tensor:
        """
        SW(i,e) = sum_k alpha_k * phi_k(i) * 1_e(i)
        Returns shape (n, m).
        """
        alpha_norm   = torch.softmax(self.alpha, dim=0)             # (K,)
        node_scores  = (eigvecs * alpha_norm.unsqueeze(0)).sum(-1)  # (n,)
        H_t          = torch.FloatTensor(H_np).to(eigvecs.device)   # (n, m)
        return node_scores.unsqueeze(1) * H_t                       # (n, m)

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor, H_np: np.ndarray,
                eigvecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns rho (n, m) and hyperedge context ctx (m, d)."""
        ctx    = self.compute_hyperedge_context(X, H_np)
        ic     = self.information_content(X, ctx, H_np)
        sw     = self.structural_weight(H_np, eigvecs)
        rho    = ic * sw                                            # (n, m)
        return rho, ctx


# =============================================================================
# SECTION 4 — SPECTRALFUSION  (Step 2)
# =============================================================================

class SpectralFusionMLP(nn.Module):
    """
    Implements:
        A^final = Phi diag(omega) Phi^T (A^hyper + A^node)

    alpha_k = softmax(w_alpha^T [lambda_k ; log|V_e| ; deg(e)])_k
    omega from fusion MLP: R^{K+2} -> R^{64} -> R^{32} -> R^K
                           with skip connections + LayerNorm
    """
    def __init__(self, K: int = 32):
        super().__init__()
        self.K      = K
        self.w_alpha = nn.Parameter(torch.randn(3))

        self.mlp = nn.Sequential(
            nn.Linear(K + 2, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32),    nn.LayerNorm(32),  nn.ReLU(),
            nn.Linear(32, K)
        )
        self.skip = nn.Linear(K + 2, K)

    def forward(self, A_sum: torch.Tensor, Phi: torch.Tensor,
                eigvals: torch.Tensor, log_he_size: float,
                mean_deg: float) -> torch.Tensor:
        """
        A_sum  : (n, n)   A^hyper + A^node
        Phi    : (n, K)   Laplacian eigenvectors
        eigvals: (K,)
        """
        # Spectral coefficients alpha_k
        feats = torch.stack([
            eigvals,
            torch.full_like(eigvals, log_he_size),
            torch.full_like(eigvals, mean_deg)
        ], dim=-1)                                                   # (K, 3)
        alpha = torch.softmax(feats @ self.w_alpha, dim=0)          # (K,)

        # Fusion MLP for learnable frequency weights omega
        inp   = torch.cat([alpha,
                           torch.tensor([log_he_size, mean_deg],
                                        device=alpha.device)])       # (K+2,)
        omega = self.mlp(inp) + self.skip(inp)                      # (K,)

        # A^final = Phi diag(omega) Phi^T * A_sum
        filtered = (Phi * omega.unsqueeze(0)) @ Phi.t()             # (n, n)
        return filtered @ A_sum                                      # (n, n)


# =============================================================================
# SECTION 5 — CO-ADAPTIVE QUANTIZER  (Step 3)
# =============================================================================

class CoAdaptiveQuantizer(nn.Module):
    """
    Per-attention-entry bit-width prediction.

    Input feature f_{ij}:
        [S^Fisher_{ij}(1) ; rho_{ij}(1) ; Structure(i,j)(1) ;
         phi_local(i,j)(4) ; s_global(4)]   -> dim=11

    Allocator MLP: R^{11} -> R^{128} -> R^{64} -> R^3
                   with BatchNorm + Dropout(0.1)

    Gumbel-Softmax: tau(t) = max(0.1, 2.0 * 0.95^(t/100))
                   Hard sampling after epoch 200.

    Bit choices: {4, 8, 16}
    """
    BIT_CHOICES = [4, 8, 16]

    def __init__(self, K: int = 32):
        super().__init__()
        self.K   = K
        d_in     = 11    # see feature description above

        self.allocator = nn.Sequential(
            nn.Linear(d_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, len(self.BIT_CHOICES))
        )

        # Learnable Structure weights gamma_k
        self.gamma = nn.Parameter(torch.ones(K) / K)

        # Fisher EMA buffer (beta=0.99)
        self.register_buffer('fisher_ema', torch.tensor(0.0))
        self.fisher_beta = 0.99

        # Step counter for temperature annealing and hard-sample switch
        self.register_buffer('step', torch.tensor(0))

    # ------------------------------------------------------------------
    def tau(self) -> float:
        """tau(t) = max(0.1, 2.0 * 0.95^(t/100))"""
        return max(0.1, 2.0 * (0.95 ** (self.step.item() / 100)))

    # ------------------------------------------------------------------
    def structure_term(self, i_ids: torch.Tensor,
                       j_ids: torch.Tensor,
                       eigvecs: torch.Tensor) -> torch.Tensor:
        """
        Structure(i,j) = sum_k gamma_k * phi_k(i) * phi_k(j)
        """
        phi_i = eigvecs[i_ids]                                      # (..., K)
        phi_j = eigvecs[j_ids]                                      # (..., K)
        return (self.gamma.unsqueeze(0) * phi_i * phi_j).sum(-1)   # (...,)

    # ------------------------------------------------------------------
    def local_features(self, i_ids: torch.Tensor,
                        j_ids: torch.Tensor,
                        degrees: torch.Tensor,
                        eigvecs: torch.Tensor,
                        A_adj: np.ndarray) -> torch.Tensor:
        """
        phi_local(i,j) = [deg(i), deg(j), |N_i ∩ N_j|, ||phi(i)-phi(j)||_2]
        """
        deg_i    = degrees[i_ids].float()
        deg_j    = degrees[j_ids].float()
        a_i      = torch.FloatTensor(A_adj[i_ids.cpu().numpy()])
        a_j      = torch.FloatTensor(A_adj[j_ids.cpu().numpy()])
        shared   = (a_i * a_j).sum(-1)
        phi_dist = torch.norm(eigvecs[i_ids] - eigvecs[j_ids], dim=-1)
        return torch.stack([deg_i, deg_j, shared, phi_dist], dim=-1)  # (..., 4)

    # ------------------------------------------------------------------
    def global_stats(self, d_v_mean: float, d_e_mean: float,
                      lam_max: float, lam_min: float,
                      budget_used: float, budget_total: float,
                      device: torch.device) -> torch.Tensor:
        """
        s_global = [d_v_mean, d_e_mean, lambda_max/lambda_min,
                    budget_used/budget_total]
        """
        ratio = lam_max / (lam_min + 1e-8)
        bu    = budget_used / (budget_total + 1e-8)
        return torch.tensor([d_v_mean, d_e_mean, ratio, bu], device=device)

    # ------------------------------------------------------------------
    def gumbel_softmax(self, logits: torch.Tensor,
                        hard: bool = False) -> torch.Tensor:
        tau     = self.tau()
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y_soft  = F.softmax((logits + gumbels) / tau, dim=-1)
        if hard:
            idx    = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    # ------------------------------------------------------------------
    def quantize_uniform(self, A: torch.Tensor, b: int) -> torch.Tensor:
        """
        Q(A; b, s) with s = max|A| / (2^{b-1} - 1)  (minimises MSE)
        """
        s      = A.abs().max().item() / ((2 ** (b - 1)) - 1 + 1e-8)
        qmin   = -(2 ** (b - 1))
        qmax   =  (2 ** (b - 1)) - 1
        return torch.round(torch.clamp(A / (s + 1e-8), qmin, qmax)) * s

    # ------------------------------------------------------------------
    def forward(self, A: torch.Tensor,
                rho_mat: torch.Tensor,
                eigvecs: torch.Tensor,
                degrees: torch.Tensor,
                A_adj: np.ndarray,
                stats: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A       : (n, n)  attention matrix to quantize
        rho_mat : (n, n)  pairwise information density
        Returns: A_quant (n,n), beta_mat (n,n,3), exp_bits (n,n)
        """
        n      = A.shape[0]
        device = A.device

        # Update Fisher EMA with current attention magnitudes (proxy)
        self.fisher_ema = (self.fisher_beta * self.fisher_ema
                           + (1 - self.fisher_beta) * (A.detach() ** 2).mean())

        # Flat index pairs
        idx_i = torch.arange(n, device=device).unsqueeze(1).expand(n, n).reshape(-1)
        idx_j = torch.arange(n, device=device).unsqueeze(0).expand(n, n).reshape(-1)

        fisher_f = self.fisher_ema.expand(n * n).unsqueeze(-1)     # (n², 1)
        rho_f    = rho_mat.reshape(-1, 1)                          # (n², 1)
        struct_f = self.structure_term(idx_i, idx_j, eigvecs).unsqueeze(-1)
        loc_f    = self.local_features(idx_i, idx_j, degrees, eigvecs, A_adj)
        s_glob   = self.global_stats(
                       stats['d_v_mean'], stats['d_e_mean'],
                       stats['lam_max'],  stats['lam_min'],
                       stats['budget_used'], stats['budget_total'], device
                   ).unsqueeze(0).expand(n * n, -1)

        feat   = torch.cat([fisher_f, rho_f, struct_f, loc_f, s_glob], dim=-1)  # (n², 11)
        logits = self.allocator(feat)                               # (n², 3)

        hard   = self.step.item() >= 200
        beta   = self.gumbel_softmax(logits, hard=hard)            # (n², 3)
        beta_mat = beta.reshape(n, n, -1)                          # (n, n, 3)

        # Q_adaptive(A_ij) = sum_b beta^(b) * Q(A_ij; b, s^(b))
        A_quant = torch.zeros_like(A)
        for k, b in enumerate(self.BIT_CHOICES):
            A_quant = A_quant + beta_mat[..., k] * self.quantize_uniform(A, b)

        bit_vals = torch.tensor(self.BIT_CHOICES, dtype=torch.float32, device=device)
        exp_bits = (beta_mat * bit_vals).sum(-1)                   # (n, n)

        self.step += 1
        return A_quant, beta_mat, exp_bits


# =============================================================================
# SECTION 6 — QAdapt CONVOLUTION LAYER
# =============================================================================

class QAdaptConv(nn.Module):
    """
    Single QAdapt layer:
        Step 1 → rho_{i,e} = IC * SW
        Step 2 → A^hyper, A^node, SpectralFusion → A^final
        Step 3 → CoAdaptiveQuantizer → A_quant
        Output → sigma(A_quant X Theta)
    """
    def __init__(self, in_features: int, out_features: int,
                 K: int = 32, N_neg: int = 64,
                 dropout: float = 0.5, alpha_scale: float = 1.0):
        super().__init__()
        self.K           = K
        self.alpha_scale = alpha_scale

        # Linear transformation Theta^(l)
        self.Theta   = nn.Linear(in_features, out_features, bias=True)

        # Step 1
        self.density = InformationDensityEstimator(in_features, K, N_neg)

        # Step 2 — projections
        self.P_e     = nn.Linear(in_features, in_features, bias=False)  # intra-hyperedge
        self.W_node  = nn.Linear(in_features, in_features, bias=False)  # node-level
        self.w_e     = nn.Parameter(torch.ones(1))                       # hyperedge agg weight
        self.fusion  = SpectralFusionMLP(K)

        # Step 3
        self.quantizer = CoAdaptiveQuantizer(K)

        self.dropout = nn.Dropout(dropout)

        # Spectral buffers — set via set_spectral() before training
        self.register_buffer('eigvecs',  None)
        self.register_buffer('eigvals',  None)
        self.register_buffer('degrees',  None)

    # ------------------------------------------------------------------
    def set_spectral(self, eigvals: np.ndarray, eigvecs: np.ndarray,
                     degrees: np.ndarray):
        self.eigvecs  = torch.FloatTensor(eigvecs)
        self.eigvals  = torch.FloatTensor(eigvals)
        self.degrees  = torch.FloatTensor(degrees)

    # ------------------------------------------------------------------
    def _intra_hyperedge_attention(self, X: torch.Tensor,
                                    rho: torch.Tensor,
                                    H_np: np.ndarray) -> torch.Tensor:
        """
        A^hyper_{ij} = softmax_j( (P_e x_i)^T(P_e x_j)/sqrt(d)
                                  + alpha * log(rho_{i,e} + eps) )
        Aggregated over all hyperedges:
            A^hyper = sum_e w_e * A^(e)   in R^{n x n}
        """
        n, m    = H_np.shape
        d       = X.shape[1]
        eps     = 1e-8
        X_proj  = self.P_e(X)                                       # (n, d)
        A_hyper = torch.zeros(n, n, device=X.device)

        for e in range(min(m, 300)):            # cap for memory
            node_ids = np.where(H_np[:, e] > 0)[0]
            k        = len(node_ids)
            if k < 2:
                continue
            xi     = X_proj[node_ids]                               # (k, d)
            scores = (xi @ xi.t()) / math.sqrt(d)                   # (k, k)
            # Bias by log-information-density (paper eq.)
            rho_e  = rho[node_ids, e]                               # (k,)
            bias   = self.alpha_scale * torch.log(rho_e.clamp(min=eps)).unsqueeze(1)
            attn   = F.softmax(scores + bias, dim=-1)               # (k, k)
            rows   = torch.LongTensor(node_ids).to(X.device)
            A_hyper[rows[:, None], rows[None, :]] += attn * self.w_e

        return A_hyper

    # ------------------------------------------------------------------
    def _node_attention(self, X: torch.Tensor,
                         rho: torch.Tensor,
                         H_np: np.ndarray) -> torch.Tensor:
        """
        A^node_{ij} = softmax_j( (W x_i)^T(W x_j)/sqrt(d)
                                 + alpha * log(rho_bar_{i,j} + eps) )
        rho_bar_{i,j} = mean rho over hyperedges containing both i and j
        """
        n, d    = X.shape
        eps     = 1e-8
        X_proj  = self.W_node(X)                                    # (n, d)
        scores  = (X_proj @ X_proj.t()) / math.sqrt(d)             # (n, n)

        H_t          = torch.FloatTensor(H_np).to(X.device)
        shared_count = (H_t @ H_t.t()).clamp(min=1)
        rho_bar      = (rho @ H_t.t()) / shared_count

        bias     = self.alpha_scale * torch.log(rho_bar.clamp(min=eps))
        A_node   = F.softmax(scores + bias, dim=-1)
        return A_node

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor, H_np: np.ndarray,
                W_e: np.ndarray) -> dict:
        assert self.eigvecs is not None, "Call set_spectral() before forward()."
        n       = X.shape[0]
        device  = X.device
        eigvecs = self.eigvecs.to(device)
        eigvals = self.eigvals.to(device)
        degrees = self.degrees.to(device)

        # ── Step 1: Information density ──────────────────────────────
        rho, ctx = self.density(X, H_np, eigvecs)                   # (n, m)

        # ── Step 2: Multi-scale attention ────────────────────────────
        A_hyper = self._intra_hyperedge_attention(X, rho, H_np)     # (n, n)
        A_node  = self._node_attention(X, rho, H_np)                # (n, n)
        A_sum   = A_hyper + A_node

        # SpectralFusion
        log_he  = math.log(max(float(np.mean((H_np > 0).sum(axis=0))), 1.0))
        deg_mean = float(degrees.mean().item())
        A_final = self.fusion(A_sum, eigvecs, eigvals, log_he, deg_mean)  # (n, n)

        # ── Step 3: Co-adaptive quantization ─────────────────────────
        H_t      = torch.FloatTensor(H_np).to(device)
        shared   = (H_t @ H_t.t()).clamp(min=1)
        rho_mat  = (rho @ H_t.t()) / shared                        # (n, n)

        D_v      = torch.FloatTensor((H_np * W_e[None, :]).sum(axis=1)).to(device)
        D_e_arr  = torch.FloatTensor(H_np.sum(axis=0)).to(device)
        stats    = {
            'd_v_mean':     D_v.mean().item(),
            'd_e_mean':     D_e_arr.mean().item(),
            'lam_max':      eigvals.max().item(),
            'lam_min':      eigvals.min().item(),
            'budget_used':  float(self.quantizer.step.item()),
            'budget_total': 1000.0,
        }
        A_adj    = (H_np @ H_np.T).astype(float)                   # adjacency proxy

        A_quant, beta_mat, exp_bits = self.quantizer(
            A_final, rho_mat, eigvecs, degrees.long(), A_adj, stats
        )

        # ── HGNN convolution with quantised attention ─────────────────
        # X^(l+1) = sigma( A_quant X Theta )
        X_out = self.dropout(self.Theta(A_quant @ X))               # (n, out_features)

        return {
            'output':   X_out,
            'A_final':  A_final,
            'A_quant':  A_quant,
            'rho':      rho,
            'beta_mat': beta_mat,
            'exp_bits': exp_bits,
        }


# =============================================================================
# SECTION 7 — FULL QAdapt NETWORK
# =============================================================================

class QAdaptNet(nn.Module):
    """
    Multi-layer QAdapt with joint training objective:
        L = L_task + lambda1 * L_compression + lambda2 * L_spectral
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 K: int = 32, N_neg: int = 64,
                 lambda1: float = 0.01, lambda2: float = 0.001):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.layers = nn.ModuleList([
            QAdaptConv(hidden_dim, hidden_dim, K=K, N_neg=N_neg, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def set_spectral(self, eigvals, eigvecs, degrees):
        for layer in self.layers:
            layer.set_spectral(eigvals, eigvecs, degrees)

    def forward(self, X: torch.Tensor, H_np: np.ndarray,
                W_e: np.ndarray) -> dict:
        X     = self.input_proj(X)
        louts = []
        for layer in self.layers:
            res = layer(X, H_np, W_e)
            X   = F.relu(res['output'])
            louts.append(res)
        return {'logits': self.output_layer(X), 'layer_outputs': louts}

    def compute_loss(self, logits, labels, layer_outputs, mask, task) -> dict:
        """
        L = L_task + lambda1 * L_compression + lambda2 * L_spectral
        L_compression = mean(expected_bits)          [lower = more compressed]
        L_spectral    = ||A_final - A_quant||_F / ||A_final||_F
        """
        if task == 'classification':
            L_task = F.cross_entropy(logits[mask], labels[mask])
        else:
            L_task = F.mse_loss(logits[mask].squeeze(), labels[mask].float())

        L_comp = torch.stack([lo['exp_bits'].mean() for lo in layer_outputs]).mean()

        L_spec = torch.stack([
            torch.norm(lo['A_final'] - lo['A_quant'], p='fro')
            / torch.norm(lo['A_final'], p='fro').clamp(min=1e-8)
            for lo in layer_outputs
        ]).mean()

        total = L_task + self.lambda1 * L_comp + self.lambda2 * L_spec
        return {'total': total, 'task': L_task,
                'compression': L_comp, 'spectral': L_spec}


# =============================================================================
# SECTION 8 — BASELINE HGNN  (for comparison)
# =============================================================================

class BaselineHGNN(nn.Module):
    """
    Standard Laplacian-based HGNN — no quantization, no information density.
    Uses the exact same HGNN convolution formula as the paper:
        X^(l+1) = sigma(D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X^(l) Theta)
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=2, dropout=0.5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder     = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, H_np, W_e):
        n, m   = H_np.shape
        D_v    = np.maximum((H_np * W_e[None, :]).sum(axis=1), 1e-8)
        D_e    = np.maximum(H_np.sum(axis=0), 1e-8)
        Dv_inv = np.diag(1.0 / np.sqrt(D_v))
        De_inv = np.diag(1.0 / D_e)
        We_d   = np.diag(W_e)
        A      = Dv_inv @ H_np @ We_d @ De_inv @ H_np.T @ Dv_inv
        A_t    = torch.FloatTensor(A).to(X.device)
        h      = self.encoder(A_t @ X)
        logits = self.output_layer(h)
        return {
            'logits': logits,
            'layer_outputs': [{
                'A_final':  A_t,
                'A_quant':  A_t,
                'exp_bits': torch.full((n, n), 16.0, device=X.device),
                'rho':      torch.zeros(n, H_np.shape[1], device=X.device),
            }]
        }


# =============================================================================
# SECTION 9 — METRICS  (all Table 1 columns)
# =============================================================================

def information_retention_score(A_orig: torch.Tensor,
                                  A_quant: torch.Tensor) -> float:
    """IR = 1 - ||A_orig - A_quant||_1 / ||A_orig||_1   in [0,1]"""
    denom = A_orig.detach().abs().sum().item()
    if denom < 1e-12:
        return 1.0
    diff = (A_orig.detach() - A_quant.detach()).abs().sum().item()
    return float(np.clip(1.0 - diff / denom, 0, 1))


def spectral_preservation_score(A_orig: torch.Tensor,
                                  A_quant: torch.Tensor) -> float:
    """SP = 1 - ||A_orig - A_quant||_F / ||A_orig||_F   in [0,1]"""
    denom = torch.norm(A_orig.detach(), p='fro').item()
    if denom < 1e-12:
        return 1.0
    num = torch.norm(A_orig.detach() - A_quant.detach(), p='fro').item()
    return float(np.clip(1.0 - num / denom, 0, 1))


def measure_inference_time(model, X, H_np, W_e, n_runs=50, warmup=10) -> float:
    """Average inference time per call in milliseconds."""
    model.eval()
    device = next(model.parameters()).device
    X = X.to(device)
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


def evaluate(model, X, H_np, W_e, labels, mask, task='classification',
             measure_time=True) -> dict:
    """
    Returns all Table 1 metrics:
        classification: acc, f1, auc, comp_ratio, info_retain, spec_pres, time_ms
        regression    : mae, rmse, r2, comp_ratio, info_retain, spec_pres, time_ms
    """
    model.eval()
    device = next(model.parameters()).device
    X = X.to(device); labels = labels.to(device)

    with torch.no_grad():
        out      = model(X, H_np, W_e)
        logits   = out['logits']
        louts    = out['layer_outputs']

        ir_scores, sp_scores, exp_bits_all = [], [], []
        for lo in louts:
            ir_scores.append(information_retention_score(lo['A_final'], lo['A_quant']))
            sp_scores.append(spectral_preservation_score(lo['A_final'], lo['A_quant']))
            exp_bits_all.append(lo['exp_bits'].mean())

        avg_exp_bits = torch.stack(exp_bits_all).mean()
        comp  = 16.0 / avg_exp_bits.item() if avg_exp_bits.item() > 1e-8 else 1.0
        avg_ir = float(np.mean(ir_scores))
        avg_sp = float(np.mean(sp_scores))

        if task == 'classification':
            probs = F.softmax(logits[mask], dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            true  = labels[mask].cpu().numpy()
            acc   = accuracy_score(true, preds)
            f1    = f1_score(true, preds, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(true, probs, multi_class='ovr', average='macro') \
                      if probs.shape[1] > 2 \
                      else roc_auc_score(true, probs[:, 1])
            except ValueError:
                auc = 0.0
            result = {'acc': acc, 'f1': f1, 'auc': auc}
        else:
            preds  = logits[mask].squeeze(-1).cpu().numpy()
            true   = labels[mask].float().cpu().numpy()
            result = {
                'mae':  mean_absolute_error(true, preds),
                'rmse': float(np.sqrt(mean_squared_error(true, preds))),
                'r2':   r2_score(true, preds),
            }

    result.update({'comp_ratio': comp, 'info_retain': avg_ir, 'spec_pres': avg_sp})
    result['time_ms'] = measure_inference_time(model, X, H_np, W_e) if measure_time else 0.0
    return result


def statistical_summary(qadapt_results: list, baseline_results: list, task: str) -> dict:
    """Paired t-test + Cohen's d for all metrics."""
    metrics = (['acc', 'f1', 'auc'] if task == 'classification' else ['mae', 'rmse', 'r2'])
    metrics += ['comp_ratio', 'info_retain', 'spec_pres', 'time_ms']
    summary = {}
    for m in metrics:
        q = np.array([r[m] for r in qadapt_results   if m in r])
        b = np.array([r[m] for r in baseline_results  if m in r])
        if not len(q):
            continue
        t, p   = ttest_rel(q, b) if len(q) > 1 else (0.0, 1.0)
        diffs  = q - b
        cohen  = diffs.mean() / (diffs.std() + 1e-10) if len(diffs) > 1 else 0.0
        summary[m] = {
            'qadapt_mean':   float(q.mean()),   'qadapt_std':    float(q.std()),
            'baseline_mean': float(b.mean()),   'baseline_std':  float(b.std()),
            't_stat': float(t), 'p_value': float(p), 'cohen_d': float(cohen),
            'significant': bool(p < 0.01),
        }
    return summary


def print_summary_table(summary: dict, task: str):
    print("\n" + "=" * 105)
    print("  RESULTS SUMMARY  |  5-Fold CV  |  Statistical significance: p < 0.01")
    print("=" * 105)
    header = f"{'Metric':<16} {'QAdapt':>20} {'Baseline':>20} {'Improvement':>13} {'p-value':>10} {'Cohen d':>9} {'Sig':>5}"
    print(header)
    print("-" * 105)
    labels_map = {
        'acc':'Accuracy', 'f1':'F1 (macro)', 'auc':'AUC (macro)',
        'mae':'MAE', 'rmse':'RMSE', 'r2':'R²',
        'comp_ratio':'Comp. Ratio', 'info_retain':'Info Retain',
        'spec_pres':'Spec Pres', 'time_ms':'Time (ms)',
    }
    lower_better = {'mae', 'rmse', 'time_ms'}
    for key, label in labels_map.items():
        if key not in summary:
            continue
        s   = summary[key]
        imp = (s['baseline_mean'] - s['qadapt_mean']) if key in lower_better \
              else (s['qadapt_mean'] - s['baseline_mean'])
        sig = "***" if s['p_value'] < 0.001 else "**" if s['p_value'] < 0.01 \
              else "*" if s['p_value'] < 0.05 else ""
        print(f"{label:<16} "
              f"{s['qadapt_mean']:>9.4f}±{s['qadapt_std']:.4f}  "
              f"{s['baseline_mean']:>9.4f}±{s['baseline_std']:.4f}  "
              f"{imp:>+12.4f}  {s['p_value']:>10.4f}  {s['cohen_d']:>9.3f}  {sig:>5}")
    print("=" * 105)


# =============================================================================
# SECTION 10 — DATA LOADING  (universal: all 5 datasets + synthetic fallback)
# =============================================================================

def _read_file(folder: str, candidates: list, cols: list,
               max_rows=None) -> pd.DataFrame:
    """
    Try to load the first matching file (xlsx or csv) from a list of candidates.
    Returns empty DataFrame if nothing found.
    """
    for fname in candidates:
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(path, usecols=lambda c: c in cols, nrows=max_rows)
            else:
                df = pd.read_excel(path, usecols=lambda c: c in cols, nrows=max_rows)
            # Keep only the columns we need (some files may have extras)
            df = df[[c for c in cols if c in df.columns]]
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            print(f"  Loaded {len(df):>6} rows  ←  {fname}")
            return df
        except Exception as e:
            print(f"  [warn] Could not read {fname}: {e}")
    return pd.DataFrame(columns=cols)


def _build_hypergraph(hyperedges: list, entity_types: dict,
                       feature_dim: int, seed: int = 42):
    """
    Convert a list of hyperedges (each = list of entity strings) into
    incidence matrix H, random feature matrix X, and entity index.
    """
    all_ents    = sorted({e for edge in hyperedges for e in edge})
    e2idx       = {e: i for i, e in enumerate(all_ents)}
    n, m        = len(all_ents), len(hyperedges)

    H = np.zeros((n, m), dtype=np.float32)
    for eid, members in enumerate(hyperedges):
        for ent in members:
            if ent in e2idx:
                H[e2idx[ent], eid] = 1.0

    rng         = np.random.default_rng(seed)
    unique_types = list(set(entity_types.values()))
    type_embeds  = {t: rng.standard_normal(feature_dim) for t in unique_types}
    X            = rng.standard_normal((n, feature_dim)).astype(np.float32)
    for i, ent in enumerate(all_ents):
        t    = entity_types.get(ent, 'unknown')
        X[i] += 0.5 * type_embeds.get(t, np.zeros(feature_dim))

    return H, X, all_ents, e2idx


def _safe_int(val):
    try:
        return int(float(val))
    except:
        return str(val)


# ─── IMDB (classification: predict movie genre) ───────────────────────────

def load_imdb(folder: str, max_rows=None, feature_dim=64):
    """
    Files needed (xlsx or csv):
        user_movies    : userID, movieID, rating
        movie_directors: movieID, directorID
        movie_actors   : movieID, actorID
        movie_genres   : movieID, genreID   ← label source
    """
    df_um = _read_file(folder, ['user_movies.xlsx',    'user_movies.csv'],
                       ['userID','movieID','rating'],   max_rows)
    df_md = _read_file(folder, ['movie_directors.xlsx','movie_directors.csv'],
                       ['movieID','directorID'],        max_rows)
    df_ma = _read_file(folder, ['movie_actors.xlsx',   'movie_actors.csv'],
                       ['movieID','actorID'],           max_rows)
    df_mg = _read_file(folder, ['movie_genres.xlsx',   'movie_genres.csv'],
                       ['movieID','genreID'],           max_rows)

    etype, hedges, genre_map = {}, [], {}

    def add(a, at, b, bt):
        etype[a] = at; etype[b] = bt; hedges.append([a, b])

    for _, r in df_um.iterrows():
        add(f"user_{_safe_int(r['userID'])}", 'user',
            f"movie_{_safe_int(r['movieID'])}", 'movie')
    for _, r in df_md.iterrows():
        add(f"movie_{_safe_int(r['movieID'])}", 'movie',
            f"director_{_safe_int(r['directorID'])}", 'director')
    for _, r in df_ma.iterrows():
        add(f"movie_{_safe_int(r['movieID'])}", 'movie',
            f"actor_{_safe_int(r['actorID'])}", 'actor')

    uniq_g = sorted(df_mg['genreID'].dropna().unique().tolist())
    g2idx  = {g: i for i, g in enumerate(uniq_g)}
    for _, r in df_mg.iterrows():
        mid, gid = _safe_int(r['movieID']), _safe_int(r['genreID'])
        add(f"movie_{mid}", 'movie', f"genre_{gid}", 'genre')
        if mid not in genre_map and gid in g2idx:
            genre_map[mid] = g2idx[gid]

    if not hedges:
        print("  [warn] No IMDB data found — using synthetic fallback.")
        return make_synthetic(task='classification')

    H, X, ents, e2idx = _build_hypergraph(hedges, etype, feature_dim)
    n = len(ents)
    labels = np.full(n, -1, dtype=np.int64)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'movie':
            try:
                mid = _safe_int(ent.split('_')[1])
                if mid in genre_map:
                    labels[i] = genre_map[mid]
            except: pass

    num_classes = len(uniq_g) if uniq_g else 1
    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges, "
          f"{num_classes} classes  [classification]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, num_classes


# ─── DBLP (classification: predict paper conference/area) ─────────────────

def load_dblp(folder: str, max_rows=None, feature_dim=64):
    """
    Files needed (xlsx or csv):
        paper_author: paperID, authorID
        paper_conf  : paperID, confID   ← label source
    """
    df_pa = _read_file(folder, ['paper_author.xlsx','paper_author.csv'],
                       ['paperID','authorID'], max_rows)
    df_pc = _read_file(folder, ['paper_conf.xlsx', 'paper_conf.csv'],
                       ['paperID','confID'],   max_rows)

    etype, hedges, label_map = {}, [], {}

    for _, r in df_pa.iterrows():
        p, a = f"paper_{_safe_int(r['paperID'])}", f"author_{_safe_int(r['authorID'])}"
        etype[p] = 'paper'; etype[a] = 'author'; hedges.append([p, a])

    uniq_c = sorted(df_pc['confID'].dropna().unique().tolist())
    c2idx  = {c: i for i, c in enumerate(uniq_c)}
    for _, r in df_pc.iterrows():
        pid, cid = _safe_int(r['paperID']), _safe_int(r['confID'])
        p, c = f"paper_{pid}", f"conf_{cid}"
        etype[p] = 'paper'; etype[c] = 'conf'; hedges.append([p, c])
        if pid not in label_map and cid in c2idx:
            label_map[pid] = c2idx[cid]

    if not hedges:
        print("  [warn] No DBLP data found — using synthetic fallback.")
        return make_synthetic(task='classification')

    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim)
    n      = len(ents)
    labels = np.full(n, -1, dtype=np.int64)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'paper':
            try:
                pid = _safe_int(ent.split('_')[1])
                if pid in label_map: labels[i] = label_map[pid]
            except: pass

    num_classes = max(len(uniq_c), 1)
    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges, "
          f"{num_classes} classes  [classification]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, num_classes


# ─── ACM (classification: predict paper subject) ──────────────────────────

def load_acm(folder: str, max_rows=None, feature_dim=64):
    """
    Files needed (xlsx or csv):
        paper_author : paperID, authorID
        paper_subject: paperID, subjectID  ← label source
    """
    df_pa = _read_file(folder, ['paper_author.xlsx', 'paper_author.csv'],
                       ['paperID','authorID'],   max_rows)
    df_ps = _read_file(folder, ['paper_subject.xlsx','paper_subject.csv'],
                       ['paperID','subjectID'],  max_rows)

    etype, hedges, label_map = {}, [], {}

    for _, r in df_pa.iterrows():
        p, a = f"paper_{_safe_int(r['paperID'])}", f"author_{_safe_int(r['authorID'])}"
        etype[p] = 'paper'; etype[a] = 'author'; hedges.append([p, a])

    uniq_s = sorted(df_ps['subjectID'].dropna().unique().tolist())
    s2idx  = {s: i for i, s in enumerate(uniq_s)}
    for _, r in df_ps.iterrows():
        pid, sid = _safe_int(r['paperID']), _safe_int(r['subjectID'])
        p, s = f"paper_{pid}", f"subject_{sid}"
        etype[p] = 'paper'; etype[s] = 'subject'; hedges.append([p, s])
        if pid not in label_map and sid in s2idx:
            label_map[pid] = s2idx[sid]

    if not hedges:
        print("  [warn] No ACM data found — using synthetic fallback.")
        return make_synthetic(task='classification')

    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim)
    n      = len(ents)
    labels = np.full(n, -1, dtype=np.int64)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'paper':
            try:
                pid = _safe_int(ent.split('_')[1])
                if pid in label_map: labels[i] = label_map[pid]
            except: pass

    num_classes = max(len(uniq_s), 1)
    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges, "
          f"{num_classes} classes  [classification]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, num_classes


# ─── AMAZON (regression: predict product rating) ──────────────────────────

def load_amazon(folder: str, max_rows=None, feature_dim=64):
    """
    Files needed (xlsx or csv):
        user_product: userID, productID, rating  ← rating is the regression target
    """
    df = _read_file(folder,
                    ['user_product.xlsx','user_product.csv',
                     'ratings.xlsx',     'ratings.csv'],
                    ['userID','productID','rating'], max_rows)

    if df.empty:
        print("  [warn] No Amazon data found — using synthetic fallback.")
        return make_synthetic(task='regression')

    etype, hedges, rating_map = {}, [], {}
    for _, r in df.iterrows():
        u  = f"user_{_safe_int(r['userID'])}"
        p  = f"product_{_safe_int(r['productID'])}"
        etype[u] = 'user'; etype[p] = 'product'
        hedges.append([u, p])
        pid = _safe_int(r['productID'])
        if pid not in rating_map:
            rating_map[pid] = float(r.get('rating', 0.0))

    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim)
    n      = len(ents)
    labels = np.zeros(n, dtype=np.float32)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'product':
            try:
                pid = _safe_int(ent.split('_')[1])
                labels[i] = rating_map.get(pid, 0.0)
            except: pass

    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges  [regression]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, 1


# ─── YELP (regression: predict business rating) ───────────────────────────

def load_yelp(folder: str, max_rows=None, feature_dim=64):
    """
    Files needed (xlsx or csv):
        user_business: userID, businessID, rating  ← rating is the regression target
    """
    df = _read_file(folder,
                    ['user_business.xlsx','user_business.csv',
                     'reviews.xlsx',      'reviews.csv'],
                    ['userID','businessID','rating'], max_rows)

    if df.empty:
        print("  [warn] No Yelp data found — using synthetic fallback.")
        return make_synthetic(task='regression')

    etype, hedges, rating_map = {}, [], {}
    for _, r in df.iterrows():
        u  = f"user_{_safe_int(r['userID'])}"
        b  = f"business_{_safe_int(r['businessID'])}"
        etype[u] = 'user'; etype[b] = 'business'
        hedges.append([u, b])
        bid = _safe_int(r['businessID'])
        if bid not in rating_map:
            rating_map[bid] = float(r.get('rating', 0.0))

    H, X, ents, _ = _build_hypergraph(hedges, etype, feature_dim)
    n      = len(ents)
    labels = np.zeros(n, dtype=np.float32)
    for i, ent in enumerate(ents):
        if etype.get(ent) == 'business':
            try:
                bid = _safe_int(ent.split('_')[1])
                labels[i] = rating_map.get(bid, 0.0)
            except: pass

    print(f"  Hypergraph: {n} nodes, {H.shape[1]} hyperedges  [regression]")
    return H, X, np.ones(H.shape[1], dtype=np.float32), labels, 1


# ─── Synthetic fallback ────────────────────────────────────────────────────

def make_synthetic(n_nodes=500, n_he=200, feat_dim=64,
                    n_classes=5, task='classification', seed=42):
    """Auto-generated hypergraph when real data is unavailable."""
    rng = np.random.default_rng(seed)
    H   = np.zeros((n_nodes, n_he), dtype=np.float32)
    for e, s in enumerate(rng.integers(2, 10, size=n_he)):
        H[rng.choice(n_nodes, s, replace=False), e] = 1.0
    X   = rng.standard_normal((n_nodes, feat_dim)).astype(np.float32)
    W_e = np.ones(n_he, dtype=np.float32)
    if task == 'classification':
        lbl = rng.integers(0, n_classes, n_nodes).astype(np.int64)
        nc  = n_classes
    else:
        lbl = rng.standard_normal(n_nodes).astype(np.float32)
        nc  = 1
    print(f"  Synthetic hypergraph: {n_nodes} nodes, {n_he} hyperedges  [{task}]")
    return H, X, W_e, lbl, nc


# ─── Unified loader ────────────────────────────────────────────────────────

LOADERS = {
    'imdb':   load_imdb,
    'dblp':   load_dblp,
    'acm':    load_acm,
    'amazon': load_amazon,
    'yelp':   load_yelp,
}

def load_dataset(name: str, data_path: str = None,
                  max_rows=None, feature_dim=64):
    """
    Universal entry point.
    If data_path is None or the folder does not exist, falls back to synthetic.
    """
    loader = LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from: {list(LOADERS.keys())}")

    if data_path and os.path.isdir(data_path):
        print(f"  Reading from: {data_path}")
        return loader(data_path, max_rows=max_rows, feature_dim=feature_dim)
    else:
        if data_path:
            print(f"  [warn] '{data_path}' not found — using synthetic data.")
        else:
            print(f"  No --data_path given — using synthetic data.")
        task = DATASET_CONFIGS[name]['task']
        return make_synthetic(task=task, feat_dim=feature_dim)


# =============================================================================
# SECTION 11 — TRAINING LOOP
# =============================================================================

def train_model(model, X, H_np, W_e, labels,
                train_mask, val_mask, test_mask,
                task='classification', num_epochs=200,
                lr=0.005, weight_decay=5e-4, patience=30,
                model_type='qadapt', verbose=True,
                mi_update_interval=5) -> dict:
    """
    Joint training of all components.
    MI networks (critic + W_ctx) updated every mi_update_interval steps.
    Gumbel temperature anneals automatically via CoAdaptiveQuantizer.step.
    """
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    X        = X.to(device)
    labels_t = (torch.LongTensor(labels) if task == 'classification'
                else torch.FloatTensor(labels)).to(device)

    # Pre-compute spectral decomposition (once, before training)
    if model_type == 'qadapt':
        if verbose: print("    Computing Laplacian eigenpairs (K=32)...")
        K    = min(32, H_np.shape[0] - 2)
        ev, evec = compute_laplacian_eigenpairs(H_np, W_e, K=K)
        degs = H_np.sum(axis=1).astype(np.float32)
        model.set_spectral(ev, evec, degs)

    # Separate MI params for independent update schedule
    mi_params, main_params = [], []
    for name, p in model.named_parameters():
        (mi_params if 'density' in name else main_params).append(p)

    opt_main = torch.optim.Adam(main_params, lr=lr, weight_decay=weight_decay)
    opt_mi   = torch.optim.Adam(mi_params, lr=lr * 0.5) if mi_params else None

    best_val, best_state, patience_ctr, step = -np.inf, None, 0, 0

    for epoch in range(num_epochs):
        model.train()
        opt_main.zero_grad()
        if opt_mi: opt_mi.zero_grad()

        out    = model(X, H_np, W_e)
        logits = out['logits']

        if model_type == 'qadapt':
            losses = model.compute_loss(logits, labels_t, out['layer_outputs'],
                                         train_mask, task)
            loss = losses['total']
        else:
            loss = (F.cross_entropy(logits[train_mask], labels_t[train_mask])
                    if task == 'classification'
                    else F.mse_loss(logits[train_mask].squeeze(), labels_t[train_mask]))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_main.step()
        if opt_mi and step % mi_update_interval == 0:
            opt_mi.step()
        step += 1

        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                vl = model(X, H_np, W_e)['logits']
                if task == 'classification':
                    vm = (vl[val_mask].argmax(1) == labels_t[val_mask]).float().mean().item()
                    improved = vm > best_val
                else:
                    vm = -F.mse_loss(vl[val_mask].squeeze(), labels_t[val_mask]).item()
                    improved = vm > best_val
            if improved:
                best_val, best_state, patience_ctr = vm, deepcopy(model.state_dict()), 0
            else:
                patience_ctr += 1
            if verbose and epoch % 20 == 0:
                if model_type == 'qadapt':
                    print(f"    Epoch {epoch:03d} | loss={loss.item():.4f} "
                          f"(task={losses['task'].item():.4f} "
                          f"comp={losses['compression'].item():.4f} "
                          f"spec={losses['spectral'].item():.4f}) val={vm:.4f}")
                else:
                    print(f"    Epoch {epoch:03d} | loss={loss.item():.4f} val={vm:.4f}")
            if patience_ctr >= patience:
                if verbose: print(f"    Early stop @ epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return evaluate(model, X, H_np, W_e, labels_t, test_mask,
                    task=task, measure_time=True)


# =============================================================================
# SECTION 12 — 5-FOLD CROSS VALIDATION + STATISTICAL TESTING
# =============================================================================

def run_five_fold_cv(H, X, W_e, labels, valid_indices, task,
                      feat_dim, hidden_dim, output_dim,
                      cfg, n_splits=5) -> Tuple[list, list]:
    """
    5-fold stratified CV (classification) or regular KFold (regression).
    Returns (qadapt_results, baseline_results).
    """
    X_t = torch.FloatTensor(X)
    if task == 'classification':
        kf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(valid_indices, labels[valid_indices]))
    else:
        kf     = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(valid_indices))

    qa_res, bl_res = [], []

    for fold, (trval_idx, test_idx) in enumerate(splits):
        print(f"\n  Fold {fold+1}/{n_splits}")
        trval      = valid_indices[trval_idx]
        test_ids   = valid_indices[test_idx]
        val_size   = max(1, len(trval) // 5)
        val_ids, train_ids = trval[:val_size], trval[val_size:]

        def mk(ids):
            m = torch.zeros(len(labels), dtype=torch.bool)
            m[ids] = True
            return m

        # QAdapt
        qm = QAdaptNet(feat_dim, hidden_dim, output_dim,
                       num_layers=cfg['num_layers'], dropout=cfg['dropout'],
                       K=cfg['K'], N_neg=cfg['N_neg'],
                       lambda1=cfg['lambda1'], lambda2=cfg['lambda2'])
        qa_res.append(train_model(
            qm, X_t, H, W_e, labels,
            mk(train_ids), mk(val_ids), mk(test_ids),
            task=task, num_epochs=cfg['num_epochs'], model_type='qadapt'
        ))

        # Baseline
        bm = BaselineHGNN(feat_dim, hidden_dim, output_dim,
                          num_layers=cfg['num_layers'], dropout=cfg['dropout'])
        bl_res.append(train_model(
            bm, X_t, H, W_e, labels,
            mk(train_ids), mk(val_ids), mk(test_ids),
            task=task, num_epochs=cfg['num_epochs'], model_type='baseline'
        ))

    return qa_res, bl_res


# =============================================================================
# SECTION 13 — DATASET CONFIGS + MAIN
# =============================================================================

DATASET_CONFIGS = {
    'imdb':   {'task':'classification', 'hidden_dim':128, 'num_layers':2,
               'dropout':0.5, 'K':32, 'N_neg':64, 'num_epochs':200,
               'lambda1':0.01, 'lambda2':0.001},
    'dblp':   {'task':'classification', 'hidden_dim':256, 'num_layers':2,
               'dropout':0.5, 'K':32, 'N_neg':64, 'num_epochs':200,
               'lambda1':0.01, 'lambda2':0.001},
    'acm':    {'task':'classification', 'hidden_dim':128, 'num_layers':2,
               'dropout':0.3, 'K':32, 'N_neg':64, 'num_epochs':200,
               'lambda1':0.01, 'lambda2':0.001},
    'amazon': {'task':'regression',     'hidden_dim':128, 'num_layers':2,
               'dropout':0.3, 'K':32, 'N_neg':64, 'num_epochs':200,
               'lambda1':0.01, 'lambda2':0.001},
    'yelp':   {'task':'regression',     'hidden_dim':128, 'num_layers':2,
               'dropout':0.3, 'K':32, 'N_neg':64, 'num_epochs':200,
               'lambda1':0.01, 'lambda2':0.001},
}


def run_dataset(name, args):
    print(f"\n{'='*70}\n  Dataset: {name.upper()}\n{'='*70}")
    cfg  = DATASET_CONFIGS.get(name, DATASET_CONFIGS['imdb'])
    task = cfg['task']

    # Resolve per-dataset subfolder automatically if --data_path is a root folder
    # e.g.  --data_path /data  →  tries /data/IMDB, /data/imdb, /data directly
    folder = None
    if args.data_path:
        candidates = [
            args.data_path,
            os.path.join(args.data_path, name.upper()),
            os.path.join(args.data_path, name),
            os.path.join(args.data_path, name.capitalize()),
        ]
        for c in candidates:
            if os.path.isdir(c):
                folder = c
                break

    H, X, W_e, labels, num_classes = load_dataset(
        name, data_path=folder,
        max_rows=args.max_rows, feature_dim=args.feature_dim
    )

    feat_dim   = X.shape[1]
    output_dim = num_classes if task == 'classification' else 1
    valid      = (np.where(labels >= 0)[0]
                  if task == 'classification'
                  else np.arange(len(labels)))

    print(f"\n  Running 5-fold cross-validation ...")
    qa_res, bl_res = run_five_fold_cv(
        H, X, W_e, labels, valid, task,
        feat_dim, cfg['hidden_dim'], output_dim, cfg
    )

    summary = statistical_summary(qa_res, bl_res, task)
    print_summary_table(summary, task)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='QAdapt — single-file experiment runner',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # With real data
  python qadapt.py --dataset imdb   --data_path /data/IMDB
  python qadapt.py --dataset amazon --data_path /data/Amazon
  python qadapt.py --dataset all    --data_path /data

  # Without data (synthetic fallback — no crash)
  python qadapt.py --dataset imdb
  python qadapt.py --dataset all

  # Quick debug
  python qadapt.py --dataset imdb --data_path /data/IMDB --max_rows 200
        """
    )
    parser.add_argument('--dataset',
                        type=str, default='imdb',
                        choices=['imdb','dblp','acm','amazon','yelp','all'],
                        help='Dataset to run (default: imdb)')
    parser.add_argument('--data_path',
                        type=str, default=None,
                        help='Path to dataset folder (optional — '
                             'uses synthetic data if not provided or folder missing)')
    parser.add_argument('--max_rows',
                        type=int, default=None,
                        help='Limit rows per file for quick debugging')
    parser.add_argument('--feature_dim',
                        type=int, default=64,
                        help='Node feature dimension (default: 64)')
    parser.add_argument('--n_nodes',
                        type=int, default=500,
                        help='Nodes for synthetic datasets (default: 500)')
    parser.add_argument('--n_hyperedges',
                        type=int, default=200,
                        help='Hyperedges for synthetic datasets (default: 200)')
    parser.add_argument('--seed',
                        type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets = ['imdb','dblp','acm','amazon','yelp'] \
               if args.dataset == 'all' else [args.dataset]

    print("\n" + "=" * 70)
    print("  QAdapt: Information-Theoretic Hypergraph Quantization")
    print(f"  Datasets : {datasets}")
    print(f"  Data path: {args.data_path or '(none — synthetic fallback)'}")
    print("=" * 70)

    all_summaries = {}
    for ds in datasets:
        try:
            all_summaries[ds] = run_dataset(ds, args)
        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    # Cross-dataset overview
    if len(all_summaries) > 1:
        print("\n\n" + "=" * 70)
        print("  CROSS-DATASET OVERVIEW")
        print("=" * 70)
        print(f"{'Dataset':<10} {'Task':<16} {'Primary Metric':>16} "
              f"{'Comp. Ratio':>13} {'Sig?':>6}")
        print("-" * 70)
        for ds, summ in all_summaries.items():
            task = DATASET_CONFIGS[ds]['task']
            pk   = 'acc' if task == 'classification' else 'mae'
            if pk not in summ: continue
            pm  = summ[pk]['qadapt_mean']
            cr  = summ.get('comp_ratio', {}).get('qadapt_mean', 0.0)
            sig = "Yes**" if summ[pk]['significant'] else "No"
            print(f"{ds:<10} {task:<16} {pm:>16.4f} {cr:>13.2f} {sig:>6}")


if __name__ == '__main__':
    main()