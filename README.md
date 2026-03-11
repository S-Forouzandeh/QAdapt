# QAdapt

# QAdapt: Information-Theoretic Adaptive Quantization for Hypergraph Neural Networks

> **Faithful implementation of the full QAdapt theoretical framework.**  
> Every equation from the paper is implemented exactly as described, correcting the misalignments present in the original prototype code.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Fixes vs. Original Code](#key-fixes-vs-original-code)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Datasets](#datasets)
8. [Metrics](#metrics)
9. [Configuration](#configuration)

---

## Overview

QAdapt is a three-stage framework for efficient hypergraph neural network inference through adaptive mixed-precision quantization:

| Stage | Component | What it does |
|---|---|---|
| **1** | Information Density Estimation | Scores each node–hyperedge pair with $\rho_{i,e} = \text{IC}(\mathbf{x}_i,\mathbf{h}_e)\cdot\text{SW}(i,e)$ via contrastive MI + spectral weights |
| **2** | Multi-Scale Attention + SpectralFusion | Combines intra-hyperedge and node-level attention, fused through $\boldsymbol{\Phi}\,\text{diag}(\boldsymbol{\omega})\,\boldsymbol{\Phi}^\top$ |
| **3** | Co-Adaptive Quantization | Predicts per-attention-entry bit-widths $\in\{4,8,16\}$ using Fisher sensitivity + $\rho$ + spectral structure features |

**Joint training objective:**
$$\mathcal{L} = \mathcal{L}_\text{task} + \lambda_1\mathcal{L}_\text{compression} + \lambda_2\mathcal{L}_\text{spectral}$$

---

## Architecture

```
Input X (n × d)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  QAdapt Layer                                                   │
│                                                                 │
│  Step 1: Information Density                                    │
│    W_ctx (shared, O(d²)) → h_e^ctx = MeanPool({W_ctx x_j})    │
│    IC via InfoNCE (N=64 negatives)                              │
│    SW(i,e) = Σ_k α_k φ_k(i) · 1_e(i)   [Laplacian eigvecs]   │
│    ρ_{i,e} = IC × SW                                           │
│                                                                 │
│  Step 2: Attention                                              │
│    A^hyper: softmax((P_e x_i)ᵀ(P_e x_j)/√d + α log ρ)        │
│    A^node:  softmax((W x_i)ᵀ(W x_j)/√d   + α log ρ̄)          │
│    SpectralFusion: Φ diag(ω) Φᵀ (A^hyper + A^node)            │
│                                                                 │
│  Step 3: Co-Adaptive Quantization                               │
│    Fisher EMA: S = EMA((∂L/∂A_ij)², β=0.99)                   │
│    MLP_alloc([S; ρ; Structure(i,j); s_global]) → β ∈ {4,8,16} │
│    Gumbel-Softmax τ: 2.0 → 0.1 (hard after epoch 200)         │
│    Q_adaptive = Σ_b β^(b) Q(A; b, s^(b))                      │
│                                                                 │
│  Output: σ(A_quant · X · Θ)                                    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
Output Layer → logits (n × num_classes)
```

---

## Key Fixes vs. Original Code

The original prototype code diverged from the paper in 7 critical ways. This implementation corrects all of them:

| # | Component | Original Code (Wrong) | This Code (Correct) |
|---|---|---|---|
| 1 | **Quantization target** | Node feature outputs $\mathbf{X}$ | Attention matrices $\mathbf{A}$ as per paper |
| 2 | **SpectralFusion** | Plain MLP fusion | $\boldsymbol{\Phi}\,\text{diag}(\boldsymbol{\omega})\,\boldsymbol{\Phi}^\top$ with K=32 Laplacian eigenvectors |
| 3 | **Information density $\rho$** | Sigmoid compatibility score | InfoNCE contrastive MI × spectral SW |
| 4 | **Structural weight SW** | Not implemented | $\sum_k \alpha_k \phi_k(i)\cdot\mathbf{1}_e(i)$ with Laplacian eigenvectors |
| 5 | **Fisher sensitivity** | `abs(attention)` (incorrect proxy) | EMA of $({\partial\mathcal{L}}/{\partial A_{ij}})^2$, $\beta=0.99$ |
| 6 | **Bit-width input features** | No spectral, no degree stats | Full $\mathbf{f}_{ij}=[S^\text{Fisher};\rho;\phi_\text{local}(i,j);s_\text{global}]$ |
| 7 | **Loss $\mathcal{L}_\text{spectral}$** | Missing | $\|\mathbf{A}_\text{final}-\mathbf{A}_\text{quant}\|_F / \|\mathbf{A}_\text{final}\|_F$ |

---

## Project Structure

```
qadapt/
├── models/
│   └── qadapt.py              # Full model: all paper equations
│       ├── compute_laplacian_eigenpairs()   # ARPACK, K=32, tol=1e-6
│       ├── CriticNetwork                   # f_θ: R^2d → R
│       ├── InformationDensityEstimator     # Step 1: ρ = IC × SW
│       ├── SpectralFusionMLP               # Step 2: Φ diag(ω) Φᵀ
│       ├── CoAdaptiveQuantizer             # Step 3: MLP_alloc + Gumbel
│       ├── QAdaptConv                      # One QAdapt layer
│       └── QAdaptNet                       # Full model + joint loss
│
├── utils/
│   └── metrics.py             # All Table 1 metrics
│       ├── information_retention_score()
│       ├── spectral_preservation_score()
│       ├── compression_ratio()
│       ├── measure_inference_time()
│       ├── evaluate()                      # All metrics in one call
│       ├── five_fold_cv()                  # 5-fold CV with stat tests
│       ├── statistical_summary()           # Paired t-test + Cohen's d
│       └── print_summary_table()
│
├── data/
│   └── loader.py              # IMDB xlsx loader + synthetic datasets
│       ├── load_imdb_data()
│       ├── build_imdb_hypergraph()         # H, X, labels
│       └── make_synthetic_dataset()        # DBLP/Amazon/Yelp/ACM fallback
│
├── experiments/
│   ├── trainer.py             # Training loop with MI update interval
│   │   ├── build_qadapt()
│   │   ├── build_baseline()              # Standard HGNN (no quantization)
│   │   └── train_model()
│   └── run_experiments.py     # Main entry point
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/qadapt.git
cd qadapt
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
openpyxl>=3.1.0
```

---

## Quick Start

### IMDB Dataset (Classification)

```bash
python experiments/run_experiments.py \
    --dataset imdb \
    --data_path /path/to/IMDB
```

### Synthetic Dataset (Regression, e.g. Yelp/Amazon)

```bash
python experiments/run_experiments.py \
    --dataset amazon \
    --n_nodes 500 \
    --n_hyperedges 200
```

### All Datasets

```bash
python experiments/run_experiments.py \
    --dataset all \
    --data_path /path/to/IMDB
```

### Debug Mode (limit rows)

```bash
python experiments/run_experiments.py \
    --dataset imdb \
    --data_path /path/to/IMDB \
    --max_rows 200
```

### Programmatic API

```python
import torch
import numpy as np
from models.qadapt import QAdaptNet, compute_laplacian_eigenpairs
from utils.metrics import evaluate

# Build model
model = QAdaptNet(
    input_dim=64, hidden_dim=128, output_dim=5,
    num_layers=2, dropout=0.5, K=32, N_neg=64,
    lambda1=0.01, lambda2=0.001
)

# Pre-compute spectral information (do once, before training)
H_np = ...       # (n, m) numpy incidence matrix
W_e  = ...       # (m,)   hyperedge weights
eigvals, eigvecs = compute_laplacian_eigenpairs(H_np, W_e, K=32)
degrees = H_np.sum(axis=1).astype(np.float32)
model.set_spectral(eigvals, eigvecs, degrees)

# Forward pass
X = torch.FloatTensor(...)   # (n, 64)
out = model(X, H_np, W_e)
logits = out['logits']          # (n, num_classes)

# Access per-layer quantization details
for i, lo in enumerate(out['layer_outputs']):
    print(f"Layer {i}: expected bits = {lo['exp_bits'].mean():.2f}")
    print(f"         rho shape      = {lo['rho'].shape}")
    print(f"         A_final shape  = {lo['A_final'].shape}")
```

---

## Datasets

| Dataset | Task | Nodes | Hyperedges | Classes | Source |
|---|---|---|---|---|---|
| **IMDB** | Classification | ~500+ | ~400+ | genres | xlsx files |
| **DBLP** | Classification | 66,543 | 22,363 | 4 | [DBLP](https://dblp.uni-trier.de/) |
| **ACM** | Classification | 3,025 | 3,025 | 3 | [ACM](https://dl.acm.org/) |
| **Amazon** | Regression | variable | variable | — | [Amazon](https://amazon.com/) |
| **Yelp** | Regression | variable | variable | — | [Yelp](https://yelp.com/) |

For datasets other than IMDB, synthetic fallback data is generated automatically when real data is unavailable (use `--dataset dblp`, `--dataset amazon`, etc.).

**Expected IMDB folder structure:**
```
/path/to/IMDB/
├── user_movies.xlsx       [userID, movieID, rating]
├── movie_directors.xlsx   [movieID, directorID]
├── movie_actors.xlsx      [movieID, actorID]
└── movie_genres.xlsx      [movieID, genreID]
```

---

## Metrics

All metrics from **Table 1** of the paper are computed:

### Classification (IMDB, DBLP, ACM)
| Metric | Description | Implementation |
|---|---|---|
| **Acc** | Top-1 accuracy | `accuracy_score` |
| **F1** | Macro-averaged F1 | `f1_score(average='macro')` |
| **AUC** | Macro OvR AUC | `roc_auc_score(multi_class='ovr')` |

### Regression (Amazon, Yelp)
| Metric | Description | Implementation |
|---|---|---|
| **MAE** | Mean absolute error | `mean_absolute_error` |
| **RMSE** | Root mean squared error | `sqrt(mean_squared_error)` |
| **R²** | Coefficient of determination | `r2_score` |

### Efficiency
| Metric | Description | Formula |
|---|---|---|
| **Time (ms)** | Avg inference time per batch | 50 runs, 10 warmup, CUDA sync |
| **Comp. Ratio** | Compression vs FP16 | `16.0 / mean(expected_bits)` |

### Theory (Information-Theoretic)
| Metric | Description | Formula |
|---|---|---|
| **Info Retain** | Information preservation | $1 - \|\rho_\text{orig}-\rho_\text{quant}\|_1 / \|\rho_\text{orig}\|_1$ |
| **Spec Pres** | Spectral preservation | $1 - \|\mathbf{A}_\text{orig}-\mathbf{A}_\text{quant}\|_F / \|\mathbf{A}_\text{orig}\|_F$ |

### Statistical Testing
- **5-fold cross-validation** (stratified for classification)
- **Paired t-test** vs. baseline, threshold $p < 0.01$
- **Cohen's d** effect size reported

---

## Configuration

Key hyperparameters (all in `experiments/run_experiments.py`):

```python
DATASET_CONFIGS = {
    'imdb': {
        'task': 'classification',
        'hidden_dim': 128,       # Model hidden dimension
        'num_layers': 2,         # Number of QAdapt convolution layers
        'dropout': 0.5,
        'K': 32,                 # Laplacian eigenpairs retained (ARPACK)
        'N_neg': 64,             # Negative samples for InfoNCE
        'num_epochs': 200,       # Training epochs (hard Gumbel after 200)
        'lambda1': 0.01,         # Compression loss weight
        'lambda2': 0.001,        # Spectral preservation loss weight
    },
    ...
}
```

**Gumbel temperature schedule:**
```
τ(t) = max(0.1, 2.0 × 0.95^(t/100))
Hard sampling: after epoch 200
```

**MI network update:** every 5 main-model iterations.

---


## License

MIT License. See `LICENSE` for details.
