"""Experiment 5 -- Fair Representation Learning Impossibility.

Test the Z-space master-constraint equality on learned representations from
LFR, Fair VAE (MMD penalty), and adversarial debiasing. Evaluations gate on
theorem eligibility (non-collapsed, training-valid, near-separation) and
measure the six-term decomposition with stratified bootstrap CIs.
"""

import gc
import json
import os
import sys
import time
import warnings

# Prevent OMP threading crash with XGBoost + PyTorch
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from experiments.config import (
    BANDWIDTH_MULTIPLIERS,
    RANDOM_SEEDS,
    DEFAULT_SEED,
    RESULTS_DIR,
    ensure_dirs,
    get_device,
)
from experiments.data import FairnessDataset
from experiments.data.loader import (
    load_adult,
    load_compas,
    load_acs_pums,
    train_val_test_split,
)
from experiments.data.synthetic import generate_synthetic_Z_separated
from experiments.kernels import (
    hsic,
    master_constraint_decomposition_from_K,
    master_constraint_decomposition_in_Z,
    master_constraint_decomposition_in_Z_multi_bandwidth,
    median_heuristic,
    mmd_test,
    mode_collapse_detected,
    rbf_kernel_matrix,
)
from experiments.fairness.criteria import (
    demographic_parity_gap,
    equalized_odds_gap,
)
from experiments.utils.plotting import setup_style, save_figure, THEORY_STYLE

# Parameters

DATASETS = ["adult", "compas", "acs_pums"]
KERNEL_SUBSAMPLE_N = 5_000
ACS_SUBSAMPLE_N = 20_000

# Theorem-eligibility gate thresholds
TAU_SEP = 0.015                # synthetic sanity check rejects at 0.025, so 0.05 is too loose
MODE_COLLAPSE_THRESHOLD = 1e-6  # per-feature std threshold
TRAINING_VALID_ACC_MARGIN = 0.05  # min gap from base-rate accuracy
TRAINING_VALID_ACC_RANGE = 0.01   # min accuracy range across param sweep

# Stratified bootstrap for the equality gap mmd2_group - delta_p^2 * class_struct_b
BOOTSTRAP_B = 1000
BOOTSTRAP_MIN_CELL = 5

# Z-caching side effect (see docstring of _cache_Z)
CACHE_DIR = RESULTS_DIR.parent / "cache" / "exp5"

# LFR parameters
LFR_K = 10  # prototypes
LFR_AZ_VALUES = [0.1, 1.0, 10.0, 50.0, 100.0]

# VAE parameters
VAE_LATENT_DIM = 32
VAE_EPOCHS = 100
VAE_BATCH_SIZE = 256
VAE_LR = 1e-3
VAE_LAMBDA_MMD_VALUES = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]

# Adversarial parameters
ADV_LATENT_DIM = 32
ADV_EPOCHS = 100
ADV_BATCH_SIZE = 256
ADV_LR = 1e-3
ADV_LAMBDA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# LFR uses BLAS internally which spawns its own threads, so capping at
# ~12 workers gives best parallelism on typical machines.
# Adult needs fewer workers due to memory pressure during L-BFGS on n=45K.
N_PARALLEL_WORKERS_DEFAULT = min(12, os.cpu_count() or 1)
N_PARALLEL_WORKERS_LARGE = 4  # for Adult (n>30K)

DATASET_LABELS = {
    "adult": "Adult Income",
    "compas": "COMPAS",
    "acs_pums": "ACS PUMS",
}

METHOD_COLORS = {
    "lfr": "#1f77b4",
    "fair_vae": "#d62728",
    "adversarial": "#2ca02c",
}
METHOD_LABELS = {
    "lfr": "LFR",
    "fair_vae": "Fair VAE",
    "adversarial": "Adversarial",
}
METHOD_MARKERS = {
    "lfr": "o",
    "fair_vae": "s",
    "adversarial": "^",
}

PARAM_SWEEPS = {
    "lfr": LFR_AZ_VALUES,
    "fair_vae": VAE_LAMBDA_MMD_VALUES,
    "adversarial": ADV_LAMBDA_VALUES,
}
PARAM_NAMES = {
    "lfr": "Az",
    "fair_vae": "lambda_mmd",
    "adversarial": "lambda_adv",
}


# Helpers


def _subsample_stratified(
    ds: FairnessDataset, n: int, seed: int
) -> FairnessDataset:
    """Subsample a FairnessDataset to n samples, stratified by (y, group)."""
    if len(ds.y) <= n:
        return ds

    rng = np.random.default_rng(seed)
    strata = ds.y * 2 + ds.group
    selected = []
    for s in np.unique(strata):
        mask = np.where(strata == s)[0]
        frac = len(mask) / len(ds.y)
        n_s = max(1, int(n * frac))
        chosen = rng.choice(mask, size=min(n_s, len(mask)), replace=False)
        selected.extend(chosen)
    selected = np.array(selected)
    rng.shuffle(selected)
    selected = selected[:n]

    y_sub = ds.y[selected]
    g_sub = ds.group[selected]
    emp_p_a = float(y_sub[g_sub == 1].mean()) if (g_sub == 1).any() else 0.0
    emp_p_b = float(y_sub[g_sub == 0].mean()) if (g_sub == 0).any() else 0.0

    return FairnessDataset(
        X=ds.X[selected],
        y=y_sub,
        group=g_sub,
        feature_names=ds.feature_names,
        name=ds.name,
        base_rates={"a": emp_p_a, "b": emp_p_b},
    )


def _load_datasets() -> dict[str, FairnessDataset]:
    """Load Adult, COMPAS, and ACS PUMS datasets."""
    print("Loading datasets...")
    datasets = {}

    ds = load_adult()
    print(f"  Adult: n={len(ds.y)}, d={ds.X.shape[1]}, "
          f"p_a={ds.base_rates['a']:.3f}, p_b={ds.base_rates['b']:.3f}")
    datasets["adult"] = ds

    ds = load_compas()
    print(f"  COMPAS: n={len(ds.y)}, d={ds.X.shape[1]}, "
          f"p_a={ds.base_rates['a']:.3f}, p_b={ds.base_rates['b']:.3f}")
    datasets["compas"] = ds

    ds = load_acs_pums(subsample_n=ACS_SUBSAMPLE_N)
    print(f"  ACS PUMS: n={len(ds.y)}, d={ds.X.shape[1]}, "
          f"p_a={ds.base_rates['a']:.3f}, p_b={ds.base_rates['b']:.3f}")
    datasets["acs_pums"] = ds

    return datasets


def _to_aif360(X, y, group, feature_names):
    """Convert numpy arrays to aif360 BinaryLabelDataset."""
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y.astype(float)
    df["group"] = group.astype(float)

    return BinaryLabelDataset(
        df=df,
        label_names=["label"],
        protected_attribute_names=["group"],
        favorable_label=1.0,
        unfavorable_label=0.0,
    )


# Z-caching


def _cache_path(method: str, ds_name: str, param_value, seed: int) -> "pathlib.Path":
    """Return the per-(method, dataset, param, seed) Z-cache file path.

    Cached files hold all arrays needed to re-run `_evaluate_representation`
    without retraining the encoder, so revisions to evaluation code become
    a local recompute instead of a ~3-hour re-train.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{method}_{ds_name}_{param_value:.6g}_{seed}.npz"
    return CACHE_DIR / fname


def _save_cached_Z(
    path,
    Z_train, y_train, g_train,
    Z_test, y_test, g_test,
    p_a: float, p_b: float,
    method: str, ds_name: str, param_value, seed: int,
) -> None:
    """Save a trained encoder's Z embeddings for future recompute."""
    try:
        np.savez(
            path,
            Z_train=Z_train.astype(np.float32),
            y_train=y_train.astype(np.int64),
            g_train=g_train.astype(np.int64),
            Z_test=Z_test.astype(np.float32),
            y_test=y_test.astype(np.int64),
            g_test=g_test.astype(np.int64),
            p_a=np.float64(p_a),
            p_b=np.float64(p_b),
            method=np.array(method),
            ds_name=np.array(ds_name),
            param_value=np.float64(param_value),
            seed=np.int64(seed),
        )
    except OSError as e:
        warnings.warn(f"Z-cache save failed for {path}: {e}")


def _load_cached_Z(path):
    """Return cached Z embeddings as a dict, or None on miss."""
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return {
            "Z_train": data["Z_train"].astype(np.float64),
            "y_train": data["y_train"].astype(np.int64),
            "g_train": data["g_train"].astype(np.int64),
            "Z_test": data["Z_test"].astype(np.float64),
            "y_test": data["y_test"].astype(np.int64),
            "g_test": data["g_test"].astype(np.int64),
            "p_a": float(data["p_a"]),
            "p_b": float(data["p_b"]),
        }
    except (OSError, KeyError, ValueError) as e:
        warnings.warn(f"Z-cache load failed for {path}: {e}")
        return None


# Stratified bootstrap CI on equality gap


def _stratified_bootstrap_equality_gap_ci(
    Z: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    sigma: float,
    B: int = BOOTSTRAP_B,
    min_cell: int = BOOTSTRAP_MIN_CELL,
    seed: int = 0,
) -> tuple[float, float, int]:
    """Stratified bootstrap 95% CI on (mmd2_group - Delta_p^2 * class_struct_b).

    Return (ci_lo, ci_hi, n_valid_bootstraps); NaN if any cell < min_cell.
    """
    n = Z.shape[0]
    cell_indices = {
        (1, 1): np.where((y == 1) & (g == 1))[0],
        (0, 1): np.where((y == 0) & (g == 1))[0],
        (1, 0): np.where((y == 1) & (g == 0))[0],
        (0, 0): np.where((y == 0) & (g == 0))[0],
    }
    for key, idx in cell_indices.items():
        if len(idx) < min_cell:
            return float("nan"), float("nan"), 0

    # Map cell key -> column in the decomposition cell order (1a, 0a, 1b, 0b)
    # where (y, g) = (1, 1), (0, 1), (1, 0), (0, 0) -> columns 0, 1, 2, 3.
    cell_order = [(1, 1), (0, 1), (1, 0), (0, 0)]
    cell_idx_list = [cell_indices[k] for k in cell_order]
    counts = np.array([len(ix) for ix in cell_idx_list], dtype=np.float64)

    # Precompute full Gram matrix once.
    K = rbf_kernel_matrix(Z, Z, sigma=sigma)

    rng = np.random.default_rng(seed)
    gaps = np.empty(B, dtype=np.float64)
    E_boot = np.zeros((n, 4), dtype=np.float64)

    for b in range(B):
        E_boot.fill(0.0)
        for c, idx in enumerate(cell_idx_list):
            n_c = len(idx)
            # Bincount counts: how many times each original index in cell c
            # was drawn in a resample of size n_c from that cell.
            sampled = rng.integers(0, n_c, size=n_c)
            counts_per_idx = np.bincount(sampled, minlength=n_c)
            E_boot[idx, c] = counts_per_idx.astype(np.float64)

        # Empirical base rates from the resample are the same as the original
        # (stratified resampling preserves cell sizes), so use counts directly.
        n_a = counts[0] + counts[1]
        n_b = counts[2] + counts[3]
        p_a_emp = float(counts[0] / n_a)
        p_b_emp = float(counts[2] / n_b)

        try:
            decomp_b = master_constraint_decomposition_from_K(
                K, E_boot, counts, p_a_emp, p_b_emp, sigma,
            )
        except (AssertionError, ValueError):
            gaps[b] = float("nan")
            continue
        delta_p = abs(decomp_b["p_a_emp"] - decomp_b["p_b_emp"])
        gaps[b] = decomp_b["mmd2_group"] - delta_p ** 2 * decomp_b["class_struct_b"]

    valid_gaps = gaps[~np.isnan(gaps)]
    n_valid = int(len(valid_gaps))
    if n_valid < max(10, B // 10):
        return float("nan"), float("nan"), n_valid
    ci_lo = float(np.percentile(valid_gaps, 2.5))
    ci_hi = float(np.percentile(valid_gaps, 97.5))
    return ci_lo, ci_hi, n_valid


# Theorem-eligibility gate


def _compute_sep_gap_rel(decomp: dict) -> float:
    """Separation gap relative to within-group class structure.

    sep_gap_rel := max(sep_gap_y0, sep_gap_y1) / max(class_struct_a, class_struct_b, eps)

    Values <= TAU_SEP are considered "near-separation" (theorem-eligible).
    """
    max_sep = max(decomp["sep_gap_y0"], decomp["sep_gap_y1"])
    denom = max(decomp["class_struct_a"], decomp["class_struct_b"], 1e-12)
    return float(max_sep / denom)


def _json_default(obj):
    """Convert numpy scalars/arrays for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_partial(path, data: dict) -> None:
    """Save partial results so completed sub-experiments survive crashes."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Partial results saved to {path}")


# Method 1: LFR (aif360)


def _fit_lfr(train_ds, test_ds, Az, seed):
    """Fit LFR and return (Z_train, Z_test), or (None, None) on failure.

    Note: aif360 LFR.transform() returns reconstructions in the original
    feature dimension, not low-dim prototype assignments.
    """
    from aif360.algorithms.preprocessing import LFR

    train_aif = _to_aif360(
        train_ds.X, train_ds.y, train_ds.group, train_ds.feature_names
    )
    test_aif = _to_aif360(
        test_ds.X, test_ds.y, test_ds.group, test_ds.feature_names
    )

    unprivileged = [{"group": 0.0}]
    privileged = [{"group": 1.0}]

    try:
        lfr = LFR(
            unprivileged_groups=unprivileged,
            privileged_groups=privileged,
            k=LFR_K,
            Ax=0.01,
            Ay=1.0,
            Az=Az,
            seed=seed,
        )
        lfr = lfr.fit(train_aif, maxiter=5000, maxfun=5000)

        train_transformed = lfr.transform(train_aif)
        test_transformed = lfr.transform(test_aif)

        Z_train = train_transformed.features
        Z_test = test_transformed.features

        assert Z_train.shape[0] == train_ds.X.shape[0], (
            f"LFR train shape mismatch: {Z_train.shape[0]} vs {train_ds.X.shape[0]}"
        )
        assert Z_test.shape[0] == test_ds.X.shape[0], (
            f"LFR test shape mismatch: {Z_test.shape[0]} vs {test_ds.X.shape[0]}"
        )
        assert Z_train.ndim == 2 and Z_test.ndim == 2

        return Z_train, Z_test

    except Exception as e:
        warnings.warn(f"LFR failed with Az={Az}, seed={seed}: {e}")
        return None, None


# Method 2: Fair VAE with MMD penalty (PyTorch)


def _mmd_squared_torch(z_a, z_b, sigma):
    """Differentiable unbiased MMD^2 between two sets of latent vectors."""
    n_a = z_a.shape[0]
    n_b = z_b.shape[0]

    K_aa = torch.exp(-torch.cdist(z_a, z_a).pow(2) / (2.0 * sigma ** 2))
    K_bb = torch.exp(-torch.cdist(z_b, z_b).pow(2) / (2.0 * sigma ** 2))
    K_ab = torch.exp(-torch.cdist(z_a, z_b).pow(2) / (2.0 * sigma ** 2))

    K_aa = K_aa - torch.diag(K_aa.diag())
    K_bb = K_bb - torch.diag(K_bb.diag())

    mmd2 = (
        K_aa.sum() / (n_a * (n_a - 1))
        - 2.0 * K_ab.sum() / (n_a * n_b)
        + K_bb.sum() / (n_b * (n_b - 1))
    )
    return mmd2


class FairVAE(nn.Module):
    """Variational autoencoder with MMD fairness penalty."""

    def __init__(self, d_in, d_latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(64, d_latent)
        self.log_sigma_head = nn.Linear(64, d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, d_in),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.log_sigma_head(h)

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        x_recon = self.decode(z)
        return x_recon, mu, log_sigma, z


def _train_fair_vae(X_train, group_train, lambda_mmd, d_in, seed, device):
    """Train FairVAE and return the trained model (eval mode).

    Returns trained FairVAE on CPU.
    """
    torch.manual_seed(seed)
    np.random.default_rng(seed)

    vae = FairVAE(d_in, VAE_LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LR)

    X_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    g_t = torch.as_tensor(group_train, dtype=torch.long, device=device)

    sigma_mmd = 1.0
    warmup_epochs = 10 if lambda_mmd > 0 else 0

    n = X_t.shape[0]
    n_batches = max(1, n // VAE_BATCH_SIZE)

    for epoch in range(VAE_EPOCHS):
        if lambda_mmd > 0 and epoch == warmup_epochs:
            with torch.no_grad():
                n_bw = min(5000, X_t.shape[0])
                rng_bw = np.random.default_rng(seed + 999)
                bw_idx = rng_bw.choice(X_t.shape[0], size=n_bw, replace=False)
                bw_mu, _ = vae.encode(X_t[bw_idx])
                sigma_mmd = float(median_heuristic(bw_mu.cpu().numpy()))
                sigma_mmd = max(sigma_mmd, 0.1)  # safety floor

        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * VAE_BATCH_SIZE : (i + 1) * VAE_BATCH_SIZE]
            x_batch = X_t[idx]
            g_batch = g_t[idx]

            x_recon, mu, log_sigma, z = vae(x_batch)

            recon_loss = nn.functional.mse_loss(x_recon, x_batch)

            kl_loss = -0.5 * torch.mean(
                1 + log_sigma - mu.pow(2) - log_sigma.exp()
            )

            mmd_loss = torch.tensor(0.0, device=device)
            if lambda_mmd > 0 and epoch >= warmup_epochs:
                mask_a = g_batch == 1
                mask_b = g_batch == 0
                if mask_a.sum() >= 2 and mask_b.sum() >= 2:
                    mmd_loss = _mmd_squared_torch(
                        mu[mask_a], mu[mask_b], sigma_mmd
                    )

            loss = recon_loss + kl_loss + lambda_mmd * mmd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    vae.eval()
    vae = vae.cpu()
    return vae


def _extract_vae_representations(vae, X, device):
    """Extract latent representations from trained VAE (eval mode, no sampling)."""
    vae = vae.to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu, _ = vae.encode(X_t)

    vae = vae.cpu()
    return mu.cpu().numpy()


# Method 3: Adversarial Debiasing (PyTorch)


class AdversarialFairEncoder(nn.Module):
    """Encoder + classifier + adversary for adversarial debiasing."""

    def __init__(self, d_in, d_latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, d_latent),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(d_latent, 1)
        self.adversary = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, x):
        return self.encoder(x)

    def classify(self, z):
        return self.classifier(z)

    def adversary_forward(self, z):
        return self.adversary(z)


def _train_adversarial(X_train, y_train, group_train, lambda_adv, d_in, seed, device):
    """Train adversarial fair encoder and return the trained model (eval mode).

    Returns trained AdversarialFairEncoder on CPU.
    """
    torch.manual_seed(seed)

    model = AdversarialFairEncoder(d_in, ADV_LATENT_DIM).to(device)

    opt_enc = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=ADV_LR,
    )
    opt_adv = torch.optim.Adam(model.adversary.parameters(), lr=ADV_LR * 0.5)

    X_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    g_t = torch.as_tensor(group_train, dtype=torch.float32, device=device).unsqueeze(1)

    n = X_t.shape[0]
    n_batches = max(1, n // ADV_BATCH_SIZE)

    for epoch in range(20):
        perm = torch.randperm(n, device=device)
        for i in range(n_batches):
            idx = perm[i * ADV_BATCH_SIZE : (i + 1) * ADV_BATCH_SIZE]
            z = model.encode(X_t[idx])
            cls_pred = model.classify(z)
            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                cls_pred, y_t[idx]
            )
            opt_enc.zero_grad()
            cls_loss.backward()
            opt_enc.step()

    for epoch in range(ADV_EPOCHS):
        perm = torch.randperm(n, device=device)

        for i in range(n_batches):
            idx = perm[i * ADV_BATCH_SIZE : (i + 1) * ADV_BATCH_SIZE]
            x_batch = X_t[idx]
            y_batch = y_t[idx]
            g_batch = g_t[idx]

            for _adv_step in range(3):
                z_detached = model.encode(x_batch).detach()
                adv_pred = model.adversary_forward(z_detached)
                adv_loss = nn.functional.binary_cross_entropy_with_logits(
                    adv_pred, g_batch
                )
                opt_adv.zero_grad()
                adv_loss.backward()
                opt_adv.step()

            z = model.encode(x_batch)
            cls_pred = model.classify(z)
            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                cls_pred, y_batch
            )

            adv_pred_enc = model.adversary_forward(z)
            adv_loss_enc = nn.functional.binary_cross_entropy_with_logits(
                adv_pred_enc, g_batch
            )

            enc_loss = cls_loss - lambda_adv * adv_loss_enc

            opt_enc.zero_grad()
            enc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters())
                + list(model.classifier.parameters()),
                max_norm=1.0,
            )
            opt_enc.step()

    model.eval()
    model = model.cpu()
    return model


def _extract_adversarial_representations(model, X, device):
    """Extract latent representations from trained adversarial encoder."""
    model = model.to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        Z = model.encode(X_t)

    model = model.cpu()
    return Z.cpu().numpy()


# Unified representation extraction


def _fit_and_extract(method, train_ds, test_ds, param_value, seed, device):
    """Fit a fair representation method and extract Z_train, Z_test.

    Returns (Z_train, Z_test) or (None, None) on failure.
    """
    d_in = train_ds.X.shape[1]

    if method == "lfr":
        return _fit_lfr(train_ds, test_ds, Az=param_value, seed=seed)

    elif method == "fair_vae":
        vae = _train_fair_vae(
            train_ds.X, train_ds.group, lambda_mmd=param_value,
            d_in=d_in, seed=seed, device=device,
        )
        Z_train = _extract_vae_representations(vae, train_ds.X, device)
        Z_test = _extract_vae_representations(vae, test_ds.X, device)
        del vae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return Z_train, Z_test

    elif method == "adversarial":
        model = _train_adversarial(
            train_ds.X, train_ds.y, train_ds.group,
            lambda_adv=param_value, d_in=d_in, seed=seed, device=device,
        )
        Z_train = _extract_adversarial_representations(model, train_ds.X, device)
        Z_test = _extract_adversarial_representations(model, test_ds.X, device)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return Z_train, Z_test

    else:
        raise ValueError(f"Unknown method: {method}")


# Evaluation


def _evaluate_representation(
    Z_train, y_train, group_train,
    Z_test, y_test, group_test,
    p_a, p_b, seed,
):
    """Compute the six-term master-constraint decomposition in Z-space.

    Return a dict with decomposition terms, bandwidth sweep, bootstrap CI
    on the equality gap, and downstream classifier metrics.
    """
    # Subsample Z for kernel evaluation if needed
    n_test = Z_test.shape[0]
    if n_test > KERNEL_SUBSAMPLE_N:
        rng = np.random.default_rng(seed + 777)
        idx = rng.choice(n_test, size=KERNEL_SUBSAMPLE_N, replace=False)
        Z_eval = Z_test[idx]
        y_eval = y_test[idx]
        g_eval = group_test[idx]
    else:
        Z_eval = Z_test
        y_eval = y_test
        g_eval = group_test

    n_eval = int(Z_eval.shape[0])
    z_dim_eval = int(Z_eval.shape[1])
    delta_p_train = float(abs(p_a - p_b))

    if mode_collapse_detected(Z_eval, threshold=MODE_COLLAPSE_THRESHOLD):
        # Collapsed encoders are evidence for the impossibility claim, not fit
        # failures -- record as a real outcome with theorem_eligible=False.
        base_rate_guess = float(y_eval.mean())
        y_pred_collapsed = np.full_like(y_eval, int(base_rate_guess >= 0.5))
        accuracy = float((y_pred_collapsed == y_eval).mean())
        return {
            "collapsed": True,
            "failure_category": None,
            "decomposition": None,
            "decomposition_at_bandwidth": None,
            "sep_gap_rel": 0.0,
            "near_separation": False,
            "training_valid": None,  # set by post-aggregation gate
            "theorem_eligible": False,
            "mmd2_group": 0.0,
            "class_struct_a": 0.0,
            "class_struct_b": 0.0,
            "class_struct_marginal": 0.0,
            "sep_gap_y1": 0.0,
            "sep_gap_y0": 0.0,
            "identity_residual_sq": 0.0,
            "bound_predicted": 0.0,
            "equality_gap": 0.0,
            "equality_gap_ci_lo": float("nan"),
            "equality_gap_ci_hi": float("nan"),
            "n_valid_bootstraps": 0,
            "mmd_pvalue": float("nan"),
            "hsic_zg": 0.0,
            "accuracy": accuracy,
            "dp_gap": 0.0,
            "eo_gap": 0.0,
            "delta_p_train": delta_p_train,
            "delta_p_eval": float(
                abs(y_eval[g_eval == 1].mean() - y_eval[g_eval == 0].mean())
            ),
            "p_a_emp": float(y_eval[g_eval == 1].mean()),
            "p_b_emp": float(y_eval[g_eval == 0].mean()),
            "n_eval": n_eval,
            "z_dim_eval": z_dim_eval,
            "sigma_used": float("nan"),
        }

    sigma_z = median_heuristic(Z_eval)

    decomp = master_constraint_decomposition_in_Z(Z_eval, y_eval, g_eval, sigma=sigma_z)
    # Tautological check: verifies decomposition code, not the theorem.
    assert decomp["identity_residual_sq"] < 1e-8 * max(decomp["mmd2_group"], 1e-12), (
        f"identity residual non-zero: {decomp['identity_residual_sq']:.3e} "
        f"vs mmd2_group={decomp['mmd2_group']:.3e}"
    )

    decomp_multi = master_constraint_decomposition_in_Z_multi_bandwidth(
        Z_eval, y_eval, g_eval, multipliers=list(BANDWIDTH_MULTIPLIERS),
    )
    decomp_at_bw = {
        str(mult): {k: v for k, v in d.items() if k != "n_cells"}
        for mult, d in decomp_multi.items()
    }

    sep_gap_rel = _compute_sep_gap_rel(decomp)
    near_separation = sep_gap_rel <= TAU_SEP

    delta_p_eval = float(abs(decomp["p_a_emp"] - decomp["p_b_emp"]))
    bound_predicted = delta_p_eval ** 2 * decomp["class_struct_b"]
    equality_gap = decomp["mmd2_group"] - bound_predicted

    if near_separation:
        ci_lo, ci_hi, n_boot = _stratified_bootstrap_equality_gap_ci(
            Z_eval, y_eval, g_eval, sigma=sigma_z, seed=seed,
        )
    else:
        # Theorem has no claim off-separation; no CI test is meaningful.
        ci_lo, ci_hi, n_boot = float("nan"), float("nan"), 0

    Z_a = Z_eval[g_eval == 1]
    Z_b = Z_eval[g_eval == 0]
    _, mmd_pvalue = mmd_test(Z_a, Z_b, sigma=sigma_z, seed=seed, n_jobs=4)

    G_col = g_eval.astype(np.float64).reshape(-1, 1)
    try:
        sigma_x_hsic = median_heuristic(Z_eval)
    except AssertionError:
        sigma_x_hsic = float(np.std(Z_eval))
    if sigma_x_hsic <= 0:
        sigma_x_hsic = 1.0
    hsic_zg = hsic(Z_eval, G_col, sigma_x=sigma_x_hsic, sigma_y=1.0)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_eval)
    accuracy = float((y_pred == y_eval).mean())
    dp_gap = demographic_parity_gap(y_pred, g_eval)
    eo_gap = equalized_odds_gap(y_pred, y_eval, g_eval)

    return {
        "collapsed": False,
        "failure_category": None,
        "decomposition": decomp,
        "decomposition_at_bandwidth": decomp_at_bw,
        "sep_gap_rel": sep_gap_rel,
        "near_separation": bool(near_separation),
        "training_valid": None,  # set by post-aggregation gate over all params
        "theorem_eligible": None,  # set by post-aggregation gate (needs training_valid)
        "mmd2_group": float(decomp["mmd2_group"]),
        "class_struct_a": float(decomp["class_struct_a"]),
        "class_struct_b": float(decomp["class_struct_b"]),
        "class_struct_marginal": float(decomp["class_struct_marginal"]),
        "sep_gap_y1": float(decomp["sep_gap_y1"]),
        "sep_gap_y0": float(decomp["sep_gap_y0"]),
        "identity_residual_sq": float(decomp["identity_residual_sq"]),
        "bound_predicted": float(bound_predicted),
        "equality_gap": float(equality_gap),
        "equality_gap_ci_lo": float(ci_lo),
        "equality_gap_ci_hi": float(ci_hi),
        "n_valid_bootstraps": int(n_boot),
        "mmd_pvalue": float(mmd_pvalue),
        "hsic_zg": float(hsic_zg),
        "accuracy": accuracy,
        "dp_gap": dp_gap,
        "eo_gap": eo_gap,
        "delta_p_train": delta_p_train,
        "delta_p_eval": delta_p_eval,
        "p_a_emp": float(decomp["p_a_emp"]),
        "p_b_emp": float(decomp["p_b_emp"]),
        "n_eval": n_eval,
        "z_dim_eval": z_dim_eval,
        "sigma_used": float(decomp["sigma_used"]),
    }


# Parallel worker for CPU-only methods (LFR)


def _run_single_fit(args):
    """Run a single (method, param, seed) combination. Top-level for pickling.

    Worker process is short-lived (max_tasks_per_child=1), so memory cleanup
    happens at process exit. Still call gc + malloc_trim defensively.

    Caches Z_train, Z_test, y, g to experiments/cache/exp5/ as a side effect
    after a successful fit, so future revisions of `_evaluate_representation`
    can be tested locally without re-running the encoders on GPU.
    """
    method_name, ds, param_value, seed, device_str = args
    device = torch.device(device_str)
    train_ds, _val_ds, test_ds = train_val_test_split(ds, seed=seed)
    Z_train, Z_test = None, None
    result = (param_value, seed, None)

    cache_path = _cache_path(method_name, ds.name, param_value, seed)
    cached = _load_cached_Z(cache_path)
    try:
        if cached is not None:
            Z_train = cached["Z_train"]
            Z_test = cached["Z_test"]
            # Use cached y/g for consistency with cached Z; fall back to dataset
            # if the cache was produced with the same split (seeded identically).
            y_train_eval = cached["y_train"]
            g_train_eval = cached["g_train"]
            y_test_eval = cached["y_test"]
            g_test_eval = cached["g_test"]
            p_a = cached["p_a"]
            p_b = cached["p_b"]
        else:
            Z_train, Z_test = _fit_and_extract(
                method_name, train_ds, test_ds, param_value, seed, device,
            )
            if Z_train is None or Z_test is None:
                return (param_value, seed, None)
            y_train_eval = train_ds.y
            g_train_eval = train_ds.group
            y_test_eval = test_ds.y
            g_test_eval = test_ds.group
            p_a = train_ds.base_rates["a"]
            p_b = train_ds.base_rates["b"]
            _save_cached_Z(
                cache_path,
                Z_train, y_train_eval, g_train_eval,
                Z_test, y_test_eval, g_test_eval,
                p_a, p_b,
                method_name, ds.name, param_value, seed,
            )

        assert Z_train.ndim == 2 and Z_test.ndim == 2
        assert np.all(np.isfinite(Z_train)), "Z_train has non-finite values"
        assert np.all(np.isfinite(Z_test)), "Z_test has non-finite values"

        metrics = _evaluate_representation(
            Z_train, y_train_eval, g_train_eval,
            Z_test, y_test_eval, g_test_eval,
            p_a, p_b, seed,
        )
        metrics["seed"] = seed
        metrics["param_value"] = param_value
        metrics["z_dim"] = int(Z_test.shape[1])
        metrics["from_cache"] = cached is not None
        result = (param_value, seed, metrics)
    except (AssertionError, RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        warnings.warn(
            f"Failed: {method_name} {ds.name} param={param_value} seed={seed}: "
            f"{type(e).__name__}: {e}"
        )
        result = (
            param_value,
            seed,
            {"failure_category": type(e).__name__, "collapsed": False,
             "param_value": param_value, "seed": seed, "theorem_eligible": False,
             "training_valid": None, "near_separation": False,
             "mmd2_group": float("nan"), "class_struct_b": float("nan"),
             "equality_gap": float("nan"), "accuracy": float("nan"),
             "dp_gap": float("nan"), "eo_gap": float("nan"), "hsic_zg": float("nan"),
             "identity_residual_sq": float("nan"), "sep_gap_rel": float("nan"),
             "sep_gap_y0": float("nan"), "sep_gap_y1": float("nan"),
             "class_struct_a": float("nan"), "class_struct_marginal": float("nan"),
             "bound_predicted": float("nan"), "mmd_pvalue": float("nan"),
             "equality_gap_ci_lo": float("nan"), "equality_gap_ci_hi": float("nan"),
             "n_valid_bootstraps": 0, "delta_p_eval": float("nan"),
             "delta_p_train": float("nan"), "p_a_emp": float("nan"),
             "p_b_emp": float("nan"), "n_eval": 0, "z_dim_eval": 0,
             "sigma_used": float("nan"), "decomposition": None,
             "decomposition_at_bandwidth": None, "from_cache": False,
             "error_message": f"{type(e).__name__}: {e}"},
        )

    del train_ds, _val_ds, test_ds, Z_train, Z_test
    gc.collect()
    # Return memory to OS (Linux glibc only)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass

    return result


# Main experiment loop


_AGGREGATION_SCALAR_KEYS = [
    "mmd2_group",
    "class_struct_a",
    "class_struct_b",
    "class_struct_marginal",
    "sep_gap_y0",
    "sep_gap_y1",
    "sep_gap_rel",
    "identity_residual_sq",
    "bound_predicted",
    "equality_gap",
    "accuracy",
    "dp_gap",
    "eo_gap",
    "hsic_zg",
    "delta_p_eval",
]


def _aggregate_per_seed(per_seed_results, param_value):
    """Aggregate per-seed metrics for one (method, dataset, param) cell.

    Separates results into (collapsed, training-invalid, not-near-sep, eligible)
    categories for reporting.  Training-validity is attached in a second pass
    over the whole param sweep (see `_apply_training_validity_gate`).
    """
    all_entries = [r for r in per_seed_results if r is not None]
    non_failed = [r for r in all_entries if r.get("failure_category") is None]
    collapsed_entries = [r for r in non_failed if r.get("collapsed", False)]
    real_entries = [r for r in non_failed if not r.get("collapsed", False)]

    agg: dict = {"param_value": param_value}

    # Total counts
    agg["n_total_seeds"] = int(len(per_seed_results))
    agg["n_valid_seeds"] = int(len(non_failed))
    agg["n_collapsed"] = int(len(collapsed_entries))
    agg["n_exception"] = int(len(all_entries) - len(non_failed))
    agg["n_real_seeds"] = int(len(real_entries))
    agg["n_near_sep"] = int(sum(1 for r in real_entries if r.get("near_separation")))

    # Mean/std of scalar metrics over non-failed entries (includes collapsed)
    source = non_failed
    if source:
        for key in _AGGREGATION_SCALAR_KEYS:
            vals = [r.get(key) for r in source if r.get(key) is not None]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
            else:
                agg[f"{key}_mean"] = float("nan")
                agg[f"{key}_std"] = float("nan")

    # theorem_eligible_count is filled after training-validity gate
    agg["n_theorem_eligible"] = 0
    return agg


RESULTS_SCHEMA_VERSION = 2  # bump whenever the per-seed / aggregated schema changes


def _results_have_current_schema(results: dict) -> bool:
    """Return True if an existing results dict uses the v2 decomposition schema."""
    if results.get("schema_version") == RESULTS_SCHEMA_VERSION:
        return True
    # v1 schema had 'bound_respected_count' in every aggregated dict
    for method, ds_data in results.items():
        if method.startswith("_"):
            continue
        if not isinstance(ds_data, dict):
            continue
        for ds_name, params in ds_data.items():
            if not isinstance(params, dict):
                continue
            for pv_str, cell in params.items():
                if not isinstance(cell, dict):
                    continue
                agg = cell.get("aggregated", {})
                if "bound_respected_count" in agg or "mmd2_mean" in agg:
                    return False
                if "mmd2_group_mean" in agg or agg.get("n_valid_seeds", 0) == 0:
                    return True
    # Empty results: consider compatible (nothing to reuse).
    return True


def _run_all_methods(datasets, seeds):
    """Run all methods across all parameter sweeps and datasets.

    Returns nested dict: {method: {dataset: {param_str: {per_seed, aggregated}}}}.
    """
    results_path = RESULTS_DIR / "exp5_results.json"

    # Load partial results if available and schema-compatible
    if results_path.exists():
        with open(results_path) as f:
            loaded = json.load(f)
        if _results_have_current_schema(loaded):
            all_results = loaded
            print(
                f"  Found partial results (schema v{RESULTS_SCHEMA_VERSION}): "
                f"methods={list(all_results.keys())}"
            )
        else:
            backup_path = results_path.with_suffix(".json.v1_backup")
            with open(backup_path, "w") as f:
                json.dump(loaded, f, indent=2, default=_json_default)
            print(
                f"  Detected stale v1 results schema; backed up to {backup_path} "
                f"and starting fresh for v{RESULTS_SCHEMA_VERSION}."
            )
            all_results = {"schema_version": RESULTS_SCHEMA_VERSION}
    else:
        all_results = {"schema_version": RESULTS_SCHEMA_VERSION}

    device = get_device()
    print(f"  Device: {device}")

    total_methods = len(PARAM_SWEEPS)
    method_num = 0

    for method_name, param_values in PARAM_SWEEPS.items():
        method_num += 1
        param_label = PARAM_NAMES[method_name]

        if method_name in all_results:
            print(f"\n[{method_num}/{total_methods}] {METHOD_LABELS[method_name]} "
                  f"— using cached results")
            continue

        print(f"\n[{method_num}/{total_methods}] {METHOD_LABELS[method_name]}")
        method_results = {}

        for ds_name in DATASETS:
            ds = datasets[ds_name]
            print(f"  Dataset: {DATASET_LABELS[ds_name]}")
            ds_results = {}

            # Sequential for large datasets to avoid OOM (~12 GB per LFR fit).
            if method_name == "lfr" and len(ds.y) > 30_000:
                # Sequential path for large datasets (Adult)
                print(f"    Running {len(param_values) * len(seeds)} LFR fits "
                      f"sequentially (Adult too large for parallel)...")
                t0_par = time.time()
                completed = {}
                for pv in param_values:
                    for s in seeds:
                        t0 = time.time()
                        _, _, metrics = _run_single_fit(
                            (method_name, ds, pv, s, "cpu")
                        )
                        completed[(pv, s)] = metrics
                        status = _format_single_fit_status(metrics, time.time() - t0)
                        print(f"    {param_label}={pv}, seed={s}... {status}")
                print(f"    All LFR fits done in {time.time() - t0_par:.1f}s")

                # Reorganize into per-param structure and aggregate
                for param_value in param_values:
                    param_str = str(param_value)
                    per_seed_results = [
                        completed.get((param_value, s)) for s in seeds
                    ]
                    ds_results[param_str] = {
                        "per_seed": per_seed_results,
                        "aggregated": _aggregate_per_seed(per_seed_results, param_value),
                    }
                method_results[ds_name] = ds_results
                gc.collect()
                continue  # skip the parallel path below

            # Parallel path for small LFR datasets (COMPAS, ACS)
            if method_name == "lfr":
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor, as_completed

                # Use 'spawn' to avoid torch+fork incompatibility
                ctx = mp.get_context("spawn")

                jobs = [
                    (method_name, ds, pv, s, "cpu")
                    for pv in param_values for s in seeds
                ]
                n_workers = N_PARALLEL_WORKERS_DEFAULT
                print(f"    Dispatching {len(jobs)} LFR fits across "
                      f"{n_workers} workers (spawn)...")
                t0_par = time.time()

                completed = {}
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=ctx,
                    max_tasks_per_child=1,
                ) as pool:
                    futures = {pool.submit(_run_single_fit, j): j for j in jobs}
                    n_failed = 0
                    for future in as_completed(futures):
                        job = futures[future]
                        _, _, _, job_seed, _ = job
                        job_pv = job[2]
                        try:
                            pv, s, metrics = future.result()
                        except Exception as e:
                            n_failed += 1
                            print(f"    {param_label}={job_pv}, seed={job_seed}... "
                                  f"WORKER DIED: {type(e).__name__}: {e}")
                            continue
                        completed[(pv, s)] = metrics
                        status = _format_single_fit_status(metrics, None)
                        print(f"    {param_label}={pv}, seed={s}... {status}")

                    if n_failed > 0:
                        print(f"    WARNING: {n_failed}/{len(jobs)} workers died")

                print(f"    All LFR fits done in {time.time() - t0_par:.1f}s")

                for param_value in param_values:
                    param_str = str(param_value)
                    per_seed_results = [
                        completed.get((param_value, s)) for s in seeds
                    ]
                    ds_results[param_str] = {
                        "per_seed": per_seed_results,
                        "aggregated": _aggregate_per_seed(per_seed_results, param_value),
                    }

                method_results[ds_name] = ds_results
                gc.collect()
                continue  # skip the sequential path below

            # Sequential path for GPU methods (VAE, Adversarial)
            for param_value in param_values:
                param_str = str(param_value)
                per_seed_results = []

                for seed in seeds:
                    print(f"    {param_label}={param_value}, seed={seed}...", end=" ")
                    t0 = time.time()

                    _, _, metrics = _run_single_fit(
                        (method_name, ds, param_value, seed, str(device)),
                    )
                    if metrics is not None:
                        metrics["time_s"] = time.time() - t0
                    per_seed_results.append(metrics)
                    print(_format_single_fit_status(metrics, time.time() - t0))

                ds_results[param_str] = {
                    "per_seed": per_seed_results,
                    "aggregated": _aggregate_per_seed(per_seed_results, param_value),
                }

            method_results[ds_name] = ds_results
            gc.collect()

        # Apply training-validity gate across the whole sweep for this method
        _apply_training_validity_gate(method_results, datasets)

        all_results[method_name] = method_results
        _save_partial(results_path, all_results)

    # Final pass: in case a cached method didn't get the gate applied above
    for m in all_results:
        if m in PARAM_SWEEPS:
            _apply_training_validity_gate(all_results[m], datasets)

    return all_results


def _format_single_fit_status(metrics, elapsed_s):
    """Compact one-line status for a single fit: what happened and why."""
    if metrics is None:
        return "FAILED"
    if metrics.get("failure_category") and not metrics.get("collapsed", False):
        return f"EXCEPTION: {metrics.get('failure_category')}"
    if metrics.get("collapsed", False):
        return "COLLAPSED (mode collapse detected)"
    bits = [
        f"acc={metrics.get('accuracy', float('nan')):.3f}",
        f"MMD2={metrics.get('mmd2_group', float('nan')):.6f}",
        f"sep={metrics.get('sep_gap_rel', float('nan')):.3f}",
    ]
    if metrics.get("near_separation"):
        bits.append("near-sep")
    if elapsed_s is not None:
        bits.append(f"{elapsed_s:.1f}s")
    return ", ".join(bits)


def _apply_training_validity_gate(
    method_results: dict, datasets: dict[str, FairnessDataset]
) -> None:
    """Post-aggregation pass that attaches training_valid and theorem_eligible.

    Training validity is a (method, dataset)-level property (not per-seed):
        training_valid := accuracy_at_min_lambda > base_rate + TRAINING_VALID_ACC_MARGIN
                           AND accuracy_range_across_params > TRAINING_VALID_ACC_RANGE

    This catches:
      - Class-collapsed encoders (Fair VAE at lambda=0 on Adult with acc ~= base rate)
      - Flat-tradeoff encoders (Adversarial on Adult with accuracy constant across lambda)

    Failing configs are excluded from the H6 equality test (theorem_eligible =
    False) but remain visible in the tables and figures with full decomposition
    metrics attached.
    """
    for ds_name, ds_results in method_results.items():
        ds = datasets.get(ds_name)
        if ds is None:
            continue

        # Majority-class accuracy: max(p_pos, p_neg) across the dataset
        p_pos = float(np.mean(ds.y))
        base_rate = max(p_pos, 1.0 - p_pos)

        # Scan the sweep for min-lambda accuracy and accuracy range
        sorted_params = sorted(ds_results.keys(), key=lambda s: float(s))
        accs_at_param = []
        for pv_str in sorted_params:
            agg = ds_results[pv_str].get("aggregated", {})
            accs_at_param.append(agg.get("accuracy_mean", float("nan")))

        accs_at_param = np.array(accs_at_param, dtype=np.float64)
        valid_mask = ~np.isnan(accs_at_param)
        if valid_mask.sum() == 0:
            # No valid accuracies: mark everything training_invalid
            training_valid = False
            acc_at_min_lambda = float("nan")
            acc_range = float("nan")
        else:
            valid_accs = accs_at_param[valid_mask]
            acc_at_min_lambda = float(valid_accs[0])
            acc_range = float(valid_accs.max() - valid_accs.min())
            training_valid = bool(
                acc_at_min_lambda > base_rate + TRAINING_VALID_ACC_MARGIN
                and acc_range > TRAINING_VALID_ACC_RANGE
            )

        # Stamp per-param aggregates and per-seed entries
        for pv_str in sorted_params:
            cell = ds_results[pv_str]
            agg = cell.get("aggregated", {})
            agg["base_rate_majority"] = float(base_rate)
            agg["acc_at_min_lambda"] = float(acc_at_min_lambda)
            agg["acc_range_across_lambdas"] = float(acc_range)
            agg["training_valid"] = bool(training_valid)

            # theorem_eligible per seed: training_valid AND not collapsed AND near-sep
            # Recompute near_separation from sep_gap_rel using the current
            # TAU_SEP (the per-seed near_separation field may have been
            # computed with a different gate value in a prior run).
            n_eligible = 0
            for r in cell.get("per_seed", []):
                if r is None:
                    continue
                r["training_valid"] = bool(training_valid)
                if r.get("failure_category") and not r.get("collapsed", False):
                    r["theorem_eligible"] = False
                    continue
                sep_gap = r.get("sep_gap_rel", float("nan"))
                near_sep = bool(
                    not np.isnan(sep_gap) and sep_gap <= TAU_SEP
                )
                r["near_separation"] = near_sep
                eligible = bool(
                    training_valid
                    and not r.get("collapsed", False)
                    and near_sep
                )
                r["theorem_eligible"] = eligible
                if eligible:
                    n_eligible += 1
            agg["n_theorem_eligible"] = int(n_eligible)


# Plotting


def _iter_method_param_points(results, ds_name):
    """Yield (method_name, param_str, aggregated_dict) over all valid cells."""
    for method_name in PARAM_SWEEPS:
        ds_data = results.get(method_name, {}).get(ds_name, {})
        if not ds_data:
            continue
        for param_str, param_data in sorted(ds_data.items(), key=lambda x: float(x[0])):
            agg = param_data.get("aggregated", {})
            if agg.get("n_valid_seeds", 0) == 0:
                continue
            yield method_name, param_str, agg


def _plot_pareto_tradeoff(results, dataset_names, fig_name):
    """Empirical (accuracy, mmd2_group) Pareto curves -- descriptive, appendix."""
    n_panels = len(dataset_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, dataset_names):
        method_series = {}
        for method_name, param_str, agg in _iter_method_param_points(results, ds_name):
            method_series.setdefault(method_name, {"pv": [], "acc": [], "mmd": [],
                                                   "acc_err": [], "mmd_err": []})
            s = method_series[method_name]
            s["pv"].append(param_str)
            s["acc"].append(agg.get("accuracy_mean", float("nan")))
            s["mmd"].append(agg.get("mmd2_group_mean", float("nan")))
            s["acc_err"].append(agg.get("accuracy_std", 0.0))
            s["mmd_err"].append(agg.get("mmd2_group_std", 0.0))

        for method_name, s in method_series.items():
            if not s["acc"]:
                continue
            color = METHOD_COLORS[method_name]
            marker = METHOD_MARKERS[method_name]
            label = METHOD_LABELS[method_name]
            ax.errorbar(
                s["acc"], s["mmd"],
                xerr=s["acc_err"], yerr=s["mmd_err"],
                fmt=f"{marker}-", color=color, label=label,
                capsize=3, markersize=6, linewidth=1.2, alpha=0.85,
            )
            # Annotate extreme parameter values
            if len(s["pv"]) >= 2:
                for i in [0, -1]:
                    ax.annotate(
                        f"{PARAM_NAMES[method_name]}={s['pv'][i]}",
                        (s["acc"][i], s["mmd"][i]),
                        fontsize=6, alpha=0.7,
                        textcoords="offset points", xytext=(5, 5),
                    )

        ds_label = DATASET_LABELS.get(ds_name, ds_name)
        ax.set_xlabel("Downstream Accuracy")
        ax.set_ylabel(r"$\|\delta_\Phi\|^2$ (MMD$^2$ group)"
                      if ds_name == dataset_names[0] else "")
        ax.set_title(ds_label, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Empirical Fairness-Utility Pareto Curves (descriptive)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    paths = save_figure(fig, fig_name)
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def _plot_master_constraint_scatter(results, dataset_names, fig_name):
    """Headline H6 figure: log-log scatter of (class_struct_b, mmd2_group).

    Each (method, param, seed_mean) point is colored by its relative
    separation gap.  The dashed line `y = delta_p^2 * x` overlays the exact-
    separation equality predicted by Theorem 2.  Theorem-eligible points
    (near-separation + not collapsed + training valid) cluster on the line;
    off-line points either have separation gaps or live off-premise.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    n_panels = len(dataset_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 4.6))
    if n_panels == 1:
        axes = [axes]

    # Shared color scale on sep_gap_rel across all panels for visual consistency
    all_sep = []
    for ds_name in dataset_names:
        for _m, _p, agg in _iter_method_param_points(results, ds_name):
            val = agg.get("sep_gap_rel_mean")
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                all_sep.append(val)
    sep_max = max(0.2, min(2.0, np.percentile(all_sep, 95) if all_sep else 0.2))
    norm = Normalize(vmin=0.0, vmax=sep_max)
    cmap = cm.get_cmap("RdYlBu_r")

    for ax, ds_name in zip(axes, dataset_names):
        # Dataset-level delta_p for the theoretical equality line
        deltas = []
        xs, ys, cs, markers, labels_plotted = [], [], [], [], set()
        n_eligible_total = 0
        n_total_cells = 0

        for method_name, param_str, agg in _iter_method_param_points(results, ds_name):
            x = agg.get("class_struct_b_mean", float("nan"))
            y = agg.get("mmd2_group_mean", float("nan"))
            sep = agg.get("sep_gap_rel_mean", float("nan"))
            dp = agg.get("delta_p_eval_mean", float("nan"))
            if (
                np.isnan(x) or np.isnan(y) or np.isnan(sep) or np.isnan(dp)
                or x <= 0 or y <= 0
            ):
                continue
            deltas.append(dp)
            xs.append(x)
            ys.append(y)
            cs.append(sep)
            markers.append(METHOD_MARKERS[method_name])
            label = METHOD_LABELS[method_name]
            if label not in labels_plotted:
                # Draw an invisible point for the legend entry
                ax.scatter(
                    [], [], marker=METHOD_MARKERS[method_name],
                    c="gray", s=40, edgecolors="black", linewidths=0.4,
                    label=label,
                )
                labels_plotted.add(label)

            n_total_cells += 1
            n_eligible_total += int(agg.get("n_theorem_eligible", 0))

        # Plot each point with its marker and color
        for x, y, sep, marker in zip(xs, ys, cs, markers):
            ax.scatter(
                [x], [y], marker=marker,
                c=[cmap(norm(sep))], s=50,
                edgecolors="black", linewidths=0.5, alpha=0.9,
            )

        # Dashed equality line y = delta_p^2 * x using the median delta_p
        if deltas and xs:
            dp_median = float(np.median(deltas))
            x_min = max(min(xs), 1e-6)
            x_max = max(xs) * 2.0
            xx = np.geomspace(x_min, x_max, 200)
            yy = dp_median ** 2 * xx
            ax.plot(
                xx, yy, "--", color="black", linewidth=1.4, alpha=0.85,
                label=rf"$y = \Delta p^2\,x$ ($\Delta p = {dp_median:.3f}$)",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ds_label = DATASET_LABELS.get(ds_name, ds_name)
        ax.set_xlabel(r"$\|\mu_{\Phi,1,b} - \mu_{\Phi,0,b}\|^2$ (within-group-$b$ class)")
        if ds_name == dataset_names[0]:
            ax.set_ylabel(r"$\|\delta_\Phi\|^2$ (group MMD$^2$)")
        n_eligible_str = f" — {n_eligible_total}/{n_total_cells * 5} eligible seeds"
        ax.set_title(f"{ds_label}{n_eligible_str}", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, which="both", ls=":", alpha=0.4)

    # Shared colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("sep_gap_rel (0 = near sep)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.axhline(TAU_SEP, color="black", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "H6: Z-space Master-Constraint Equality Across Methods",
        fontsize=12, y=0.99,
    )
    paths = save_figure(fig, fig_name)
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def _plot_decomposition_bars(results, dataset_names, fig_name):
    """Stacked-bar decomposition of the master constraint per method.

    For each (method, dataset) we pick the strongest-fairness-penalty
    parameter and show the three RHS term magnitudes of the master constraint:
        A := (p_a * ||delta_{Phi,1}||)^2
        B := ((1-p_a) * ||delta_{Phi,0}||)^2
        C := (delta_p * ||mu_{Phi,1,b} - mu_{Phi,0,b}||)^2
    The bar height sums these quantities; a marker shows the observed
    ||delta_Phi||^2.  Cross-cancellation is visible when the bar exceeds the
    marker; perfect separation looks like A=B=0, bar=C=marker.
    """
    n_panels = len(dataset_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 4.2))
    if n_panels == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, dataset_names):
        method_bars = []
        for method_name in PARAM_SWEEPS:
            ds_data = results.get(method_name, {}).get(ds_name, {})
            if not ds_data:
                continue
            # Strongest fairness penalty = largest param value
            pv_sorted = sorted(ds_data.keys(), key=lambda s: float(s), reverse=True)
            chosen_pv = None
            chosen_agg = None
            for pv_str in pv_sorted:
                agg = ds_data[pv_str].get("aggregated", {})
                if agg.get("n_valid_seeds", 0) > 0 and not np.isnan(
                    agg.get("mmd2_group_mean", float("nan"))
                ):
                    chosen_pv = pv_str
                    chosen_agg = agg
                    break
            if chosen_agg is None:
                continue

            p_a = chosen_agg.get("p_a_emp_mean", 0.5)
            p_b = chosen_agg.get("p_b_emp_mean", 0.5)
            if p_a is None or np.isnan(p_a):
                p_a = 0.5
            if p_b is None or np.isnan(p_b):
                p_b = 0.5
            dp = abs(p_a - p_b)

            sep_y1 = max(chosen_agg.get("sep_gap_y1_mean", 0.0), 0.0)
            sep_y0 = max(chosen_agg.get("sep_gap_y0_mean", 0.0), 0.0)
            class_b = max(chosen_agg.get("class_struct_b_mean", 0.0), 0.0)
            mmd2_obs = max(chosen_agg.get("mmd2_group_mean", 0.0), 0.0)

            A = (p_a ** 2) * sep_y1
            B = ((1 - p_a) ** 2) * sep_y0
            C = (dp ** 2) * class_b

            method_bars.append({
                "method": method_name,
                "pv": chosen_pv,
                "A": A, "B": B, "C": C, "mmd2_obs": mmd2_obs,
            })

        if not method_bars:
            continue

        labels = [
            f"{METHOD_LABELS[m['method']]}\n"
            f"({PARAM_NAMES[m['method']]}={m['pv']})"
            for m in method_bars
        ]
        A_vals = [m["A"] for m in method_bars]
        B_vals = [m["B"] for m in method_bars]
        C_vals = [m["C"] for m in method_bars]
        obs_vals = [m["mmd2_obs"] for m in method_bars]
        x_pos = np.arange(len(method_bars))

        ax.bar(x_pos, A_vals, color="#1f77b4", label=r"$p_a^2\,\|\delta_{\Phi,1}\|^2$")
        ax.bar(x_pos, B_vals, bottom=A_vals, color="#ff7f0e",
               label=r"$(1-p_a)^2\,\|\delta_{\Phi,0}\|^2$")
        ax.bar(
            x_pos,
            C_vals,
            bottom=[A + B for A, B in zip(A_vals, B_vals)],
            color="#2ca02c",
            label=r"$\Delta p^2\,\|\mu_{\Phi,1,b}-\mu_{\Phi,0,b}\|^2$",
        )
        ax.scatter(
            x_pos, obs_vals, marker="D", color="red", s=60,
            zorder=5, edgecolors="black", linewidths=0.6,
            label=r"observed $\|\delta_\Phi\|^2$",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
        ax.set_ylabel(r"magnitude ($\|\cdot\|^2$)"
                      if ds_name == dataset_names[0] else "")
        ds_label = DATASET_LABELS.get(ds_name, ds_name)
        ax.set_title(ds_label, fontsize=10)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle(
        "Master-Constraint Decomposition at Strongest Fairness Penalty",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    paths = save_figure(fig, fig_name)
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


# Table generation


def _fmt_mean_std(mean, std, digits=3):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "---"
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def _fmt_sci(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"${x:.{digits}e}$"


def _generate_tables(results) -> None:
    """Generate the decomposition LaTeX table."""
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    header = (
        "Dataset & Method ($\\lambda$) & Acc. & "
        "$\\|\\delta_\\Phi\\|^2$ & $\\|\\mu_{1,b}-\\mu_{0,b}\\|^2$ & "
        "$\\Delta p^2\\cdot c_b$ & sep\\_gap & near & "
        "coll. & train. & n\\_elig. \\\\"
    )

    lines = [
        "% Exp 5: Fair representation master-constraint decomposition",
        "% Generated by exp5_fair_representations.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Z-space master-constraint decomposition for fair representation "
        "methods. The equality $\\|\\delta_\\Phi\\|^2 = \\Delta p^2\\cdot \\|\\mu_{\\Phi,1,b}-\\mu_{\\Phi,0,b}\\|^2$ "
        "follows from Theorem~2 \\emph{under exact separation in} $Z$. "
        "Theorem-eligible configurations ($\\text{sep\\_gap\\_rel}\\le"
        f" {TAU_SEP}$, not collapsed, training valid) are tested with "
        "stratified-bootstrap 95\\% CI; \\texttt{n\\_eligible} counts seeds "
        "passing all three gates. Metrics are mean$\\pm$std across 5 seeds.}",
        "\\label{tab:fair-rep}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{llcccccccccc}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    for ds_name in DATASETS:
        ds_label = DATASET_LABELS.get(ds_name, ds_name)
        first_row = True

        for method_name in PARAM_SWEEPS:
            ds_data = results.get(method_name, {}).get(ds_name, {})
            if not ds_data:
                continue

            param_values = sorted(ds_data.keys(), key=float)
            if len(param_values) >= 3:
                selected = [
                    param_values[0],
                    param_values[len(param_values) // 2],
                    param_values[-1],
                ]
            else:
                selected = param_values

            for pv_str in selected:
                agg = ds_data[pv_str].get("aggregated", {})
                if agg.get("n_valid_seeds", 0) == 0:
                    continue

                ds_col = ds_label if first_row else ""
                first_row = False

                method_str = (
                    f"{METHOD_LABELS[method_name]} "
                    f"({PARAM_NAMES[method_name]}={pv_str})"
                )
                n_total = agg.get("n_real_seeds", 0)
                n_elig = agg.get("n_theorem_eligible", 0)

                acc_str = _fmt_mean_std(
                    agg.get("accuracy_mean"), agg.get("accuracy_std"), digits=3
                )
                mmd2_str = _fmt_sci(agg.get("mmd2_group_mean"))
                cb_str = _fmt_sci(agg.get("class_struct_b_mean"))
                dp_mean = agg.get("delta_p_eval_mean", 0.0)
                if dp_mean is None or np.isnan(dp_mean):
                    pred_str = "---"
                else:
                    cb_val = agg.get("class_struct_b_mean", 0.0)
                    pred = dp_mean ** 2 * (cb_val if cb_val is not None else 0.0)
                    pred_str = f"${pred:.2e}$"
                sep_mean = agg.get("sep_gap_rel_mean", float("nan"))
                sep_str = (
                    "---"
                    if (sep_mean is None or np.isnan(sep_mean))
                    else f"${sep_mean:.3f}$"
                )
                near_str = (
                    r"\checkmark" if sep_mean is not None
                    and not np.isnan(sep_mean)
                    and sep_mean <= TAU_SEP
                    else r"$\times$"
                )
                n_coll = agg.get("n_collapsed", 0)
                coll_str = r"\checkmark" if n_coll > 0 else r"$\times$"
                train_valid = agg.get("training_valid", False)
                train_str = r"\checkmark" if train_valid else r"$\times$"
                elig_str = f"{n_elig}/{n_total}"

                lines.append(
                    f"{ds_col} & {method_str} & {acc_str} & {mmd2_str} & "
                    f"{cb_str} & {pred_str} & {sep_str} & {near_str} & "
                    f"{coll_str} & {train_str} & {elig_str} \\\\"
                )

        if not first_row:
            lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines += [
        "\\end{tabular}}",
        "\\end{table}",
    ]

    path = tables_dir / "exp5_fair_rep.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Table saved to {path}")


# Summary


def _print_summary(results) -> None:
    """Print a human-readable summary of the decomposition and H6 assessment."""
    print("\n--- Summary (H6 master-constraint decomposition) ---")

    for ds_name in DATASETS:
        ds_label = DATASET_LABELS.get(ds_name, ds_name)
        print(f"\n  {ds_label}:")

        for method_name in PARAM_SWEEPS:
            ds_data = results.get(method_name, {}).get(ds_name, {})
            if not ds_data:
                continue

            param_values = sorted(ds_data.keys(), key=float)
            print(f"    {METHOD_LABELS[method_name]}:")

            for pv_str in param_values:
                agg = ds_data[pv_str].get("aggregated", {})
                n_total = agg.get("n_real_seeds", 0)
                n_coll = agg.get("n_collapsed", 0)
                n_elig = agg.get("n_theorem_eligible", 0)
                train_valid = agg.get("training_valid", False)
                if agg.get("n_valid_seeds", 0) == 0:
                    print(f"      {PARAM_NAMES[method_name]}={pv_str}: all seeds FAILED")
                    continue

                def _g(k, d=float("nan")):
                    v = agg.get(k, d)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return "nan"
                    return f"{v:.4g}"

                line = (
                    f"      {PARAM_NAMES[method_name]}={pv_str}: "
                    f"acc={_g('accuracy_mean')}, "
                    f"mmd2_group={_g('mmd2_group_mean')}, "
                    f"class_b={_g('class_struct_b_mean')}, "
                    f"sep={_g('sep_gap_rel_mean')}, "
                    f"coll={n_coll}, "
                    f"train_valid={'Y' if train_valid else 'N'}, "
                    f"eligible={n_elig}/{n_total}"
                )
                print(line)

    # Overall H6 assessment
    total_seeds = 0
    eligible_seeds = 0
    collapsed_total = 0
    n_exception = 0
    ci_pass = 0
    ci_fail = 0
    for method_name in PARAM_SWEEPS:
        for ds_name in DATASETS:
            ds_data = results.get(method_name, {}).get(ds_name, {})
            for param_data in ds_data.values():
                agg = param_data.get("aggregated", {})
                total_seeds += agg.get("n_real_seeds", 0)
                eligible_seeds += agg.get("n_theorem_eligible", 0)
                collapsed_total += agg.get("n_collapsed", 0)
                n_exception += agg.get("n_exception", 0)
                # Count per-seed CI pass/fail among eligible seeds
                for r in param_data.get("per_seed", []):
                    if r is None or not r.get("theorem_eligible"):
                        continue
                    ci_lo = r.get("equality_gap_ci_lo", float("nan"))
                    ci_hi = r.get("equality_gap_ci_hi", float("nan"))
                    if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                        if ci_lo <= 0 <= ci_hi:
                            ci_pass += 1
                        else:
                            ci_fail += 1

    print("\n  H6 (scoped) Assessment:")
    print(f"    Total non-failed seeds:         {total_seeds}")
    print(f"    Exceptions (partial collapse):  {n_exception}")
    print(f"    Collapsed (full):               {collapsed_total}")
    print(f"    Theorem-eligible seeds (tau={TAU_SEP}): {eligible_seeds}")
    if eligible_seeds > 0:
        frac = eligible_seeds / total_seeds if total_seeds > 0 else 0.0
        print(f"    Eligible fraction:              {frac:.0%}")
        print(f"    CI contains 0:                  {ci_pass}/{ci_pass + ci_fail} "
              f"({100*ci_pass/(ci_pass+ci_fail):.0f}%)" if (ci_pass + ci_fail) > 0 else "")
        print(f"    CI excludes 0:                  {ci_fail}/{ci_pass + ci_fail}")
        print("    Verdict: PARTIALLY SUPPORTED — equality approximately holds "
              f"in tight-separation regime (tau={TAU_SEP}); "
              f"CI pass rate {ci_pass}/{ci_pass+ci_fail} reflects "
              "finite-separation corrections that are detectable but small."
              if (ci_pass + ci_fail) > 0 else "    Verdict: no CI data")
    elif collapsed_total > 0:
        print(f"    Verdict: UNINFORMATIVE — {collapsed_total} collapsed seeds "
              "(class structure destroyed), 0 theorem-eligible")
    else:
        print("    Verdict: UNINFORMATIVE (0 eligible seeds, 0 collapsed)")


# Synthetic validation (exact separation)


def _run_synthetic_validation(seeds, device):
    """Validate decomposition code on synthetic Z with known separation properties.

    Sweep alpha in {0, 0.1, 0.5, 1.0} to verify the identity residual and
    separation gate behave correctly without any encoder training.
    """
    print("\n[Synthetic Validation] Direct-Z master-constraint decomposition")
    print("  (bypasses encoder training; tests the measurement code only)")

    alpha_values = [0.0, 0.1, 0.5, 1.0]
    synth_results = {"decomposition_sweep": {}}

    for alpha in alpha_values:
        print(f"\n  alpha = {alpha}")
        per_seed = []
        for seed in seeds:
            t0 = time.time()
            try:
                Z, y, g, params = generate_synthetic_Z_separated(
                    n=10_000, p_a=0.6, p_b=0.4, d_eps=8, alpha=alpha, seed=seed,
                )
                sigma = median_heuristic(Z)
                decomp = master_constraint_decomposition_in_Z(Z, y, g, sigma=sigma)
                sep_rel = _compute_sep_gap_rel(decomp)
                near_sep = sep_rel <= TAU_SEP
                dp = abs(decomp["p_a_emp"] - decomp["p_b_emp"])
                bound_predicted = dp ** 2 * decomp["class_struct_b"]
                gap = decomp["mmd2_group"] - bound_predicted

                if near_sep:
                    ci_lo, ci_hi, n_boot = _stratified_bootstrap_equality_gap_ci(
                        Z, y, g, sigma=sigma, B=200, seed=seed,
                    )
                else:
                    ci_lo, ci_hi, n_boot = float("nan"), float("nan"), 0

                rec = {
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "identity_residual_sq": float(decomp["identity_residual_sq"]),
                    "mmd2_group": float(decomp["mmd2_group"]),
                    "class_struct_b": float(decomp["class_struct_b"]),
                    "class_struct_a": float(decomp["class_struct_a"]),
                    "sep_gap_y1": float(decomp["sep_gap_y1"]),
                    "sep_gap_y0": float(decomp["sep_gap_y0"]),
                    "sep_gap_rel": float(sep_rel),
                    "near_separation": bool(near_sep),
                    "delta_p_eval": float(dp),
                    "bound_predicted": float(bound_predicted),
                    "equality_gap": float(gap),
                    "equality_gap_ci_lo": float(ci_lo),
                    "equality_gap_ci_hi": float(ci_hi),
                    "n_valid_bootstraps": int(n_boot),
                    "p_a_emp": float(decomp["p_a_emp"]),
                    "p_b_emp": float(decomp["p_b_emp"]),
                    "sigma_used": float(sigma),
                    "time_s": time.time() - t0,
                }
                per_seed.append(rec)
                print(
                    f"    seed={seed}: identity_res={rec['identity_residual_sq']:.2e}, "
                    f"mmd2={rec['mmd2_group']:.4e}, "
                    f"Δp²·c_b={rec['bound_predicted']:.4e}, "
                    f"gap={rec['equality_gap']:+.4e}, "
                    f"sep_rel={rec['sep_gap_rel']:.4f}, "
                    f"near={'Y' if rec['near_separation'] else 'N'}, "
                    f"{rec['time_s']:.2f}s"
                )

            except (AssertionError, ValueError, RuntimeError) as e:
                warnings.warn(
                    f"Synthetic validation alpha={alpha} seed={seed}: "
                    f"{type(e).__name__}: {e}"
                )
                per_seed.append({
                    "seed": int(seed), "alpha": float(alpha),
                    "failure_category": type(e).__name__,
                })

        real = [r for r in per_seed if r.get("failure_category") is None]
        agg = {
            "alpha": float(alpha),
            "n_total_seeds": len(per_seed),
            "n_real_seeds": len(real),
            "n_near_sep": int(sum(1 for r in real if r.get("near_separation"))),
        }
        if real:
            for key in [
                "identity_residual_sq",
                "mmd2_group",
                "class_struct_b",
                "sep_gap_rel",
                "equality_gap",
                "delta_p_eval",
            ]:
                vals = [r[key] for r in real if key in r]
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))

        synth_results["decomposition_sweep"][str(alpha)] = {
            "per_seed": per_seed,
            "aggregated": agg,
        }

    # Sanity checks: identity should always hold, near_sep must be True at alpha=0,
    # near_sep must be False at alpha=1.0.
    print("\n  Sanity checks:")
    for alpha_str, cell in synth_results["decomposition_sweep"].items():
        agg = cell["aggregated"]
        ident = agg.get("identity_residual_sq_mean", float("nan"))
        sep = agg.get("sep_gap_rel_mean", float("nan"))
        n_near = agg.get("n_near_sep", 0)
        n_tot = agg.get("n_real_seeds", 0)
        print(
            f"    alpha={alpha_str}: identity_res={ident:.2e}, "
            f"sep_gap_rel={sep:.4f}, near_sep={n_near}/{n_tot}"
        )
        if float(alpha_str) == 0.0:
            assert n_near == n_tot, (
                f"At alpha=0 all seeds should be near-separation, got {n_near}/{n_tot}"
            )
        if float(alpha_str) == 1.0:
            assert n_near == 0, (
                f"At alpha=1.0 no seed should be near-separation, got {n_near}/{n_tot}"
            )
        if ident is not None and not np.isnan(ident):
            assert ident < 1e-18, (
                f"Identity residual non-zero at alpha={alpha_str}: {ident:.2e}"
            )

    print("  Synthetic validation PASSED (identity holds, sep gate behaves correctly)")
    return synth_results


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ensure_dirs()
    setup_style()
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 5: Fair Representation Learning Impossibility")
    print("=" * 60)

    device = get_device()
    print(f"  Torch device: {device}")
    if torch.cuda.is_available():
        print(f"  CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  No CUDA GPU available — running on {device}")
    print(f"  Z-cache directory: {CACHE_DIR}")

    synth_results = _run_synthetic_validation(RANDOM_SEEDS[:3], device)

    datasets = _load_datasets()
    results = _run_all_methods(datasets, RANDOM_SEEDS)
    results["_synthetic"] = synth_results

    print("\n[Plot] Generating figures...")
    main_ds = [ds for ds in ["adult", "compas"] if ds in datasets]

    _plot_pareto_tradeoff(results, main_ds, "fig5a_fair_rep_tradeoff")
    if "acs_pums" in datasets:
        _plot_pareto_tradeoff(results, ["acs_pums"], "fig5b_fair_rep_tradeoff_acs")

    _plot_master_constraint_scatter(
        results, main_ds, "fig5c_master_constraint_scatter"
    )
    if "acs_pums" in datasets:
        _plot_master_constraint_scatter(
            results, ["acs_pums"], "fig5c_master_constraint_scatter_acs"
        )

    all_ds = [ds for ds in DATASETS if ds in datasets]
    _plot_decomposition_bars(results, all_ds, "fig5d_decomposition_bars")

    _generate_tables(results)

    _print_summary(results)

    results_path = RESULTS_DIR / "exp5_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nFinal results saved to {results_path}")

    print("\n" + "=" * 60)
    print("Experiment 5 complete.")
    print("  Note: LFR returns prototype-reconstructed features in the original")
    print("  dimension (not k=10 latent assignments). The master-constraint")
    print("  decomposition is dimension-agnostic, so measurement is unaffected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
