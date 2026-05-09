"""Synthetic validation of the master constraint identity and finite-criteria impossibility.

Sub-experiments:
  1a  Master constraint -- algebraic identity verified to float64 precision.
  1b  Finite-criteria impossibility -- residual MMD vs number of fairness criteria,
      with linear kernel control to isolate the dimensional transition.
  1c  Bandwidth robustness for 1b across 0.5x/1x/2x median heuristic.
  1d  Dimensional transition -- k_99 vs n (single seed).
  1e  Rich-DGP n-sweep -- multi-directional population delta (5 seeds with error bars).

Produces:
  figures/fig1a_master_constraint_sweep.{pdf,png}
  figures/fig1a_master_constraint_bars.{pdf,png}
  figures/fig1b_finite_criteria.{pdf,png}
  figures/fig1c_bandwidth_robustness.{pdf,png}
  figures/fig1d_dimensional_transition.{pdf,png}
  figures/fig1e_rich_transition.{pdf,png}
  results/exp1_results.json
  results/tables/exp1_dimensional_transition.tex
  results/tables/exp1_rich_transition.tex
"""

import json
import time

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from experiments.config import (
    RANDOM_SEEDS,
    DEFAULT_SEED,
    BANDWIDTH_MULTIPLIERS,
    RESULTS_DIR,
    ensure_dirs,
)
from experiments.data.synthetic import generate_synthetic, generate_rich_synthetic
from experiments.kernels import rbf_kernel_matrix, median_heuristic
from experiments.utils.plotting import setup_style, save_figure

# Parameters

# Exp 1a
EXP1A_DELTA_PS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_SAMPLES = 10_000
D_FEATURES = 10

# Exp 1b
EXP1B_DELTA_P = 0.20
MAX_CRITERIA = 50
N_EIGENVALUES = 1000  # enough so residual at k=50 reflects true RBF incompressibility
N_RANDOM_TRIALS = 10

# Linear kernel control for dimensional transition contrast
EXP1B_RUN_LINEAR_CONTROL = True

# Exp 1c: extended bandwidth sweep including narrow sigmas
EXP1C_BANDWIDTH_MULTIPLIERS = [0.01, 0.1, 0.5, 1.0, 2.0]

# Exp 1d: k_c vs n
EXP1D_N_VALUES = [500, 2000, 10000]
EXP1D_N_EIG_TRANSITION = 300
EXP1D_CAPTURE_TARGETS = [0.90, 0.95, 0.99]

# Exp 1e: rich-DGP n-sweep with multi-directional population delta
EXP1E_RUN = True
EXP1E_N_VALUES = [200, 500, 1000, 2000, 5000, 10000]
EXP1E_D_FEATURES = 20
EXP1E_DECAY = 0.5
EXP1E_TOTAL_SHIFT = 0.5
EXP1E_N_EIG_CAP = 800
EXP1E_CAPTURE_TARGETS = [0.90, 0.95, 0.99]
EXP1E_SEEDS = None  # None -> use RANDOM_SEEDS; override for smoke tests

# Exp 1a bar chart
BAR_CHART_DELTA_P = 0.20


# Exp 1a: Master Constraint Verification


def run_exp1a(
    seeds: list[int], delta_ps: list[float]
) -> list[dict]:
    """Verify the projected master constraint across base-rate gaps."""
    results = []
    for dp in delta_ps:
        for seed in seeds:
            ds, params = generate_synthetic(
                n=N_SAMPLES, delta_p=dp, d=D_FEATURES, seed=seed
            )
            true_beta = params["true_beta"]
            scores = ds.X @ true_beta

            group = ds.group
            y = ds.y

            p_a = float(y[group == 1].mean())
            p_b = float(y[group == 0].mean())

            mu_a = float(scores[group == 1].mean())
            mu_b = float(scores[group == 0].mean())
            mu_1a = float(scores[(group == 1) & (y == 1)].mean())
            mu_1b = float(scores[(group == 0) & (y == 1)].mean())
            mu_0a = float(scores[(group == 1) & (y == 0)].mean())
            mu_0b = float(scores[(group == 0) & (y == 0)].mean())

            lhs = mu_a - mu_b
            term1 = p_a * (mu_1a - mu_1b)
            term2 = (1 - p_a) * (mu_0a - mu_0b)
            term3 = (p_a - p_b) * (mu_1b - mu_0b)
            rhs = term1 + term2 + term3
            residual = abs(lhs - rhs)

            results.append({
                "delta_p": dp,
                "seed": seed,
                "lhs": lhs,
                "term1": term1,
                "term2": term2,
                "term3": term3,
                "rhs": rhs,
                "residual": residual,
                "p_a": p_a,
                "p_b": p_b,
            })

    return results


# Exp 1b: Finite-Criteria Impossibility


def _linear_kernel_matrix(X: np.ndarray) -> np.ndarray:
    """Compute K(x, x') = X @ X^T as a non-characteristic kernel control."""
    X = np.asarray(X, dtype=np.float64)
    return X @ X.T


def _residual_curves(
    K: np.ndarray,
    group: np.ndarray,
    max_k: int,
    n_eig: int,
    n_random_trials: int,
    seed: int,
    min_capture_frac: float = 0.90,
) -> dict:
    """Compute residual MMD fraction curves for greedy, KPCA, and random strategies.

    Returns greedy/kpca/random arrays, mmd2_biased, eigenvalue_capture_fraction,
    and k_90/k_95/k_99 summary stats.
    """
    n = K.shape[0]
    assert K.shape == (n, n), f"Expected ({n}, {n}), got {K.shape}"

    n_a = int((group == 1).sum())
    n_b = int((group == 0).sum())
    assert n_a + n_b == n, f"n_a={n_a} + n_b={n_b} != {n}"

    c = np.zeros(n, dtype=np.float64)
    c[group == 1] = 1.0 / n_a
    c[group == 0] = -1.0 / n_b

    mmd2 = float(c @ K @ c)
    assert mmd2 > 0, f"Biased MMD^2 should be positive, got {mmd2}"

    n_eig_actual = min(n_eig, n)
    eigenvalues, eigenvectors = eigh(
        K, subset_by_index=[n - n_eig_actual, n - 1]
    )
    eigenvalues = eigenvalues[::-1].copy()
    eigenvectors = eigenvectors[:, ::-1].copy()

    valid = eigenvalues > 1e-10
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]
    n_valid = len(eigenvalues)
    assert n_valid > 0, "No eigenvalues above threshold"

    projections = eigenvectors.T @ c  # (n_valid,)
    d_coords = np.sqrt(eigenvalues) * projections  # (n_valid,)
    d_sq = d_coords ** 2

    capture_frac = float(d_sq.sum() / mmd2)
    truncation_residual = 1.0 - capture_frac
    assert capture_frac > min_capture_frac, (
        f"Eigendecomposition only captures {capture_frac:.4f} of MMD^2 "
        f"(need >{min_capture_frac}). Try increasing N_EIGENVALUES."
    )

    k_at_target = {
        f"k_{int(t * 100)}": _capture_k(d_sq, mmd2, t)
        for t in (0.90, 0.95, 0.99)
    }

    k_max = min(max_k, n_valid)

    # Greedy: sort by |d_i| descending
    greedy_order = np.argsort(-np.abs(d_coords))
    greedy_cumsum = np.cumsum(d_sq[greedy_order])
    greedy_residual = np.ones(k_max + 1)
    greedy_residual[1:] = 1.0 - greedy_cumsum[:k_max] / mmd2

    # KPCA: already in eigenvalue order (descending)
    kpca_cumsum = np.cumsum(d_sq)
    kpca_residual = np.ones(k_max + 1)
    kpca_residual[1:] = 1.0 - kpca_cumsum[:k_max] / mmd2

    # Random: random orthonormal directions in eigenspace
    random_residuals = np.zeros((n_random_trials, k_max + 1))
    for trial in range(n_random_trials):
        rng = np.random.default_rng(seed + 1000 + trial)
        Z = rng.standard_normal((n_valid, n_valid))
        Q, _ = np.linalg.qr(Z)
        rotated = Q.T @ d_coords  # (n_valid,)
        rot_sq = rotated ** 2
        perm = rng.permutation(n_valid)
        rot_sq_perm = rot_sq[perm]
        cum = np.cumsum(rot_sq_perm)
        random_residuals[trial, 0] = 1.0
        random_residuals[trial, 1:] = 1.0 - cum[:k_max] / mmd2

    random_mean = random_residuals.mean(axis=0)

    return {
        "greedy": greedy_residual.tolist(),
        "kpca": kpca_residual.tolist(),
        "random": random_mean.tolist(),
        "random_all": random_residuals.tolist(),
        "mmd2_biased": mmd2,
        "eigenvalue_capture_fraction": capture_frac,
        "truncation_residual": truncation_residual,
        "n_valid_eigenvalues": n_valid,
        "k_max": k_max,
        **k_at_target,  # k_90, k_95, k_99
    }


def run_exp1b(
    seeds: list[int],
    delta_p: float = EXP1B_DELTA_P,
    sigma_multiplier: float = 1.0,
    run_linear_control: bool = EXP1B_RUN_LINEAR_CONTROL,
    min_capture_frac: float = 0.90,
) -> dict:
    """Run finite-criteria impossibility across seeds for RBF and optionally linear kernel."""
    per_seed = []

    for seed in seeds:
        print(f"  Exp 1b: seed={seed}, sigma_mult={sigma_multiplier}")
        ds, params = generate_synthetic(
            n=N_SAMPLES, delta_p=delta_p, d=D_FEATURES, seed=seed
        )

        # RBF (characteristic)
        base_sigma = median_heuristic(ds.X)
        sigma = sigma_multiplier * base_sigma

        K = rbf_kernel_matrix(ds.X, sigma=sigma)
        assert K.shape == (N_SAMPLES, N_SAMPLES), (
            f"Expected ({N_SAMPLES}, {N_SAMPLES}), got {K.shape}"
        )

        curves = _residual_curves(
            K, ds.group, MAX_CRITERIA, N_EIGENVALUES, N_RANDOM_TRIALS, seed,
            min_capture_frac=min_capture_frac,
        )
        del K

        seed_entry = {
            "seed": seed,
            "sigma": sigma,
            **curves,
        }

        # Linear (non-characteristic, rank d) control
        if run_linear_control:
            K_linear = _linear_kernel_matrix(ds.X)
            assert K_linear.shape == (N_SAMPLES, N_SAMPLES), (
                f"Expected ({N_SAMPLES}, {N_SAMPLES}), got {K_linear.shape}"
            )
            # d+5 eigenvectors: captures rank-d eigenspace plus buffer
            curves_linear = _residual_curves(
                K_linear, ds.group, MAX_CRITERIA,
                n_eig=D_FEATURES + 5,
                n_random_trials=N_RANDOM_TRIALS,
                seed=seed,
            )
            del K_linear
            seed_entry["linear_control"] = curves_linear
            assert curves_linear["n_valid_eigenvalues"] <= D_FEATURES, (
                f"Linear kernel rank should be <= {D_FEATURES}, "
                f"got {curves_linear['n_valid_eigenvalues']}"
            )

        per_seed.append(seed_entry)

    # Aggregate RBF across seeds
    k_max = min(r["k_max"] for r in per_seed)
    greedy_all = np.array([r["greedy"][:k_max + 1] for r in per_seed])
    kpca_all = np.array([r["kpca"][:k_max + 1] for r in per_seed])
    random_all = np.array([r["random"][:k_max + 1] for r in per_seed])

    # When a target fraction cannot be reached within the eigenbasis,
    # _capture_k returns -1; substitute n_valid as a lower bound.
    trunc_all = np.array([r["truncation_residual"] for r in per_seed])
    n_valid_all = np.array(
        [r["n_valid_eigenvalues"] for r in per_seed], dtype=float
    )

    def _agg_k(target_key: str):
        raw = np.array([r[target_key] for r in per_seed], dtype=float)
        reached = (raw >= 0)
        vals = np.where(reached, raw, n_valid_all)
        return vals, bool(reached.all())

    k90_all, k90_reached = _agg_k("k_90")
    k95_all, k95_reached = _agg_k("k_95")
    k99_all, k99_reached = _agg_k("k_99")

    out = {
        "delta_p": delta_p,
        "sigma_multiplier": sigma_multiplier,
        "per_seed": per_seed,
        "k_max": k_max,
        "greedy_mean": greedy_all.mean(axis=0).tolist(),
        "greedy_std": greedy_all.std(axis=0).tolist(),
        "kpca_mean": kpca_all.mean(axis=0).tolist(),
        "kpca_std": kpca_all.std(axis=0).tolist(),
        "random_mean": random_all.mean(axis=0).tolist(),
        "random_std": random_all.std(axis=0).tolist(),
        "truncation_floor_mean": float(trunc_all.mean()),
        "truncation_floor_std": float(trunc_all.std()),
        "truncation_floor_per_seed": trunc_all.tolist(),
        "k_90_mean": float(k90_all.mean()),
        "k_90_std": float(k90_all.std()),
        "k_90_reached": k90_reached,
        "k_95_mean": float(k95_all.mean()),
        "k_95_std": float(k95_all.std()),
        "k_95_reached": k95_reached,
        "k_99_mean": float(k99_all.mean()),
        "k_99_std": float(k99_all.std()),
        "k_99_reached": k99_reached,
    }

    # Aggregate linear control across seeds
    if run_linear_control:
        lin_k_max = min(r["linear_control"]["k_max"] for r in per_seed)
        lin_greedy_all = np.array(
            [r["linear_control"]["greedy"][:lin_k_max + 1] for r in per_seed]
        )
        lin_kpca_all = np.array(
            [r["linear_control"]["kpca"][:lin_k_max + 1] for r in per_seed]
        )
        lin_random_all = np.array(
            [r["linear_control"]["random"][:lin_k_max + 1] for r in per_seed]
        )
        out["linear_k_max"] = lin_k_max
        out["linear_greedy_mean"] = lin_greedy_all.mean(axis=0).tolist()
        out["linear_greedy_std"] = lin_greedy_all.std(axis=0).tolist()
        out["linear_kpca_mean"] = lin_kpca_all.mean(axis=0).tolist()
        out["linear_kpca_std"] = lin_kpca_all.std(axis=0).tolist()
        out["linear_random_mean"] = lin_random_all.mean(axis=0).tolist()
        out["linear_random_std"] = lin_random_all.std(axis=0).tolist()

    return out


# Exp 1c: Bandwidth Robustness


def run_exp1c() -> dict:
    """Run Exp 1b at five bandwidth multipliers across all seeds.

    Narrow bandwidths use permissive capture threshold since the kernel
    becomes near-identity and the truncated eigenbasis cannot fully capture delta_hat.
    """
    results = {}
    for mult in EXP1C_BANDWIDTH_MULTIPLIERS:
        print(f"  Exp 1c: bandwidth multiplier = {mult}")
        min_cf = 0.0 if mult <= 0.1 else 0.90
        r = run_exp1b(
            list(RANDOM_SEEDS),
            sigma_multiplier=mult,
            run_linear_control=False,
            min_capture_frac=min_cf,
        )
        results[str(mult)] = r
    return results


# Exp 1d: Dimensional Transition (n sweep)


def _capture_k(d_sq: np.ndarray, mmd2: float, target: float) -> int:
    """Return smallest k with cumulative greedy capture >= target * mmd2, or -1."""
    if mmd2 <= 0:
        return -1
    order = np.argsort(-d_sq)
    cum = np.cumsum(d_sq[order])
    threshold = target * mmd2
    idx = np.searchsorted(cum, threshold)
    if idx >= len(cum):
        return -1
    return int(idx) + 1  # 1-indexed


def run_exp1d_dimensional_transition(
    n_values: list[int] = EXP1D_N_VALUES,
    delta_p: float = EXP1B_DELTA_P,
    seed: int = DEFAULT_SEED,
    capture_targets: list[float] = EXP1D_CAPTURE_TARGETS,
) -> dict:
    """Measure k_c(RBF) vs k_c(linear) across sample sizes for a single seed."""
    out = {
        "delta_p": delta_p,
        "seed": seed,
        "n_values": list(n_values),
        "capture_targets": list(capture_targets),
        "results": {},
    }

    for n in n_values:
        print(f"  Exp 1d: n={n}")
        ds, _ = generate_synthetic(n=n, delta_p=delta_p, d=D_FEATURES, seed=seed)

        n_a = int((ds.group == 1).sum())
        n_b = int((ds.group == 0).sum())
        c = np.zeros(n, dtype=np.float64)
        c[ds.group == 1] = 1.0 / n_a
        c[ds.group == 0] = -1.0 / n_b

        per_n = {}

        for kernel_name in ("rbf", "linear"):
            if kernel_name == "rbf":
                sigma = median_heuristic(ds.X)
                K = rbf_kernel_matrix(ds.X, sigma=sigma)
            else:
                K = _linear_kernel_matrix(ds.X)

            mmd2 = float(c @ K @ c)
            assert mmd2 > 0, (
                f"Biased MMD^2 should be positive (n={n}, kernel={kernel_name}), "
                f"got {mmd2}"
            )

            n_eig_actual = min(EXP1D_N_EIG_TRANSITION, n)
            eigenvalues, eigenvectors = eigh(
                K, subset_by_index=[n - n_eig_actual, n - 1]
            )
            del K
            eigenvalues = eigenvalues[::-1].copy()
            eigenvectors = eigenvectors[:, ::-1].copy()
            valid = eigenvalues > 1e-10
            eigenvalues = eigenvalues[valid]
            eigenvectors = eigenvectors[:, valid]

            projections = eigenvectors.T @ c
            d_coords = np.sqrt(eigenvalues) * projections
            d_sq = d_coords ** 2
            n_valid = int(len(eigenvalues))

            capture_k = {
                f"k_{int(target * 100)}": _capture_k(d_sq, mmd2, target)
                for target in capture_targets
            }
            capture_k["n_valid"] = n_valid
            capture_k["mmd2_biased"] = float(mmd2)
            capture_k["captured_fraction"] = float(d_sq.sum() / mmd2)
            if kernel_name == "rbf":
                capture_k["sigma"] = float(sigma)

            per_n[kernel_name] = capture_k

        out["results"][str(n)] = per_n

    return out


# Exp 1e: Rich-DGP n-sweep


def run_exp1e_rich_dgp_n_sweep(
    n_values: list[int] = EXP1E_N_VALUES,
    d: int = EXP1E_D_FEATURES,
    decay: float = EXP1E_DECAY,
    total_shift: float = EXP1E_TOTAL_SHIFT,
    delta_p: float = EXP1B_DELTA_P,
    seeds: list[int] | None = None,
    capture_targets: list[float] = EXP1E_CAPTURE_TARGETS,
    n_eig_cap: int = EXP1E_N_EIG_CAP,
) -> dict:
    """Multi-seed n-sweep with power-law-decay mean shift across all d features.

    Demonstrates k_c(RBF) growth vs k_c(linear) <= d bound.
    """
    if seeds is None:
        seeds = list(RANDOM_SEEDS)

    target_keys = [f"k_{int(t * 100)}" for t in capture_targets]

    out = {
        "delta_p": delta_p,
        "d": d,
        "decay": decay,
        "total_shift": total_shift,
        "n_values": list(n_values),
        "capture_targets": list(capture_targets),
        "seeds": list(seeds),
        "results": {},
    }

    for n in n_values:
        print(f"  Exp 1e: n={n}")
        n_eig_actual = min(n_eig_cap, n - 1)

        per_seed_per_kernel = {
            "rbf":    {k: [] for k in target_keys + ["mmd2", "n_valid", "capture", "sigma"]},
            "linear": {k: [] for k in target_keys + ["mmd2", "n_valid", "capture"]},
        }

        for seed in seeds:
            ds, _ = generate_rich_synthetic(
                n=n,
                delta_p=delta_p,
                d=d,
                decay=decay,
                total_shift=total_shift,
                seed=seed,
            )

            n_a = int((ds.group == 1).sum())
            n_b = int((ds.group == 0).sum())
            c_vec = np.zeros(n, dtype=np.float64)
            c_vec[ds.group == 1] = 1.0 / n_a
            c_vec[ds.group == 0] = -1.0 / n_b

            # RBF kernel pass
            sigma = median_heuristic(ds.X)
            K_rbf = rbf_kernel_matrix(ds.X, sigma=sigma)
            assert K_rbf.shape == (n, n)

            mmd2_rbf = float(c_vec @ K_rbf @ c_vec)
            assert mmd2_rbf > 0, (
                f"RBF biased MMD^2 non-positive at n={n}, seed={seed}: {mmd2_rbf}"
            )

            eigvals, eigvecs = eigh(
                K_rbf, subset_by_index=[n - n_eig_actual, n - 1]
            )
            del K_rbf
            eigvals = eigvals[::-1].copy()
            eigvecs = eigvecs[:, ::-1].copy()
            valid = eigvals > 1e-10
            eigvals = eigvals[valid]
            eigvecs = eigvecs[:, valid]
            d_sq = (np.sqrt(eigvals) * (eigvecs.T @ c_vec)) ** 2
            n_valid_rbf = int(len(eigvals))
            capture_rbf = float(d_sq.sum() / mmd2_rbf)
            assert capture_rbf > 0.99, (
                f"RBF truncation insufficient at n={n}, seed={seed}: "
                f"capture={capture_rbf:.4f} (need > 0.99). "
                f"Bump EXP1E_N_EIG_CAP."
            )
            for tgt, key in zip(capture_targets, target_keys):
                per_seed_per_kernel["rbf"][key].append(
                    _capture_k(d_sq, mmd2_rbf, tgt)
                )
            per_seed_per_kernel["rbf"]["mmd2"].append(mmd2_rbf)
            per_seed_per_kernel["rbf"]["n_valid"].append(n_valid_rbf)
            per_seed_per_kernel["rbf"]["capture"].append(capture_rbf)
            per_seed_per_kernel["rbf"]["sigma"].append(float(sigma))

            # Linear kernel pass (rank <= d)
            K_lin = _linear_kernel_matrix(ds.X)
            assert K_lin.shape == (n, n)

            mmd2_lin = float(c_vec @ K_lin @ c_vec)
            assert mmd2_lin > 0, (
                f"linear biased MMD^2 non-positive at n={n}, seed={seed}: {mmd2_lin}"
            )

            n_eig_lin = min(d + 5, n - 1)
            eigvals, eigvecs = eigh(
                K_lin, subset_by_index=[n - n_eig_lin, n - 1]
            )
            del K_lin
            eigvals = eigvals[::-1].copy()
            eigvecs = eigvecs[:, ::-1].copy()
            valid = eigvals > 1e-10
            eigvals = eigvals[valid]
            eigvecs = eigvecs[:, valid]
            d_sq = (np.sqrt(eigvals) * (eigvecs.T @ c_vec)) ** 2
            n_valid_lin = int(len(eigvals))
            capture_lin = float(d_sq.sum() / mmd2_lin)
            assert capture_lin > 0.99, (
                f"Linear truncation insufficient at n={n}, seed={seed}: "
                f"capture={capture_lin:.4f} (need > 0.99)."
            )
            assert n_valid_lin <= d, (
                f"Linear kernel rank should be <= {d}, got {n_valid_lin}"
            )
            for tgt, key in zip(capture_targets, target_keys):
                per_seed_per_kernel["linear"][key].append(
                    _capture_k(d_sq, mmd2_lin, tgt)
                )
            per_seed_per_kernel["linear"]["mmd2"].append(mmd2_lin)
            per_seed_per_kernel["linear"]["n_valid"].append(n_valid_lin)
            per_seed_per_kernel["linear"]["capture"].append(capture_lin)

        per_n = {}
        for kernel_name, metrics in per_seed_per_kernel.items():
            agg: dict = {}
            for key, vals in metrics.items():
                arr = np.asarray(vals, dtype=np.float64)
                agg[f"{key}_mean"] = float(arr.mean())
                agg[f"{key}_std"]  = float(arr.std())
                agg[f"{key}_min"]  = float(arr.min())
                agg[f"{key}_max"]  = float(arr.max())
                agg[f"{key}_all"]  = [float(v) for v in arr]
            per_n[kernel_name] = agg
        out["results"][str(n)] = per_n

    return out


# Plotting

_TERM_COLORS = {
    "LHS": "black",
    "Term 1": "#1f77b4",   # blue
    "Term 2": "#ff7f0e",   # orange
    "Term 3": "#2ca02c",   # green
}


def plot_exp1a_sweep(results: list[dict]) -> None:
    """Two-panel figure: constraint terms vs delta_p (left) and residual (right)."""
    delta_ps = sorted(set(r["delta_p"] for r in results))
    lhs_by_dp = {dp: [] for dp in delta_ps}
    t1_by_dp = {dp: [] for dp in delta_ps}
    t2_by_dp = {dp: [] for dp in delta_ps}
    t3_by_dp = {dp: [] for dp in delta_ps}
    res_by_dp = {dp: [] for dp in delta_ps}

    for r in results:
        dp = r["delta_p"]
        lhs_by_dp[dp].append(r["lhs"])
        t1_by_dp[dp].append(r["term1"])
        t2_by_dp[dp].append(r["term2"])
        t3_by_dp[dp].append(r["term3"])
        res_by_dp[dp].append(r["residual"])

    dp_arr = np.array(delta_ps)

    def _stats(vals_by_dp):
        means = np.array([np.mean(vals_by_dp[dp]) for dp in delta_ps])
        stds = np.array([np.std(vals_by_dp[dp]) for dp in delta_ps])
        return means, stds

    lhs_m, lhs_s = _stats(lhs_by_dp)
    t1_m, t1_s = _stats(t1_by_dp)
    t2_m, t2_s = _stats(t2_by_dp)
    t3_m, t3_s = _stats(t3_by_dp)
    res_m, res_s = _stats(res_by_dp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for label, m, s, color in [
        ("LHS: $\\mu_a - \\mu_b$", lhs_m, lhs_s, _TERM_COLORS["LHS"]),
        ("Term 1: $p_a(\\mu_{1a} - \\mu_{1b})$", t1_m, t1_s, _TERM_COLORS["Term 1"]),
        ("Term 2: $(1-p_a)(\\mu_{0a} - \\mu_{0b})$", t2_m, t2_s, _TERM_COLORS["Term 2"]),
        ("Term 3: $(p_a-p_b)(\\mu_{1b} - \\mu_{0b})$", t3_m, t3_s, _TERM_COLORS["Term 3"]),
    ]:
        ax1.errorbar(dp_arr, m, yerr=s, marker="o", markersize=5, capsize=3,
                     label=label, color=color)

    ax1.set_xlabel("Base-rate gap $\\Delta p$")
    ax1.set_ylabel("Score difference")
    ax1.set_title("Master Constraint Terms vs $\\Delta p$")
    ax1.legend(fontsize=8)

    ax2.errorbar(dp_arr, res_m, yerr=res_s, marker="o", markersize=5, capsize=3,
                 color="red")
    ax2.set_yscale("log")
    ax2.set_xlabel("Base-rate gap $\\Delta p$")
    ax2.set_ylabel("$|\\mathrm{LHS} - \\mathrm{RHS}|$")
    ax2.set_title("Master Constraint Residual")

    fig.tight_layout()
    paths = save_figure(fig, "fig1a_master_constraint_sweep")
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def plot_exp1a_bars(results: list[dict], target_dp: float) -> None:
    """Bar decomposition for a single delta_p value."""
    subset = [r for r in results if abs(r["delta_p"] - target_dp) < 1e-8]
    assert len(subset) > 0, f"No results for delta_p={target_dp}"

    lhs_vals = [r["lhs"] for r in subset]
    t1_vals = [r["term1"] for r in subset]
    t2_vals = [r["term2"] for r in subset]
    t3_vals = [r["term3"] for r in subset]
    res_vals = [r["residual"] for r in subset]

    means = [np.mean(lhs_vals), np.mean(t1_vals), np.mean(t2_vals), np.mean(t3_vals)]
    stds = [np.std(lhs_vals), np.std(t1_vals), np.std(t2_vals), np.std(t3_vals)]
    labels = ["LHS", "Term 1\n(separation)", "Term 2\n(separation)", "Term 3\n(base-rate)"]
    colors = [_TERM_COLORS["LHS"], _TERM_COLORS["Term 1"],
              _TERM_COLORS["Term 2"], _TERM_COLORS["Term 3"]]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    rhs_sum = np.mean(t1_vals) + np.mean(t2_vals) + np.mean(t3_vals)
    ax.axhline(rhs_sum, linestyle="--", color="gray", linewidth=1.5, label="Sum of RHS terms")

    mean_res = np.mean(res_vals)
    ax.text(0.98, 0.95, f"|LHS $-$ RHS| = {mean_res:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score difference")
    ax.set_title(f"Master Constraint Decomposition ($\\Delta p = {target_dp:.2f}$)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    paths = save_figure(fig, "fig1a_master_constraint_bars")
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def plot_exp1b(results: dict) -> None:
    """Plot residual MMD fraction vs k for RBF and optional linear control."""
    k_max = results["k_max"]
    ks = np.arange(k_max + 1)

    greedy_m = np.array(results["greedy_mean"])
    greedy_s = np.array(results["greedy_std"])
    kpca_m = np.array(results["kpca_mean"])
    kpca_s = np.array(results["kpca_std"])
    random_m = np.array(results["random_mean"])
    random_s = np.array(results["random_std"])

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(ks, greedy_m, color="#d62728",
            label="Greedy — RBF (characteristic)", linewidth=1.5)
    ax.fill_between(ks, greedy_m - greedy_s, greedy_m + greedy_s,
                     color="#d62728", alpha=0.15)

    ax.plot(ks, kpca_m, color="#1f77b4",
            label="Kernel PCA — RBF", linewidth=1.5)
    ax.fill_between(ks, kpca_m - kpca_s, kpca_m + kpca_s,
                     color="#1f77b4", alpha=0.15)

    ax.plot(ks, random_m, color="#7f7f7f",
            label="Random directions — RBF", linewidth=1.5)
    ax.fill_between(ks, random_m - random_s, random_m + random_s,
                     color="#7f7f7f", alpha=0.15)

    if "linear_greedy_mean" in results:
        lin_m = np.array(results["linear_greedy_mean"])
        lin_s = np.array(results["linear_greedy_std"])
        ks_lin = np.arange(len(lin_m))
        ax.plot(ks_lin, lin_m, color="#9467bd", linestyle="--",
                label=f"Greedy — linear (non-char., rank $d{{=}}{D_FEATURES}$)",
                linewidth=1.8)
        ax.fill_between(ks_lin, lin_m - lin_s, lin_m + lin_s,
                        color="#9467bd", alpha=0.15)
        ax.axvline(D_FEATURES, linestyle=":", color="#9467bd",
                   linewidth=1, alpha=0.7)
        ax.annotate(f"$k{{=}}d{{=}}{D_FEATURES}$:\nlinear $\\to 0$",
                    xy=(D_FEATURES, 0.0), xytext=(D_FEATURES + 4, 0.18),
                    fontsize=8, color="#9467bd",
                    arrowprops=dict(arrowstyle="->", color="#9467bd"))

    ax.axvline(3, linestyle="--", color="gray", linewidth=1, alpha=0.7)
    ax.annotate("$k=3$ (reference)",
                xy=(3, 0.5), xytext=(12, 0.72),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"),
                color="gray")

    trunc_floor = results.get("truncation_floor_mean")
    if trunc_floor is not None and trunc_floor > 0:
        ax.axhline(trunc_floor, linestyle=":", color="#8c564b",
                   linewidth=1.5, alpha=0.9,
                   label=f"Truncation floor ($1-\\mathrm{{capture}}$)"
                         f" $= {trunc_floor:.1e}$")

    ax.set_xlabel("Number of kernel-eigenvector directions $k$")
    ax.set_ylabel("Residual fraction $\\|\\delta_\\perp\\|^2 / \\|\\delta\\|^2$")
    ax.set_title("Finite-Criteria Impossibility: Dimensional Transition")
    ax.set_xlim(0, k_max)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    paths = save_figure(fig, "fig1b_finite_criteria")
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def plot_exp1c(results: dict) -> None:
    """Overlay greedy residual curves at five bandwidths (left) and k_99 vs sigma (right)."""
    fig, (ax_curves, ax_rank) = plt.subplots(1, 2, figsize=(11, 4.5))

    bw_order = sorted(results.keys(), key=float)
    bw_colors_list = ["#000000", "#9467bd", "#2ca02c", "#1f77b4", "#d62728"]
    bw_color = dict(zip(bw_order, bw_colors_list[:len(bw_order)]))

    for mult_str in bw_order:
        r = results[mult_str]
        k_max = r["k_max"]
        ks = np.arange(k_max + 1)
        greedy_m = np.array(r["greedy_mean"])
        greedy_s = np.array(r["greedy_std"])
        color = bw_color[mult_str]
        trunc = r.get("truncation_floor_mean", 0.0)
        label = f"${mult_str}\\times$ median"
        if trunc > 0.05:
            label += f" (cap={1 - trunc:.2f})"
        ax_curves.plot(ks, greedy_m, color=color, label=label, linewidth=1.5)
        ax_curves.fill_between(ks, np.maximum(greedy_m - greedy_s, 1e-8),
                               greedy_m + greedy_s, color=color, alpha=0.15)

    ax_curves.axvline(3, linestyle="--", color="gray", linewidth=1, alpha=0.7)
    ax_curves.set_xlabel("Number of kernel-eigenvector directions $k$")
    ax_curves.set_ylabel("Residual fraction $\\|\\delta_\\perp\\|^2 / \\|\\delta\\|^2$")
    ax_curves.set_title("Greedy Residual vs Bandwidth (5 seeds)")
    ax_curves.set_xlim(0, MAX_CRITERIA)
    ax_curves.set_yscale("log")
    ax_curves.set_ylim(1e-5, 1.1)
    ax_curves.legend(fontsize=8, loc="lower left")
    ax_curves.grid(True, which="both", alpha=0.3)

    mults = np.array([float(m) for m in bw_order])
    k99_means = np.array([results[m]["k_99_mean"] for m in bw_order])
    k99_stds = np.array([results[m]["k_99_std"] for m in bw_order])
    k99_reached = np.array([results[m].get("k_99_reached", True) for m in bw_order])

    clean_mask = k99_reached
    ax_rank.errorbar(mults[clean_mask], k99_means[clean_mask],
                     yerr=k99_stds[clean_mask], fmt="o", color="#d62728",
                     markersize=8, capsize=4, linewidth=1.5,
                     label="RBF $k_{99}$ (well-captured)")
    if (~clean_mask).any():
        ax_rank.errorbar(mults[~clean_mask], k99_means[~clean_mask],
                         yerr=k99_stds[~clean_mask], fmt="o",
                         markerfacecolor="white", markeredgecolor="#d62728",
                         markersize=8, capsize=4, linewidth=1.5, color="#d62728",
                         label="RBF $k_{99}$ (lower bound: trunc.-limited)")

    ax_rank.plot(mults, k99_means, color="#d62728", linewidth=1,
                 linestyle="--", alpha=0.6)

    ax_rank.axhline(D_FEATURES, linestyle=":", color="#9467bd",
                    linewidth=1.2, alpha=0.8,
                    label=f"Linear rank $= d = {D_FEATURES}$")
    ax_rank.axhline(N_EIGENVALUES, linestyle=":", color="gray",
                    linewidth=1, alpha=0.5,
                    label=f"N_EIGENVALUES $= {N_EIGENVALUES}$")

    ax_rank.set_xscale("log")
    ax_rank.set_xlabel("Bandwidth multiplier $\\sigma / \\sigma_{\\mathrm{median}}$")
    ax_rank.set_ylabel("Effective witness rank $k_{99}$")
    ax_rank.set_title("Witness Space Grows as $\\sigma$ Shrinks")
    ax_rank.legend(fontsize=8, loc="upper right")
    ax_rank.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    paths = save_figure(fig, "fig1c_bandwidth_robustness")
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def write_exp1c_table(results: dict) -> None:
    """Emit a LaTeX table for the exp1c bandwidth sweep."""
    bw_order = sorted(results.keys(), key=float)

    rows = []
    any_trunc_limited = False
    for mult_str in bw_order:
        r = results[mult_str]
        cap = 1.0 - r["truncation_floor_mean"]
        dagger = ""
        if not r.get("k_99_reached", True) or cap < 0.95:
            any_trunc_limited = True
            dagger = "$^{\\dagger}$"

        def _fmt_k(tag: str) -> str:
            m = r[f"k_{tag}_mean"]
            s = r[f"k_{tag}_std"]
            reached = r.get(f"k_{tag}_reached", True)
            if not reached:
                return f"$\\geq {int(round(m))}$"
            return f"${m:.1f} \\pm {s:.1f}$"

        k90 = _fmt_k("90")
        k95 = _fmt_k("95")
        k99 = _fmt_k("99")
        rows.append(
            f"${mult_str}${dagger} & {k90} & {k95} & {k99} & ${cap:.4f}$ \\\\"
        )

    caption = (
        "Bandwidth sweep ($\\Delta p=0.20$, $n=10{,}000$, RBF kernel, 5 seeds, "
        f"top-{N_EIGENVALUES} eigendecomposition). The effective witness rank "
        "$k_\\theta$ (smallest $k$ in the greedy order capturing $\\theta$ of "
        "$\\|\\hat\\delta\\|^2$) grows monotonically as $\\sigma$ shrinks, "
        "consistent with the Hermite-expansion intuition: at narrow bandwidth "
        "the RBF kernel weights high-order Hermite polynomials more heavily "
        "and the effective rank of $\\hat\\delta$ in the kernel eigenbasis "
        "grows. The linear (non-characteristic) kernel's structural rank "
        f"bound is $d = {D_FEATURES}$ regardless of any scale parameter."
    )
    if any_trunc_limited:
        caption += (
            " $^{\\dagger}$ At very narrow bandwidth the RBF kernel is "
            "near-identity on this DGP (mean shift $\\approx 0.5 \\gg \\sigma$), "
            f"and the top-{N_EIGENVALUES} eigenbasis captures only a fraction "
            "of MMD$^2$. The reported $k_\\theta$ values for those rows are "
            "\\emph{lower bounds} on the true effective rank; the true "
            "$k_{99}$ at $\\sigma = 0.01 \\sigma_\\mathrm{med}$ is higher "
            f"than {N_EIGENVALUES}."
        )

    table = (
        "% Exp 1c: Bandwidth sweep -- effective witness rank vs sigma multiplier\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:exp1-bandwidth}\n"
        "\\begin{tabular}{ccccc}\n"
        "\\toprule\n"
        "$\\sigma / \\sigma_{\\mathrm{med}}$ & $k_{90}$ & $k_{95}$ & $k_{99}$"
        " & Capture \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    out_path = RESULTS_DIR / "tables" / "exp1_bandwidth.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"  Saved: {out_path}")


def plot_exp1d_dimensional_transition(results: dict) -> None:
    """Plot k_c vs n for RBF and linear kernels."""
    n_values = results["n_values"]
    capture_targets = results["capture_targets"]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    target_styles = {
        0.90: dict(linestyle=":", marker="o"),
        0.95: dict(linestyle="--", marker="s"),
        0.99: dict(linestyle="-",  marker="^"),
    }

    for target in capture_targets:
        key = f"k_{int(target * 100)}"
        rbf_ks = [results["results"][str(n)]["rbf"][key] for n in n_values]
        lin_ks = [results["results"][str(n)]["linear"][key] for n in n_values]
        style = target_styles.get(target, dict(linestyle="-", marker="o"))
        ax.plot(n_values, rbf_ks, color="#d62728",
                label=f"RBF $k_{{{int(target*100)}}}$",
                linewidth=1.5, markersize=5, **style)
        ax.plot(n_values, lin_ks, color="#9467bd",
                label=f"Linear $k_{{{int(target*100)}}}$",
                linewidth=1.5, markersize=5, **style)

    ax.axhline(D_FEATURES, color="#9467bd", linestyle=":", alpha=0.5,
               linewidth=1)
    ax.text(n_values[-1], D_FEATURES + 0.5,
            f"linear rank $= d = {D_FEATURES}$",
            ha="right", va="bottom", color="#9467bd", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Smallest $k$ capturing fraction of $\\|\\delta\\|^2$")
    ax.set_title("Dimensional Transition: Characteristic vs Non-Characteristic")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    paths = save_figure(fig, "fig1d_dimensional_transition")
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def write_exp1d_table(results: dict) -> None:
    """Emit a LaTeX table summarising the dimensional transition."""
    n_values = results["n_values"]
    targets = results["capture_targets"]
    target_pcts = [int(t * 100) for t in targets]

    header = (
        "$n$ & "
        + " & ".join(f"$k_{{{p}}}^{{\\mathrm{{RBF}}}}$" for p in target_pcts)
        + " & "
        + " & ".join(f"$k_{{{p}}}^{{\\mathrm{{lin}}}}$" for p in target_pcts)
        + " \\\\"
    )

    rows = []
    for n in n_values:
        rbf = results["results"][str(n)]["rbf"]
        lin = results["results"][str(n)]["linear"]
        rbf_cells = [
            ("$-$" if rbf[f"k_{p}"] < 0 else str(rbf[f"k_{p}"]))
            for p in target_pcts
        ]
        lin_cells = [
            ("$-$" if lin[f"k_{p}"] < 0 else str(lin[f"k_{p}"]))
            for p in target_pcts
        ]
        rows.append(
            f"{n:,} & " + " & ".join(rbf_cells) + " & " + " & ".join(lin_cells) + " \\\\"
        )

    n_cols = 1 + 2 * len(target_pcts)
    col_spec = "r" + "c" * (n_cols - 1)

    table = (
        "% Exp 1d: Dimensional transition -- k_c vs n for RBF and linear kernels\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Dimensional transition: smallest $k$ capturing $90/95/99\\%$ "
        f"of $\\|\\hat\\delta\\|^2$ for the RBF (characteristic) and linear "
        f"(non-characteristic, rank $d{{=}}{D_FEATURES}$) kernels at "
        f"$\\Delta p = {results['delta_p']}$, single seed. "
        "The linear kernel is \\emph{structurally} bounded by $k_c \\le d$ "
        f"for all $n$ and all targets (rank constraint); the RBF is not "
        "bounded \\emph{a priori} and at small $n$ exceeds $d$ substantially "
        "(e.g.\\ $k_{99}^{\\mathrm{RBF}}{=}48$ at $n{=}500$, i.e.\\ nearly $5\\times$ "
        "the linear rank). For this DGP the class signal is concentrated in a "
        "single feature, so at large $n$ the empirical $\\hat\\delta$ "
        "re-concentrates in few directions for both kernels; the structural "
        f"bound $k_c^{{\\mathrm{{lin}}}} \\le d{{=}}{D_FEATURES}$ is the robust "
        "signature of the dimensional transition.}\n"
        "\\label{tab:dim-transition}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    out_path = RESULTS_DIR / "tables" / "exp1_dimensional_transition.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"  Saved: {out_path}")


def plot_exp1e_rich_transition(results: dict) -> None:
    """Two-panel figure: k_c vs n (left) and RBF-linear gap vs n (right)."""
    n_values = np.asarray(results["n_values"], dtype=np.float64)
    capture_targets = results["capture_targets"]
    d = results["d"]
    n_seeds = len(results.get("seeds", []))

    target_styles = {
        0.90: dict(marker="o", alpha_factor=0.55),
        0.95: dict(marker="s", alpha_factor=0.75),
        0.99: dict(marker="^", alpha_factor=1.00),
    }
    rbf_color = "#d62728"
    lin_color = "#9467bd"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    for target in capture_targets:
        key = f"k_{int(target * 100)}"
        style = target_styles.get(target, dict(marker="o", alpha_factor=1.0))

        rbf_mean = np.array([
            results["results"][str(int(n))]["rbf"][f"{key}_mean"] for n in n_values
        ])
        rbf_std = np.array([
            results["results"][str(int(n))]["rbf"][f"{key}_std"] for n in n_values
        ])
        lin_mean = np.array([
            results["results"][str(int(n))]["linear"][f"{key}_mean"] for n in n_values
        ])
        lin_std = np.array([
            results["results"][str(int(n))]["linear"][f"{key}_std"] for n in n_values
        ])

        ax_left.plot(
            n_values, rbf_mean, color=rbf_color, linestyle="-",
            marker=style["marker"], markersize=6, linewidth=1.5,
            alpha=style["alpha_factor"],
            label=f"RBF  $k_{{{int(target*100)}}}$",
        )
        ax_left.fill_between(
            n_values, rbf_mean - rbf_std, rbf_mean + rbf_std,
            color=rbf_color, alpha=0.12 * style["alpha_factor"],
        )
        ax_left.plot(
            n_values, lin_mean, color=lin_color, linestyle="--",
            marker=style["marker"], markersize=6, linewidth=1.5,
            alpha=style["alpha_factor"],
            label=f"Linear  $k_{{{int(target*100)}}}$",
        )
        ax_left.fill_between(
            n_values, lin_mean - lin_std, lin_mean + lin_std,
            color=lin_color, alpha=0.12 * style["alpha_factor"],
        )

    ax_left.axhline(d, color=lin_color, linestyle=":", alpha=0.6, linewidth=1)
    ax_left.text(
        n_values[-1], d + 1,
        f"linear rank bound $= d = {d}$",
        ha="right", va="bottom", color=lin_color, fontsize=8,
    )
    ax_left.set_xscale("log")
    ax_left.set_xlabel("Sample size $n$")
    ax_left.set_ylabel("Smallest $k$ capturing fraction of $\\|\\hat\\delta\\|^2$")
    ax_left.set_title(
        f"Rich-DGP n-sweep: $k_c$ vs $n$  ($d={d}$, {n_seeds} seeds)"
    )
    ax_left.legend(fontsize=7, loc="upper left", ncol=2)
    ax_left.grid(True, which="both", alpha=0.3)

    for target in capture_targets:
        key = f"k_{int(target * 100)}"
        style = target_styles.get(target, dict(marker="o", alpha_factor=1.0))
        rbf_mean = np.array([
            results["results"][str(int(n))]["rbf"][f"{key}_mean"] for n in n_values
        ])
        lin_mean = np.array([
            results["results"][str(int(n))]["linear"][f"{key}_mean"] for n in n_values
        ])
        rbf_std = np.array([
            results["results"][str(int(n))]["rbf"][f"{key}_std"] for n in n_values
        ])
        lin_std = np.array([
            results["results"][str(int(n))]["linear"][f"{key}_std"] for n in n_values
        ])
        gap = rbf_mean - lin_mean
        gap_std = np.sqrt(rbf_std ** 2 + lin_std ** 2)  # independent additive
        ax_right.plot(
            n_values, gap, color="#2ca02c", linestyle="-",
            marker=style["marker"], markersize=6, linewidth=1.5,
            alpha=style["alpha_factor"],
            label=f"Gap at $k_{{{int(target*100)}}}$",
        )
        ax_right.fill_between(
            n_values, gap - gap_std, gap + gap_std,
            color="#2ca02c", alpha=0.10 * style["alpha_factor"],
        )

    ax_right.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax_right.set_xscale("log")
    ax_right.set_xlabel("Sample size $n$")
    ax_right.set_ylabel("$k_c(\\mathrm{RBF}) - k_c(\\mathrm{linear})$")
    ax_right.set_title("Finite-sample spread gap (RBF $-$ linear)")
    ax_right.legend(fontsize=8, loc="upper right")
    ax_right.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    paths = save_figure(fig, "fig1e_rich_transition")
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def write_exp1e_table(results: dict) -> None:
    """Emit a LaTeX table summarising Exp 1e rich-DGP n-sweep results."""
    n_values = results["n_values"]
    targets = results["capture_targets"]
    target_pcts = [int(t * 100) for t in targets]
    d = results["d"]

    header = (
        "$n$ & "
        + " & ".join(
            f"$k_{{{p}}}^{{\\mathrm{{RBF}}}}$" for p in target_pcts
        )
        + " & "
        + " & ".join(
            f"$k_{{{p}}}^{{\\mathrm{{lin}}}}$" for p in target_pcts
        )
        + " \\\\"
    )

    rows = []
    for n in n_values:
        rbf = results["results"][str(n)]["rbf"]
        lin = results["results"][str(n)]["linear"]
        rbf_cells = [
            f"${rbf[f'k_{p}_mean']:.1f} \\pm {rbf[f'k_{p}_std']:.1f}$"
            for p in target_pcts
        ]
        lin_cells = [
            f"${lin[f'k_{p}_mean']:.1f} \\pm {lin[f'k_{p}_std']:.1f}$"
            for p in target_pcts
        ]
        rows.append(
            f"{n:,} & " + " & ".join(rbf_cells) + " & "
            + " & ".join(lin_cells) + " \\\\"
        )

    n_cols = 1 + 2 * len(target_pcts)
    col_spec = "r" + "c" * (n_cols - 1)

    n_seeds = len(results.get("seeds", []))
    table = (
        "% Exp 1e: Rich-DGP n-sweep -- k_c vs n for RBF and linear kernels\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Rich-DGP n-sweep: smallest $k$ capturing "
        "$90/95/99\\%$ of $\\|\\hat\\delta\\|^2$ for RBF and linear "
        f"(rank $d{{=}}{d}$) kernels as sample size grows, mean $\\pm$ std "
        f"across {n_seeds} seeds. The DGP has a multi-directional population "
        f"$\\delta$ with mean shift spread across all $d = {d}$ features "
        f"via power-law decay "
        f"($\\mu_b[i] \\propto (i+1)^{{-{results['decay']}}}$, "
        f"$\\|\\mu_b - \\mu_a\\| = {results['total_shift']}$). At every "
        "$n$ the RBF's empirical $k_{99}$ substantially exceeds the linear "
        f"kernel's structural bound $k \\le d{{=}}{d}$, demonstrating that "
        "the characteristic kernel's finite-sample eigenbasis resolves "
        "more directions of $\\hat\\delta$ than the non-characteristic "
        "rank-$d$ kernel. The gap is non-monotone in $n$: it grows as the "
        "empirical RBF kernel starts resolving its richer eigenstructure "
        "(peak near $n{=}1000$), then shrinks as both kernels converge "
        "toward their (finite) population spectra for this specific DGP. "
        "The primary dimensional-transition signature for the paper "
        "remains the $k{=}d$ residual floor from Exp~1b "
        "(Fig.~\\ref{fig:finite-criteria}), where the linear kernel "
        "reaches $0$ exactly at $k{=}d$ and the RBF does not.}\n"
        "\\label{tab:rich-transition}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    out_path = RESULTS_DIR / "tables" / "exp1_rich_transition.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"  Saved: {out_path}")


# JSON serialization helper


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Main


def main():
    ensure_dirs()
    setup_style()

    print("=" * 60)
    print("Experiment 1: Synthetic Validation")
    print("=" * 60)

    t0 = time.time()
    print("\n[Exp 1a] Master constraint verification...")
    exp1a_results = run_exp1a(RANDOM_SEEDS, EXP1A_DELTA_PS)

    max_residual = max(r["residual"] for r in exp1a_results)
    print(f"  Max residual: {max_residual:.2e}")
    assert max_residual < 1e-10, (
        f"Master constraint residual too large: {max_residual:.2e} (expected < 1e-10)"
    )

    plot_exp1a_sweep(exp1a_results)
    plot_exp1a_bars(exp1a_results, BAR_CHART_DELTA_P)
    print(f"  Exp 1a done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("\n[Exp 1b] Finite-criteria impossibility...")
    exp1b_results = run_exp1b(RANDOM_SEEDS)

    greedy_at_3 = exp1b_results["greedy_mean"][3]
    greedy_at_10 = exp1b_results["greedy_mean"][10]
    greedy_std_3 = exp1b_results["greedy_std"][3]
    greedy_std_10 = exp1b_results["greedy_std"][10]
    random_at_3 = exp1b_results["random_mean"][3]
    trunc_floor = exp1b_results["truncation_floor_mean"]
    k99_mean = exp1b_results["k_99_mean"]
    k99_std = exp1b_results["k_99_std"]
    print(f"  Greedy residual at k=3:  {greedy_at_3:.4f} +/- {greedy_std_3:.4f} "
          f"(CV={greedy_std_3/max(greedy_at_3, 1e-12):.2f} -- unstable, do not cite)")
    print(f"  Greedy residual at k=10: {greedy_at_10:.4f} +/- {greedy_std_10:.4f} "
          f"(stable -- cite this as headline)")
    print(f"  Random residual at k=3:  {random_at_3:.4f}")
    print(f"  Effective witness rank k_99: {k99_mean:.1f} +/- {k99_std:.1f}")
    print(f"  Truncation floor (1 - capture): {trunc_floor:.2e}")

    plot_exp1b(exp1b_results)
    print(f"  Exp 1b done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("\n[Exp 1c] Bandwidth robustness (5 seeds x 5 bandwidths)...")
    exp1c_results = run_exp1c()

    for mult_str in sorted(exp1c_results.keys(), key=float):
        r = exp1c_results[mult_str]
        cap = 1.0 - r["truncation_floor_mean"]
        print(
            f"  sigma/sigma_med={mult_str:>5}: "
            f"k_90={r['k_90_mean']:6.1f}+/-{r['k_90_std']:4.1f}  "
            f"k_95={r['k_95_mean']:6.1f}+/-{r['k_95_std']:4.1f}  "
            f"k_99={r['k_99_mean']:6.1f}+/-{r['k_99_std']:4.1f}  "
            f"capture={cap:.4f}"
        )

    plot_exp1c(exp1c_results)
    write_exp1c_table(exp1c_results)
    print(f"  Exp 1c done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("\n[Exp 1d] Dimensional transition (n sweep)...")
    exp1d_results = run_exp1d_dimensional_transition(seed=DEFAULT_SEED)

    for n in EXP1D_N_VALUES:
        rbf = exp1d_results["results"][str(n)]["rbf"]
        lin = exp1d_results["results"][str(n)]["linear"]
        print(f"  n={n:>5}:  RBF k99={rbf['k_99']:>3}  "
              f"linear k99={lin['k_99']:>3}  "
              f"(n_valid: rbf={rbf['n_valid']}, lin={lin['n_valid']})")

    plot_exp1d_dimensional_transition(exp1d_results)
    write_exp1d_table(exp1d_results)
    print(f"  Exp 1d done in {time.time() - t0:.1f}s")

    if "linear_greedy_mean" in exp1b_results:
        lin_at_d = exp1b_results["linear_greedy_mean"][D_FEATURES]
        print(f"\n[verify] linear greedy residual at k=d={D_FEATURES}: "
              f"{lin_at_d:.2e} (expected ~0)")
        assert lin_at_d < 1e-6, (
            f"Linear kernel control should fully capture delta_hat at k=d, "
            f"got residual {lin_at_d:.2e}"
        )

    exp1e_results = None
    if EXP1E_RUN:
        t0 = time.time()
        print("\n[Exp 1e] Rich-DGP n-sweep (multi-directional delta)...")
        seeds_1e = EXP1E_SEEDS if EXP1E_SEEDS is not None else list(RANDOM_SEEDS)
        exp1e_results = run_exp1e_rich_dgp_n_sweep(seeds=seeds_1e)

        print(f"  DGP: d={EXP1E_D_FEATURES}, decay={EXP1E_DECAY}, "
              f"total_shift={EXP1E_TOTAL_SHIFT}, seeds={len(seeds_1e)}")
        for n in EXP1E_N_VALUES:
            r = exp1e_results["results"][str(n)]
            print(
                f"  n={n:>5}:  "
                f"k99(RBF)={r['rbf']['k_99_mean']:5.1f}+/-{r['rbf']['k_99_std']:4.1f}  "
                f"k99(lin)={r['linear']['k_99_mean']:4.1f}+/-{r['linear']['k_99_std']:4.1f}  "
                f"gap={r['rbf']['k_99_mean'] - r['linear']['k_99_mean']:+6.1f}"
            )

        plot_exp1e_rich_transition(exp1e_results)
        write_exp1e_table(exp1e_results)
        print(f"  Exp 1e done in {time.time() - t0:.1f}s")

        for n in EXP1E_N_VALUES:
            lin_k99 = exp1e_results["results"][str(n)]["linear"]["k_99_max"]
            assert lin_k99 <= EXP1E_D_FEATURES, (
                f"Linear k_99 exceeded d={EXP1E_D_FEATURES} at n={n}: {lin_k99}"
            )

    all_results = {
        "exp1a": exp1a_results,
        "exp1b": exp1b_results,
        "exp1c": exp1c_results,
        "exp1d": exp1d_results,
    }
    if exp1e_results is not None:
        all_results["exp1e"] = exp1e_results
    results_path = RESULTS_DIR / "exp1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("Experiment 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
