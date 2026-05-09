"""Experiment 4 -- Spectral Analysis of Cross-Covariance Operators.

Decomposes the group difference delta = mu_a - mu_b in the kernel PCA eigenbasis
on real fairness datasets. Sub-experiments: (a) spectral decay of delta,
(b) HSIC(S, G) before and after fairness interventions.
"""

import gc
import json
import os
import time
import warnings

# Prevent OMP threading crash with XGBoost + PyTorch (macOS only)
import sys
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from experiments.config import (
    BANDWIDTH_MULTIPLIERS,
    RANDOM_SEEDS,
    DEFAULT_SEED,
    RESULTS_DIR,
    ensure_dirs,
)
from experiments.data import FairnessDataset
from experiments.data.loader import (
    load_adult,
    load_compas,
    load_acs_pums,
    train_val_test_split,
)
from experiments.fairness.interventions import apply_method
from experiments.kernels import (
    hsic,
    hsic_test,
    median_heuristic,
    rbf_kernel_matrix,
)
from experiments.utils.plotting import setup_style, save_figure, THEORY_STYLE

# Parameters

DATASETS = ["adult", "compas", "acs_pums"]
KERNEL_SUBSAMPLE_N = 10_000
N_EIGENVALUES = 200
ACS_SUBSAMPLE_N = 20_000
HSIC_N_JOBS = os.cpu_count() or 1  # parallelize permutation test

# Classifiers for Analysis B
CLASSIFIERS = [
    "unconstrained_lr",
    "unconstrained_xgb",
    "hardt_postprocessing",
    "exponentiated_gradient_dp",
    "exponentiated_gradient_eo",
]

# Methods whose y_proba is a pass-through from the base LR, so score-level
# metrics are undefined. Skipped from HSIC computation.
NO_SCORE_METHODS = {"hardt_postprocessing"}
CLASSIFIER_LABELS = {
    "unconstrained_lr": "LR",
    "unconstrained_xgb": "XGBoost",
    "hardt_postprocessing": "Hardt (EO)",
    "exponentiated_gradient_dp": "EG (DP)",
    "exponentiated_gradient_eo": "EG (EO)",
}
CLASSIFIER_COLORS = {
    "unconstrained_lr": "black",
    "unconstrained_xgb": "#333333",
    "hardt_postprocessing": "#2ca02c",
    "exponentiated_gradient_dp": "#1f77b4",
    "exponentiated_gradient_eo": "#ff7f0e",
}

DATASET_LABELS = {
    "adult": "Adult Income",
    "compas": "COMPAS",
    "acs_pums": "ACS PUMS",
}

# Capture thresholds to report
CAPTURE_THRESHOLDS = [0.90, 0.95, 0.99]

# Key k values for table
TABLE_K_VALUES = [3, 10, 50, 100]


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
    """Load all three real datasets."""
    print("Loading datasets...")
    datasets = {}

    ds = load_adult()
    print(f"  Adult: n={len(ds.y)}, p_a={ds.base_rates['a']:.3f}, p_b={ds.base_rates['b']:.3f}")
    datasets["adult"] = ds

    ds = load_compas()
    print(f"  COMPAS: n={len(ds.y)}, p_a={ds.base_rates['a']:.3f}, p_b={ds.base_rates['b']:.3f}")
    datasets["compas"] = ds

    ds = load_acs_pums(subsample_n=ACS_SUBSAMPLE_N)
    print(f"  ACS PUMS: n={len(ds.y)}, p_a={ds.base_rates['a']:.3f}, p_b={ds.base_rates['b']:.3f}")
    datasets["acs_pums"] = ds

    return datasets


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
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


def _eigendecompose(K: np.ndarray, n_eig: int, center: bool = True):
    """Return (eigenvalues, eigenvectors) in descending order, thresholded > 1e-10.

    Double-centering is invariant for c with c^T 1 = 0 (group-difference vectors).
    """
    n = K.shape[0]
    if center:
        row_mean = K.mean(axis=0)
        K = K - row_mean[None, :] - row_mean[:, None] + row_mean.mean()
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

    return eigenvalues, eigenvectors


def _compute_spectral_energy(
    K: np.ndarray, c: np.ndarray, eigenvalues: np.ndarray,
    eigenvectors: np.ndarray, mmd2: float,
):
    """Project coefficient vector onto eigenbasis and compute spectral energy.

    Returns dict with greedy/kpca cumulative capture, d_coords, d_sq, etc.
    """
    n_valid = len(eigenvalues)

    # RKHS coordinates of delta_hat: d_i = sqrt(lambda_i) * (U_i^T c)
    projections = eigenvectors.T @ c
    d_coords = np.sqrt(eigenvalues) * projections
    d_sq = d_coords ** 2

    capture_frac = float(d_sq.sum() / mmd2)
    assert capture_frac > 0.90, (
        f"Eigendecomposition only captures {capture_frac:.4f} of MMD^2 "
        f"(need >0.90). Try increasing N_EIGENVALUES."
    )

    # Greedy ordering: sort by |d_i| descending
    greedy_order = np.argsort(-np.abs(d_coords))
    greedy_cumcapture = np.cumsum(d_sq[greedy_order]) / mmd2

    kpca_cumcapture = np.cumsum(d_sq) / mmd2

    # Find k needed for each threshold
    k_thresholds = {}
    for thresh in CAPTURE_THRESHOLDS:
        greedy_k = int(np.searchsorted(greedy_cumcapture, thresh)) + 1
        kpca_k = int(np.searchsorted(kpca_cumcapture, thresh)) + 1
        k_thresholds[str(thresh)] = {
            "greedy": min(greedy_k, n_valid),
            "kpca": min(kpca_k, n_valid),
        }

    # Capture at key k values
    capture_at_k = {}
    for k in TABLE_K_VALUES:
        if k <= n_valid:
            capture_at_k[str(k)] = {
                "greedy": float(greedy_cumcapture[k - 1]),
                "kpca": float(kpca_cumcapture[k - 1]),
            }

    return {
        "d_sq": d_sq.tolist(),
        "greedy_cumcapture": greedy_cumcapture.tolist(),
        "kpca_cumcapture": kpca_cumcapture.tolist(),
        "greedy_order": greedy_order.tolist(),
        "capture_frac": capture_frac,
        "k_thresholds": k_thresholds,
        "capture_at_k": capture_at_k,
        "n_valid": n_valid,
    }


# Analysis A: Spectral Decomposition


def run_spectral_analysis(
    datasets: dict[str, FairnessDataset], seeds: list[int]
) -> dict:
    """Compute spectral decomposition of group difference on real data.

    Returns {dataset: {per_seed: [...], aggregated: {...}}}.
    """
    results = {}
    total = len(datasets) * len(seeds)
    run_num = 0

    for ds_name, ds in datasets.items():
        per_seed = []
        for seed in seeds:
            run_num += 1
            print(f"  [{run_num}/{total}] Spectral: {ds_name}, seed={seed}")
            t0 = time.time()

            # Subsample for kernel computation
            ds_sub = _subsample_stratified(ds, KERNEL_SUBSAMPLE_N, seed)
            n = len(ds_sub.y)
            group = ds_sub.group
            y = ds_sub.y

            n_a = int((group == 1).sum())
            n_b = int((group == 0).sum())
            assert n_a > 0 and n_b > 0, f"Need both groups: n_a={n_a}, n_b={n_b}"

            # Compute bandwidth on subsample to avoid OOM
            if n > 5000:
                rng_bw = np.random.default_rng(seed + 999)
                bw_idx = rng_bw.choice(n, size=5000, replace=False)
                sigma = median_heuristic(ds_sub.X[bw_idx])
            else:
                sigma = median_heuristic(ds_sub.X)

            K = rbf_kernel_matrix(ds_sub.X, sigma=sigma)
            assert K.shape == (n, n), f"Expected ({n}, {n}), got {K.shape}"

            c = np.zeros(n, dtype=np.float64)
            c[group == 1] = 1.0 / n_a
            c[group == 0] = -1.0 / n_b

            # V-statistic (biased) MMD^2 for cross-experiment comparability with exp2
            mmd2 = float(c @ K @ c)
            assert mmd2 > 0, f"Biased MMD^2 should be positive, got {mmd2}"

            # Eigendecompose
            eigenvalues, eigenvectors = _eigendecompose(K, N_EIGENVALUES)

            # Project group difference onto eigenbasis
            spectral = _compute_spectral_energy(
                K, c, eigenvalues, eigenvectors, mmd2
            )

            seed_result = {
                "seed": seed,
                "n": n,
                "n_a": n_a,
                "n_b": n_b,
                "sigma": sigma,
                "mmd2": mmd2,
                "estimator": "biased_V_statistic",
                "eigenvalues": eigenvalues.tolist(),
                **spectral,
                "time_s": time.time() - t0,
            }

            per_seed.append(seed_result)
            print(f"    MMD^2={mmd2:.6f}, capture={spectral['capture_frac']:.4f}, "
                  f"k_90={spectral['k_thresholds']['0.9']['greedy']}, "
                  f"k_95={spectral['k_thresholds']['0.95']['greedy']}, "
                  f"k_99={spectral['k_thresholds']['0.99']['greedy']}, "
                  f"time={seed_result['time_s']:.1f}s")

            del K, ds_sub, eigenvalues, eigenvectors
            gc.collect()

        # When n <= KERNEL_SUBSAMPLE_N, std reflects bandwidth-subsample noise only
        used_full_dataset = bool(len(ds.y) <= KERNEL_SUBSAMPLE_N)
        ds_block = {
            "per_seed": per_seed,
            "estimator": "biased_V_statistic",
            "used_full_dataset": used_full_dataset,
        }
        if used_full_dataset:
            ds_block["std_caveat"] = (
                f"bandwidth subsample noise only — n={len(ds.y)} <= "
                f"KERNEL_SUBSAMPLE_N={KERNEL_SUBSAMPLE_N}"
            )
        results[ds_name] = ds_block

    return results


def run_conditional_spectra(
    datasets: dict[str, FairnessDataset], seeds: list[int]
) -> dict:
    """Compute spectral decomposition within Y=0 and Y=1 subgroups.

    Returns {dataset: {per_seed: [{y0: {...}, y1: {...}}, ...]}}.
    """
    results = {}
    total = len(datasets) * len(seeds)
    run_num = 0

    for ds_name, ds in datasets.items():
        per_seed = []
        for seed in seeds:
            run_num += 1
            print(f"  [{run_num}/{total}] Conditional spectra: {ds_name}, seed={seed}")

            ds_sub = _subsample_stratified(ds, KERNEL_SUBSAMPLE_N, seed)
            n = len(ds_sub.y)
            group = ds_sub.group
            y = ds_sub.y

            seed_result = {"seed": seed}

            for y_class in [0, 1]:
                label = f"y{y_class}"
                mask = y == y_class
                n_class = int(mask.sum())

                X_cls = ds_sub.X[mask]
                group_cls = group[mask]
                n_a_cls = int((group_cls == 1).sum())
                n_b_cls = int((group_cls == 0).sum())

                if n_a_cls == 0 or n_b_cls == 0:
                    warnings.warn(
                        f"{ds_name} seed={seed} Y={y_class}: empty subgroup "
                        f"(n_a={n_a_cls}, n_b={n_b_cls}), skipping"
                    )
                    seed_result[label] = None
                    continue

                if n_class > 5000:
                    rng_bw = np.random.default_rng(seed + 999 + y_class)
                    bw_idx = rng_bw.choice(n_class, size=5000, replace=False)
                    sigma = median_heuristic(X_cls[bw_idx])
                else:
                    sigma = median_heuristic(X_cls)

                n_eig_cls = min(N_EIGENVALUES, n_class)
                K_cls = rbf_kernel_matrix(X_cls, sigma=sigma)

                c_cls = np.zeros(n_class, dtype=np.float64)
                c_cls[group_cls == 1] = 1.0 / n_a_cls
                c_cls[group_cls == 0] = -1.0 / n_b_cls

                mmd2_cls = float(c_cls @ K_cls @ c_cls)

                if mmd2_cls <= 0:
                    warnings.warn(
                        f"{ds_name} seed={seed} Y={y_class}: mmd2={mmd2_cls}, skipping"
                    )
                    seed_result[label] = None
                    del K_cls
                    gc.collect()
                    continue

                eigenvalues, eigenvectors = _eigendecompose(K_cls, n_eig_cls)
                spectral = _compute_spectral_energy(
                    K_cls, c_cls, eigenvalues, eigenvectors, mmd2_cls
                )

                seed_result[label] = {
                    "n": n_class,
                    "n_a": n_a_cls,
                    "n_b": n_b_cls,
                    "sigma": sigma,
                    "mmd2": mmd2_cls,
                    **spectral,
                }

                del K_cls, eigenvalues, eigenvectors
                gc.collect()

            per_seed.append(seed_result)
            del ds_sub
            gc.collect()

        results[ds_name] = {"per_seed": per_seed}

    return results


def run_bandwidth_robustness(
    datasets: dict[str, FairnessDataset],
) -> dict:
    """Run spectral analysis at 0.5x, 1x, 2x median bandwidth on Adult (seed=42).

    Returns {multiplier: {spectral results}}.
    """
    ds = datasets.get("adult")
    if ds is None:
        return {}

    seed = DEFAULT_SEED
    results = {}

    ds_sub = _subsample_stratified(ds, KERNEL_SUBSAMPLE_N, seed)
    n = len(ds_sub.y)
    group = ds_sub.group

    n_a = int((group == 1).sum())
    n_b = int((group == 0).sum())

    if n > 5000:
        rng_bw = np.random.default_rng(seed + 999)
        bw_idx = rng_bw.choice(n, size=5000, replace=False)
        base_sigma = median_heuristic(ds_sub.X[bw_idx])
    else:
        base_sigma = median_heuristic(ds_sub.X)

    c = np.zeros(n, dtype=np.float64)
    c[group == 1] = 1.0 / n_a
    c[group == 0] = -1.0 / n_b

    for mult in BANDWIDTH_MULTIPLIERS:
        sigma = mult * base_sigma
        print(f"  Bandwidth robustness: mult={mult}, sigma={sigma:.4f}")

        K = rbf_kernel_matrix(ds_sub.X, sigma=sigma)
        mmd2 = float(c @ K @ c)
        assert mmd2 > 0, f"MMD^2 should be positive at mult={mult}, got {mmd2}"

        eigenvalues, eigenvectors = _eigendecompose(K, N_EIGENVALUES)
        spectral = _compute_spectral_energy(
            K, c, eigenvalues, eigenvectors, mmd2
        )

        results[str(mult)] = {
            "sigma": sigma,
            "mmd2": mmd2,
            **spectral,
        }

        del K, eigenvalues, eigenvectors
        gc.collect()

    del ds_sub
    gc.collect()
    return results


# Analysis B: HSIC Before/After Interventions


def _train_and_score(
    clf_name: str,
    train_ds: FairnessDataset,
    test_ds: FairnessDataset,
    seed: int,
) -> np.ndarray | None:
    """Train a classifier and return test probability scores, or None on failure."""
    try:
        if clf_name == "exponentiated_gradient_dp":
            _, y_proba = apply_method(
                "exponentiated_gradient",
                train_ds.X, train_ds.y, train_ds.group,
                test_ds.X, test_ds.group,
                seed=seed,
                constraint="demographic_parity",
                eps=0.01,
            )
        elif clf_name == "exponentiated_gradient_eo":
            _, y_proba = apply_method(
                "exponentiated_gradient",
                train_ds.X, train_ds.y, train_ds.group,
                test_ds.X, test_ds.group,
                seed=seed,
                constraint="equalized_odds",
                eps=0.01,
            )
        elif clf_name == "hardt_postprocessing":
            _, y_proba = apply_method(
                clf_name,
                train_ds.X, train_ds.y, train_ds.group,
                test_ds.X, test_ds.group,
                seed=seed,
            )
        else:
            _, y_proba = apply_method(
                clf_name,
                train_ds.X, train_ds.y, train_ds.group,
                test_ds.X, test_ds.group,
                seed=seed,
            )
        return y_proba
    except Exception as e:
        warnings.warn(f"{clf_name} failed: {e}")
        return None


def _fisher_combine(p_values: list[float]) -> float:
    """Fisher's method for combining k independent one-sided p-values.

    chi2 = -2 * sum(log p), df = 2k. Drops NaN entries before combining.
    Matches the implementation in exp2_residual_unfairness.py.
    """
    from scipy.stats import chi2

    ps = [p for p in p_values if p is not None and not np.isnan(p)]
    if len(ps) == 0:
        return float("nan")
    ps_arr = np.clip(np.asarray(ps, dtype=np.float64), 1e-12, 1.0)
    stat = -2.0 * float(np.sum(np.log(ps_arr)))
    return float(chi2.sf(stat, df=2 * len(ps_arr)))


def run_hsic_interventions(
    datasets: dict[str, FairnessDataset], seeds: list[int]
) -> dict:
    """Compute HSIC(S, G) for each classifier before and after interventions.

    Uses a single shared LR-reference bandwidth per dataset so cross-classifier
    magnitudes are comparable. Permutation p-values are Fisher-combined across seeds.
    """
    results = {}
    total = len(datasets) * len(seeds)
    run_num = 0

    for ds_name, ds in datasets.items():
        ref_train, _, ref_test = train_val_test_split(ds, seed=DEFAULT_SEED)
        ref_scores = _train_and_score(
            "unconstrained_lr", ref_train, ref_test, DEFAULT_SEED
        )
        assert ref_scores is not None, (
            f"LR reference scores failed on {ds_name} — cannot compute "
            "sigma_x_shared. Investigate _train_and_score before retrying."
        )
        sigma_x_shared = float(median_heuristic(ref_scores.reshape(-1, 1)))
        assert sigma_x_shared > 0, (
            f"sigma_x_shared = {sigma_x_shared} on {ds_name}; LR scores "
            "should never produce a zero median heuristic."
        )
        print(f"  [{ds_name}] sigma_x_shared = {sigma_x_shared:.6f} "
              f"(LR-reference, seed={DEFAULT_SEED})")

        per_seed = []
        # Collect per-seed p-values for Fisher combination
        per_seed_pvals: dict[str, list[float]] = {
            c: [] for c in CLASSIFIERS if c not in NO_SCORE_METHODS
        }

        for seed in seeds:
            run_num += 1
            print(f"  [{run_num}/{total}] HSIC: {ds_name}, seed={seed}")

            train_ds, _val_ds, test_ds = train_val_test_split(ds, seed=seed)
            group_test = test_ds.group

            seed_result = {"seed": seed, "n_test": len(test_ds.y)}
            seed_perm_p = {}

            for clf_name in CLASSIFIERS:
                if clf_name in NO_SCORE_METHODS:
                    seed_result[clf_name] = None
                    continue

                scores = _train_and_score(clf_name, train_ds, test_ds, seed)
                if scores is None:
                    seed_result[clf_name] = None
                    continue

                S = scores.reshape(-1, 1)
                G = group_test.astype(np.float64).reshape(-1, 1)

                n_unique = len(np.unique(scores))
                if n_unique <= 1:
                    warnings.warn(f"{clf_name}: constant scores, skipping HSIC")
                    seed_result[clf_name] = None
                    continue

                hsic_val = hsic(
                    S, G, sigma_x=sigma_x_shared, sigma_y=1.0,
                )
                seed_result[clf_name] = float(hsic_val)

                hsic_obs, p_val = hsic_test(
                    S, G, sigma_x=sigma_x_shared, sigma_y=1.0, seed=seed,
                    n_jobs=HSIC_N_JOBS,
                )
                seed_perm_p[clf_name] = float(p_val)
                per_seed_pvals[clf_name].append(float(p_val))
                print(f"    {clf_name}: HSIC={hsic_val:.6f}, p={p_val:.4f}")

            seed_result["perm_p"] = seed_perm_p
            per_seed.append(seed_result)
            gc.collect()

        # Combine per-seed p-values via Fisher's method
        perm_results = {}
        for clf_name, pvals in per_seed_pvals.items():
            if not pvals:
                continue
            p_fisher = _fisher_combine(pvals)
            p_floor = 1.0 / (1 + 999)  # n_perm default
            saturated = all(p <= 1.5 * p_floor for p in pvals)
            perm_results[clf_name] = {
                "p_fisher": float(p_fisher),
                "per_seed_p": pvals,
                "saturated": saturated,
            }
            print(f"  [{ds_name}] {clf_name}: p_fisher={p_fisher:.6f}, "
                  f"per_seed_p={[f'{p:.4f}' for p in pvals]}"
                  f"{' (saturated)' if saturated else ''}")

        results[ds_name] = {
            "per_seed": per_seed,
            "perm_test": perm_results,
            "sigma_x_shared": sigma_x_shared,
        }

    return results


# Aggregation


def aggregate_results(
    spectral: dict, conditional: dict, hsic_results: dict
) -> dict:
    """Aggregate across seeds for all analyses.

    Returns dict with mean/std for spectral capture and HSIC values.
    """
    agg = {}

    # Spectral aggregation
    for ds_name, ds_data in spectral.items():
        per_seed = ds_data["per_seed"]
        n_seeds = len(per_seed)

        # Stack greedy cumulative capture arrays (varying length -- use min)
        min_len = min(len(r["greedy_cumcapture"]) for r in per_seed)
        greedy_all = np.array([r["greedy_cumcapture"][:min_len] for r in per_seed])
        kpca_all = np.array([r["kpca_cumcapture"][:min_len] for r in per_seed])

        # Per-component normalized energy in greedy order, for fig4a log-log panel
        per_comp_all = []
        for r in per_seed:
            d_sq = np.array(r["d_sq"])
            order = np.array(r["greedy_order"])
            mmd2_r = float(r["mmd2"])
            per_comp_all.append((d_sq[order][:min_len] / mmd2_r))
        per_comp_all = np.array(per_comp_all)

        # Capture at k=3 (greedy)
        capture_at_3 = [
            r["capture_at_k"].get("3", {}).get("greedy", float("nan"))
            for r in per_seed
        ]

        # k thresholds
        k_thresholds_agg = {}
        for thresh_str in ["0.9", "0.95", "0.99"]:
            greedy_ks = [
                r["k_thresholds"].get(thresh_str, {}).get("greedy", float("nan"))
                for r in per_seed
            ]
            k_thresholds_agg[thresh_str] = {
                "greedy_mean": float(np.mean(greedy_ks)),
                "greedy_std": float(np.std(greedy_ks)),
            }

        # Infer used_full_dataset for legacy cached results
        ns = [r.get("n") for r in per_seed if r.get("n") is not None]
        full_threshold = int(0.95 * KERNEL_SUBSAMPLE_N)
        inferred_full = (
            bool(ns) and max(ns) < full_threshold and min(ns) == max(ns)
        )
        used_full = ds_data.get("used_full_dataset", inferred_full)
        std_caveat = ds_data.get("std_caveat")
        if used_full and std_caveat is None:
            std_caveat = (
                f"bandwidth subsample noise only — n={ns[0] if ns else 'unknown'} "
                f"<= KERNEL_SUBSAMPLE_N={KERNEL_SUBSAMPLE_N}"
            )

        agg_ds = {
            "greedy_cumcapture_mean": greedy_all.mean(axis=0).tolist(),
            "greedy_cumcapture_std": greedy_all.std(axis=0).tolist(),
            "kpca_cumcapture_mean": kpca_all.mean(axis=0).tolist(),
            "kpca_cumcapture_std": kpca_all.std(axis=0).tolist(),
            "per_comp_mean": per_comp_all.mean(axis=0).tolist(),
            "per_comp_std": per_comp_all.std(axis=0).tolist(),
            "mmd2_mean": float(np.mean([r["mmd2"] for r in per_seed])),
            "mmd2_std": float(np.std([r["mmd2"] for r in per_seed])),
            "capture_at_3_mean": float(np.mean(capture_at_3)),
            "capture_at_3_std": float(np.std(capture_at_3)),
            "k_thresholds": k_thresholds_agg,
            "n_seeds": n_seeds,
            "min_len": min_len,
            "estimator": ds_data.get("estimator", "biased_V_statistic"),
            "used_full_dataset": used_full,
            "std_caveat": std_caveat,
        }

        # Capture at all table k values
        for k_str in [str(k) for k in TABLE_K_VALUES]:
            vals = [
                r["capture_at_k"].get(k_str, {}).get("greedy")
                for r in per_seed
                if k_str in r.get("capture_at_k", {})
            ]
            if vals:
                agg_ds[f"capture_at_{k_str}_mean"] = float(np.mean(vals))
                agg_ds[f"capture_at_{k_str}_std"] = float(np.std(vals))
            else:
                agg_ds[f"capture_at_{k_str}_mean"] = float("nan")
                agg_ds[f"capture_at_{k_str}_std"] = float("nan")

        agg[ds_name] = agg_ds

    # HSIC aggregation
    for ds_name, ds_data in hsic_results.items():
        per_seed = ds_data["per_seed"]
        if ds_name not in agg:
            agg[ds_name] = {}

        hsic_agg = {}
        for clf_name in CLASSIFIERS:
            vals = [
                r.get(clf_name) for r in per_seed
                if r.get(clf_name) is not None
            ]
            if vals:
                vals_arr = np.array(vals)
                mean_val = float(np.mean(vals_arr))
                std_val = float(np.std(vals_arr))
                cv = float(std_val / mean_val) if mean_val > 0 else float("nan")
                hsic_agg[clf_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "cv": cv,
                    "per_seed_values": [float(v) for v in vals],
                    "n": len(vals),
                }

        agg[ds_name]["hsic"] = hsic_agg
        agg[ds_name]["hsic_perm_test"] = ds_data.get("perm_test", {})
        if "sigma_x_shared" in ds_data:
            agg[ds_name]["sigma_x_shared"] = ds_data["sigma_x_shared"]

    return agg


# Plotting


def plot_spectral_decay(
    spectral: dict, aggregated: dict,
    dataset_names: list[str], fig_name: str,
) -> None:
    """Plot per-component energy on log-log axes with cumulative capture inset."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    n_panels = len(dataset_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, dataset_names):
        agg = aggregated[ds_name]

        min_len = agg["min_len"]
        ks = np.arange(1, min_len + 1)

        per_comp_m = np.array(agg["per_comp_mean"])
        per_comp_s = np.array(agg["per_comp_std"])

        # Avoid log(0): clip to a small positive floor for plotting only
        eps = 1e-12
        per_comp_m_clip = np.maximum(per_comp_m, eps)

        ax.plot(ks, per_comp_m_clip, color="#d62728", linewidth=1.5,
                marker="o", markersize=3, label="Greedy (best alignment)")
        ax.fill_between(
            ks,
            np.maximum(per_comp_m - per_comp_s, eps),
            per_comp_m + per_comp_s,
            color="#d62728", alpha=0.15,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, min_len)
        ax.set_ylim(eps, 2.0)

        ds_label = DATASET_LABELS.get(ds_name, ds_name)
        if agg.get("used_full_dataset"):
            ds_label = f"{ds_label} (bw noise)"
        mmd2_str = f"MMD$^2$={agg['mmd2_mean']:.4f}"
        k99_val = agg.get("k_thresholds", {}).get("0.99", {}).get("greedy_mean", float("nan"))
        cap3_val = agg.get("capture_at_3_mean", float("nan"))
        subtitle = f"{mmd2_str}, top-3 = {cap3_val:.0%}, $k_{{99}}$ = {k99_val:.0f}"
        ax.set_xlabel("Component index $k$ (log)")
        if ds_name == dataset_names[0]:
            ax.set_ylabel("Per-component energy $|c_k|^2 / \\|\\delta\\|^2$ (log)")
        ax.set_title(f"{ds_label}\n({subtitle})", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        inset = inset_axes(
            ax, width="42%", height="38%", loc="upper right", borderpad=1.2,
        )

        greedy_m = np.array(agg["greedy_cumcapture_mean"])
        greedy_s = np.array(agg["greedy_cumcapture_std"])
        kpca_m = np.array(agg["kpca_cumcapture_mean"])

        inset_max = min(50, min_len)
        ks_inset = ks[:inset_max]

        inset.plot(ks_inset, greedy_m[:inset_max], color="#d62728", linewidth=1.2)
        inset.fill_between(
            ks_inset,
            greedy_m[:inset_max] - greedy_s[:inset_max],
            np.minimum(greedy_m[:inset_max] + greedy_s[:inset_max], 1.0),
            color="#d62728", alpha=0.15,
        )
        inset.plot(
            ks_inset, kpca_m[:inset_max], color="#1f77b4",
            linewidth=1.0, linestyle="--",
        )

        for thresh in CAPTURE_THRESHOLDS:
            inset.axhline(
                thresh, linestyle=":", color="gray", linewidth=0.6, alpha=0.6,
            )

        capture_3 = agg["capture_at_3_mean"]
        inset.axvline(3, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
        inset.annotate(
            f"$k=3$\n{capture_3:.0%}",
            xy=(3, capture_3),
            xytext=(8, max(capture_3 - 0.25, 0.05)),
            fontsize=6,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
            color="gray",
        )

        inset.set_xlim(1, inset_max)
        inset.set_ylim(0.0, 1.02)
        inset.set_xlabel("$k$", fontsize=6, labelpad=1)
        inset.set_ylabel("cum.", fontsize=6, labelpad=1)
        inset.tick_params(labelsize=5, pad=1)
        inset.set_title("cumulative", fontsize=6, pad=2)

    fig.suptitle(
        "Spectral Decay of Group Difference $\\delta = \\mu_a - \\mu_b$",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    paths = save_figure(fig, fig_name)
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def plot_hsic_bars(
    aggregated: dict, dataset_names: list[str], fig_name: str,
) -> None:
    """Plot grouped bar chart of HSIC(S, G) per classifier per dataset."""
    n_panels = len(dataset_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.0))
    if n_panels == 1:
        axes = [axes]

    clfs_to_plot = [c for c in CLASSIFIERS if c not in NO_SCORE_METHODS]
    n_clfs = len(clfs_to_plot)
    x = np.arange(n_clfs)
    bar_width = 0.6

    for ax, ds_name in zip(axes, dataset_names):
        hsic_data = aggregated[ds_name].get("hsic", {})
        perm_data = aggregated[ds_name].get("hsic_perm_test", {})
        sigma_shared = aggregated[ds_name].get("sigma_x_shared")

        means = []
        stds = []
        colors = []
        labels = []

        for clf_name in clfs_to_plot:
            clf_data = hsic_data.get(clf_name, {})
            means.append(clf_data.get("mean", 0.0))
            stds.append(clf_data.get("std", 0.0))
            colors.append(CLASSIFIER_COLORS.get(clf_name, "gray"))
            labels.append(CLASSIFIER_LABELS.get(clf_name, clf_name))

        ax.bar(x, means, bar_width, yerr=stds, capsize=3,
               color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

        for i, clf_name in enumerate(clfs_to_plot):
            perm = perm_data.get(clf_name, {})
            p_val = perm.get("p_fisher", perm.get("p_value"))
            if p_val is not None and p_val < 0.05:
                ax.text(i, means[i] + stds[i] + 0.0002, "*",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("HSIC$(S, G)$" if ds_name == dataset_names[0] else "")
        title = DATASET_LABELS.get(ds_name, ds_name)
        if sigma_shared is not None:
            title = f"{title}\n($\\sigma_x={sigma_shared:.3f}$, LR-ref)"
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        "Score-Group Dependence Before/After Fairness Interventions\n"
        "(Hardt omitted: pass-through scores from base LR)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    paths = save_figure(fig, fig_name)
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


# Table generation


def _generate_tables(aggregated: dict) -> None:
    """Generate LaTeX table with spectral capture and HSIC values."""
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ds_labels = {"adult": "Adult", "compas": "COMPAS", "acs_pums": "ACS PUMS"}

    # Detect any dataset with the full-dataset caveat for the footnote
    has_full_dataset_caveat = any(
        aggregated.get(ds, {}).get("used_full_dataset")
        for ds in ["adult", "compas", "acs_pums"]
    )

    lines = [
        "% Exp 4: Spectral analysis of group difference",
        "% Generated by exp4_spectral_analysis.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Spectral capture of the group difference "
        "$\\delta = \\mu_a - \\mu_b$ in the kernel-PCA basis and score-group "
        "dependence under a shared LR-reference HSIC bandwidth (mean $\\pm$ "
        "std, 5 seeds). $k_{\\theta}$ = components needed to reach $\\theta$ "
        "fraction of MMD$^2$ (greedy ordering by alignment with $\\delta$). "
        "HSIC p-values are Fisher-combined across all 5 seeds (999 perms/seed). "
        "MMD$^2$ reported using the V-statistic (biased) estimator, "
        "consistent with Exp 2c."
        + (" $\\dagger$~Dataset uses the full $n$ across all 5 seeds "
           "(below KERNEL\\_SUBSAMPLE\\_N$=10\\,000$); reported values have "
           "zero resampling variability (std reflects bandwidth-subsample "
           "noise only)."
           if has_full_dataset_caveat else "")
        + "}",
        "\\label{tab:spectral-analysis}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Dataset & MMD$^2$ & $k_{90}$ & $k_{95}$ & $k_{99}$ "
        "& Cap.@$k{=}3$ & HSIC(LR) & HSIC(EG-DP) \\\\",
        "\\midrule",
    ]

    for ds_name in ["adult", "compas", "acs_pums"]:
        if ds_name not in aggregated:
            continue
        agg = aggregated[ds_name]
        label = ds_labels.get(ds_name, ds_name)
        if agg.get("used_full_dataset"):
            label = label + "$^{\\dagger}$"

        mmd2_m = agg.get("mmd2_mean", float("nan"))
        cap3_m = agg.get("capture_at_3_mean", float("nan"))
        cap3_s = agg.get("capture_at_3_std", float("nan"))

        k90 = agg.get("k_thresholds", {}).get("0.9", {}).get("greedy_mean", float("nan"))
        k95 = agg.get("k_thresholds", {}).get("0.95", {}).get("greedy_mean", float("nan"))
        k99 = agg.get("k_thresholds", {}).get("0.99", {}).get("greedy_mean", float("nan"))

        hsic_lr = agg.get("hsic", {}).get("unconstrained_lr", {})
        hsic_dp = agg.get("hsic", {}).get("exponentiated_gradient_dp", {})

        is_full = agg.get("used_full_dataset", False)

        def _fmt_pm(mean_val: float, std_val: float) -> str:
            if is_full and std_val < 0.0005:
                return f"${mean_val:.3f}$\\textsuperscript{{$\\dagger$}}"
            return f"${mean_val:.3f} \\pm {std_val:.3f}$"

        def _fmt_hsic(data: dict) -> str:
            if not data:
                return "---"
            m, s = data.get("mean", 0), data.get("std", 0)
            if is_full and s < 0.00005:
                return f"${m:.4f}$\\textsuperscript{{$\\dagger$}}"
            return f"${m:.4f} \\pm {s:.4f}$"

        cap3_str = _fmt_pm(cap3_m, cap3_s)
        lr_str = _fmt_hsic(hsic_lr)
        dp_str = _fmt_hsic(hsic_dp)

        lines.append(
            f"{label} & ${mmd2_m:.4f}$ & ${k90:.1f}$ & ${k95:.1f}$ & ${k99:.1f}$ "
            f"& {cap3_str} "
            f"& {lr_str} & {dp_str} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    path = tables_dir / "exp4_spectral.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Table saved to {path}")


# Main


def main():
    ensure_dirs()
    setup_style()
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 4: Spectral Analysis of Cross-Covariance Operators")
    print("=" * 60)

    results_path = RESULTS_DIR / "exp4_results.json"
    partial = {}
    if results_path.exists():
        with open(results_path) as f:
            partial = json.load(f)
        print(f"  Found partial results: {list(partial.keys())}")

    datasets = _load_datasets()

    # --- Analysis A: Spectral decomposition ---
    if "spectral" in partial:
        spectral = partial["spectral"]
        print(f"\n[Analysis A] Using cached spectral results.")
    else:
        t0 = time.time()
        print("\n[Analysis A] Spectral decomposition of group difference...")
        spectral = run_spectral_analysis(datasets, RANDOM_SEEDS)
        print(f"  Spectral analysis done in {time.time() - t0:.1f}s")
        _save_partial(results_path, {"spectral": spectral})

    # --- Conditional spectra ---
    if "conditional" in partial:
        conditional = partial["conditional"]
        print(f"\n[Conditional] Using cached conditional spectra.")
    else:
        t0 = time.time()
        print("\n[Conditional] Within-class spectral decomposition...")
        conditional = run_conditional_spectra(datasets, RANDOM_SEEDS)
        print(f"  Conditional spectra done in {time.time() - t0:.1f}s")
        _save_partial(results_path, {"spectral": spectral, "conditional": conditional})

    # --- Bandwidth robustness ---
    if "bandwidth_robustness" in partial:
        bandwidth_robust = partial["bandwidth_robustness"]
        print(f"\n[Bandwidth] Using cached bandwidth robustness results.")
    else:
        t0 = time.time()
        print("\n[Bandwidth] Robustness check at 0.5x/1x/2x median bandwidth...")
        bandwidth_robust = run_bandwidth_robustness(datasets)
        print(f"  Bandwidth robustness done in {time.time() - t0:.1f}s")
        _save_partial(results_path, {
            "spectral": spectral, "conditional": conditional,
            "bandwidth_robustness": bandwidth_robust,
        })

    # --- Analysis B: HSIC before/after ---
    if "hsic" in partial:
        hsic_results = partial["hsic"]
        print(f"\n[Analysis B] Using cached HSIC results.")
    else:
        t0 = time.time()
        print("\n[Analysis B] HSIC(S, G) before/after interventions...")
        hsic_results = run_hsic_interventions(datasets, RANDOM_SEEDS)
        print(f"  HSIC analysis done in {time.time() - t0:.1f}s")
        _save_partial(results_path, {
            "spectral": spectral, "conditional": conditional,
            "bandwidth_robustness": bandwidth_robust, "hsic": hsic_results,
        })

    # --- Aggregate ---
    print("\n[Aggregate] Computing cross-seed statistics...")
    aggregated = aggregate_results(spectral, conditional, hsic_results)

    # --- Print summary ---
    print("\n--- Summary ---")
    for ds_name in DATASETS:
        if ds_name not in aggregated:
            continue
        agg = aggregated[ds_name]
        print(f"  {DATASET_LABELS.get(ds_name, ds_name)}:")
        print(f"    MMD^2 = {agg.get('mmd2_mean', 0):.6f} +/- {agg.get('mmd2_std', 0):.6f}")
        print(f"    Capture@k=3 (greedy) = {agg.get('capture_at_3_mean', 0):.4f} +/- {agg.get('capture_at_3_std', 0):.4f}")
        for thresh_str in ["0.9", "0.95", "0.99"]:
            kdata = agg.get("k_thresholds", {}).get(thresh_str, {})
            print(f"    k_{thresh_str} (greedy) = {kdata.get('greedy_mean', 0):.1f} +/- {kdata.get('greedy_std', 0):.1f}")
        hsic_data = agg.get("hsic", {})
        for clf_name in ["unconstrained_lr", "exponentiated_gradient_dp"]:
            cd = hsic_data.get(clf_name, {})
            if cd:
                print(f"    HSIC({CLASSIFIER_LABELS.get(clf_name, clf_name)}) = {cd['mean']:.6f} +/- {cd['std']:.6f}")

    # --- Plot ---
    print("\n[Plot] Generating figures...")
    main_ds = [ds for ds in ["adult", "compas"] if ds in aggregated]
    if main_ds:
        plot_spectral_decay(spectral, aggregated, main_ds, "fig4a_spectral_decay")
    if "acs_pums" in aggregated:
        plot_spectral_decay(spectral, aggregated, ["acs_pums"], "fig4a_spectral_decay_acs")

    main_ds_hsic = [ds for ds in ["adult", "compas"] if ds in aggregated and "hsic" in aggregated[ds]]
    if main_ds_hsic:
        plot_hsic_bars(aggregated, main_ds_hsic, "fig4b_intervention_hsic")

    # --- Save final results ---
    all_results = {
        "spectral": spectral,
        "conditional": conditional,
        "bandwidth_robustness": bandwidth_robust,
        "hsic": hsic_results,
        "aggregated": aggregated,
    }
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {results_path}")

    # --- Tables ---
    _generate_tables(aggregated)

    print("\n" + "=" * 60)
    print("Experiment 4 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
