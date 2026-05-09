"""Experiment 2 -- Residual Unfairness on Real Data.

Sub-experiments 2a (master constraint on real scores), 2b (residual unfairness
heatmap with significance testing), and 2c (RKHS projection analysis).
"""

import gc
import json
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from experiments.config import (
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
from experiments.fairness.criteria import (
    demographic_parity_gap,
    calibration_error,
    master_constraint_residual,
)
from experiments.fairness.interventions import apply_method
from experiments.kernels import rbf_kernel_matrix, median_heuristic
from experiments.utils.plotting import setup_style, save_figure, GROUP_COLORS
from experiments.utils.stats import bh_correction

# Parameters

DATASETS = ["adult", "compas", "acs_pums"]
ACS_SUBSAMPLE_N = 20_000

METHODS = [
    "unconstrained_lr",
    "unconstrained_xgb",
    "hardt_postprocessing",
    "platt_scaling",
    "reweighting",
    "exponentiated_gradient",
]
METHOD_LABELS = {
    "unconstrained_lr": "LR (unconstrained)",
    "unconstrained_xgb": "XGBoost (unconstrained)",
    "hardt_postprocessing": "Hardt post-proc. (EO)",
    "platt_scaling": "Platt scaling (Cal.)",
    "reweighting": "Reweighting (DP)",
    "exponentiated_gradient": "Exp. Gradient (EO)",
}

CRITERIA = ["accuracy", "dp_gap", "tpr_gap", "fpr_gap", "cal_error"]
CRITERIA_LABELS = {
    "accuracy": "Accuracy",
    "dp_gap": "DP Gap",
    "tpr_gap": "TPR Gap",
    "fpr_gap": "FPR Gap",
    "cal_error": "Cal. Error",
}

DECISION_CRITERIA = ["accuracy", "dp_gap", "tpr_gap", "fpr_gap"]
SCORE_CRITERIA = ["cal_error"]

# Only decision-level gap metrics are testable under the group-label permutation null.
SIG_TEST_CRITERIA = {"dp_gap", "tpr_gap", "fpr_gap"}

# Diagonal cells: method explicitly targets this criterion, so skip sig. test.
METHOD_TARGETS = {
    "unconstrained_lr":       {"accuracy"},
    "unconstrained_xgb":      {"accuracy"},
    "hardt_postprocessing":   {"fpr_gap", "tpr_gap"},   # equalized odds
    "platt_scaling":          {"cal_error"},
    "reweighting":            {"dp_gap"},
    "exponentiated_gradient": {"fpr_gap", "tpr_gap"},   # EO constraint by default
}

# Decision-only methods whose cal_error cells are masked to NaN.
# Hardt's ThresholdOptimizer returns the base LR's predict_proba unchanged.
NO_SCORE_METHODS = {"hardt_postprocessing"}

N_PERMUTATIONS = 999
BH_ALPHA = 0.05

KERNEL_SUBSAMPLE_N = 10_000

_TERM_COLORS = {
    "LHS": "black",
    "Term 1": "#1f77b4",
    "Term 2": "#ff7f0e",
    "Term 3": "#2ca02c",
}


# Helpers


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


def _is_binary_proba(y_proba: np.ndarray) -> bool:
    """Detect the fairlearn `_pmf_predict` fallback where y_proba is effectively {0.0, 1.0}."""
    if y_proba is None or len(y_proba) == 0:
        return True
    uniq = np.unique(y_proba)
    return len(uniq) <= 2 and bool(np.all(np.isin(uniq, [0.0, 1.0])))


def _compute_criteria(
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    group: np.ndarray,
    method_name: str = "",
) -> dict[str, float]:
    """Compute all 5 fairness criteria for a single method run."""
    for g in [0, 1]:
        for y_val in [0, 1]:
            count = ((group == g) & (y_true == y_val)).sum()
            assert count > 0, (
                f"Subgroup (group={g}, y={y_val}) is empty -- need at least 1"
            )

    acc = float((y_pred == y_true).mean())
    dp = demographic_parity_gap(y_pred, group)

    tpr_a = float(y_pred[(group == 1) & (y_true == 1)].mean())
    tpr_b = float(y_pred[(group == 0) & (y_true == 1)].mean())
    fpr_a = float(y_pred[(group == 1) & (y_true == 0)].mean())
    fpr_b = float(y_pred[(group == 0) & (y_true == 0)].mean())
    tpr_gap = float(abs(tpr_a - tpr_b))
    fpr_gap = float(abs(fpr_a - fpr_b))

    if method_name in NO_SCORE_METHODS or _is_binary_proba(y_pred_proba):
        cal_avg = float("nan")
    else:
        cal = calibration_error(y_pred_proba, y_true, group)
        cal_avg = float((cal["a"] + cal["b"]) / 2)

    return {
        "accuracy": acc,
        "dp_gap": dp,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "cal_error": cal_avg,
    }


# Significance testing helpers (Exp 2b)


def _cell_statistic(
    crit: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    group: np.ndarray,
) -> float:
    """Compute a single decision-level gap criterion for the permutation loop."""
    if crit == "dp_gap":
        m_a = y_pred[group == 1].mean()
        m_b = y_pred[group == 0].mean()
        return float(abs(m_a - m_b))
    if crit == "tpr_gap":
        m_a = y_pred[(group == 1) & (y_true == 1)].mean()
        m_b = y_pred[(group == 0) & (y_true == 1)].mean()
        return float(abs(m_a - m_b))
    if crit == "fpr_gap":
        m_a = y_pred[(group == 1) & (y_true == 0)].mean()
        m_b = y_pred[(group == 0) & (y_true == 0)].mean()
        return float(abs(m_a - m_b))
    raise ValueError(
        f"Unsupported criterion for permutation test: {crit}. "
        f"Supported: dp_gap, tpr_gap, fpr_gap."
    )


def _permutation_pvalue(
    crit: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    group: np.ndarray,
    n_perm: int = N_PERMUTATIONS,
    seed: int = 42,
) -> float:
    """One-sided permutation p-value under the null that group indep (y_pred, y_true)."""
    obs = _cell_statistic(crit, y_pred, y_true, group)
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        g_perm = rng.permutation(group)
        perm_stats[i] = _cell_statistic(crit, y_pred, y_true, g_perm)
    return float((1 + np.sum(perm_stats >= obs)) / (1 + n_perm))


def _fisher_combine(p_values: list[float]) -> float:
    """Fisher's method: chi2 = -2 * sum(log p), df = 2k. Drops NaN entries."""
    from scipy.stats import chi2

    ps = [p for p in p_values if p is not None and not np.isnan(p)]
    if len(ps) == 0:
        return float("nan")
    ps_arr = np.clip(np.asarray(ps, dtype=np.float64), 1e-12, 1.0)
    stat = -2.0 * float(np.sum(np.log(ps_arr)))
    return float(chi2.sf(stat, df=2 * len(ps_arr)))


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


# Exp 2a: Master Constraint on Real Data


def run_exp2a(
    datasets: dict[str, FairnessDataset], seeds: list[int]
) -> list[dict]:
    """Verify the projected master constraint on real classifier scores."""
    results = []
    for ds_name, ds in datasets.items():
        for seed in seeds:
            print(f"  Exp 2a: dataset={ds_name}, seed={seed}")
            train_ds, _val_ds, test_ds = train_val_test_split(ds, seed=seed)

            y_pred, y_pred_proba = apply_method(
                "unconstrained_lr",
                train_ds.X, train_ds.y, train_ds.group,
                test_ds.X, test_ds.group,
                seed=seed,
            )

            residual = master_constraint_residual(
                y_pred_proba, test_ds.y, test_ds.group
            )

            scores = y_pred_proba
            group = test_ds.group
            y = test_ds.y

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

            accuracy = float((y_pred == test_ds.y).mean())

            assert residual < 1e-10, (
                f"Master constraint residual too large on {ds_name} seed={seed}: "
                f"{residual:.2e}"
            )

            results.append({
                "dataset": ds_name,
                "seed": seed,
                "lhs": lhs,
                "term1": term1,
                "term2": term2,
                "term3": term3,
                "rhs": rhs,
                "residual": residual,
                "p_a": p_a,
                "p_b": p_b,
                "accuracy": accuracy,
            })

    return results


# Exp 2b: Residual Unfairness Heatmap


def run_exp2b(
    datasets: dict[str, FairnessDataset], seeds: list[int]
) -> dict:
    """Train all methods on each dataset x seed, compute criteria and significance tests.

    Returns nested dict keyed by dataset -> method -> criterion -> {mean, std, values}.
    """
    results = {}
    for ds_name, ds in datasets.items():
        results[ds_name] = {}
        for method_name in METHODS:
            all_criteria = {c: [] for c in CRITERIA}
            per_seed_preds: dict[int, tuple] = {}

            for seed in seeds:
                print(f"  Exp 2b: dataset={ds_name}, seed={seed}, method={method_name}")
                train_ds, _val_ds, test_ds = train_val_test_split(ds, seed=seed)

                try:
                    y_pred, y_proba = apply_method(
                        method_name,
                        train_ds.X, train_ds.y, train_ds.group,
                        test_ds.X, test_ds.group,
                        seed=seed,
                    )
                except Exception as e:
                    warnings.warn(
                        f"Method {method_name} failed on {ds_name} seed={seed}: {e}"
                    )
                    continue

                crit = _compute_criteria(
                    y_pred, y_proba, test_ds.y, test_ds.group,
                    method_name=method_name,
                )
                for c in CRITERIA:
                    all_criteria[c].append(crit[c])
                per_seed_preds[seed] = (
                    np.asarray(y_pred),
                    np.asarray(y_proba) if y_proba is not None else None,
                    np.asarray(test_ds.y),
                    np.asarray(test_ds.group),
                )

            method_result = {}
            for c in CRITERIA:
                vals = all_criteria[c]
                if len(vals) > 0:
                    vals_arr = np.asarray(vals, dtype=np.float64)
                    if np.all(np.isnan(vals_arr)):
                        method_result[c] = {
                            "mean": float("nan"),
                            "std": float("nan"),
                            "values": vals,
                        }
                    else:
                        method_result[c] = {
                            "mean": float(np.nanmean(vals_arr)),
                            "std": float(np.nanstd(vals_arr)),
                            "values": vals,
                        }
                else:
                    method_result[c] = {
                        "mean": float("nan"),
                        "std": float("nan"),
                        "values": [],
                    }

            sig_results = {}
            targets = METHOD_TARGETS.get(method_name, set())
            for c in CRITERIA:
                if c not in SIG_TEST_CRITERIA:
                    sig_results[c] = {
                        "p_fisher": None,
                        "per_seed_p": None,
                        "is_target": c in targets,
                        "q_value": None,
                        "significant": None,
                    }
                    continue
                if c in targets:
                    sig_results[c] = {
                        "p_fisher": None,
                        "per_seed_p": None,
                        "is_target": True,
                        "q_value": None,
                        "significant": None,
                    }
                    continue
                if np.isnan(method_result[c]["mean"]):
                    sig_results[c] = {
                        "p_fisher": None,
                        "per_seed_p": None,
                        "is_target": False,
                        "q_value": None,
                        "significant": None,
                    }
                    continue
                per_seed_p = []
                for seed in seeds:
                    if seed not in per_seed_preds:
                        continue
                    yp, _, yt, g = per_seed_preds[seed]
                    p = _permutation_pvalue(c, yp, yt, g, seed=seed)
                    per_seed_p.append(p)
                p_fisher = _fisher_combine(per_seed_p)
                # Flag when all per-seed p-values hit the permutation floor
                p_floor = 1.0 / (1 + N_PERMUTATIONS)
                saturated = all(p <= 1.5 * p_floor for p in per_seed_p)
                sig_results[c] = {
                    "p_fisher": p_fisher,
                    "per_seed_p": per_seed_p,
                    "is_target": False,
                    "q_value": None,  # filled after BH
                    "significant": None,  # filled after BH
                    "saturated": saturated,
                }

            method_result["_significance"] = sig_results
            results[ds_name][method_name] = method_result

        # BH correction per dataset over non-target, non-NaN p-values
        flat = []  # (method, criterion, p)
        for m in METHODS:
            sig_map = results[ds_name][m].get("_significance", {})
            for c in CRITERIA:
                sig = sig_map.get(c)
                if sig is None:
                    continue
                if sig["is_target"]:
                    continue
                p = sig["p_fisher"]
                if p is None or np.isnan(p):
                    continue
                flat.append((m, c, p))
        if flat:
            pvals = np.array([f[2] for f in flat], dtype=np.float64)
            rejected, corrected = bh_correction(pvals, alpha=BH_ALPHA)
            for (m, c, _), rej, q in zip(flat, rejected, corrected):
                results[ds_name][m]["_significance"][c]["q_value"] = float(q)
                results[ds_name][m]["_significance"][c]["significant"] = bool(rej)
            n_sig = int(np.sum(rejected))
            n_sat = sum(
                1 for m, c, _ in flat
                if results[ds_name][m]["_significance"][c].get("saturated")
            )
            sat_note = (
                f" ({n_sat} at permutation floor, report q<0.001)"
                if n_sat > 0 else ""
            )
            BASELINE_METHODS = {"unconstrained_lr", "unconstrained_xgb"}
            interv_cells = [
                (m, c, r) for (m, c, _), r in zip(flat, rejected)
                if m not in BASELINE_METHODS
            ]
            base_cells = [
                (m, c, r) for (m, c, _), r in zip(flat, rejected)
                if m in BASELINE_METHODS
            ]
            n_interv_sig = sum(1 for _, _, r in interv_cells if r)
            n_base_sig = sum(1 for _, _, r in base_cells if r)
            print(
                f"  Exp 2b [{ds_name}]: {n_sig}/{len(flat)} off-diagonal cells "
                f"significant at BH q<{BH_ALPHA}{sat_note}"
            )
            print(
                f"    Intervention-leak: {n_interv_sig}/{len(interv_cells)}, "
                f"baseline gaps: {n_base_sig}/{len(base_cells)}"
            )

    return results


# Exp 2c: Addressed vs Residual Unfairness


def run_exp2c(
    datasets: dict[str, FairnessDataset], seeds: list[int]
) -> dict:
    """Project delta = mu_a - mu_b onto separation directions and report residual fraction.

    Uses the unbiased U-statistic estimator (kernel diagonal zeroed).
    """
    results = {}
    for ds_name, ds in datasets.items():
        per_seed = []
        for seed in seeds:
            print(f"  Exp 2c: dataset={ds_name}, seed={seed}")

            ds_sub = _subsample_stratified(ds, KERNEL_SUBSAMPLE_N, seed)
            n = len(ds_sub.y)
            group = ds_sub.group
            y = ds_sub.y

            n_a = int((group == 1).sum())
            n_b = int((group == 0).sum())
            n_1a = int(((group == 1) & (y == 1)).sum())
            n_1b = int(((group == 0) & (y == 1)).sum())
            n_0a = int(((group == 1) & (y == 0)).sum())
            n_0b = int(((group == 0) & (y == 0)).sum())

            assert n_a > 0 and n_b > 0, f"Need both groups: n_a={n_a}, n_b={n_b}"
            assert n_1a > 0 and n_1b > 0, f"Need Y=1 in both groups: n_1a={n_1a}, n_1b={n_1b}"
            assert n_0a > 0 and n_0b > 0, f"Need Y=0 in both groups: n_0a={n_0a}, n_0b={n_0b}"

            # Bandwidth on a subsample to avoid OOM (10K x 10K upper triangle)
            if n > 5000:
                rng_bw = np.random.default_rng(seed + 999)
                bw_idx = rng_bw.choice(n, size=5000, replace=False)
                sigma = median_heuristic(ds_sub.X[bw_idx])
            else:
                sigma = median_heuristic(ds_sub.X)
            K = rbf_kernel_matrix(ds_sub.X, sigma=sigma)
            assert K.shape == (n, n), f"Expected ({n}, {n}), got {K.shape}"

            # delta: group difference mu_a - mu_b
            c = np.zeros(n, dtype=np.float64)
            c[group == 1] = 1.0 / n_a
            c[group == 0] = -1.0 / n_b

            # delta_1: conditional on Y=1
            c1 = np.zeros(n, dtype=np.float64)
            c1[(group == 1) & (y == 1)] = 1.0 / n_1a
            c1[(group == 0) & (y == 1)] = -1.0 / n_1b

            # delta_0: conditional on Y=0
            c0 = np.zeros(n, dtype=np.float64)
            c0[(group == 1) & (y == 0)] = 1.0 / n_0a
            c0[(group == 0) & (y == 0)] = -1.0 / n_0b

            delta_sq_biased = float(c @ K @ c)

            # Zero kernel diagonal to remove O(1/n) V-statistic bias
            np.fill_diagonal(K, 0.0)

            delta_sq = float(c @ K @ c)
            d1_sq = float(c1 @ K @ c1)
            d0_sq = float(c0 @ K @ c0)
            d_d1 = float(c @ K @ c1)
            d_d0 = float(c @ K @ c0)
            d1_d0 = float(c1 @ K @ c0)

            assert delta_sq > 0, (
                f"delta_sq (unbiased) must be positive (groups must differ), "
                f"got {delta_sq}"
            )

            # 2x2 Gram matrix of {delta_1, delta_0}
            G = np.array([[d1_sq, d1_d0], [d1_d0, d0_sq]])
            b = np.array([d_d1, d_d0])

            # Projection: ||proj||^2 = b^T G^{-1} b
            proj_sq = float(b @ np.linalg.solve(G, b))
            residual_frac = 1.0 - proj_sq / delta_sq

            residual_frac = float(np.clip(residual_frac, 0.0, 1.0))

            diag_bias = float(np.sum(c ** 2))  # = 1/n_a + 1/n_b

            per_seed.append({
                "seed": seed,
                "n": n,
                "delta_sq": delta_sq,
                "delta_sq_biased": delta_sq_biased,
                "diag_bias": diag_bias,
                "proj_sq": proj_sq,
                "residual_frac": residual_frac,
                "estimator": "unbiased_U_statistic",
                "sigma": sigma,
            })

            del K, ds_sub
            gc.collect()

        fracs = [r["residual_frac"] for r in per_seed]
        # When n <= KERNEL_SUBSAMPLE_N, all seeds use identical data;
        # std reflects only bandwidth subsample noise, not sampling variability.
        used_full = bool(len(ds.y) <= KERNEL_SUBSAMPLE_N)
        results[ds_name] = {
            "residual_frac_mean": float(np.mean(fracs)),
            "residual_frac_std": float(np.std(fracs)),
            "used_full_dataset": used_full,
            "std_caveat": (
                "Full dataset used across all seeds (n <= KERNEL_SUBSAMPLE_N); "
                "per-seed std reflects kernel-bandwidth subsample noise, "
                "not sampling variability."
                if used_full else None
            ),
            "per_seed": per_seed,
        }

    return results


# Plotting


def plot_exp2a(results: list[dict]) -> None:
    """Bar chart showing master constraint decomposition per dataset."""
    dataset_names = sorted(set(r["dataset"] for r in results))

    fig, axes = plt.subplots(1, len(dataset_names), figsize=(14, 4))
    if len(dataset_names) == 1:
        axes = [axes]

    ds_labels = {"adult": "Adult Income", "compas": "COMPAS", "acs_pums": "ACS PUMS"}

    for ax, ds_name in zip(axes, dataset_names):
        subset = [r for r in results if r["dataset"] == ds_name]
        assert len(subset) > 0, f"No results for {ds_name}"

        lhs_vals = [r["lhs"] for r in subset]
        t1_vals = [r["term1"] for r in subset]
        t2_vals = [r["term2"] for r in subset]
        t3_vals = [r["term3"] for r in subset]
        res_vals = [r["residual"] for r in subset]
        p_a = np.mean([r["p_a"] for r in subset])
        p_b = np.mean([r["p_b"] for r in subset])

        means = [np.mean(lhs_vals), np.mean(t1_vals), np.mean(t2_vals), np.mean(t3_vals)]
        stds = [np.std(lhs_vals), np.std(t1_vals), np.std(t2_vals), np.std(t3_vals)]
        labels = ["LHS", "Term 1", "Term 2", "Term 3"]
        colors = [_TERM_COLORS[l] for l in labels]

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors,
               alpha=0.85, edgecolor="black", linewidth=0.5)

        rhs_sum = np.mean(t1_vals) + np.mean(t2_vals) + np.mean(t3_vals)
        ax.axhline(rhs_sum, linestyle="--", color="gray", linewidth=1.5)

        mean_res = np.mean(res_vals)
        ax.text(
            0.98, 0.95, f"|LHS $-$ RHS| = {mean_res:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Score difference" if ds_name == dataset_names[0] else "")
        ax.set_title(
            f"{ds_labels.get(ds_name, ds_name)}\n"
            f"($p_a={p_a:.2f}$, $p_b={p_b:.2f}$)",
            fontsize=10,
        )

    fig.suptitle("Master Constraint Decomposition on Real Data", fontsize=12, y=1.02)
    fig.tight_layout()
    paths = save_figure(fig, "fig2a_master_constraint_real")
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def plot_exp2b(results: dict) -> None:
    """Heatmap figure: Adult + COMPAS (main), ACS (appendix)."""
    main_datasets = [ds for ds in ["adult", "compas"] if ds in results]
    if main_datasets:
        _plot_heatmap(results, main_datasets, "fig2b_residual_heatmap")

    if "acs_pums" in results:
        _plot_heatmap(results, ["acs_pums"], "fig2b_residual_heatmap_acs")


def _plot_heatmap(results: dict, dataset_names: list[str], fig_name: str) -> None:
    """Draw a split heatmap (decision-level | score-level) with significance stars."""
    n_datasets = len(dataset_names)
    fig_width = 6.0 * n_datasets if n_datasets <= 2 else 6.0
    fig, axes = plt.subplots(1, n_datasets, figsize=(fig_width, 5.0))
    if n_datasets == 1:
        axes = [axes]

    ds_labels = {"adult": "Adult Income", "compas": "COMPAS", "acs_pums": "ACS PUMS"}
    n_decision = len(DECISION_CRITERIA)

    for ax, ds_name in zip(axes, dataset_names):
        ds_results = results[ds_name]
        n_methods = len(METHODS)
        n_criteria = len(CRITERIA)

        matrix = np.full((n_methods, n_criteria), np.nan, dtype=np.float64)
        annotations = [[""] * n_criteria for _ in range(n_methods)]

        for i, method in enumerate(METHODS):
            for j, crit in enumerate(CRITERIA):
                m = ds_results[method][crit]["mean"]
                s = ds_results[method][crit]["std"]
                matrix[i, j] = m
                if np.isnan(m):
                    annotations[i][j] = "N/A"
                else:
                    ann = f"{m:.3f}\n({s:.3f})"
                    sig = ds_results[method].get("_significance", {}).get(crit, {})
                    if sig and sig.get("significant") is True:
                        ann += " *"
                    annotations[i][j] = ann

        # Normalize per column; invert accuracy so high = good = green
        norm_matrix = np.full_like(matrix, np.nan)
        for j in range(n_criteria):
            col = matrix[:, j]
            valid = ~np.isnan(col)
            if valid.sum() == 0:
                continue
            col_min = col[valid].min()
            col_max = col[valid].max()
            if col_max - col_min < 1e-12:
                norm_matrix[valid, j] = 0.5
            else:
                norm_matrix[valid, j] = (col[valid] - col_min) / (col_max - col_min)

            if CRITERIA[j] == "accuracy":
                norm_matrix[valid, j] = 1.0 - norm_matrix[valid, j]

        cmap = plt.cm.RdYlGn_r.copy()
        cmap.set_bad(color="#dddddd")
        masked = np.ma.masked_invalid(norm_matrix)
        im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        for i in range(n_methods):
            for j in range(n_criteria):
                if np.isnan(matrix[i, j]):
                    color = "#555555"
                else:
                    nv = norm_matrix[i, j]
                    color = "white" if nv > 0.7 or nv < 0.3 else "black"
                ax.text(
                    j, i, annotations[i][j],
                    ha="center", va="center",
                    fontsize=7, color=color,
                )

        ax.set_xticks(range(n_criteria))
        ax.set_xticklabels(
            [CRITERIA_LABELS[c] for c in CRITERIA],
            fontsize=8, rotation=30, ha="right",
        )
        ax.set_yticks(range(n_methods))
        ax.set_yticklabels(
            [METHOD_LABELS[m] for m in METHODS] if ds_name == dataset_names[0] else [],
            fontsize=8,
        )
        ax.set_title(ds_labels.get(ds_name, ds_name), fontsize=11, pad=18)

        ax.axvline(x=n_decision - 0.5, color="black", linewidth=1.8)

        dec_center = (n_decision - 1) / 2
        sco_center = n_decision + (len(SCORE_CRITERIA) - 1) / 2
        y_label = -0.9
        ax.text(
            dec_center, y_label, "Decision-level",
            ha="center", va="bottom", fontsize=9, fontstyle="italic",
        )
        ax.text(
            sco_center, y_label, "Score-level",
            ha="center", va="bottom", fontsize=9, fontstyle="italic",
        )
        ax.set_ylim(n_methods - 0.5, -1.2)

    footnote = (
        "N/A: method does not produce calibrated score-level output "
        "(Hardt: decisions only; EG/Reweighting: binary _pmf_predict fallback).   "
        "* : significant at q<0.05 (BH-corrected per dataset over off-diagonal "
        "cells; Fisher-combined per-seed permutation p-values, 999 perms/seed)."
    )
    fig.text(
        0.5, 0.005, footnote,
        ha="center", va="bottom", fontsize=7, color="#333333",
        wrap=True,
    )

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    paths = save_figure(fig, fig_name)
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def plot_exp2c(results: dict) -> None:
    """Stacked bar chart: addressed vs residual unfairness per dataset."""
    ds_labels = {"adult": "Adult Income", "compas": "COMPAS", "acs_pums": "ACS PUMS"}
    dataset_names = [ds for ds in DATASETS if ds in results]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(dataset_names))
    bar_width = 0.5

    addressed = []
    residual = []
    for ds_name in dataset_names:
        r = results[ds_name]
        res_frac = r["residual_frac_mean"]
        addr_frac = 1.0 - res_frac
        addressed.append(addr_frac)
        residual.append(res_frac)

    bars_addr = ax.bar(x, addressed, bar_width, label="Addressed", color="#1f77b4", alpha=0.85)
    bars_res = ax.bar(x, residual, bar_width, bottom=addressed, label="Residual", color="#d62728", alpha=0.85)

    for i, ds_name in enumerate(dataset_names):
        r = results[ds_name]
        addr_pct = (1.0 - r["residual_frac_mean"]) * 100
        res_pct = r["residual_frac_mean"] * 100
        res_std = r["residual_frac_std"] * 100

        ax.text(i, addressed[i] / 2, f"{addr_pct:.1f}%",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        # For tiny residuals, annotate above the bar instead of inside
        std_label = "bw noise" if r.get("used_full_dataset") else f"{res_std:.1f}%"
        res_text = f"{res_pct:.1f}%\n({std_label})"
        if residual[i] > 0.02:
            ax.text(i, addressed[i] + residual[i] / 2, res_text,
                    ha="center", va="center", fontsize=8, color="white")
        else:
            ax.annotate(
                res_text, xy=(i, 1.0), xytext=(i, 1.04),
                ha="center", va="bottom", fontsize=7, color="#d62728",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([ds_labels.get(ds, ds) for ds in dataset_names], fontsize=10)
    ax.set_ylabel("Fraction of $\\|\\delta\\|^2$")
    ax.set_title("Unfairness Addressed by Separation Directions ($\\delta_1, \\delta_0$)")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    paths = save_figure(fig, "fig2c_addressed_vs_residual")
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


# LaTeX tables


def _generate_tables(all_results: dict) -> None:
    """Generate LaTeX tables for the paper appendix."""
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _table_master_constraint(all_results["exp2a"], tables_dir)
    _table_residual_heatmap(all_results["exp2b"], tables_dir)
    _table_addressed_residual(all_results["exp2c"], tables_dir)


def _table_master_constraint(results: list[dict], tables_dir) -> None:
    """Master constraint verification table."""
    ds_labels = {"adult": "Adult", "compas": "COMPAS", "acs_pums": "ACS PUMS"}
    dataset_names = sorted(set(r["dataset"] for r in results))

    lines = [
        "% Exp 2a: Master constraint on real data",
        "% Generated by exp2_residual_unfairness.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Master constraint verification on real datasets (5 seeds, LR classifier).}",
        "\\label{tab:master-constraint-real}",
        "\\begin{tabular}{lcccccr}",
        "\\toprule",
        "Dataset & Acc. & LHS & Term~1 & Term~2 & Term~3 & $|\\text{LHS} - \\text{RHS}|$ \\\\",
        "\\midrule",
    ]

    for ds_name in dataset_names:
        subset = [r for r in results if r["dataset"] == ds_name]
        acc = np.mean([r["accuracy"] for r in subset])
        lhs = np.mean([r["lhs"] for r in subset])
        t1 = np.mean([r["term1"] for r in subset])
        t2 = np.mean([r["term2"] for r in subset])
        t3 = np.mean([r["term3"] for r in subset])
        res = np.mean([r["residual"] for r in subset])
        label = ds_labels.get(ds_name, ds_name)
        lines.append(
            f"{label} & ${acc:.3f}$ & ${lhs:+.3f}$ & ${t1:+.3f}$ & "
            f"${t2:+.3f}$ & ${t3:+.3f}$ & ${res:.1e}$ \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path = tables_dir / "exp2_master_constraint.tex"
    path.write_text("\n".join(lines))
    print(f"  Table saved: {path}")


def _table_residual_heatmap(results: dict, tables_dir) -> None:
    """Full numerical heatmap table with N/A masking and significance stars."""
    lines = [
        "% Exp 2b: Residual unfairness heatmap",
        "% Generated by exp2_residual_unfairness.py",
    ]

    ds_labels = {"adult": "Adult", "compas": "COMPAS", "acs_pums": "ACS PUMS"}
    caption_note = (
        "N/A: method does not produce a calibrated score-level output (Hardt "
        "post-processing inherits the base LR's probability; EG and Reweighting "
        "fall back to binary outputs when fairlearn's \\texttt{\\_pmf\\_predict} "
        "is unavailable). "
        "Cells marked with $^*$ are significant at $q<0.05$ after BH correction "
        "per dataset across all off-diagonal cells (Fisher-combined per-seed "
        "permutation p-values, $999$ perms/seed, shuffling test-set group labels)."
    )

    for ds_name in results:
        ds_results = results[ds_name]
        label = ds_labels.get(ds_name, ds_name)
        lines += [
            "",
            f"% --- {label} ---",
            "\\begin{table}[t]",
            "\\centering",
            (
                f"\\caption{{Fairness criteria by method --- {label} "
                f"(mean $\\pm$ std, 5 seeds). {caption_note}}}"
            ),
            f"\\label{{tab:heatmap-{ds_name}}}",
            "\\begin{tabular}{l|cccc|c}",
            "\\toprule",
            (
                "& \\multicolumn{4}{c|}{\\emph{Decision-level}} "
                "& \\emph{Score-level} \\\\"
            ),
            "\\cmidrule(lr){2-5} \\cmidrule(lr){6-6}",
            "Method & Acc. & DP Gap & TPR Gap & FPR Gap & Cal. Err. \\\\",
            "\\midrule",
        ]

        sig_map = {m: ds_results[m].get("_significance", {}) for m in METHODS}
        for method in METHODS:
            row_label = METHOD_LABELS[method]
            cells = []
            for crit in CRITERIA:
                m_val = ds_results[method][crit]["mean"]
                s_val = ds_results[method][crit]["std"]
                if np.isnan(m_val):
                    cells.append("N/A")
                    continue
                sig = sig_map[method].get(crit, {})
                star = "^{*}" if sig.get("significant") is True else ""
                cells.append(f"${m_val:.3f} \\pm {s_val:.3f}{star}$")
            lines.append(f"{row_label} & " + " & ".join(cells) + " \\\\")

        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    path = tables_dir / "exp2_residual_heatmap.tex"
    path.write_text("\n".join(lines))
    print(f"  Table saved: {path}")


def _table_addressed_residual(results: dict, tables_dir) -> None:
    """Addressed vs residual fractions table."""
    ds_labels = {"adult": "Adult", "compas": "COMPAS", "acs_pums": "ACS PUMS"}

    any_full = any(results[ds].get("used_full_dataset") for ds in results)

    lines = [
        "% Exp 2c: Addressed vs residual unfairness",
        "% Generated by exp2_residual_unfairness.py",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Fraction of group disparity $\\|\\delta\\|^2$ addressed by "
            "separation directions $\\delta_1, \\delta_0$ (equalized odds). "
            "Unbiased U-statistic estimator (kernel diagonal zeroed). "
            "RBF kernel with median heuristic, 5 seeds."
            + (
                " $^{\\dagger}$ Full dataset used across all seeds "
                "($n \\le 10{,}000$); reported std reflects kernel-bandwidth "
                "subsample noise, not sampling variability."
                if any_full else ""
            )
            + "}"
        ),
        "\\label{tab:addressed-residual}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dataset & Addressed (\\%) & Residual (\\%) & $\\|\\delta\\|^2$ \\\\",
        "\\midrule",
    ]

    for ds_name in results:
        r = results[ds_name]
        label = ds_labels.get(ds_name, ds_name)
        if r.get("used_full_dataset"):
            label = label + "$^{\\dagger}$"
        addr = (1.0 - r["residual_frac_mean"]) * 100
        addr_std = r["residual_frac_std"] * 100
        res = r["residual_frac_mean"] * 100
        delta_sq_mean = np.mean([s["delta_sq"] for s in r["per_seed"]])
        lines.append(
            f"{label} & ${addr:.1f} \\pm {addr_std:.1f}$ & "
            f"${res:.1f} \\pm {addr_std:.1f}$ & ${delta_sq_mean:.4f}$ \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path = tables_dir / "exp2_addressed_residual.tex"
    path.write_text("\n".join(lines))
    print(f"  Table saved: {path}")


# JSON serialization helper


def _save_partial(path, data: dict) -> None:
    """Save partial results so completed sub-experiments survive crashes."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Partial results saved to {path}")


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
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 2: Residual Unfairness on Real Data")
    print("=" * 60)

    results_path = RESULTS_DIR / "exp2_results.json"
    partial = {}
    if results_path.exists():
        with open(results_path) as f:
            partial = json.load(f)
        print(f"  Found partial results: {list(partial.keys())}")

    datasets = _load_datasets()

    if "exp2a" in partial:
        exp2a_results = partial["exp2a"]
        print("\n[Exp 2a] Using cached results.")
    else:
        t0 = time.time()
        print("\n[Exp 2a] Master constraint on real data...")
        exp2a_results = run_exp2a(datasets, RANDOM_SEEDS)
        max_res = max(r["residual"] for r in exp2a_results)
        print(f"  Max residual: {max_res:.2e}")
        assert max_res < 1e-10, f"Master constraint residual too large: {max_res:.2e}"
        plot_exp2a(exp2a_results)
        print(f"  Exp 2a done in {time.time() - t0:.1f}s")
        _save_partial(results_path, {"exp2a": exp2a_results})

    if "exp2b" in partial:
        exp2b_results = partial["exp2b"]
        print("\n[Exp 2b] Using cached results.")
    else:
        t0 = time.time()
        print("\n[Exp 2b] Residual unfairness heatmap...")
        exp2b_results = run_exp2b(datasets, RANDOM_SEEDS)
        plot_exp2b(exp2b_results)
        print(f"  Exp 2b done in {time.time() - t0:.1f}s")
        _save_partial(results_path, {"exp2a": exp2a_results, "exp2b": exp2b_results})

    if "exp2c" in partial:
        exp2c_results = partial["exp2c"]
        print("\n[Exp 2c] Using cached results.")
    else:
        gc.collect()
        t0 = time.time()
        print("\n[Exp 2c] Addressed vs residual unfairness...")
        exp2c_results = run_exp2c(datasets, RANDOM_SEEDS)
        for ds_name, r in exp2c_results.items():
            print(f"  {ds_name}: residual = {r['residual_frac_mean']:.1%} +/- {r['residual_frac_std']:.1%}")
        plot_exp2c(exp2c_results)
        print(f"  Exp 2c done in {time.time() - t0:.1f}s")

    all_results = {"exp2a": exp2a_results, "exp2b": exp2b_results, "exp2c": exp2c_results}
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {results_path}")

    _generate_tables(all_results)

    print("\n" + "=" * 60)
    print("Experiment 2 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
