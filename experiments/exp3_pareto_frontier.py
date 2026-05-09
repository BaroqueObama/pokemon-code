"""Experiment 3 -- Fairness-Accuracy Pareto Frontier.

Sweeps EG constraint strength across DP and EO, plots methods on a
(fairness violation, error) plane with the theoretical bound overlay.
"""

import gc
import json
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from experiments.config import (
    RANDOM_SEEDS,
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
)
from experiments.fairness.interventions import apply_method
from experiments.utils.plotting import setup_style, save_figure, THEORY_STYLE

DATASETS = ["adult", "compas", "acs_pums"]
ACS_SUBSAMPLE_N = 20_000

EPS_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]

CONSTRAINTS = ["demographic_parity", "equalized_odds"]

# Decision-only methods whose y_proba must not enter calibration metrics.
NO_SCORE_METHODS = {"hardt_postprocessing"}

# Methods that explicitly enforce separation (<w, delta_y> = 0) as a constraint.
# EG is included only when its constraint is equalized_odds (see _is_separation_enforcing).
SEPARATION_ENFORCING_METHODS = {
    "hardt_postprocessing",
}

# Near-separation EO tolerance; tau = 0.02 is a strict sensitivity check.
EO_SEPARATION_TOLERANCE = 0.05
EO_TOLERANCE_STRICT = 0.02

BOOTSTRAP_B = 1000

BASELINES = [
    "unconstrained_lr",
    "unconstrained_xgb",
    "hardt_postprocessing",
    "platt_scaling",
    "reweighting",
]

CONSTRAINT_LABELS = {
    "demographic_parity": "EG (DP)",
    "equalized_odds": "EG (EO)",
}
BASELINE_LABELS = {
    "unconstrained_lr": "LR",
    "unconstrained_xgb": "XGBoost",
    "hardt_postprocessing": "Hardt (EO)",
    "platt_scaling": "Platt",
    "reweighting": "K\\&C Reweight (DP)",
}
DATASET_LABELS = {
    "adult": "Adult Income",
    "compas": "COMPAS",
    "acs_pums": "ACS PUMS",
}

SWEEP_COLORS = {
    "demographic_parity": "#1f77b4",
    "equalized_odds": "#ff7f0e",
}
BASELINE_MARKERS = {
    "unconstrained_lr": ("o", "black"),
    "unconstrained_xgb": ("^", "black"),
    "hardt_postprocessing": ("D", "#2ca02c"),
    "platt_scaling": ("s", "#9467bd"),
    "reweighting": ("P", "#d62728"),
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
    """Return True if y_proba is effectively {0.0, 1.0} (fairlearn _pmf_predict fallback)."""
    if y_proba is None or len(y_proba) == 0:
        return True
    uniq = np.unique(y_proba)
    return len(uniq) <= 2 and bool(np.all(np.isin(uniq, [0.0, 1.0])))


def _bootstrap_error_ci(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    B: int = BOOTSTRAP_B,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (lo, hi) percentile bootstrap CI on test-set error rate."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n = len(y_true)
    assert len(y_pred) == n, (
        f"bootstrap_error_ci length mismatch: {len(y_pred)} vs {n}"
    )
    wrong = (y_pred != y_true).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    errs = wrong[idx].mean(axis=1)
    lo = float(np.percentile(errs, 100.0 * alpha / 2))
    hi = float(np.percentile(errs, 100.0 * (1.0 - alpha / 2)))
    return lo, hi


def _compute_criteria(
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    group: np.ndarray,
    method_name: str,
) -> dict[str, float]:
    """Compute accuracy, DP gap, TPR/FPR gaps, and calibration error.

    Masks cal_error to NaN for decision-only methods or binary-proba fallback.
    """
    for g in [0, 1]:
        for y_val in [0, 1]:
            count = ((group == g) & (y_true == y_val)).sum()
            assert count > 0, (
                f"Subgroup (group={g}, y={y_val}) is empty — need at least 1"
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


# Core sweep

def run_sweep(
    datasets: dict[str, FairnessDataset], seeds: list[int]
) -> list[dict]:
    """Sweep all methods and constraint strengths.

    Returns list of dicts, one per (dataset, seed, method, constraint, eps) combo.
    Each dict has keys: dataset, seed, method, constraint, eps,
                        accuracy, error, dp_gap, eo_gap, tpr_gap, fpr_gap, cal_error.
    """
    results = []
    total_fits = (len(BASELINES) + len(CONSTRAINTS) * len(EPS_SWEEP)) * len(datasets) * len(seeds)
    fit_num = 0

    for ds_name, ds in datasets.items():
        for seed in seeds:
            train_ds, _val_ds, test_ds = train_val_test_split(ds, seed=seed)

            for method_name in BASELINES:
                fit_num += 1
                print(f"  [{fit_num}/{total_fits}] {ds_name} seed={seed} {method_name}")
                try:
                    y_pred, y_proba = apply_method(
                        method_name,
                        train_ds.X, train_ds.y, train_ds.group,
                        test_ds.X, test_ds.group,
                        seed=seed,
                    )
                    crit = _compute_criteria(
                        y_pred, y_proba, test_ds.y, test_ds.group, method_name
                    )
                    error_ci_lo, error_ci_hi = _bootstrap_error_ci(
                        y_pred, test_ds.y, seed=seed
                    )
                    results.append({
                        "dataset": ds_name,
                        "seed": seed,
                        "method": method_name,
                        "constraint": "none",
                        "eps": None,
                        **crit,
                        "error": 1.0 - crit["accuracy"],
                        "eo_gap": max(crit["tpr_gap"], crit["fpr_gap"]),
                        "error_ci_lo": error_ci_lo,
                        "error_ci_hi": error_ci_hi,
                    })
                except Exception as e:
                    warnings.warn(f"{method_name} failed on {ds_name} seed={seed}: {e}")

            for constraint_name in CONSTRAINTS:
                for eps in EPS_SWEEP:
                    fit_num += 1
                    print(f"  [{fit_num}/{total_fits}] {ds_name} seed={seed} EG({constraint_name}) eps={eps}")
                    try:
                        y_pred, y_proba = apply_method(
                            "exponentiated_gradient",
                            train_ds.X, train_ds.y, train_ds.group,
                            test_ds.X, test_ds.group,
                            seed=seed,
                            constraint=constraint_name,
                            eps=eps,
                        )
                        crit = _compute_criteria(
                            y_pred, y_proba, test_ds.y, test_ds.group,
                            "exponentiated_gradient",
                        )
                        error_ci_lo, error_ci_hi = _bootstrap_error_ci(
                            y_pred, test_ds.y, seed=seed
                        )
                        results.append({
                            "dataset": ds_name,
                            "seed": seed,
                            "method": "exponentiated_gradient",
                            "constraint": constraint_name,
                            "eps": eps,
                            **crit,
                            "error": 1.0 - crit["accuracy"],
                            "eo_gap": max(crit["tpr_gap"], crit["fpr_gap"]),
                            "error_ci_lo": error_ci_lo,
                            "error_ci_hi": error_ci_hi,
                        })
                    except Exception as e:
                        warnings.warn(
                            f"EG({constraint_name}, eps={eps}) failed on "
                            f"{ds_name} seed={seed}: {e}"
                        )

            gc.collect()

    return results


def compute_theoretical_bound(
    p_a: float, p_b: float, p_overall: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute error lower bound under exact separation (star_w projection).

    Derives min_error(DP_gap) = min(p, 1-p) * (1 - DP_gap / |p_a - p_b|).
    Valid only for classifiers enforcing separation (<w, delta_y> = 0);
    non-separation methods can legitimately cross this curve.
    """
    delta_p = abs(p_a - p_b)
    assert delta_p > 0, f"Base rates must differ: p_a={p_a}, p_b={p_b}"

    trivial_error = min(p_overall, 1.0 - p_overall)

    dp_range = np.linspace(0, delta_p, 200)
    min_error = trivial_error * (1.0 - dp_range / delta_p)

    return dp_range, min_error


def _is_separation_enforcing(method: str, constraint: str) -> bool:
    """Return True if this (method, constraint) pair enforces separation in training."""
    if method in SEPARATION_ENFORCING_METHODS:
        return True
    if method == "exponentiated_gradient" and constraint == "equalized_odds":
        return True
    return False


def _per_seed_test_base_rates(
    ds: "FairnessDataset", seed: int
) -> tuple[float, float, float]:
    """Compute test-set base rates for a specific seed's train/val/test split."""
    _, _, test_ds = train_val_test_split(ds, seed=seed)
    g = test_ds.group
    y = test_ds.y
    p_a = float(y[g == 1].mean())
    p_b = float(y[g == 0].mean())
    p_overall = float(y.mean())
    return p_a, p_b, p_overall


def aggregate_violations(
    sweep_results: list[dict],
    datasets: dict[str, FairnessDataset],
) -> dict:
    """Count per-seed bound violations for separation-enforcing methods.

    Gate and violation test applied per seed (not seed-averaged) using
    per-seed test-set base rates. Returns dict keyed by dataset name.
    """
    from collections import defaultdict

    by_ds = defaultdict(list)
    for r in sweep_results:
        by_ds[r["dataset"]].append(r)

    test_rates_cache: dict[tuple[str, int], tuple[float, float, float]] = {}

    def _get_test_rates(ds_name: str, seed: int) -> tuple[float, float, float]:
        key = (ds_name, seed)
        if key not in test_rates_cache:
            test_rates_cache[key] = _per_seed_test_base_rates(
                datasets[ds_name], seed
            )
        return test_rates_cache[key]

    def _bound_at(dp_gap: float, delta_p: float, trivial_error: float) -> float:
        if dp_gap < 0 or dp_gap > delta_p:
            return 0.0
        return trivial_error * (1.0 - dp_gap / delta_p)

    def _run_at_tolerance(records: list[dict], ds_name: str, tau: float) -> dict:
        """Run per-seed violation analysis at a given EO tolerance."""
        per_seed_records = []
        n_total = 0
        n_sep = 0
        n_violations_raw = 0
        n_violations_ci = 0
        worst_violation = 0.0

        seen = set()

        for r in records:
            seed = r["seed"]
            sig = (
                r["method"], r["constraint"], seed,
                round(r["error"], 6), round(r["dp_gap"], 6),
                round(r["eo_gap"], 6),
            )
            if sig in seen:
                continue
            seen.add(sig)

            n_total += 1
            p_a, p_b, p_overall = _get_test_rates(ds_name, seed)
            delta_p = abs(p_a - p_b)
            trivial_error = min(p_overall, 1.0 - p_overall)

            dp = r["dp_gap"]
            err = r["error"]
            err_ci_hi = r.get("error_ci_hi", err)
            eo = r["eo_gap"]
            b = _bound_at(dp, delta_p, trivial_error)

            sep_enforcing = _is_separation_enforcing(
                r["method"], r["constraint"]
            )
            small_eo = eo <= tau
            is_near_sep = sep_enforcing and small_eo
            violates_raw = err < b
            violates_ci = err_ci_hi < b

            if is_near_sep:
                n_sep += 1
                if violates_raw:
                    n_violations_raw += 1
                if violates_ci:
                    n_violations_ci += 1
                    worst_violation = max(worst_violation, b - err_ci_hi)

            per_seed_records.append({
                "method": r["method"],
                "constraint": r["constraint"],
                "eps": r.get("eps"),
                "seed": seed,
                "dp_gap": dp,
                "eo_gap": eo,
                "error": err,
                "error_ci_hi": err_ci_hi,
                "bound": b,
                "p_a_test": p_a,
                "p_b_test": p_b,
                "p_overall_test": p_overall,
                "violates_raw": bool(violates_raw),
                "violates_ci": bool(violates_ci),
                "near_sep": bool(is_near_sep),
                "sep_enforcing": bool(sep_enforcing),
            })

        return {
            "n_total": n_total,
            "n_sep": n_sep,
            "n_violations_raw": n_violations_raw,
            "n_violations_ci": n_violations_ci,
            "worst_violation": worst_violation,
            "eo_tolerance": tau,
            "per_seed_records": per_seed_records,
        }

    out = {}
    for ds_name, records in by_ds.items():
        primary = _run_at_tolerance(records, ds_name, EO_SEPARATION_TOLERANCE)
        strict = _run_at_tolerance(records, ds_name, EO_TOLERANCE_STRICT)
        out[ds_name] = {
            "primary": primary,
            "strict": strict,
        }
    return out


def aggregate_results(results: list[dict]) -> dict:
    """Group by (dataset, method, constraint, eps) and compute mean/std across seeds.

    Returns dict keyed by dataset, with lists of aggregated points.
    Each point: {method, constraint, eps, error_mean, error_std,
                 dp_gap_mean, dp_gap_std, eo_gap_mean, eo_gap_std, ...}
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["method"], r["constraint"],
               r["eps"] if r["eps"] is not None else "none")
        groups[key].append(r)

    aggregated = defaultdict(list)
    for (ds, method, constraint, eps), runs in groups.items():
        point = {
            "method": method,
            "constraint": constraint,
            "eps": eps if eps != "none" else None,
            "n_seeds": len(runs),
        }
        for metric in [
            "accuracy", "error", "dp_gap", "eo_gap",
            "tpr_gap", "fpr_gap", "cal_error",
            "error_ci_lo", "error_ci_hi",
        ]:
            # Some metrics may be NaN (cal_error on masked methods). Use
            # nanmean/nanstd so partial NaN runs don't wipe out the mean.
            vals = np.array(
                [r[metric] for r in runs if metric in r], dtype=np.float64
            )
            if vals.size == 0 or np.all(np.isnan(vals)):
                point[f"{metric}_mean"] = float("nan")
                point[f"{metric}_std"] = float("nan")
            else:
                point[f"{metric}_mean"] = float(np.nanmean(vals))
                point[f"{metric}_std"] = float(np.nanstd(vals))
        aggregated[ds].append(point)

    return dict(aggregated)


# Plotting

def plot_pareto(
    aggregated: dict,
    violations: dict,
    datasets: dict[str, FairnessDataset],
    dataset_names: list[str],
    fig_name: str,
) -> None:
    """Plot Pareto frontier with theoretical bound overlay and EO-gap colorbar."""
    n = len(dataset_names)
    fig_width = 5.5 * n if n <= 2 else 5.5
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 4.5))
    if n == 1:
        axes = [axes]

    eo_vmin, eo_vmax = 0.0, 0.25
    eo_norm = Normalize(vmin=eo_vmin, vmax=eo_vmax)
    eo_cmap = plt.get_cmap("viridis_r")

    for ax, ds_name in zip(axes, dataset_names):
        ds = datasets[ds_name]
        points = aggregated[ds_name]

        p_a = ds.base_rates["a"]
        p_b = ds.base_rates["b"]
        p_overall = float(ds.y.mean())

        dp_theory, err_theory = compute_theoretical_bound(p_a, p_b, p_overall)
        ax.plot(
            dp_theory, err_theory,
            label=r"Bound under exact separation ($\bigstar_w$)",
            **THEORY_STYLE,
        )
        ax.plot(
            dp_theory, err_theory,
            linewidth=6.0, alpha=0.06, color="red",
            zorder=1,
            label=f"Valid for EO gap $\\leq {EO_SEPARATION_TOLERANCE:.2f}$",
        )

        for constraint_name in CONSTRAINTS:
            sweep_pts = [
                p for p in points
                if p["method"] == "exponentiated_gradient"
                and p["constraint"] == constraint_name
            ]
            if not sweep_pts:
                continue
            sweep_pts.sort(key=lambda p: p["eps"])
            dp_means = np.array([p["dp_gap_mean"] for p in sweep_pts])
            err_means = np.array([p["error_mean"] for p in sweep_pts])
            dp_stds = np.array([p["dp_gap_std"] for p in sweep_pts])
            err_stds = np.array([p["error_std"] for p in sweep_pts])
            eo_means = np.array([p["eo_gap_mean"] for p in sweep_pts])

            line_color = SWEEP_COLORS[constraint_name]
            ax.errorbar(
                dp_means, err_means,
                xerr=dp_stds, yerr=err_stds,
                fmt="none", ecolor=line_color, elinewidth=0.7,
                alpha=0.35, capsize=1.5, zorder=2,
            )
            ax.plot(
                dp_means, err_means,
                color=line_color, linewidth=1.0, alpha=0.55,
                zorder=2,
                label=CONSTRAINT_LABELS[constraint_name],
            )
            ax.scatter(
                dp_means, err_means,
                c=eo_means, cmap=eo_cmap, norm=eo_norm,
                s=38, edgecolors=line_color, linewidths=0.9,
                zorder=3,
            )

        for method_name in BASELINES:
            baseline_pts = [p for p in points if p["method"] == method_name]
            if not baseline_pts:
                continue
            pt = baseline_pts[0]
            marker, _ = BASELINE_MARKERS[method_name]
            ax.errorbar(
                [pt["dp_gap_mean"]], [pt["error_mean"]],
                xerr=[pt["dp_gap_std"]], yerr=[pt["error_std"]],
                fmt="none", ecolor="black", elinewidth=0.7,
                alpha=0.5, capsize=2, zorder=3,
            )
            ax.scatter(
                [pt["dp_gap_mean"]], [pt["error_mean"]],
                c=[pt["eo_gap_mean"]], cmap=eo_cmap, norm=eo_norm,
                marker=marker, s=90, edgecolors="black", linewidths=0.9,
                zorder=4,
                label=BASELINE_LABELS[method_name],
            )

        trivial_error = min(p_overall, 1.0 - p_overall)
        ax.plot(
            0, trivial_error, marker="*", color="gray",
            markersize=10, zorder=5, label="Trivial classifier",
        )

        v = violations.get(ds_name, {}).get("primary", {})
        n_sep = v.get("n_sep", 0)
        n_total = v.get("n_total", 0)
        n_viol = v.get("n_violations_ci", 0)

        ax.set_xlabel("DP Gap")
        ax.set_ylabel("Error Rate" if ds_name == dataset_names[0] else "")
        ax.set_title(
            f"{DATASET_LABELS.get(ds_name, ds_name)}  "
            f"($p_a={p_a:.2f}$, $p_b={p_b:.2f}$, "
            f"$|\\Delta p|={abs(p_a - p_b):.2f}$)\n"
            f"Per-seed: {n_sep}/{n_total} near-sep (EO $\\leq {EO_SEPARATION_TOLERANCE:.2f}$), "
            f"{n_viol} CI violations",
            fontsize=9,
        )
        ax.set_xlim(left=-0.01)
        ax.set_ylim(bottom=-0.01)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.85)

    sm = ScalarMappable(norm=eo_norm, cmap=eo_cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes, location="right", shrink=0.78, pad=0.02, aspect=18,
    )
    cbar.set_label("Test-set EO gap", fontsize=9)

    fig.suptitle("Fairness-Accuracy Pareto Frontier", fontsize=12, y=1.02)
    paths = save_figure(fig, fig_name)
    plt.close(fig)
    print(f"  Saved: {[str(p) for p in paths]}")


def _fmt_cal(pt: dict) -> str:
    """Format Cal. Err. column, masking NaN rows to '---'."""
    m = pt.get("cal_error_mean", float("nan"))
    s = pt.get("cal_error_std", float("nan"))
    if m != m or s != s:  # NaN check
        return "---"
    return f"${m:.3f} \\pm {s:.3f}$"


# Table generation

def _generate_tables(all_results: list[dict], aggregated: dict) -> None:
    """Generate LaTeX table with key Pareto points per dataset."""
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ds_labels = {"adult": "Adult", "compas": "COMPAS", "acs_pums": "ACS PUMS"}
    highlight_eps = [0.01, 0.05, 0.10, 0.50]
    baseline_order = [
        "unconstrained_lr",
        "unconstrained_xgb",
        "hardt_postprocessing",
        "platt_scaling",
        "reweighting",
    ]

    lines = [
        "% Exp 3: Pareto frontier key operating points",
        "% Generated by exp3_pareto_frontier.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Selected operating points on the fairness-accuracy Pareto "
        "frontier (mean $\\pm$ std, 5 seeds). EG = ExponentiatedGradient. "
        "`---' in Cal.\\ Err.\\ denotes methods whose output is decision-level "
        "only (Hardt) or where fairlearn's \\_pmf\\_predict fallback produced "
        "binary probabilities; reporting calibration on those values would be "
        "a non-comparable artifact of the proba-exposure path.}",
        "\\label{tab:pareto-frontier}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Method & $\\varepsilon$ & Error & DP Gap & EO Gap & Cal.~Err. \\\\",
        "\\midrule",
    ]

    for ds_name in ["adult", "compas", "acs_pums"]:
        if ds_name not in aggregated:
            continue
        points = aggregated[ds_name]
        label = ds_labels.get(ds_name, ds_name)
        first_row = True

        for method_name in baseline_order:
            bpts = [p for p in points if p["method"] == method_name]
            if not bpts:
                continue
            pt = bpts[0]
            row_label = label if first_row else ""
            first_row = False
            lines.append(
                f"{row_label} & {BASELINE_LABELS[method_name]} & --- "
                f"& ${pt['error_mean']:.3f} \\pm {pt['error_std']:.3f}$ "
                f"& ${pt['dp_gap_mean']:.3f} \\pm {pt['dp_gap_std']:.3f}$ "
                f"& ${pt['eo_gap_mean']:.3f} \\pm {pt['eo_gap_std']:.3f}$ "
                f"& {_fmt_cal(pt)} \\\\"
            )

        for constraint_name in CONSTRAINTS:
            for eps in highlight_eps:
                epts = [p for p in points
                        if p["method"] == "exponentiated_gradient"
                        and p["constraint"] == constraint_name
                        and p["eps"] == eps]
                if not epts:
                    continue
                pt = epts[0]
                row_label = label if first_row else ""
                first_row = False
                lines.append(
                    f"{row_label} & {CONSTRAINT_LABELS[constraint_name]} & {eps} "
                    f"& ${pt['error_mean']:.3f} \\pm {pt['error_std']:.3f}$ "
                    f"& ${pt['dp_gap_mean']:.3f} \\pm {pt['dp_gap_std']:.3f}$ "
                    f"& ${pt['eo_gap_mean']:.3f} \\pm {pt['eo_gap_std']:.3f}$ "
                    f"& {_fmt_cal(pt)} \\\\"
                )

        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"

    lines += [
        "\\end{tabular}",
        "\\end{table}",
    ]

    path = tables_dir / "exp3_pareto.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Table saved to {path}")


# Main

def main():
    ensure_dirs()
    setup_style()
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 3: Fairness-Accuracy Pareto Frontier")
    print("=" * 60)

    results_path = RESULTS_DIR / "exp3_results.json"
    partial = {}
    if results_path.exists():
        with open(results_path) as f:
            partial = json.load(f)
        print(f"  Found partial results: {list(partial.keys())}")
        cached_sweep = partial.get("sweep")
        if cached_sweep and (
            not isinstance(cached_sweep, list)
            or len(cached_sweep) == 0
            or "error_ci_lo" not in cached_sweep[0]
        ):
            print(
                "  [!] Cached sweep predates current schema -- discarding "
                "and re-running from scratch."
            )
            partial.pop("sweep", None)
            partial.pop("violations", None)

    datasets = _load_datasets()

    if "sweep" in partial:
        sweep_results = partial["sweep"]
        print(f"\n[Sweep] Using cached results ({len(sweep_results)} points).")
    else:
        t0 = time.time()
        print("\n[Sweep] Running all methods and constraint strengths...")
        sweep_results = run_sweep(datasets, RANDOM_SEEDS)
        print(f"  Sweep done in {time.time() - t0:.1f}s ({len(sweep_results)} points)")
        _save_partial(results_path, {"sweep": sweep_results})

    aggregated = aggregate_results(sweep_results)

    violations = aggregate_violations(sweep_results, datasets)
    print("\n[Violations] Per-seed check (primary tau={:.2f}, strict tau={:.2f}):".format(
        EO_SEPARATION_TOLERANCE, EO_TOLERANCE_STRICT
    ))
    for ds_name, v in violations.items():
        for label, key in [("primary", "primary"), ("strict", "strict")]:
            vv = v[key]
            print(
                f"  {ds_name} ({label} tau={vv['eo_tolerance']:.2f}): "
                f"{vv['n_sep']}/{vv['n_total']} near-sep per-seed points; "
                f"{vv['n_violations_ci']} CI violations, "
                f"{vv['n_violations_raw']} raw violations."
            )
            if vv["n_violations_ci"] > 0:
                print(
                    f"    worst near-sep CI violation margin: "
                    f"{vv['worst_violation']:.6f}"
                )

    print("\n[Plot] Generating Pareto frontier figures...")
    main_ds = [ds for ds in ["adult", "compas"] if ds in aggregated]
    if main_ds:
        plot_pareto(aggregated, violations, datasets, main_ds, "fig3_pareto_frontier")
    if "acs_pums" in aggregated:
        plot_pareto(aggregated, violations, datasets, ["acs_pums"], "fig3_pareto_frontier_acs")

    all_results = {
        "sweep": sweep_results,
        "violations": violations,
    }
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {results_path}")

    _generate_tables(sweep_results, aggregated)

    print("\n" + "=" * 60)
    print("Experiment 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
