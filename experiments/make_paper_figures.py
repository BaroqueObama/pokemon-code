"""Render the 9 paper figures (fig_mw_*, fig_pareto_*, fig_fairrep_*).

Read cached JSON results from exp3, exp4, and exp5 and produce three figure
types -- m-width decay, separation-conditional Pareto bound, and fair-
representation forbidden corner -- for each of the three datasets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.config import RESULTS_DIR, ensure_dirs
from experiments.utils.plotting import save_figure, setup_style

DATASETS = ["adult", "compas", "acs_pums"]
DATASET_LABELS = {
    "adult": "Adult Income",
    "compas": "COMPAS",
    "acs_pums": "ACS PUMS",
}

MW_M_MAX = 200


def _residual_curve(per_seed: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (m, residual_mean, residual_std) averaged across seeds."""
    min_len = min(len(r["kpca_cumcapture"]) for r in per_seed)
    cum = np.array([r["kpca_cumcapture"][:min_len] for r in per_seed])
    residual = 1.0 - cum
    return np.arange(1, min_len + 1), residual.mean(0), residual.std(0)


def _k99_median(per_seed: list[dict]) -> int:
    """Return median budget m at which KPCA capture first reaches 99%."""
    ks = []
    for r in per_seed:
        cum = np.asarray(r["kpca_cumcapture"])
        idx = np.searchsorted(cum, 0.99)
        ks.append(int(idx) + 1)
    return int(np.median(ks))


def plot_fig_mw(ds_name: str, spectral: dict) -> None:
    per_seed = spectral[ds_name]["per_seed"]
    m, mean, std = _residual_curve(per_seed)
    k99 = _k99_median(per_seed)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(m, mean, color="#1f77b4", linewidth=1.5)
    ax.fill_between(
        m,
        np.clip(mean - std, 1e-12, None),
        np.clip(mean + std, 1e-12, None),
        color="#1f77b4", alpha=0.15, linewidth=0,
    )

    ax.axvline(k99, linestyle="--", color="#444444", linewidth=1.5)
    ax.annotate(
        f"$k_{{99}} \\approx {k99}$",
        xy=(k99, 1e-2),
        xytext=(5, 0),
        textcoords="offset points",
        color="#444444", fontsize=10, fontweight="bold",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, MW_M_MAX)
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlabel("criterion budget $m$")
    ax.set_ylabel(r"residual $\|P_{V_m^\perp}\hat\delta\|^2 / \|\hat\delta\|^2$")
    ax.set_title(DATASET_LABELS[ds_name])
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    paths = save_figure(fig, f"fig_mw_{ds_name}", dpi=1200)
    plt.close(fig)
    print(f"  Saved: {[p.name for p in paths]}")


PARETO_METHODS = [
    (("hardt_postprocessing", "none"), "Hardt", "#d62728"),
    (("exponentiated_gradient", "equalized_odds"), "ExpGrad (EO)", "#2ca02c"),
]


def plot_fig_pareto(ds_name: str, exp3: dict) -> None:
    """Plot (DP_gap, error) scatter for separation-enforcing classifiers."""
    near_sep = [
        r for r in exp3["violations"][ds_name]["primary"]["per_seed_records"]
        if r.get("near_sep") and r.get("sep_enforcing")
    ]
    if not near_sep:
        print(f"  fig_pareto_{ds_name}: no near-sep records, skipping")
        return

    # Pull CI bounds from the full sweep record.
    sweep_index = {
        (r["dataset"], r["method"], r["constraint"], r["eps"], r["seed"]): r
        for r in exp3["sweep"]
    }

    p_a = near_sep[0]["p_a_test"]
    p_b = near_sep[0]["p_b_test"]
    p_overall = near_sep[0]["p_overall_test"]
    delta_p = abs(p_a - p_b)
    trivial_error = min(p_overall, 1.0 - p_overall)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    dp_grid = np.linspace(0.0, delta_p, 200)
    bound = trivial_error * (1.0 - dp_grid / delta_p)
    ax.plot(
        dp_grid, bound,
        linestyle="--", color="black", linewidth=2.0, label="theoretical bound",
    )
    ax.fill_between(dp_grid, 0.0, bound, color="#d62728", alpha=0.10, linewidth=0)

    ax.axvline(delta_p, linestyle=":", color="gray", linewidth=0.8, alpha=0.7)

    for (method, constraint), label, color in PARETO_METHODS:
        rows = [
            r for r in near_sep
            if r["method"] == method and r["constraint"] == constraint
        ]
        if not rows:
            continue
        dp = np.array([r["dp_gap"] for r in rows])
        err = np.array([r["error"] for r in rows])
        ci_lo, ci_hi = [], []
        for r in rows:
            sw = sweep_index[(ds_name, r["method"], r["constraint"], r["eps"], r["seed"])]
            ci_lo.append(sw["error_ci_lo"])
            ci_hi.append(sw["error_ci_hi"])
        yerr = np.vstack([err - np.array(ci_lo), np.array(ci_hi) - err])

        ax.errorbar(
            dp, err, yerr=yerr,
            fmt="o", color=color, markersize=8,
            markeredgecolor="black", markeredgewidth=0.5,
            ecolor=color, elinewidth=1.0, capsize=0, alpha=0.9,
            label=f"{label} ({len(rows)})",
        )

    ax.set_xlim(left=-0.005)
    ax.set_ylim(bottom=-0.005)
    ax.set_xlabel("DP_gap")
    ax.set_ylabel("test error")
    ax.set_title(f"{DATASET_LABELS[ds_name]}  ($|\\Delta p|$ = {delta_p:.2f})")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    paths = save_figure(fig, f"fig_pareto_{ds_name}", dpi=1200)
    plt.close(fig)
    print(f"  Saved: {[p.name for p in paths]}")


RHO = 0.15

FAIRREP_METHODS = [
    ("lfr", "LFR", "#1f77b4"),
    ("fair_vae", "Fair-VAE", "#d62728"),
    ("adversarial", "Adversarial", "#2ca02c"),
]


def _gather_fairrep_points(method_block: dict) -> list[tuple[float, float]]:
    """Return (parity_gap, max_class_signal) pairs that pass the rho gate."""
    points = []
    for _lam, cell in method_block.items():
        for ps in cell.get("per_seed", []):
            if ps.get("collapsed"):
                continue
            sg1, sg0 = ps.get("sep_gap_y1"), ps.get("sep_gap_y0")
            if sg1 is None or sg0 is None:
                continue
            if math.isnan(sg1) or math.isnan(sg0):
                continue
            if max(sg1, sg0) > RHO ** 2:
                continue
            mmd2 = ps.get("mmd2_group")
            csa, csb = ps.get("class_struct_a"), ps.get("class_struct_b")
            if mmd2 is None or csa is None or csb is None:
                continue
            x = math.sqrt(max(mmd2, 0.0))
            y = math.sqrt(max(max(csa, csb), 0.0))
            points.append((x, y))
    return points


def plot_fig_fairrep(ds_name: str, exp5: dict, exp3: dict) -> None:
    p_a = exp3["violations"][ds_name]["primary"]["per_seed_records"][0]["p_a_test"]
    p_b = exp3["violations"][ds_name]["primary"]["per_seed_records"][0]["p_b_test"]
    delta_p = abs(p_a - p_b)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x_max = 0.0
    for key, label, color in FAIRREP_METHODS:
        block = exp5[key].get(ds_name)
        if not block:
            continue
        pts = _gather_fairrep_points(block)
        if not pts:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        x_max = max(x_max, xs.max())
        ax.scatter(
            xs, ys,
            color=color, s=55, alpha=0.85,
            edgecolors="black", linewidths=0.5,
            label=f"{label} ({len(pts)})",
        )

    x_grid = np.linspace(0.0, max(x_max * 1.15, RHO), 200)
    bound = (x_grid + RHO) / delta_p
    ax.plot(
        x_grid, bound,
        linestyle="--", color="black", linewidth=2.0, label="theoretical bound",
    )
    ax.set_yscale("log")
    y_lo, y_hi = 7e-2, max(bound.max() * 1.4, 4.0)
    ax.set_ylim(y_lo, y_hi)
    ax.fill_between(x_grid, bound, y_hi, color="#d62728", alpha=0.10, linewidth=0)

    ax.set_xlim(left=0.0, right=x_grid[-1])
    ax.set_xlabel(r"$\|\hat\mu_{\Phi, a} - \hat\mu_{\Phi, b}\|$")
    ax.set_ylabel(r"$\|\hat\mu_{\Phi, 1} - \hat\mu_{\Phi, 0}\|$")
    ax.set_title(
        f"{DATASET_LABELS[ds_name]}  "
        f"($|\\Delta p|$ = {delta_p:.2f}, $\\rho$ = {RHO:.2f})"
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    paths = save_figure(fig, f"fig_fairrep_{ds_name}", dpi=1200)
    plt.close(fig)
    print(f"  Saved: {[p.name for p in paths]}")


def _load_json(name: str) -> dict:
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def main() -> None:
    ensure_dirs()
    setup_style()

    exp3 = _load_json("exp3_results.json")
    exp4 = _load_json("exp4_results.json")
    exp5 = _load_json("exp5_results.json")

    print("Rendering fig_mw_*")
    for ds in DATASETS:
        plot_fig_mw(ds, exp4["spectral"])

    print("Rendering fig_pareto_*")
    for ds in DATASETS:
        plot_fig_pareto(ds, exp3)

    print("Rendering fig_fairrep_*")
    for ds in DATASETS:
        plot_fig_fairrep(ds, exp5, exp3)


if __name__ == "__main__":
    main()
