"""Synthetic Gaussian mixture generators for fairness experiments."""

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from experiments.data import FairnessDataset


def generate_synthetic(
    n: int = 10_000,
    delta_p: float = 0.2,
    d: int = 10,
    seed: int = 42,
    group_ratio: float = 0.5,
) -> tuple[FairnessDataset, dict]:
    """Generate a two-group Gaussian mixture with controllable base-rate gap."""
    assert 0 < delta_p < 1, f"delta_p must be in (0,1), got {delta_p}"
    assert 0 < group_ratio < 1, f"group_ratio must be in (0,1), got {group_ratio}"
    assert d >= 2, f"Need d >= 2 for sparse beta, got {d}"

    rng = np.random.default_rng(seed)

    group = (rng.random(n) < group_ratio).astype(np.int64)
    n_a = group.sum()
    n_b = n - n_a
    assert n_a > 0 and n_b > 0, f"Degenerate groups: n_a={n_a}, n_b={n_b}"

    mu_a = np.zeros(d)
    mu_b = np.zeros(d)
    mu_b[0] = 0.5

    X = np.empty((n, d), dtype=np.float64)
    X[group == 1] = rng.normal(loc=mu_a, scale=1.0, size=(n_a, d))
    X[group == 0] = rng.normal(loc=mu_b, scale=1.0, size=(n_b, d))

    beta = np.zeros(d)
    beta[0] = 1.0
    beta[1] = 0.5

    p_a = 0.6
    p_b = p_a - delta_p
    assert p_b > 0, f"p_b = {p_b} <= 0 -- delta_p too large"

    # Solve per-group intercepts numerically (Jensen makes closed-form wrong).
    from scipy.optimize import brentq

    beta_var = float(np.sum(beta**2))  # ||beta||^2 = Var(X@beta)
    beta_std = np.sqrt(beta_var)

    def _expected_base_rate(intercept: float, mu_g: np.ndarray) -> float:
        """E[sigmoid(intercept + X@beta)] via Gauss-Hermite quadrature."""
        mean_z = intercept + mu_g @ beta
        nodes, weights = np.polynomial.hermite.hermgauss(30)
        z_vals = mean_z + beta_std * np.sqrt(2) * nodes
        return float(np.sum(weights * expit(z_vals)) / np.sqrt(np.pi))

    intercept_a = brentq(
        lambda c: _expected_base_rate(c, mu_a) - p_a, -10, 10
    )
    intercept_b = brentq(
        lambda c: _expected_base_rate(c, mu_b) - p_b, -10, 10
    )

    logits = np.empty(n)
    logits[group == 1] = X[group == 1] @ beta + intercept_a
    logits[group == 0] = X[group == 0] @ beta + intercept_b

    probs = expit(logits)
    y = (rng.random(n) < probs).astype(np.int64)

    emp_p_a = float(y[group == 1].mean())
    emp_p_b = float(y[group == 0].mean())

    se_a = np.sqrt(p_a * (1 - p_a) / n_a)
    se_b = np.sqrt(p_b * (1 - p_b) / n_b)
    assert abs(emp_p_a - p_a) < 3 * se_a, (
        f"Empirical p_a={emp_p_a:.4f} too far from target {p_a} "
        f"(3 SE = {3*se_a:.4f})"
    )
    assert abs(emp_p_b - p_b) < 3 * se_b, (
        f"Empirical p_b={emp_p_b:.4f} too far from target {p_b} "
        f"(3 SE = {3*se_b:.4f})"
    )

    assert np.all(np.isfinite(X)), f"Found non-finite values in X"
    assert set(np.unique(y)).issubset({0, 1}), f"y has values outside {{0,1}}"
    assert set(np.unique(group)).issubset({0, 1}), f"group has values outside {{0,1}}"

    feature_names = [f"x{i}" for i in range(d)]
    base_rates = {"a": emp_p_a, "b": emp_p_b}

    ds = FairnessDataset(
        X=X,
        y=y,
        group=group,
        feature_names=feature_names,
        name="synthetic",
        base_rates=base_rates,
    )

    params = {
        "true_beta": beta,
        "intercepts": {"a": float(intercept_a), "b": float(intercept_b)},
        "target_base_rates": {"a": p_a, "b": p_b},
        "mu_a": mu_a,
        "mu_b": mu_b,
    }

    return ds, params


def generate_rich_synthetic(
    n: int = 10_000,
    delta_p: float = 0.2,
    d: int = 20,
    decay: float = 0.5,
    total_shift: float = 0.5,
    seed: int = 42,
    group_ratio: float = 0.5,
) -> tuple[FairnessDataset, dict]:
    """Two-group Gaussian mixture with power-law-decaying mean shift across all features.

    Spreads the group mean difference over d components so the population delta
    has full-rank spectral support in the RBF-RKHS, unlike the rank-1 shift in
    ``generate_synthetic``.  Label model is identical (sparse logistic).
    """
    assert 0 < delta_p < 1, f"delta_p must be in (0,1), got {delta_p}"
    assert 0 < group_ratio < 1, f"group_ratio must be in (0,1), got {group_ratio}"
    assert d >= 2, f"Need d >= 2 for sparse beta, got {d}"
    assert decay > 0, f"decay must be positive, got {decay}"
    assert total_shift > 0, f"total_shift must be positive, got {total_shift}"

    rng = np.random.default_rng(seed)

    group = (rng.random(n) < group_ratio).astype(np.int64)
    n_a = int(group.sum())
    n_b = n - n_a
    assert n_a > 0 and n_b > 0, f"Degenerate groups: n_a={n_a}, n_b={n_b}"

    raw_weights = np.array(
        [(i + 1) ** (-decay) for i in range(d)], dtype=np.float64
    )
    weight_norm = float(np.linalg.norm(raw_weights))
    assert weight_norm > 0, "Power-law weights degenerate"
    shift_vec = raw_weights * (total_shift / weight_norm)
    actual_shift = float(np.linalg.norm(shift_vec))
    assert abs(actual_shift - total_shift) < 1e-10, (
        f"Shift magnitude {actual_shift} != target {total_shift}"
    )

    mu_a = np.zeros(d)
    mu_b = shift_vec.copy()

    X = np.empty((n, d), dtype=np.float64)
    X[group == 1] = rng.normal(loc=mu_a, scale=1.0, size=(n_a, d))
    X[group == 0] = rng.normal(loc=mu_b, scale=1.0, size=(n_b, d))

    beta = np.zeros(d)
    beta[0] = 1.0
    beta[1] = 0.5

    p_a = 0.6
    p_b = p_a - delta_p
    assert p_b > 0, f"p_b = {p_b} <= 0 -- delta_p too large"

    from scipy.optimize import brentq

    beta_var = float(np.sum(beta ** 2))
    beta_std = np.sqrt(beta_var)

    def _expected_base_rate(intercept: float, mu_g: np.ndarray) -> float:
        """E[sigmoid(intercept + X@beta)] via Gauss-Hermite quadrature."""
        mean_z = intercept + mu_g @ beta
        nodes, weights = np.polynomial.hermite.hermgauss(30)
        z_vals = mean_z + beta_std * np.sqrt(2) * nodes
        return float(np.sum(weights * expit(z_vals)) / np.sqrt(np.pi))

    intercept_a = brentq(
        lambda c: _expected_base_rate(c, mu_a) - p_a, -10, 10
    )
    intercept_b = brentq(
        lambda c: _expected_base_rate(c, mu_b) - p_b, -10, 10
    )

    logits = np.empty(n)
    logits[group == 1] = X[group == 1] @ beta + intercept_a
    logits[group == 0] = X[group == 0] @ beta + intercept_b

    probs = expit(logits)
    y = (rng.random(n) < probs).astype(np.int64)

    emp_p_a = float(y[group == 1].mean())
    emp_p_b = float(y[group == 0].mean())

    se_a = np.sqrt(p_a * (1 - p_a) / n_a)
    se_b = np.sqrt(p_b * (1 - p_b) / n_b)
    assert abs(emp_p_a - p_a) < 3 * se_a, (
        f"Empirical p_a={emp_p_a:.4f} too far from target {p_a} "
        f"(3 SE = {3*se_a:.4f})"
    )
    assert abs(emp_p_b - p_b) < 3 * se_b, (
        f"Empirical p_b={emp_p_b:.4f} too far from target {p_b} "
        f"(3 SE = {3*se_b:.4f})"
    )

    assert np.all(np.isfinite(X)), "Found non-finite values in X"
    assert set(np.unique(y)).issubset({0, 1}), f"y has values outside {{0,1}}"
    assert set(np.unique(group)).issubset({0, 1}), (
        f"group has values outside {{0,1}}"
    )

    feature_names = [f"x{i}" for i in range(d)]
    base_rates = {"a": emp_p_a, "b": emp_p_b}

    ds = FairnessDataset(
        X=X,
        y=y,
        group=group,
        feature_names=feature_names,
        name="synthetic_rich",
        base_rates=base_rates,
    )

    params = {
        "true_beta": beta,
        "intercepts": {"a": float(intercept_a), "b": float(intercept_b)},
        "target_base_rates": {"a": p_a, "b": p_b},
        "mu_a": mu_a,
        "mu_b": mu_b,
        "shift_vec": shift_vec,
        "decay": decay,
        "total_shift": total_shift,
    }

    return ds, params


def generate_separable_synthetic(
    n: int = 10_000,
    p_a: float = 0.6,
    p_b: float = 0.4,
    d: int = 10,
    seed: int = 42,
    group_ratio: float = 0.5,
) -> tuple[FairnessDataset, dict]:
    """Generate two-group data with deterministic Y = 1(x[0] > 0) for Theorem 2 validation."""
    assert 0 < p_a < 1 and 0 < p_b < 1
    assert d >= 2
    assert 0 < group_ratio < 1

    rng = np.random.default_rng(seed)

    group = (rng.random(n) < group_ratio).astype(np.int64)
    n_a = int(group.sum())
    n_b = n - n_a
    assert n_a > 0 and n_b > 0

    mu_a = np.zeros(d)
    mu_a[0] = float(norm.ppf(p_a))
    mu_b = np.zeros(d)
    mu_b[0] = float(norm.ppf(p_b))

    X = np.empty((n, d), dtype=np.float64)
    X[group == 1] = rng.normal(loc=mu_a, scale=1.0, size=(n_a, d))
    X[group == 0] = rng.normal(loc=mu_b, scale=1.0, size=(n_b, d))

    y = (X[:, 0] > 0).astype(np.int64)

    emp_p_a = float(y[group == 1].mean())
    emp_p_b = float(y[group == 0].mean())

    se_a = np.sqrt(p_a * (1 - p_a) / n_a)
    se_b = np.sqrt(p_b * (1 - p_b) / n_b)
    assert abs(emp_p_a - p_a) < 4 * se_a, (
        f"Empirical p_a={emp_p_a:.4f} too far from target {p_a}"
    )
    assert abs(emp_p_b - p_b) < 4 * se_b, (
        f"Empirical p_b={emp_p_b:.4f} too far from target {p_b}"
    )

    feature_names = [f"x{i}" for i in range(d)]

    ds = FairnessDataset(
        X=X,
        y=y,
        group=group,
        feature_names=feature_names,
        name="synthetic_separable",
        base_rates={"a": emp_p_a, "b": emp_p_b},
    )

    params = {
        "mu_a": mu_a,
        "mu_b": mu_b,
        "target_base_rates": {"a": p_a, "b": p_b},
        "delta_p": abs(p_a - p_b),
        "separation_rule": "Y = 1(x[0] > 0)",
    }

    return ds, params


def generate_synthetic_Z_separated(
    n: int = 10_000,
    p_a: float = 0.6,
    p_b: float = 0.4,
    d_eps: int = 8,
    alpha: float = 0.0,
    seed: int = 42,
    group_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a representation Z analytically with controllable separation.

    Z is constructed directly (no encoder) so that the Theorem 2 separation
    premise (Y perp G | Z) holds exactly at alpha=0 and is violated for
    alpha > 0.  Returns (Z, y, g, params) -- no FairnessDataset because
    there is no underlying feature space X.
    """
    assert 0 < p_a < 1 and 0 < p_b < 1, f"base rates must be in (0,1), got {p_a}, {p_b}"
    assert 0 < group_ratio < 1, f"group_ratio must be in (0,1), got {group_ratio}"
    assert d_eps >= 2, f"d_eps must be >= 2 (need noise + output dim), got {d_eps}"
    assert alpha >= 0, f"alpha must be >= 0, got {alpha}"
    assert n >= 100, f"need n >= 100 to populate all 4 cells, got {n}"

    rng = np.random.default_rng(seed)

    g = (rng.random(n) < group_ratio).astype(np.int64)
    n_a = int(g.sum())
    n_b = n - n_a
    assert n_a > 0 and n_b > 0, (
        f"empty group(s) at n={n}, group_ratio={group_ratio}: n_a={n_a}, n_b={n_b}"
    )

    y = np.empty(n, dtype=np.int64)
    y[g == 1] = (rng.random(n_a) < p_a).astype(np.int64)
    y[g == 0] = (rng.random(n_b) < p_b).astype(np.int64)

    # Every (y, g) cell must be non-empty for the decomposition helper
    for yy in [0, 1]:
        for gg in [0, 1]:
            count = int(((y == yy) & (g == gg)).sum())
            assert count >= 1, (
                f"cell (y={yy}, g={gg}) is empty; try larger n or adjusted base rates"
            )

    eta = rng.standard_normal(size=(n, d_eps))

    Z = np.empty((n, d_eps), dtype=np.float64)
    Z[:, 0] = (
        y.astype(np.float64)
        + 0.5 * eta[:, 0]
        + alpha * (g.astype(np.float64) - 0.5) * eta[:, 1]
    )
    Z[:, 1:] = eta[:, 1:]

    emp_p_a = float(y[g == 1].mean())
    emp_p_b = float(y[g == 0].mean())

    params = {
        "n": int(n),
        "d_eps": int(d_eps),
        "alpha": float(alpha),
        "target_base_rates": {"a": float(p_a), "b": float(p_b)},
        "empirical_base_rates": {"a": emp_p_a, "b": emp_p_b},
        "delta_p": abs(emp_p_a - emp_p_b),
        "group_ratio": float(group_ratio),
        "separation_rule": (
            "exact (alpha=0)" if alpha == 0 else f"alpha={alpha} violation in Z[:, 0]"
        ),
    }

    return Z, y, g, params
