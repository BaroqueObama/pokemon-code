"""Bootstrap CI, permutation tests, and BH correction."""

import numpy as np
from statsmodels.stats.multitest import multipletests


def bootstrap_ci(
    statistic_fn,
    X: np.ndarray,
    Y: np.ndarray,
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    paired: bool = False,
) -> tuple[float, float]:
    """Return percentile bootstrap CI for statistic_fn(X, Y).

    Parameters
    ----------
    statistic_fn : callable(X, Y) -> float
    paired : if True, resample shared indices; if False, resample independently.
    """
    rng = np.random.default_rng(seed)
    n_x, n_y = len(X), len(Y)
    stats = np.empty(B)

    for b in range(B):
        if paired:
            assert n_x == n_y, (
                f"Paired bootstrap requires equal lengths: {n_x} vs {n_y}"
            )
            idx = rng.integers(0, n_x, size=n_x)
            stats[b] = statistic_fn(X[idx], Y[idx])
        else:
            idx_x = rng.integers(0, n_x, size=n_x)
            idx_y = rng.integers(0, n_y, size=n_y)
            stats[b] = statistic_fn(X[idx_x], Y[idx_y])

    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def permutation_test(
    statistic_fn,
    X: np.ndarray,
    Y: np.ndarray,
    n_perm: int = 999,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (observed_statistic, p_value) via permutation test.

    For MMD/HSIC prefer the kernel-matrix-reusing versions in kernels.py.
    """
    observed = statistic_fn(X, Y)
    n = len(X)
    pooled = np.concatenate([X, Y], axis=0)

    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        X_perm = pooled[idx[:n]]
        Y_perm = pooled[idx[n:]]
        perm_stats[i] = statistic_fn(X_perm, Y_perm)

    p_value = (1 + np.sum(perm_stats >= observed)) / (1 + n_perm)
    return (float(observed), float(p_value))


def bh_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rejected, corrected_p_values) after Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=np.float64)
    assert np.all((p_values >= 0) & (p_values <= 1)), (
        f"p-values must be in [0,1], got min={p_values.min()}, max={p_values.max()}"
    )

    rejected, corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return (np.asarray(rejected), np.asarray(corrected))
