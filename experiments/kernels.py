"""Core kernel toolkit: RBF kernel, median heuristic, MMD, HSIC."""

import numpy as np
import torch
from scipy.spatial.distance import pdist

from experiments.config import get_device

# Bandwidth selection


def median_heuristic(X: np.ndarray, Y: np.ndarray | None = None) -> float:
    """Compute the median-heuristic bandwidth over the pooled sample.

    Subsamples to 20k points when necessary to keep O(n^2) memory bounded.
    """
    assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D with shape {X.shape}"

    if Y is not None:
        assert Y.ndim == 2, f"Expected 2D array, got {Y.ndim}D with shape {Y.shape}"
        assert X.shape[1] == Y.shape[1], (
            f"Feature dims must match: {X.shape[1]} vs {Y.shape[1]}"
        )
        Z = np.vstack([X, Y])
    else:
        Z = X

    n = Z.shape[0]
    max_exact = 20_000

    if n > max_exact:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_exact, replace=False)
        Z = Z[idx]

    if Z.shape[0] > max_exact:
        sq_dists = pdist(Z, metric="sqeuclidean")
    else:
        Z_t = torch.as_tensor(Z, dtype=torch.float64, device=torch.device("cpu"))
        n_z = Z_t.shape[0]
        tri_i, tri_j = torch.triu_indices(n_z, n_z, offset=1)
        diffs = Z_t[tri_i] - Z_t[tri_j]
        sq_dists = (diffs * diffs).sum(dim=1).numpy()

    sigma = float(np.sqrt(np.median(sq_dists)))
    assert sigma > 0, "Median heuristic produced sigma=0 (all points identical?)"
    return sigma


# RBF kernel matrix


def rbf_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    sigma: float | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute the Gaussian RBF kernel matrix K(X, Y)."""
    assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D with shape {X.shape}"
    self_kernel = Y is None
    if Y is not None:
        assert Y.ndim == 2, f"Expected 2D array, got {Y.ndim}D with shape {Y.shape}"
        assert X.shape[1] == Y.shape[1], (
            f"Feature dims must match: {X.shape[1]} vs {Y.shape[1]}"
        )

    if sigma is None:
        sigma = median_heuristic(X, Y)

    if device is None:
        device = get_device()

    # MPS lacks float64; fall back to CPU for permutation-test precision
    use_device = torch.device("cpu") if device.type == "mps" else device

    X_t = torch.as_tensor(X, dtype=torch.float64, device=use_device)
    Y_t = (
        X_t
        if self_kernel
        else torch.as_tensor(Y, dtype=torch.float64, device=use_device)
    )

    sq_dists = torch.cdist(X_t, Y_t).pow(2)
    K = torch.exp(-sq_dists / (2.0 * sigma**2))
    K_np = K.cpu().numpy()

    # exp(-0) can drift from 1.0 in float64 due to cdist rounding
    if self_kernel:
        np.fill_diagonal(K_np, 1.0)

    n_rows, n_cols = X.shape[0], (X.shape[0] if self_kernel else Y.shape[0])
    assert K_np.shape == (n_rows, n_cols), (
        f"Expected shape ({n_rows}, {n_cols}), got {K_np.shape}"
    )
    assert np.all(K_np >= 0) and np.all(K_np <= 1.0 + 1e-10), (
        f"Kernel values out of [0,1]: min={K_np.min()}, max={K_np.max()}"
    )

    return K_np


# MMD


def _mmd_squared_from_kernel(K: np.ndarray, n: int, m: int) -> float:
    """Compute unbiased MMD^2 from a pooled (n+m, n+m) kernel matrix with zeroed diagonal."""
    K_XX = K[:n, :n]
    K_XY = K[:n, n:]
    K_YY = K[n:, n:]

    mmd2 = (
        K_XX.sum() / (n * (n - 1))
        - 2.0 * K_XY.sum() / (n * m)
        + K_YY.sum() / (m * (m - 1))
    )
    return float(mmd2)


def mmd_squared(
    X: np.ndarray, Y: np.ndarray, sigma: float | None = None
) -> float:
    """Compute unbiased MMD^2 between samples X and Y with an RBF kernel."""
    assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D with shape {X.shape}"
    assert Y.ndim == 2, f"Expected 2D array, got {Y.ndim}D with shape {Y.shape}"
    n, m = X.shape[0], Y.shape[0]
    assert n >= 2, f"Need n >= 2, got {n}"
    assert m >= 2, f"Need m >= 2, got {m}"

    if sigma is None:
        sigma = median_heuristic(X, Y)

    Z = np.vstack([X, Y])
    K = rbf_kernel_matrix(Z, Z, sigma=sigma)
    np.fill_diagonal(K, 0.0)
    return _mmd_squared_from_kernel(K, n, m)


def mmd_squared_biased(
    X: np.ndarray, Y: np.ndarray, sigma: float | None = None
) -> float:
    """Compute biased (V-statistic) MMD^2 between X and Y.

    Keeps the diagonal and divides by n^2, m^2 so the result equals
    ||mu_hat_X - mu_hat_Y||^2 exactly on sample mean embeddings.
    Use for algebraic identity verification; use the unbiased
    `mmd_squared` for hypothesis testing.
    """
    assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D with shape {X.shape}"
    assert Y.ndim == 2, f"Expected 2D array, got {Y.ndim}D with shape {Y.shape}"
    n, m = X.shape[0], Y.shape[0]
    assert n >= 1, f"Need n >= 1, got {n}"
    assert m >= 1, f"Need m >= 1, got {m}"

    if sigma is None:
        sigma = median_heuristic(X, Y)

    Z = np.vstack([X, Y])
    K = rbf_kernel_matrix(Z, Z, sigma=sigma)
    K_XX = K[:n, :n]
    K_XY = K[:n, n:]
    K_YY = K[n:, n:]

    mmd2 = (
        K_XX.sum() / (n * n)
        - 2.0 * K_XY.sum() / (n * m)
        + K_YY.sum() / (m * m)
    )
    return float(mmd2)


def mmd_test(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float | None = None,
    n_perm: int = 999,
    seed: int = 42,
    n_jobs: int = 1,
) -> tuple[float, float]:
    """Two-sample MMD permutation test. Returns (observed_mmd2, p_value)."""
    assert X.ndim == 2 and Y.ndim == 2
    n, m = X.shape[0], Y.shape[0]
    assert n >= 2 and m >= 2

    if sigma is None:
        sigma = median_heuristic(X, Y)

    Z = np.vstack([X, Y])
    K = rbf_kernel_matrix(Z, Z, sigma=sigma)
    K_zeroed = K.copy()
    np.fill_diagonal(K_zeroed, 0.0)

    observed = _mmd_squared_from_kernel(K_zeroed, n, m)

    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n + m) for _ in range(n_perm)]

    def _compute_one(perm):
        K_perm = K_zeroed[np.ix_(perm, perm)]
        return _mmd_squared_from_kernel(K_perm, n, m)

    if n_jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            perm_stats = np.array(list(pool.map(_compute_one, perms)))
    else:
        perm_stats = np.array([_compute_one(p) for p in perms])

    p_value = (1 + np.sum(perm_stats >= observed)) / (1 + n_perm)
    return (observed, float(p_value))


def mmd_multi_bandwidth(
    X: np.ndarray,
    Y: np.ndarray,
    multipliers: list[float] | None = None,
    n_perm: int = 999,
    seed: int = 42,
) -> dict[float, tuple[float, float]]:
    """Run MMD permutation tests at multiple bandwidth scales. Returns {multiplier: (mmd2, p_value)}."""
    if multipliers is None:
        from experiments.config import BANDWIDTH_MULTIPLIERS

        multipliers = BANDWIDTH_MULTIPLIERS

    assert X.ndim == 2 and Y.ndim == 2
    n, m = X.shape[0], Y.shape[0]

    base_sigma = median_heuristic(X, Y)
    Z = np.vstack([X, Y])

    Z_t = torch.as_tensor(Z, dtype=torch.float64, device=torch.device("cpu"))
    sq_dists = torch.cdist(Z_t, Z_t).pow(2).numpy()

    rng = np.random.default_rng(seed)
    results = {}

    for mult in multipliers:
        sigma = mult * base_sigma
        K = np.exp(-sq_dists / (2.0 * sigma**2))
        np.fill_diagonal(K, 0.0)

        observed = _mmd_squared_from_kernel(K, n, m)

        perm_stats = np.empty(n_perm)
        for i in range(n_perm):
            idx = rng.permutation(n + m)
            K_perm = K[np.ix_(idx, idx)]
            perm_stats[i] = _mmd_squared_from_kernel(K_perm, n, m)

        p_value = (1 + np.sum(perm_stats >= observed)) / (1 + n_perm)
        results[mult] = (float(observed), float(p_value))

    return results


# HSIC


def _hsic_from_kernels(K_X: np.ndarray, K_Y: np.ndarray) -> float:
    """Compute biased HSIC from pre-computed kernel matrices."""
    n = K_X.shape[0]
    assert K_X.shape == (n, n), f"Expected ({n},{n}), got {K_X.shape}"
    assert K_Y.shape == (n, n), f"Expected ({n},{n}), got {K_Y.shape}"

    K_Xc = (
        K_X
        - K_X.mean(axis=0, keepdims=True)
        - K_X.mean(axis=1, keepdims=True)
        + K_X.mean()
    )
    K_Yc = (
        K_Y
        - K_Y.mean(axis=0, keepdims=True)
        - K_Y.mean(axis=1, keepdims=True)
        + K_Y.mean()
    )

    # sum(A * B) == tr(A @ B) for symmetric matrices, avoiding O(n^3) matmul
    hsic_val = float(np.sum(K_Xc * K_Yc)) / (n * n)
    return float(hsic_val)


def hsic(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
) -> float:
    """Compute biased HSIC between X and Y with RBF kernels."""
    assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D with shape {X.shape}"
    assert Y.ndim == 2, f"Expected 2D array, got {Y.ndim}D with shape {Y.shape}"
    assert X.shape[0] == Y.shape[0], (
        f"Sample sizes must match: {X.shape[0]} vs {Y.shape[0]}"
    )

    K_X = rbf_kernel_matrix(X, sigma=sigma_x)
    K_Y = rbf_kernel_matrix(Y, sigma=sigma_y)
    return _hsic_from_kernels(K_X, K_Y)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix: K_c = H K H where H = I - (1/n)11^T."""
    return (
        K
        - K.mean(axis=0, keepdims=True)
        - K.mean(axis=1, keepdims=True)
        + K.mean()
    )


def hsic_test(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    n_perm: int = 999,
    seed: int = 42,
    n_jobs: int = 1,
) -> tuple[float, float]:
    """HSIC independence test via permutation. Returns (observed_hsic, p_value).

    Pre-centers both kernels before the loop because centering commutes
    with permutation (H is permutation-invariant), saving 4*O(n^2) per
    permutation.
    """
    n = X.shape[0]
    assert X.shape[0] == Y.shape[0]

    K_X = rbf_kernel_matrix(X, sigma=sigma_x)
    K_Y = rbf_kernel_matrix(Y, sigma=sigma_y)

    K_Xc = _center_kernel(K_X)
    K_Yc = _center_kernel(K_Y)
    n_sq = n * n

    observed = float(np.sum(K_Xc * K_Yc)) / n_sq

    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n) for _ in range(n_perm)]

    def _compute_one(perm):
        K_Yc_perm = K_Yc[np.ix_(perm, perm)]
        return float(np.sum(K_Xc * K_Yc_perm)) / n_sq

    if n_jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            perm_stats = np.array(list(pool.map(_compute_one, perms)))
    else:
        perm_stats = np.array([_compute_one(p) for p in perms])

    p_value = (1 + np.sum(perm_stats >= observed)) / (1 + n_perm)
    return (observed, float(p_value))


# Projected master constraint


def projected_master_constraint_residual(
    scores: np.ndarray, y_true: np.ndarray, group: np.ndarray
) -> float:
    """Return |LHS - RHS| of the projected master constraint on classifier scores.

    The identity is algebraic (law of total expectation), so the residual
    should be ~0 up to floating-point noise.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true)
    group = np.asarray(group)

    for g in [0, 1]:
        for y in [0, 1]:
            mask = (group == g) & (y_true == y)
            count = mask.sum()
            assert count > 0, (
                f"Subgroup (group={g}, y={y}) is empty — "
                f"need all 4 subgroups non-empty"
            )

    p_a = y_true[group == 1].mean()
    p_b = y_true[group == 0].mean()

    mu_a = scores[group == 1].mean()
    mu_b = scores[group == 0].mean()
    mu_1a = scores[(group == 1) & (y_true == 1)].mean()
    mu_1b = scores[(group == 0) & (y_true == 1)].mean()
    mu_0a = scores[(group == 1) & (y_true == 0)].mean()
    mu_0b = scores[(group == 0) & (y_true == 0)].mean()

    lhs = mu_a - mu_b
    rhs = (
        p_a * (mu_1a - mu_1b)
        + (1 - p_a) * (mu_0a - mu_0b)
        + (p_a - p_b) * (mu_1b - mu_0b)
    )

    return float(abs(lhs - rhs))


# Master constraint decomposition in Z-space (RKHS vector form)

# Cell index convention used by the decomposition helpers:
# 0 = (y=1, g=a), 1 = (y=0, g=a), 2 = (y=1, g=b), 3 = (y=0, g=b)
# Project convention: g=1 encodes "a", g=0 encodes "b".
_CELL_1A, _CELL_0A, _CELL_1B, _CELL_0B = 0, 1, 2, 3


def _decomposition_from_cell_kernel(
    M: np.ndarray,
    counts: np.ndarray,
    p_a_emp: float,
    p_b_emp: float,
    sigma: float,
) -> dict:
    """Assemble the master-constraint decomposition from a 4x4 cell kernel-sum matrix.

    Uses biased V-statistic norms so the identity holds to machine precision.
    """
    assert M.shape == (4, 4), f"Expected 4x4 cell kernel, got {M.shape}"
    assert counts.shape == (4,), f"Expected counts shape (4,), got {counts.shape}"

    def inner_product(S_cells: list[int], T_cells: list[int]) -> float:
        n_S = counts[S_cells].sum()
        n_T = counts[T_cells].sum()
        kernel_sum = M[np.ix_(S_cells, T_cells)].sum()
        return float(kernel_sum / (n_S * n_T))

    def norm_sq_diff(S_cells: list[int], T_cells: list[int]) -> float:
        ip_SS = inner_product(S_cells, S_cells)
        ip_TT = inner_product(T_cells, T_cells)
        ip_ST = inner_product(S_cells, T_cells)
        return float(ip_SS - 2.0 * ip_ST + ip_TT)

    CELLS_A = [_CELL_1A, _CELL_0A]
    CELLS_B = [_CELL_1B, _CELL_0B]
    CELLS_Y1 = [_CELL_1A, _CELL_1B]
    CELLS_Y0 = [_CELL_0A, _CELL_0B]

    mmd2_group = norm_sq_diff(CELLS_A, CELLS_B)
    sep_gap_y1 = norm_sq_diff([_CELL_1A], [_CELL_1B])
    sep_gap_y0 = norm_sq_diff([_CELL_0A], [_CELL_0B])
    class_struct_b = norm_sq_diff([_CELL_1B], [_CELL_0B])
    class_struct_a = norm_sq_diff([_CELL_1A], [_CELL_0A])
    class_struct_marginal = norm_sq_diff(CELLS_Y1, CELLS_Y0)

    # Coefficient vector c so that ||sum_i c_i mu_i||^2 is the identity residual.
    # Identically zero under empirical base rates (law of total expectation).
    c = np.zeros(4, dtype=np.float64)
    # +delta
    c[_CELL_1A] += p_a_emp
    c[_CELL_0A] += (1.0 - p_a_emp)
    c[_CELL_1B] -= p_b_emp
    c[_CELL_0B] -= (1.0 - p_b_emp)
    # -p_a * delta_1
    c[_CELL_1A] -= p_a_emp
    c[_CELL_1B] += p_a_emp
    # -(1-p_a) * delta_0
    c[_CELL_0A] -= (1.0 - p_a_emp)
    c[_CELL_0B] += (1.0 - p_a_emp)
    # -(p_a - p_b) * (mu_{1,b} - mu_{0,b})
    c[_CELL_1B] -= (p_a_emp - p_b_emp)
    c[_CELL_0B] += (p_a_emp - p_b_emp)

    c_norm = c / counts
    identity_residual_sq = float(c_norm @ M @ c_norm)

    return {
        "mmd2_group": mmd2_group,
        "sep_gap_y1": sep_gap_y1,
        "sep_gap_y0": sep_gap_y0,
        "class_struct_b": class_struct_b,
        "class_struct_a": class_struct_a,
        "class_struct_marginal": class_struct_marginal,
        "identity_residual_sq": identity_residual_sq,
        "p_a_emp": float(p_a_emp),
        "p_b_emp": float(p_b_emp),
        "sigma_used": float(sigma),
        "estimator": "biased_V_statistic",
    }


def _cell_one_hot_matrix(
    y: np.ndarray, g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n, 4) one-hot cell membership matrix and counts vector."""
    y = np.asarray(y).astype(np.int64)
    g = np.asarray(g).astype(np.int64)
    n = y.shape[0]
    assert g.shape == (n,), f"y and g must have shape ({n},), got g={g.shape}"
    unique_y = np.unique(y)
    unique_g = np.unique(g)
    assert set(unique_y.tolist()).issubset({0, 1}), (
        f"y must be binary {{0,1}}, got unique values {unique_y}"
    )
    assert set(unique_g.tolist()).issubset({0, 1}), (
        f"g must be binary {{0,1}}, got unique values {unique_g}"
    )

    E = np.zeros((n, 4), dtype=np.float64)
    E[(y == 1) & (g == 1), _CELL_1A] = 1.0
    E[(y == 0) & (g == 1), _CELL_0A] = 1.0
    E[(y == 1) & (g == 0), _CELL_1B] = 1.0
    E[(y == 0) & (g == 0), _CELL_0B] = 1.0

    counts = E.sum(axis=0)
    for k, name in enumerate(["(y=1,a)", "(y=0,a)", "(y=1,b)", "(y=0,b)"]):
        assert counts[k] >= 1.0, (
            f"Cell {name} is empty — need all 4 (y, g) cells non-empty "
            f"(counts={counts.astype(int).tolist()})"
        )
    return E, counts


def master_constraint_decomposition_from_K(
    K: np.ndarray,
    E: np.ndarray,
    counts: np.ndarray,
    p_a_emp: float,
    p_b_emp: float,
    sigma: float,
) -> dict:
    """Decomposition from a precomputed Gram matrix, for bootstrap reuse."""
    assert K.ndim == 2 and K.shape[0] == K.shape[1], f"K must be square, got {K.shape}"
    assert E.ndim == 2 and E.shape[0] == K.shape[0] and E.shape[1] == 4, (
        f"E shape mismatch: K={K.shape}, E={E.shape}"
    )
    assert counts.shape == (4,), f"counts shape {counts.shape} != (4,)"
    M = E.T @ K @ E
    return _decomposition_from_cell_kernel(M, counts, p_a_emp, p_b_emp, sigma)


def master_constraint_decomposition_in_Z(
    Z: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    sigma: float,
) -> dict:
    """Compute the Z-space master-constraint decomposition for a learned representation.

    Returns all six RKHS norm-squared quantities from the master constraint
    (biased V-statistic so the identity holds to machine precision).
    For hypothesis testing use `mmd_test` / `mmd_squared` instead.
    """
    assert Z.ndim == 2, f"Expected 2D Z, got shape {Z.shape}"
    n = Z.shape[0]
    assert n >= 4, f"Need at least 4 samples (one per cell), got n={n}"

    E, counts = _cell_one_hot_matrix(y, g)
    assert E.shape[0] == n, f"E shape {E.shape} does not match Z shape {Z.shape}"

    n_a = counts[_CELL_1A] + counts[_CELL_0A]
    n_b = counts[_CELL_1B] + counts[_CELL_0B]
    p_a_emp = float(counts[_CELL_1A] / n_a)
    p_b_emp = float(counts[_CELL_1B] / n_b)

    K = rbf_kernel_matrix(Z, Z, sigma=sigma)
    M = E.T @ K @ E

    result = _decomposition_from_cell_kernel(M, counts, p_a_emp, p_b_emp, sigma)
    result["n_cells"] = {
        "1a": int(counts[_CELL_1A]),
        "0a": int(counts[_CELL_0A]),
        "1b": int(counts[_CELL_1B]),
        "0b": int(counts[_CELL_0B]),
    }
    return result


def master_constraint_decomposition_in_Z_multi_bandwidth(
    Z: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    multipliers: list[float] | None = None,
) -> dict[float, dict]:
    """Run the Z-space master-constraint decomposition at multiple bandwidths.

    Returns {multiplier: decomposition_dict}.
    """
    if multipliers is None:
        from experiments.config import BANDWIDTH_MULTIPLIERS

        multipliers = BANDWIDTH_MULTIPLIERS

    assert Z.ndim == 2, f"Expected 2D Z, got shape {Z.shape}"
    n = Z.shape[0]
    assert n >= 4, f"Need at least 4 samples, got n={n}"

    E, counts = _cell_one_hot_matrix(y, g)
    n_a = counts[_CELL_1A] + counts[_CELL_0A]
    n_b = counts[_CELL_1B] + counts[_CELL_0B]
    p_a_emp = float(counts[_CELL_1A] / n_a)
    p_b_emp = float(counts[_CELL_1B] / n_b)

    base_sigma = median_heuristic(Z)

    Z_t = torch.as_tensor(Z, dtype=torch.float64, device=torch.device("cpu"))
    sq_dists = torch.cdist(Z_t, Z_t).pow(2).numpy()

    results: dict[float, dict] = {}
    for mult in multipliers:
        sigma = mult * base_sigma
        K = np.exp(-sq_dists / (2.0 * sigma**2))
        M = E.T @ K @ E
        decomp = _decomposition_from_cell_kernel(M, counts, p_a_emp, p_b_emp, sigma)
        decomp["n_cells"] = {
            "1a": int(counts[_CELL_1A]),
            "0a": int(counts[_CELL_0A]),
            "1b": int(counts[_CELL_1B]),
            "0b": int(counts[_CELL_0B]),
        }
        decomp["bandwidth_multiplier"] = float(mult)
        results[float(mult)] = decomp

    return results


def mode_collapse_detected(Z: np.ndarray, threshold: float = 1e-6) -> bool:
    """Detect whether a learned representation has collapsed.

    Checks both full collapse (all stds below threshold) and partial
    collapse (median pairwise distance near zero, which would make
    median_heuristic return sigma ~ 0 and crash kernel computations).
    """
    assert Z.ndim == 2, f"Expected 2D Z, got shape {Z.shape}"
    if Z.shape[0] == 0:
        return True
    if Z.std(axis=0).max() < threshold:
        return True
    n = Z.shape[0]
    max_sub = 2000
    if n > max_sub:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_sub, replace=False)
        Z_sub = Z[idx]
    else:
        Z_sub = Z
    sq_dists = np.sum((Z_sub[:, None, :] - Z_sub[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(sq_dists, np.inf)
    median_sq_dist = float(np.median(sq_dists))
    if median_sq_dist < threshold ** 2:
        return True
    return False
