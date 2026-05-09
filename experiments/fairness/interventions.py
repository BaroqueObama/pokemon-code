"""Fairness intervention wrappers.

Each method trains a classifier and returns (y_pred, y_proba) on the test set.
sklearn is used here because fairlearn requires sklearn estimators.

Score-level caveats:
- Hardt: y_proba is the base LR's probability (decisions-only method).
- EG: uses fairlearn's _pmf_predict; falls back to binary {0, 1} if unavailable.
- Reweighting: plain LR probabilities, cal_error is meaningful.

ExponentiatedGradient has no random_state param and consumes the global RNG,
so _seeded_numpy_rng is used for determinism.
"""

import warnings
from contextlib import contextmanager

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    DemographicParity,
    EqualizedOdds,
    ExponentiatedGradient,
)


@contextmanager
def _seeded_numpy_rng(seed: int):
    """Temporarily install a numpy seed; restore the previous state on exit.

    Fairlearn's ExponentiatedGradient does not accept random_state and
    consumes np.random globally during fit, so we have to guard it this way
    to make repeated calls with the same seed reproducible.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def apply_method(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    group_train: np.ndarray,
    X_test: np.ndarray,
    group_test: np.ndarray | None = None,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a fairness method and predict on test data.

    Parameters
    ----------
    name : one of 'unconstrained_lr', 'unconstrained_xgb', 'hardt_postprocessing',
           'platt_scaling', 'reweighting', 'exponentiated_gradient'
    group_test : required for methods that need group at predict time (e.g. Hardt)

    Returns
    -------
    (y_pred, y_pred_proba) : binary predictions and probability estimates on X_test
    """
    dispatch = {
        "unconstrained_lr": _unconstrained_lr,
        "unconstrained_xgb": _unconstrained_xgb,
        "hardt_postprocessing": _hardt_postprocessing,
        "platt_scaling": _platt_scaling,
        "reweighting": _reweighting,
        "exponentiated_gradient": _exponentiated_gradient,
    }
    assert name in dispatch, (
        f"Unknown method '{name}'. Available: {list(dispatch.keys())}"
    )

    return dispatch[name](
        X_train, y_train, group_train, X_test, group_test, seed, **kwargs
    )


def _unconstrained_lr(X_train, y_train, group_train, X_test, group_test, seed,
                      **kwargs):
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def _unconstrained_xgb(X_train, y_train, group_train, X_test, group_test, seed,
                       **kwargs):
    xgb = XGBClassifier(
        n_estimators=kwargs.get("n_estimators", 100),
        max_depth=kwargs.get("max_depth", 6),
        random_state=seed,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def _hardt_postprocessing(X_train, y_train, group_train, X_test, group_test, seed,
                          **kwargs):
    assert group_test is not None, (
        "Hardt postprocessing requires group_test at predict time"
    )
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train, y_train)

    to = ThresholdOptimizer(
        estimator=lr,
        constraints="equalized_odds",
        prefit=True,
        predict_method="predict_proba",
    )
    to.fit(X_train, y_train, sensitive_features=group_train)
    y_pred = to.predict(X_test, sensitive_features=group_test)

    y_proba = lr.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def _platt_scaling(X_train, y_train, group_train, X_test, group_test, seed,
                   **kwargs):
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    cal = CalibratedClassifierCV(estimator=lr, method="sigmoid", cv=5)
    cal.fit(X_train, y_train)
    y_pred = cal.predict(X_test)
    y_proba = cal.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def _reweighting(X_train, y_train, group_train, X_test, group_test, seed,
                 **kwargs):
    """Kamiran & Calders 2012 instance reweighting for demographic parity.

    Per-cell weight w(g, y) = P(G=g) * P(Y=y) / P(G=g, Y=y). Under these
    weights the (group, outcome) joint looks like the product of marginals,
    which removes the base-rate asymmetry driving DP gap. Plain LR with
    sample_weight is then fit on the reweighted training data.

    Unlike the previous implementation (which was ExponentiatedGradient in
    disguise), this returns continuous LR probabilities, so downstream
    calibration metrics are meaningful and the baseline is genuinely distinct
    from the EG(DP) sweep points.
    """
    group_train = np.asarray(group_train)
    y_train = np.asarray(y_train)
    n = len(y_train)
    assert len(group_train) == n, (
        f"group_train/y_train length mismatch: {len(group_train)} vs {n}"
    )

    p_g = {g: float((group_train == g).mean()) for g in (0, 1)}
    p_y = {y: float((y_train == y).mean()) for y in (0, 1)}
    sample_weight = np.empty(n, dtype=np.float64)
    for g in (0, 1):
        for y in (0, 1):
            mask = (group_train == g) & (y_train == y)
            p_gy = float(mask.mean())
            assert p_gy > 0, (
                f"Cell (G={g}, Y={y}) is empty — cannot compute K&C weight"
            )
            sample_weight[mask] = p_g[g] * p_y[y] / p_gy

    for g in (0, 1):
        mean_w = float(sample_weight[group_train == g].mean())
        assert abs(mean_w - 1.0) < 1e-9, (
            f"K&C weight group mean drift: group {g} mean_w={mean_w}, expected 1.0"
        )

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def _exponentiated_gradient(X_train, y_train, group_train, X_test, group_test, seed,
                            **kwargs):
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    constraint_name = kwargs.get("constraint", "equalized_odds")
    eps = kwargs.get("eps", 0.01)

    constraint_map = {
        "equalized_odds": EqualizedOdds(difference_bound=eps),
        "demographic_parity": DemographicParity(difference_bound=eps),
    }
    assert constraint_name in constraint_map, (
        f"Unknown constraint '{constraint_name}'. "
        f"Available: {list(constraint_map.keys())}"
    )

    eg = ExponentiatedGradient(
        estimator=lr,
        constraints=constraint_map[constraint_name],
    )
    with _seeded_numpy_rng(seed):
        eg.fit(X_train, y_train, sensitive_features=group_train)
        y_pred = eg.predict(X_test)
        try:
            y_proba = eg._pmf_predict(X_test)[:, 1]
        except (AttributeError, IndexError):
            warnings.warn(
                "_pmf_predict fallback triggered for exponentiated_gradient; "
                "returning binary y_proba -- downstream calibration will be "
                "masked to NaN.",
                stacklevel=2,
            )
            y_proba = y_pred.astype(np.float64)
    return y_pred, y_proba
