"""Fairness metric computation.

All criteria expressed as inner products with the mean-embedding gap delta,
matching the theoretical framework in the paper.

Group convention: group=1 is 'a' (privileged), group=0 is 'b'.
"""

import numpy as np


def demographic_parity_gap(y_pred: np.ndarray, group: np.ndarray) -> float:
    """Absolute difference in positive prediction rates between groups.

    |E[Y_hat | G=a] - E[Y_hat | G=b]|  =  |<w, delta>|
    """
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    assert len(y_pred) == len(group), (
        f"Length mismatch: y_pred={len(y_pred)}, group={len(group)}"
    )
    assert set(np.unique(group)).issubset({0, 1}), (
        f"group must be binary, got values {np.unique(group)}"
    )
    n_a = (group == 1).sum()
    n_b = (group == 0).sum()
    assert n_a > 0 and n_b > 0, f"Both groups must be non-empty: n_a={n_a}, n_b={n_b}"

    rate_a = y_pred[group == 1].mean()
    rate_b = y_pred[group == 0].mean()
    return float(abs(rate_a - rate_b))


def equalized_odds_gap(
    y_pred: np.ndarray, y_true: np.ndarray, group: np.ndarray
) -> float:
    """Max of |TPR gap| and |FPR gap| between groups.

    Separation:  <w, delta_1> = 0  AND  <w, delta_0> = 0
    We return max(|<w, delta_1>|, |<w, delta_0>|).
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    group = np.asarray(group)
    assert len(y_pred) == len(y_true) == len(group)

    tpr = {}
    fpr = {}
    for g, label in [(1, "a"), (0, "b")]:
        pos_mask = (group == g) & (y_true == 1)
        neg_mask = (group == g) & (y_true == 0)
        assert pos_mask.sum() > 0, (
            f"Group {label} has no positive examples (need at least 1)"
        )
        assert neg_mask.sum() > 0, (
            f"Group {label} has no negative examples (need at least 1)"
        )
        tpr[label] = y_pred[pos_mask].mean()
        fpr[label] = y_pred[neg_mask].mean()

    tpr_gap = abs(tpr["a"] - tpr["b"])
    fpr_gap = abs(fpr["a"] - fpr["b"])
    return float(max(tpr_gap, fpr_gap))


def calibration_error(
    y_pred_proba: np.ndarray, y_true: np.ndarray, group: np.ndarray
) -> dict[str, float]:
    """Group-conditional calibration error (unbiasedness).

    Per group: |E[S | G=g] - E[Y | G=g]|
    This measures condition C (sufficiency) from the theory.

    Returns {'a': err_a, 'b': err_b}.
    """
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    y_true = np.asarray(y_true)
    group = np.asarray(group)
    assert len(y_pred_proba) == len(y_true) == len(group)

    result = {}
    for g, label in [(1, "a"), (0, "b")]:
        mask = group == g
        assert mask.sum() > 0, f"Group {label} is empty"
        err = abs(y_pred_proba[mask].mean() - y_true[mask].mean())
        result[label] = float(err)

    return result


def master_constraint_residual(
    scores: np.ndarray, y_true: np.ndarray, group: np.ndarray
) -> float:
    """Verify the projected master constraint (star_w) on classifier scores.

    The identity (law of total expectation):
        E[S|G=a] - E[S|G=b]  =  p_a*(E[S|Y=1,G=a] - E[S|Y=1,G=b])
                                + (1-p_a)*(E[S|Y=0,G=a] - E[S|Y=0,G=b])
                                + (p_a - p_b)*(E[S|Y=1,G=b] - E[S|Y=0,G=b])

    Returns |LHS - RHS|, which should be ~0 (algebraic identity).
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
