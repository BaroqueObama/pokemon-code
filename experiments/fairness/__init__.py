"""Fairness module -- re-exports for convenient imports."""

from experiments.fairness.criteria import (
    demographic_parity_gap,
    equalized_odds_gap,
    calibration_error,
    master_constraint_residual,
)
from experiments.fairness.interventions import apply_method
