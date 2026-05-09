"""Utilities -- re-exports for convenient imports."""

from experiments.utils.stats import bootstrap_ci, permutation_test, bh_correction
from experiments.utils.plotting import (
    setup_style,
    save_figure,
    add_error_bars,
    GROUP_COLORS,
    THEORY_STYLE,
    EMPIRICAL_STYLE,
)
