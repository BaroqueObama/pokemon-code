"""Data module -- shared FairnessDataset type and loaders."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FairnessDataset:
    """Container for a binary-classification fairness dataset.

    Group convention (must be consistent everywhere):
        group = 1  ->  group 'a' (privileged / higher base rate)
        group = 0  ->  group 'b'
    """

    X: np.ndarray  # (n, d) float64 features
    y: np.ndarray  # (n,) int {0, 1} labels
    group: np.ndarray  # (n,) int {0, 1} group membership
    feature_names: list[str]
    name: str
    base_rates: dict[str, float]  # {'a': p_a, 'b': p_b}
