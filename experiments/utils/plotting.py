"""NeurIPS-style plotting utilities."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from experiments.config import FIGURE_DPI, FIGURE_FORMATS, FIGURE_SIZE, FIGURES_DIR

GROUP_COLORS = {"a": "#1f77b4", "b": "#ff7f0e"}  # tab:blue, tab:orange
THEORY_STYLE = {"linestyle": "--", "color": "#7f7f7f", "linewidth": 1.5}
EMPIRICAL_STYLE = {"marker": "o", "markersize": 5, "capsize": 3}


def setup_style() -> None:
    """Configure matplotlib rcParams for NeurIPS figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.labelcolor": "black",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.fontsize": 9,
            "figure.figsize": FIGURE_SIZE,
            "figure.dpi": 150,
            "savefig.dpi": FIGURE_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "black",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "text.color": "black",
            "text.usetex": False,
        }
    )


def save_figure(
    fig: plt.Figure,
    name: str,
    formats: list[str] | None = None,
    dpi: int | None = None,
) -> list[Path]:
    """Save figure to FIGURES_DIR in all requested formats and return paths."""
    if formats is None:
        formats = FIGURE_FORMATS
    if dpi is None:
        dpi = FIGURE_DPI

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def add_error_bars(
    ax: plt.Axes,
    x,
    y,
    ci_lower,
    ci_upper,
    **kwargs,
) -> None:
    """Plot data points with error bars using EMPIRICAL_STYLE defaults."""
    style = {**EMPIRICAL_STYLE, **kwargs}
    capsize = style.pop("capsize", 3)
    markersize = style.pop("markersize", 5)

    yerr_lower = [yi - lo for yi, lo in zip(y, ci_lower)]
    yerr_upper = [hi - yi for yi, hi in zip(y, ci_upper)]

    ax.errorbar(
        x,
        y,
        yerr=[yerr_lower, yerr_upper],
        capsize=capsize,
        markersize=markersize,
        **style,
    )
