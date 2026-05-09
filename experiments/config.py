"""Shared configuration constants, paths, and device selection."""

from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = _EXPERIMENTS_DIR / "figures"
RESULTS_DIR = _EXPERIMENTS_DIR / "results"
CACHE_DIR = _EXPERIMENTS_DIR / "cache"
DATA_RAW_DIR = _EXPERIMENTS_DIR / "data" / "raw"

RANDOM_SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_SEED = 42

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 999
ALPHA = 0.05
FDR_THRESHOLD = 0.05

BANDWIDTH_MULTIPLIERS = [0.5, 1.0, 2.0]

FIGURE_FORMATS = ["pdf", "png"]
FIGURE_DPI = 300
FIGURE_SIZE = (5.5, 4.0)  # NeurIPS single-column width


def get_device():
    """Return the best available torch device (MPS > CUDA > CPU)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return "cpu"


def ensure_dirs():
    """Create output directories. Call from experiment scripts, not at import."""
    for d in (FIGURES_DIR, RESULTS_DIR, CACHE_DIR, DATA_RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)
