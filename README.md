# The Pokémon Theorem — Code Release

Code for [*The Pokémon Theorem and other Fairness Impossibility Results*](http://arxiv.org/abs/2605.09221) (submitted NeurIPS 2026). Reproduces the nine panels of Figure 1 on Adult Income, COMPAS, and ACS PUMS.

## Abstract

Fairness impossibility results often look like distinct scalar incompatibility statements. We show that several share one RKHS geometry: fairness criteria are linear constraints on conditional mean embeddings, and unequal base rates make the law of total expectation overdetermine those constraints.

This view yields four results. The Kleinberg--Mullainathan--Raghavan dichotomy needs only group-conditional unbiasedness, not full calibration. The *Pokémon theorem* shows that a distinct group pair satisfying any finite collection of linear mean-fairness criteria leaves a residual violation witnessed by the MMD, decaying at the Kolmogorov $m$-width rate under spectral regularity. The same tools prove an impossibility for fair feature learning: parity and class-conditional separation in representation space force class collapse under unequal base rates. The approximate relaxations yield signal and error frontiers, allowing a trade-off between real-world estimators and fairness goals. Experiments on standard fairness benchmarks are consistent with our bounds.

## Setup

```bash
uv sync
```

`pip install -e .` also works. Python ≥3.12 required; pinned dependencies in `uv.lock`. Datasets download on first run (~150 MB for ACS PUMS).

All seeds are passed explicitly to numpy, torch, and sklearn. MPS (Apple Silicon) doesn't support float64, so kernel matrices fall back to CPU on MPS. CUDA paths run in float64 throughout.

## Reproducing Figure 1

Run the three main experiments, then render the 3×3 grid:

```bash
uv run python -m experiments.exp3_pareto_frontier
uv run python -m experiments.exp4_spectral_analysis
uv run python -m experiments.exp5_fair_representations
uv run python -m experiments.make_paper_figures
```

| Figure 1 row | Script | Output |
|---|---|---|
| Top — m-width residual vs criterion budget | `exp4_spectral_analysis.py` | `figures/fig_mw_{dataset}.pdf` |
| Middle — separation-conditional Pareto bound | `exp3_pareto_frontier.py` | `figures/fig_pareto_{dataset}.pdf` |
| Bottom — forbidden corner (fair representations) | `exp5_fair_representations.py` | `figures/fig_fairrep_{dataset}.pdf` |

`make_paper_figures.py` reads the cached result JSONs in `experiments/results/` — it doesn't recompute anything.

`exp1_synthetic_validation.py` and `exp2_residual_unfairness.py` are auxiliary sanity checks (master constraint verification, residual unfairness on real data). They don't feed any paper figure but are included for completeness.

## Compute

Everything was run on an RTX 3070. Measured runtimes:

| Script | Time | Notes |
|---|---|---|
| `exp1_synthetic_validation.py` | ~43 min | pure CPU, synthetic data only |
| `exp2_residual_unfairness.py` | ~6 min | CPU-bound (Adult / COMPAS / ACS PUMS) |
| `exp3_pareto_frontier.py` | ~45 min | sklearn + fairlearn + xgboost |
| `exp4_spectral_analysis.py` | ~30 min | dominated by `eigh` on n=10,000 Gram matrices |
| `exp5_fair_representations.py` | ~4 hrs | Adult LFR is ~90 min sequential (scipy L-BFGS); Fair-VAE and adversarial use CUDA |

`config.get_device()` picks MPS / CUDA / CPU automatically. Five seeds `{42, 123, 456, 789, 1011}`, B=1000 bootstrap resamples, 999 permutations, all pinned in `config.py`.

## License

MIT (see `LICENSE`). Underlying datasets inherit their original licenses (UCI Adult: public domain; COMPAS: ProPublica; ACS PUMS: U.S. Census public-use).
