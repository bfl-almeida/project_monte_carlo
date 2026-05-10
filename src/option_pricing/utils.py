"""Statistical utilities for Monte Carlo research experiments.

Provides helpers for confidence intervals, efficiency ratios, convergence
rate estimation, and structured convergence tables — building blocks shared
across all experiments in :mod:`option_pricing.experiments`.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .black_scholes import bs_call_price
from .monte_carlo import MonteCarloResult, mc_european_option_price


# ---------------------------------------------------------------------------
# Core statistical primitives
# ---------------------------------------------------------------------------

def confidence_interval(
    result: MonteCarloResult,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return a (1 − alpha) asymptotic confidence interval for a MC estimate.

    The interval is derived from the central limit theorem:

        price ± z_{1-alpha/2} * standard_error

    Parameters
    ----------
    result:
        A :class:`MonteCarloResult` containing a price estimate and its
        standard error (SE = sample std / sqrt(N)).
    alpha:
        Two-sided significance level; default 0.05 gives a 95 % CI.

    Returns
    -------
    tuple[float, float]
        (lower bound, upper bound) of the confidence interval.
    """
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    half_width = z * result.standard_error
    return (result.price - half_width, result.price + half_width)


def efficiency_ratio(
    baseline: MonteCarloResult,
    improved: MonteCarloResult,
    time_baseline: float,
    time_improved: float,
) -> float:
    """Variance-reduction efficiency ratio adjusted for compute cost.

    Defined as the ratio of *work-normalised* variances:

        efficiency = (Var_baseline * t_baseline) / (Var_improved * t_improved)

    A value > 1 means the improved estimator delivers more precision per unit
    of wall-clock time.  For antithetic variates on smooth payoffs the
    theoretical value approaches 2 when correlation ρ → −1.

    Parameters
    ----------
    baseline, improved:
        MC results for the two estimators being compared.
    time_baseline, time_improved:
        Wall-clock runtimes in seconds for the respective estimators.

    Returns
    -------
    float
        Work-normalised efficiency ratio.
    """
    var_baseline = baseline.standard_error ** 2
    var_improved = improved.standard_error ** 2
    if var_improved == 0.0:
        return float("inf")
    return (var_baseline * time_baseline) / (var_improved * time_improved)


def estimate_convergence_rate(
    n_grid: Sequence[int],
    errors: Sequence[float],
) -> float:
    """Estimate the empirical convergence rate β via OLS log-log regression.

    Fits the model  log(error) = β * log(N) + const  and returns β.
    For standard MC the theoretical value is β ≈ −0.5.

    Parameters
    ----------
    n_grid:
        Sequence of simulation sizes N (must be strictly positive).
    errors:
        Corresponding absolute errors |price_MC − price_BS|.

    Returns
    -------
    float
        Estimated slope β (expected to be close to −0.5 for standard MC).
    """
    log_n = np.log(np.asarray(n_grid, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    finite = np.isfinite(log_n) & np.isfinite(log_e)
    if finite.sum() < 2:
        return float("nan")
    x = log_n[finite]
    y = log_e[finite]
    # Manual OLS: slope = Cov(x,y) / Var(x) — avoids LAPACK entirely
    x_mean = x.mean()
    slope = float(np.dot(x - x_mean, y - y.mean()) / np.dot(x - x_mean, x - x_mean))
    return slope


# ---------------------------------------------------------------------------
# Structured convergence table
# ---------------------------------------------------------------------------

def convergence_table(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    path_grid: list[int],
    antithetic: bool = True,
    random_seed: int | None = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build a convergence table comparing MC estimates against the BS price.

    For each value of *N* in *path_grid* the function records:

    * ``n_paths``   — simulation budget
    * ``mc_price``  — Monte Carlo price estimate
    * ``bs_price``  — Black-Scholes analytical price (constant benchmark)
    * ``abs_error`` — |MC − BS|
    * ``rel_error`` — |MC − BS| / BS (in %)
    * ``std_error`` — standard error of the MC estimator (SE = σ / √N)
    * ``ci_lower``, ``ci_upper`` — asymptotic (1 − alpha) CI bounds
    * ``runtime_s`` — wall-clock time for that row's simulation (seconds)

    Parameters
    ----------
    S0, K, T, r, sigma:
        Black-Scholes market parameters.
    path_grid:
        Ordered list of simulation sizes to sweep.
    antithetic:
        Whether to use antithetic variates.
    random_seed:
        Fixed seed for reproducibility.
    alpha:
        Significance level for the confidence interval.

    Returns
    -------
    pd.DataFrame
        One row per element of *path_grid*.
    """
    benchmark = bs_call_price(S0, K, T, r, sigma)
    rows = []

    for n_paths in path_grid:
        t0 = time.perf_counter()
        result = mc_european_option_price(
            S0=S0, K=K, T=T, r=r, sigma=sigma,
            n_paths=n_paths,
            antithetic=antithetic,
            random_seed=random_seed,
        )
        runtime = time.perf_counter() - t0

        lo, hi = confidence_interval(result, alpha=alpha)
        abs_err = abs(result.price - benchmark)

        rows.append({
            "n_paths":   n_paths,
            "mc_price":  result.price,
            "bs_price":  benchmark,
            "abs_error": abs_err,
            "rel_error": abs_err / benchmark * 100.0 if benchmark != 0 else float("nan"),
            "std_error": result.standard_error,
            "ci_lower":  lo,
            "ci_upper":  hi,
            "runtime_s": runtime,
        })

    return pd.DataFrame(rows)