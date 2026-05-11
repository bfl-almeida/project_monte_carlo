"""Reproducible quantitative research experiments for Monte Carlo option pricing.

Each public function corresponds to one self-contained research question.  All
experiments use fixed random seeds and return tidy :class:`pandas.DataFrame`
objects suitable for publication-quality tables and log-log plots.

Research questions addressed
-----------------------------
1. :func:`run_convergence_experiment`
   How does MC pricing error scale with simulation budget *N*?

2. :func:`run_variance_reduction_experiment`
   What efficiency gain does antithetic sampling provide, adjusted for cost?

3. :func:`run_ci_coverage_experiment`
   Do the 95 % asymptotic CIs achieve nominal coverage across a parameter grid?

4. :func:`run_discretisation_bias_experiment`
   How does time-step resolution affect barrier option pricing bias?
"""

from __future__ import annotations
from collections import defaultdict

import time
from typing import Sequence

import numpy as np
import pandas as pd

from .black_scholes import bs_call_price, bs_put_price
from .monte_carlo import (
    MonteCarloResult,
    mc_barrier_option_price,
    mc_european_option_price,
)
from .utils import confidence_interval, estimate_convergence_rate


# ---------------------------------------------------------------------------
# Default parameter set used across experiments unless overridden
# ---------------------------------------------------------------------------

_BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)

_DEFAULT_PATH_GRID: list[int] = [
    500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000,
]

_DEFAULT_STEP_GRID: list[int] = [2, 4, 8, 16, 32, 64, 128, 252, 504]


# ---------------------------------------------------------------------------
# Experiment 1 — Convergence rate
# ---------------------------------------------------------------------------

def run_convergence_experiment(
    S0: float = _BASE["S0"],
    K: float = _BASE["K"],
    T: float = _BASE["T"],
    r: float = _BASE["r"],
    sigma: float = _BASE["sigma"],
    path_grid: list[int] = _DEFAULT_PATH_GRID,
    random_seed: int = 42,
    n_seeds: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Convergence of MC pricing error as a function of simulation budget N.

    For each N in *path_grid* the experiment runs **both** plain MC and
    antithetic MC and records price, absolute error, standard error, 95 % CI
    and runtime.  A log-log OLS regression over the full grid yields the
    empirical convergence rate β (theoretical: −0.5).

    When *n_seeds* > 1, the full N-grid is repeated for seeds
    ``random_seed, random_seed+1, …, random_seed+n_seeds−1``.  Each seed
    produces its own OLS slope; the reported ``conv_rate`` is the mean slope
    across seeds and ``conv_rate_std`` is its standard deviation, giving a
    robust, seed-averaged estimate of the convergence rate.

    Parameters
    ----------
    S0, K, T, r, sigma:
        Black-Scholes market parameters.
    path_grid:
        Ordered sequence of simulation sizes to sweep (should span at least
        two orders of magnitude for a meaningful regression).
    random_seed:
        Base seed.  When n_seeds == 1 this is the only seed used.
    n_seeds:
        Number of independent seeds to average over.  Higher values reduce
        the influence of any single lucky/unlucky draw sequence on the
        reported convergence rate.  20–50 seeds is sufficient in practice.
    alpha:
        Significance level for the asymptotic CI.

    Returns
    -------
    pd.DataFrame
        Columns: ``n_paths``, ``method``, ``mc_price``, ``bs_price``,
        ``abs_error``, ``rel_error_pct``, ``std_error``, ``ci_lower``,
        ``ci_upper``, ``runtime_s``, ``conv_rate``, ``conv_rate_std``
        (``conv_rate_std`` is NaN when n_seeds == 1).
    """
    benchmark = bs_call_price(S0, K, T, r, sigma)
    rows: list[dict] = []

    # Per-(method, n_paths) accumulator for seed-averaged price & error
    seed_errors: dict[tuple, list[float]] = defaultdict(list)
    seed_prices: dict[tuple, list[float]] = defaultdict(list)

    for seed_offset in range(n_seeds):
        seed = random_seed + seed_offset
        for antithetic, label in [(False, "standard"), (True, "antithetic")]:
            for n_paths in path_grid:
                t0 = time.perf_counter()
                res = mc_european_option_price(
                    S0=S0, K=K, T=T, r=r, sigma=sigma,
                    option_type="call",
                    n_paths=n_paths,
                    antithetic=antithetic,
                    random_seed=seed,
                )
                rt = time.perf_counter() - t0

                lo, hi = confidence_interval(res, alpha=alpha)
                abs_err = abs(res.price - benchmark)

                seed_errors[(label, n_paths)].append(abs_err)
                seed_prices[(label, n_paths)].append(res.price)

                rows.append({
                    "seed":          seed,
                    "n_paths":       n_paths,
                    "method":        label,
                    "mc_price":      res.price,
                    "bs_price":      benchmark,
                    "abs_error":     abs_err,
                    "rel_error_pct": abs_err / benchmark * 100.0,
                    "std_error":     res.standard_error,
                    "ci_lower":      lo,
                    "ci_upper":      hi,
                    "runtime_s":     rt,
                })

    df = pd.DataFrame(rows)

    # Compute seed-averaged convergence rate per method
    rates_mean: dict[str, float] = {}
    rates_std: dict[str, float] = {}

    for antithetic, label in [(False, "standard"), (True, "antithetic")]:
        per_seed_rates: list[float] = []
        for seed_offset in range(n_seeds):
            seed = random_seed + seed_offset
            seed_rows = df[(df["method"] == label) & (df["seed"] == seed)]
            rate = estimate_convergence_rate(
                seed_rows["n_paths"].tolist(),
                seed_rows["abs_error"].tolist(),
            )
            per_seed_rates.append(rate)
        rates_mean[label] = float(np.mean(per_seed_rates))
        rates_std[label]  = float(np.std(per_seed_rates, ddof=1)) if n_seeds > 1 else float("nan")

    df["conv_rate"]     = df["method"].map(rates_mean)
    df["conv_rate_std"] = df["method"].map(rates_std)

    return df


# ---------------------------------------------------------------------------
# Experiment 2 — Variance reduction effectiveness
# ---------------------------------------------------------------------------

def run_variance_reduction_experiment(
    S0: float = _BASE["S0"],
    K: float = _BASE["K"],
    T: float = _BASE["T"],
    r: float = _BASE["r"],
    sigma: float = _BASE["sigma"],
    path_grid: list[int] = _DEFAULT_PATH_GRID,
    n_replications: int = 50,
    base_seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Antithetic variates efficiency gain over repeated independent replications.

    For each N in *path_grid*, the experiment runs *n_replications* independent
    MC estimates for both plain and antithetic estimators (each replication uses
    a different seed derived from *base_seed*).  From these replications it
    computes:

    * empirical variance of the price estimator,
    * mean runtime per replication,
    * **variance reduction factor** VRF = Var_standard / Var_antithetic,
    * **work-normalised efficiency ratio** = VRF × (t_standard / t_antithetic).

    Parameters
    ----------
    n_replications:
        Number of independent MC runs per (N, method) cell.  More replications
        give a stable variance estimate; 50 is sufficient for relative errors
        below 10 %.
    base_seed:
        Seeds are set to ``base_seed + rep_index`` for each replication.

    Returns
    -------
    pd.DataFrame
        Columns: ``n_paths``, ``mean_price_standard``, ``var_standard``,
        ``mean_price_antithetic``, ``var_antithetic``, ``vrf``,
        ``mean_rt_standard``, ``mean_rt_antithetic``, ``efficiency_ratio``.
    """
    rows: list[dict] = []

    for n_paths in path_grid:
        prices_std, prices_anti = [], []
        times_std, times_anti = [], []

        for rep in range(n_replications):
            seed = base_seed + rep

            t0 = time.perf_counter()
            r_std = mc_european_option_price(
                S0=S0, K=K, T=T, r=r, sigma=sigma,
                option_type="call", n_paths=n_paths,
                antithetic=False, random_seed=seed,
            )
            times_std.append(time.perf_counter() - t0)
            prices_std.append(r_std.price)

            t0 = time.perf_counter()
            r_anti = mc_european_option_price(
                S0=S0, K=K, T=T, r=r, sigma=sigma,
                option_type="call", n_paths=n_paths,
                antithetic=True, random_seed=seed,
            )
            times_anti.append(time.perf_counter() - t0)
            prices_anti.append(r_anti.price)

        var_std  = float(np.var(prices_std,  ddof=1))
        var_anti = float(np.var(prices_anti, ddof=1))
        vrf = var_std / var_anti if var_anti > 0 else float("inf")

        mean_rt_std  = float(np.mean(times_std))
        mean_rt_anti = float(np.mean(times_anti))
        eff = vrf * (mean_rt_std / mean_rt_anti) if mean_rt_anti > 0 else float("inf")

        rows.append({
            "n_paths":              n_paths,
            "mean_price_standard":  float(np.mean(prices_std)),
            "var_standard":         var_std,
            "mean_price_antithetic":float(np.mean(prices_anti)),
            "var_antithetic":       var_anti,
            "vrf":                  vrf,
            "mean_rt_standard":     mean_rt_std,
            "mean_rt_antithetic":   mean_rt_anti,
            "efficiency_ratio":     eff,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 3 — Confidence interval coverage
# ---------------------------------------------------------------------------

def run_ci_coverage_experiment(
    param_grid: list[dict] | None = None,
    n_paths: int = 10_000,
    n_replications: int = 200,
    base_seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Empirical coverage of asymptotic CLT-based confidence intervals.

    For each parameter combination in *param_grid*, the experiment generates
    *n_replications* independent MC estimates and checks whether each CI
    contains the Black-Scholes price.  The **empirical coverage** is the
    fraction of CIs that succeed.  Nominal coverage is (1 − alpha).

    A well-calibrated estimator should have empirical coverage ≈ 95 % when
    alpha = 0.05.  Systematic deviations indicate non-normality of the
    estimator (small N) or model mis-specification.

    Parameters
    ----------
    param_grid:
        List of dicts with keys ``S0``, ``K``, ``T``, ``r``, ``sigma``.
        Defaults to a 6-point grid varying moneyness and volatility.
    n_paths:
        Simulation budget per replication (10 000 is sufficient for CLT).
    n_replications:
        Number of independent CI constructions per parameter combination.
    base_seed:
        Replication seeds are ``base_seed + rep``.
    alpha:
        Nominal significance level.

    Returns
    -------
    pd.DataFrame
        Columns: ``S0``, ``K``, ``T``, ``r``, ``sigma``, ``moneyness``,
        ``bs_price``, ``nominal_coverage``, ``empirical_coverage``,
        ``coverage_std_error``, ``runtime_total_s``, ``runtime_per_rep_s``.
    """
    if param_grid is None:
        param_grid = [
            dict(S0=100, K=90,  T=1.0, r=0.05, sigma=0.20),  # ITM
            dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.20),  # ATM
            dict(S0=100, K=110, T=1.0, r=0.05, sigma=0.20),  # OTM
            dict(S0=100, K=100, T=0.25, r=0.05, sigma=0.20), # short tenor
            dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.10),  # low vol
            dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.40),  # high vol
        ]

    rows: list[dict] = []

    for params in param_grid:
        S0, K, T, r, sigma = (
            params["S0"], params["K"], params["T"],
            params["r"], params["sigma"],
        )
        benchmark = bs_call_price(S0, K, T, r, sigma)
        moneyness = S0 / K

        hits = 0
        t_scenario_start = time.perf_counter()
        for rep in range(n_replications):
            res = mc_european_option_price(
                S0=S0, K=K, T=T, r=r, sigma=sigma,
                option_type="call", n_paths=n_paths,
                antithetic=False, random_seed=base_seed + rep,
            )
            lo, hi = confidence_interval(res, alpha=alpha)
            if lo <= benchmark <= hi:
                hits += 1
        runtime_total = time.perf_counter() - t_scenario_start

        emp_cov = hits / n_replications
        # Standard error of a proportion
        cov_se = np.sqrt(emp_cov * (1.0 - emp_cov) / n_replications)

        rows.append({
            "S0":                params["S0"],
            "K":                 params["K"],
            "T":                 params["T"],
            "r":                 params["r"],
            "sigma":             params["sigma"],
            "moneyness":         round(moneyness, 3),
            "bs_price":          benchmark,
            "nominal_coverage":  1.0 - alpha,
            "empirical_coverage":emp_cov,
            "coverage_std_error":float(cov_se),
            "runtime_total_s":   runtime_total,
            "runtime_per_rep_s": runtime_total / n_replications,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 4 — Discretisation bias in barrier options
# ---------------------------------------------------------------------------

def run_discretisation_bias_experiment(
    S0: float = _BASE["S0"],
    K: float = _BASE["K"],
    barrier: float = 120.0,
    T: float = _BASE["T"],
    r: float = _BASE["r"],
    sigma: float = _BASE["sigma"],
    step_grid: list[int] = _DEFAULT_STEP_GRID,
    n_paths: int = 100_000,
    antithetic: bool = True,
    random_seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Discretisation bias of barrier option prices as a function of n_steps.

    Path-dependent options are priced on a discrete time grid; the knock-out
    condition is checked only at grid points, so sparse grids *under-count*
    barrier crossings and **overstate** the option price.  As n_steps → ∞
    the discrete-monitoring price converges to the continuous-monitoring price.

    This experiment sweeps *step_grid* and records the estimated price, SE and
    CI at each resolution.  The **bias** is measured relative to the finest
    grid (used as a proxy for the continuous price).

    Parameters
    ----------
    barrier:
        Up-and-out barrier level (must be > S0).
    step_grid:
        Number of monitoring steps per year to sweep; spans from coarse (2) to
        daily (252) and sub-daily (504).
    n_paths:
        Simulation budget (constant across all step counts).

    Returns
    -------
    pd.DataFrame
        Columns: ``n_steps``, ``dt``, ``mc_price``, ``std_error``,
        ``ci_lower``, ``ci_upper``, ``runtime_s``, ``bias_vs_finest``.
    """
    rows: list[dict] = []

    for n_steps in step_grid:
        t0 = time.perf_counter()
        res = mc_barrier_option_price(
            S0=S0, K=K, barrier=barrier, T=T, r=r, sigma=sigma,
            option_type="call",
            barrier_type="up-and-out",
            n_paths=n_paths,
            n_steps=n_steps,
            antithetic=antithetic,
            random_seed=random_seed,
        )
        rt = time.perf_counter() - t0

        lo, hi = confidence_interval(res, alpha=alpha)
        rows.append({
            "n_steps":   n_steps,
            "dt":        T / n_steps,
            "mc_price":  res.price,
            "std_error": res.standard_error,
            "ci_lower":  lo,
            "ci_upper":  hi,
            "runtime_s": rt,
        })

    df = pd.DataFrame(rows)

    # Bias relative to the finest grid (last row)
    finest_price = df["mc_price"].iloc[-1]
    df["bias_vs_finest"] = df["mc_price"] - finest_price

    return df
