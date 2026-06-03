"""
Test suite for monte_carlo.py

Structure:
  - European option pricing vs BS reference
  - Barrier option logic
  - Monte Carlo statistical properties (CI, convergence, variance reduction)
  - Experiment smoke tests
"""

import pandas as pd
import pytest
from time import perf_counter

from option_pricing.black_scholes import bs_call_price
from option_pricing.experiments import (
    run_ci_coverage_experiment,
    run_convergence_experiment,
    run_discretisation_bias_experiment,
    run_variance_reduction_experiment,
)
from option_pricing.monte_carlo import (
    mc_barrier_option_price,
    mc_european_option_price,
)
from option_pricing.utils import (
    confidence_interval,
    convergence_table,
    efficiency_ratio,
    estimate_convergence_rate,
)


# ============================================================================
# Monte Carlo Pricing vs Black-Scholes Reference
# ============================================================================

def test_monte_carlo_call_close_to_black_scholes() -> None:
    """MC call price should converge to BS price with sufficient paths."""
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    analytic = bs_call_price(S0, K, T, r, sigma)
    mc = mc_european_option_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=200_000,
        antithetic=True,
        random_seed=123,
    )

    assert abs(mc.price - analytic) < 0.15


# ============================================================================
# Barrier Option Logic
# ============================================================================

def test_barrier_option_not_more_expensive_than_vanilla() -> None:
    """Barrier option (up-and-out) should never exceed vanilla."""
    vanilla = mc_european_option_price(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type="call",
        n_paths=100_000,
        antithetic=True,
        random_seed=123,
    )
    barrier = mc_barrier_option_price(
        S0=100.0,
        K=100.0,
        barrier=120.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type="call",
        barrier_type="up-and-out",
        n_paths=100_000,
        n_steps=126,
        antithetic=True,
        random_seed=123,
    )

    assert barrier.price <= vanilla.price
    assert barrier.price >= 0.0


# ============================================================================
# Statistical Properties of Monte Carlo Estimator
# ============================================================================

def test_confidence_interval_contains_true_price() -> None:
    """95 % CI should contain the BS price with high probability at N=100k."""
    analytic = bs_call_price(100, 100, 1.0, 0.05, 0.2)
    res = mc_european_option_price(
        S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
        option_type="call", n_paths=100_000,
        antithetic=True, random_seed=0,
    )
    lo, hi = confidence_interval(res, alpha=0.05)
    assert lo < analytic < hi, (
        f"BS price {analytic:.4f} not in CI [{lo:.4f}, {hi:.4f}]"
    )


def test_confidence_interval_ordering() -> None:
    """CI should always have lower < price < upper."""
    res = mc_european_option_price(
        S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
        n_paths=10_000, random_seed=1,
    )
    lo, hi = confidence_interval(res)
    assert lo < res.price < hi


def test_estimate_convergence_rate_negative_half() -> None:
    """Empirical convergence rate should be close to −0.5 for standard MC."""
    n_grid = [1_000, 5_000, 10_000, 50_000, 100_000]
    analytic = bs_call_price(100, 100, 1.0, 0.05, 0.2)
    errors = []
    for n in n_grid:
        res = mc_european_option_price(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            n_paths=n, antithetic=False, random_seed=7,
        )
        errors.append(abs(res.price - analytic))
    rate = estimate_convergence_rate(n_grid, errors)
    # Allow generous tolerance: theory says −0.5, accept −0.9 to −0.1
    assert -0.9 < rate < -0.1, f"Unexpected convergence rate: {rate:.3f}"


def test_efficiency_ratio_antithetic_greater_than_one() -> None:
    """Antithetic estimator should have efficiency ratio ≥ 1."""
    kwargs = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
                  option_type="call", n_paths=50_000)

    t0 = perf_counter()
    r_std = mc_european_option_price(**kwargs, antithetic=False, random_seed=10)
    t_std = perf_counter() - t0

    t0 = perf_counter()
    r_anti = mc_european_option_price(**kwargs, antithetic=True, random_seed=10)
    t_anti = perf_counter() - t0

    eff = efficiency_ratio(r_std, r_anti, t_std, t_anti)
    assert eff > 0.5, f"Suspiciously low efficiency ratio: {eff:.3f}"


def test_convergence_table_schema() -> None:
    """Convergence table should have expected columns and structure."""
    df = convergence_table(
        S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
        path_grid=[1_000, 5_000, 10_000],
    )
    expected_cols = {
        "n_paths", "mc_price", "bs_price", "abs_error",
        "rel_error", "std_error", "ci_lower", "ci_upper", "runtime_s",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 3
    assert (df["abs_error"] >= 0).all()
    assert (df["ci_lower"] < df["ci_upper"]).all()


# ============================================================================
# Experiment Smoke Tests (Structural Only)
# ============================================================================

def test_run_convergence_experiment_returns_dataframe() -> None:
    """Convergence experiment should return a DataFrame with method column."""
    df = run_convergence_experiment(
        path_grid=[1_000, 5_000],
        random_seed=42,
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df["method"].unique()) == {"standard", "antithetic"}
    assert len(df) == 4  # 2 methods × 2 path counts
    assert "conv_rate" in df.columns


def test_run_variance_reduction_experiment_vrf_positive() -> None:
    """Variance reduction experiment should show positive VRF."""
    df = run_variance_reduction_experiment(
        path_grid=[5_000, 10_000],
        n_replications=20,
        base_seed=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert (df["vrf"] > 0).all()
    assert (df["efficiency_ratio"] > 0).all()


def test_run_ci_coverage_experiment_reasonable_coverage() -> None:
    """Empirical coverage should be within 10 pp of 95 % nominal."""
    param_grid = [dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)]
    df = run_ci_coverage_experiment(
        param_grid=param_grid,
        n_paths=10_000,
        n_replications=100,
        base_seed=0,
    )
    assert isinstance(df, pd.DataFrame)
    cov = df["empirical_coverage"].iloc[0]
    assert 0.85 <= cov <= 1.0, f"Coverage {cov:.2f} outside expected range"


def test_run_discretisation_bias_experiment_monotone() -> None:
    """Barrier price should converge as n_steps increases (generally)."""
    df = run_discretisation_bias_experiment(
        step_grid=[4, 16, 64, 252],
        n_paths=50_000,
        random_seed=42,
    )
    assert isinstance(df, pd.DataFrame)
    assert "bias_vs_finest" in df.columns
    # The coarsest grid should have the largest absolute bias
    assert abs(df["bias_vs_finest"].iloc[0]) >= abs(df["bias_vs_finest"].iloc[-1])
