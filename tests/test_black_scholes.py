"""
Test suite for black_scholes.py

Structure:
  - Exact reference values (quantitative): locked to known-good implementations
  - Sign and parity properties (qualitative): structural invariants, always true
  - Edge cases: T <= 0, sigma <= 0
"""

import math
import pytest
from scipy.stats import norm

from option_pricing.black_scholes import (
    bs_call_price, bs_put_price,
    bs_call_delta, bs_put_delta,
    bs_gamma, bs_vega,
    bs_call_theta, bs_put_theta,
    bs_call_rho, bs_put_rho,
)


# ============================================================================
# Reference Parameters
# ============================================================================

# Canonical reference point: ATM, 1 year, 5% rate, 20% vol
REF_PARAMS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


# ============================================================================
# Exact Reference Values (Quantitative Tests)
# ============================================================================
# These are computed via verified formulas and locked to prevent regressions.

def test_bs_call_price_exact_reference() -> None:
    """Call price at canonical reference point."""
    result = bs_call_price(**REF_PARAMS)
    # Computed via: S0*N(d1) - K*exp(-r*T)*N(d2) with d1=0.35, d2=0.15
    expected = 10.450583572185565  # scipy verified
    assert result == pytest.approx(expected, abs=1e-10)


def test_bs_put_price_exact_reference() -> None:
    """Put price at canonical reference point."""
    result = bs_put_price(**REF_PARAMS)
    # Computed via: K*exp(-r*T)*N(-d2) - S0*N(-d1) with d1=0.35, d2=0.15
    expected = 5.573526022256971
    assert result == pytest.approx(expected, abs=1e-10)


def test_bs_call_delta_exact_reference() -> None:
    """Delta for call at canonical point: N(d1)."""
    result = bs_call_delta(**REF_PARAMS)
    expected = norm.cdf(0.35)  # N(d1) where d1 = 0.35
    assert result == pytest.approx(expected, abs=1e-12)


def test_bs_put_delta_exact_reference() -> None:
    """Delta for put at canonical point: N(d1) - 1."""
    result = bs_put_delta(**REF_PARAMS)
    expected = norm.cdf(0.35) - 1.0
    assert result == pytest.approx(expected, abs=1e-12)


def test_bs_gamma_exact_reference() -> None:
    """Gamma at canonical point: N'(d1) / (S0 * sigma * sqrt(T))."""
    result = bs_gamma(**REF_PARAMS)
    expected = norm.pdf(0.35) / (100.0 * 0.20 * 1.0)
    assert result == pytest.approx(expected, abs=1e-12)


def test_bs_vega_exact_reference() -> None:
    """Vega at canonical point: S0 * N'(d1) * sqrt(T) / 100 (per 1% vol)."""
    result = bs_vega(**REF_PARAMS)
    expected = 100.0 * norm.pdf(0.35) * 1.0 / 100.0
    assert result == pytest.approx(expected, abs=1e-12)


def test_bs_call_theta_exact_reference() -> None:
    """Call theta at canonical point (per trading day)."""
    result = bs_call_theta(**REF_PARAMS)
    # theta_call = [-(S0*N'(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2)] / 252
    d1 = 0.35
    d2 = 0.15
    decay = -100.0 * norm.pdf(d1) * 0.20 / (2.0 * 1.0)
    carry = -0.05 * 100.0 * math.exp(-0.05 * 1.0) * norm.cdf(d2)
    expected = (decay + carry) / 252.0
    assert result == pytest.approx(expected, abs=1e-12)


def test_bs_put_theta_exact_reference() -> None:
    """Put theta at canonical point (per trading day)."""
    result = bs_put_theta(**REF_PARAMS)
    # theta_put = [-(S0*N'(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)] / 252
    d1 = 0.35
    d2 = 0.15
    decay = -100.0 * norm.pdf(d1) * 0.20 / (2.0 * 1.0)
    carry = 0.05 * 100.0 * math.exp(-0.05 * 1.0) * norm.cdf(-d2)
    expected = (decay + carry) / 252.0
    assert result == pytest.approx(expected, abs=1e-12)


def test_bs_call_rho_exact_reference() -> None:
    """Call rho at canonical point (per 1% rate move)."""
    result = bs_call_rho(**REF_PARAMS)
    # rho_call = K*T*exp(-r*T)*N(d2) / 100
    d2 = 0.15
    expected = 100.0 * 1.0 * math.exp(-0.05 * 1.0) * norm.cdf(d2) / 100.0
    assert result == pytest.approx(expected, abs=1e-12)


def test_bs_put_rho_exact_reference() -> None:
    """Put rho at canonical point (per 1% rate move)."""
    result = bs_put_rho(**REF_PARAMS)
    # rho_put = -K*T*exp(-r*T)*N(-d2) / 100
    d2 = 0.15
    expected = -100.0 * 1.0 * math.exp(-0.05 * 1.0) * norm.cdf(-d2) / 100.0
    assert result == pytest.approx(expected, abs=1e-12)


# ============================================================================
# Price Parity (Qualitative / Structural)
# ============================================================================

def test_bs_put_call_parity() -> None:
    """Put-call parity: C - P = S0 - K*exp(-r*T)."""
    call = bs_call_price(**REF_PARAMS)
    put = bs_put_price(**REF_PARAMS)
    lhs = call - put
    rhs = REF_PARAMS["S0"] - REF_PARAMS["K"] * math.exp(-REF_PARAMS["r"] * REF_PARAMS["T"])
    assert lhs == pytest.approx(rhs, abs=1e-10)


# ============================================================================
# Greek Sign Properties (Qualitative / Parametrized)
# ============================================================================

@pytest.mark.parametrize("S0,K,T,r,sigma", [
    (100.0, 100.0, 1.0, 0.05, 0.20),  # ATM
    (100.0, 120.0, 1.0, 0.05, 0.20),  # OTM call
    (100.0, 80.0, 1.0, 0.05, 0.20),   # ITM call
])
def test_bs_gamma_always_positive(S0, K, T, r, sigma) -> None:
    """Gamma is strictly positive for all inputs (same for call and put)."""
    assert bs_gamma(S0, K, T, r, sigma) > 0.0


@pytest.mark.parametrize("S0,K,T,r,sigma", [
    (100.0, 100.0, 1.0, 0.05, 0.20),
    (100.0, 120.0, 1.0, 0.05, 0.20),
    (100.0, 80.0, 1.0, 0.05, 0.20),
])
def test_bs_vega_always_positive(S0, K, T, r, sigma) -> None:
    """Vega is strictly positive for all inputs (same for call and put)."""
    assert bs_vega(S0, K, T, r, sigma) > 0.0


@pytest.mark.parametrize("S0,K,T,r,sigma", [
    (100.0, 100.0, 1.0, 0.05, 0.20),
    (100.0, 120.0, 1.0, 0.05, 0.20),
    (100.0, 80.0, 1.0, 0.05, 0.20),
])
def test_bs_call_theta_negative(S0, K, T, r, sigma) -> None:
    """Call theta is negative (long call loses to time decay)."""
    assert bs_call_theta(S0, K, T, r, sigma) < 0.0


@pytest.mark.parametrize("S0,K,T,r,sigma", [
    (100.0, 100.0, 1.0, 0.05, 0.20),  # ATM
    (100.0, 80.0, 1.0, 0.05, 0.20),   # ITM (S0 > K, so put is ITM)
])
def test_bs_put_theta_negative(S0, K, T, r, sigma) -> None:
    """Put theta is negative for ATM and ITM (long put loses to time decay). 
    OTM puts can have positive theta due to carry term dominating."""
    assert bs_put_theta(S0, K, T, r, sigma) < 0.0


# ============================================================================
# Delta Parity: call_delta - put_delta = 1.0
# ============================================================================

@pytest.mark.parametrize("S0,K,T,r,sigma", [
    (100.0, 100.0, 1.0, 0.05, 0.20),
    (100.0, 80.0, 0.5, 0.02, 0.30),
    (100.0, 120.0, 2.0, 0.01, 0.15),
])
def test_bs_delta_parity(S0, K, T, r, sigma) -> None:
    """Put-call delta parity: call_delta - put_delta = 1.0 exactly."""
    diff = bs_call_delta(S0, K, T, r, sigma) - bs_put_delta(S0, K, T, r, sigma)
    assert diff == pytest.approx(1.0, abs=1e-12)


# ============================================================================
# Edge Cases: T <= 0, sigma <= 0
# ============================================================================

def test_bs_greeks_expired_option_t_zero() -> None:
    """All Greeks return 0 when T <= 0 (option has expired)."""
    S0, K, r = 100.0, 100.0, 0.05
    assert bs_gamma(S0, K, 0.0, r, 0.20) == 0.0
    assert bs_vega(S0, K, 0.0, r, 0.20) == 0.0
    assert bs_call_theta(S0, K, 0.0, r, 0.20) == 0.0
    assert bs_put_theta(S0, K, 0.0, r, 0.20) == 0.0
    assert bs_call_rho(S0, K, 0.0, r, 0.20) == 0.0
    assert bs_put_rho(S0, K, 0.0, r, 0.20) == 0.0


def test_bs_greeks_zero_vol() -> None:
    """Gamma and Vega return 0 when sigma <= 0 (no randomness)."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    assert bs_gamma(S0, K, T, r, 0.0) == 0.0
    assert bs_vega(S0, K, T, r, 0.0) == 0.0


def test_bs_call_delta_zero_vol_step_function() -> None:
    """Call delta becomes a step function when sigma → 0."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    # ITM call: delta → 1
    assert bs_call_delta(120.0, K, T, r, 0.0) == 1.0
    # OTM call: delta → 0
    assert bs_call_delta(80.0, K, T, r, 0.0) == 0.0


def test_bs_put_delta_zero_vol_step_function() -> None:
    """Put delta becomes a step function when sigma → 0."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    # ITM put: delta → -1
    assert bs_put_delta(80.0, K, T, r, 0.0) == -1.0
    # OTM put: delta → 0
    assert bs_put_delta(120.0, K, T, r, 0.0) == 0.0


# ============================================================================
# ATM Delta
# ============================================================================

def test_bs_call_delta_atm_short_dated() -> None:
    """At-the-money call delta ≈ 0.5 for short-dated options (d1 → 0)."""
    S0 = K = 100.0
    T = 1.0 / 365  # 1 day: drift term negligible, d1 ≈ 0
    r = 0.05
    sigma = 0.20
    delta = bs_call_delta(S0, K, T, r, sigma)
    assert delta == pytest.approx(0.5, abs=0.01)
