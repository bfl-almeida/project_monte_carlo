from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal: N(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Probability density function of the standard normal: N'(x)."""
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price for a European call option."""
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return math.exp(-r * T) * max(S0 * math.exp(r * T) - K, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price for a European put option."""
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        return math.exp(-r * T) * max(K - S0 * math.exp(r * T), 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)


def bs_call_delta(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Delta for a European call option: N(d1)."""
    if T <= 0:
        return 1.0 if S0 > K else 0.0
    if sigma <= 0:
        return 1.0 if S0 > K * math.exp(-r * T) else 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def bs_put_delta(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Delta for a European put option: N(d1) - 1."""
    if T <= 0:
        return -1.0 if S0 < K else 0.0
    if sigma <= 0:
        return -1.0 if S0 < K * math.exp(-r * T) else 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) - 1.0


def bs_gamma(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Gamma for a European option (identical for call and put): N'(d1) / (S0 * sigma * sqrt(T))."""
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _norm_pdf(d1) / (S0 * sigma * math.sqrt(T))


def bs_vega(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Vega for a European option (identical for call and put): S0 * N'(d1) * sqrt(T), per 1% move in vol."""
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S0 * _norm_pdf(d1) * math.sqrt(T) / 100.0


def bs_call_theta(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Theta for a European call option, per calendar day: -(S0*N'(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-rT)*N(d2), divided by 365."""
    if T <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    decay = -S0 * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    carry = -r * K * math.exp(-r * T) * _norm_cdf(d2)
    return (decay + carry) / 365.0


def bs_put_theta(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Theta for a European put option, per calendar day: -(S0*N'(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-rT)*N(-d2), divided by 365."""
    if T <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    decay = -S0 * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    carry = r * K * math.exp(-r * T) * _norm_cdf(-d2)
    return (decay + carry) / 365.0


def bs_call_rho(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Rho for a European call option, per 1% move in rates: K*T*exp(-rT)*N(d2), divided by 100."""
    if T <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * T * math.exp(-r * T) * _norm_cdf(d2) / 100.0


def bs_put_rho(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Rho for a European put option, per 1% move in rates: -K*T*exp(-rT)*N(-d2), divided by 100."""
    if T <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100.0
