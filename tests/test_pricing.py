import math

from option_pricing.black_scholes import bs_call_price, bs_put_price
from option_pricing.monte_carlo import (
    mc_barrier_option_price,
    mc_european_option_price,
)



def test_black_scholes_put_call_parity() -> None:
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    call = bs_call_price(S0, K, T, r, sigma)
    put = bs_put_price(S0, K, T, r, sigma)

    lhs = call - put
    rhs = S0 - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10



def test_monte_carlo_call_close_to_black_scholes() -> None:
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



def test_barrier_option_not_more_expensive_than_vanilla() -> None:
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
