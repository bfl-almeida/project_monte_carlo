from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]
BarrierType = Literal["up-and-out", "down-and-out"]


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float



def _build_normal_draws(
    n_paths: int,
    n_steps: int,
    antithetic: bool,
    random_seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        draws = np.vstack([z, -z])
        if draws.shape[0] < n_paths:
            extra = rng.standard_normal((1, n_steps))
            draws = np.vstack([draws, extra])
        return draws[:n_paths]

    return rng.standard_normal((n_paths, n_steps))



def simulate_terminal_price(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    antithetic: bool = False,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal prices under geometric Brownian motion."""
    z = _build_normal_draws(
        n_paths=n_paths,
        n_steps=1,
        antithetic=antithetic,
        random_seed=random_seed,
    ).reshape(-1)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)



def mc_european_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    n_paths: int = 10_000,
    antithetic: bool = False,
    random_seed: int | None = None,
) -> MonteCarloResult:
    """Monte Carlo price for a European call or put option."""
    ST = simulate_terminal_price(
        S0=S0,
        T=T,
        r=r,
        sigma=sigma,
        n_paths=n_paths,
        antithetic=antithetic,
        random_seed=random_seed,
    )

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    standard_error = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    return MonteCarloResult(price=price, standard_error=standard_error)



def simulate_price_paths(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    antithetic: bool = False,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate full geometric Brownian motion paths."""
    dt = T / n_steps
    z = _build_normal_draws(
        n_paths=n_paths,
        n_steps=n_steps,
        antithetic=antithetic,
        random_seed=random_seed,
    )

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return S0 * np.exp(log_paths)



def mc_barrier_option_price(
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    barrier_type: BarrierType = "up-and-out",
    n_paths: int = 20_000,
    n_steps: int = 252,
    antithetic: bool = False,
    random_seed: int | None = None,
) -> MonteCarloResult:
    """Monte Carlo price for simple knock-out barrier options."""
    paths = simulate_price_paths(
        S0=S0,
        T=T,
        r=r,
        sigma=sigma,
        n_paths=n_paths,
        n_steps=n_steps,
        antithetic=antithetic,
        random_seed=random_seed,
    )
    ST = paths[:, -1]

    if option_type == "call":
        vanilla_payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        vanilla_payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if barrier_type == "up-and-out":
        knocked_out = np.any(paths >= barrier, axis=1)
    elif barrier_type == "down-and-out":
        knocked_out = np.any(paths <= barrier, axis=1)
    else:
        raise ValueError("barrier_type must be 'up-and-out' or 'down-and-out'")

    payoffs = np.where(knocked_out, 0.0, vanilla_payoff)
    discounted = np.exp(-r * T) * payoffs

    price = float(np.mean(discounted))
    standard_error = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    return MonteCarloResult(price=price, standard_error=standard_error)
