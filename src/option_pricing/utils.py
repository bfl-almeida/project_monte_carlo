import pandas as pd

from .black_scholes import bs_call_price
from .monte_carlo import mc_european_call_price


def convergence_table(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    path_grid: list[int],
    antithetic: bool = True,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """
    Build a convergence table comparing Monte Carlo estimates
    against the Black-Scholes analytical price.
    """
    benchmark = bs_call_price(S0, K, T, r, sigma)
    rows = []

    for n_paths in path_grid:
        mc_price, std_error = mc_european_call_price(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            n_paths=n_paths,
            antithetic=antithetic,
            random_seed=random_seed,
        )

        rows.append(
            {
                "n_paths": n_paths,
                "mc_price": mc_price,
                "bs_price": benchmark,
                "abs_error": abs(mc_price - benchmark),
                "std_error": std_error,
            }
        )

    return pd.DataFrame(rows)