from .experiments import (
    run_ci_coverage_experiment,
    run_convergence_experiment,
    run_discretisation_bias_experiment,
    run_variance_reduction_experiment,
)
from .utils import (
    confidence_interval,
    convergence_table,
    efficiency_ratio,
    estimate_convergence_rate,
)
from .black_scholes import bs_call_price, bs_put_price
from .monte_carlo import (
    MonteCarloResult,
    mc_barrier_option_price,
    mc_european_option_price,
    simulate_price_paths,
    simulate_terminal_price,
)

__all__ = [
    # Black-Scholes
    "bs_call_price",
    "bs_put_price",
    # Monte Carlo engine
    "MonteCarloResult",
    "simulate_terminal_price",
    "simulate_price_paths",
    "mc_european_option_price",
    "mc_barrier_option_price",
    # Statistical utilities
    "confidence_interval",
    "efficiency_ratio",
    "estimate_convergence_rate",
    "convergence_table",
    # Research experiments
    "run_convergence_experiment",
    "run_variance_reduction_experiment",
    "run_ci_coverage_experiment",
    "run_discretisation_bias_experiment",
]
