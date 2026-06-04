"""Research experiments and reproducible demos for Monte Carlo option pricing."""

from .experiments import (
    aggregate_greeks_experiment,
    run_ci_coverage_experiment,
    run_convergence_experiment,
    run_discretisation_bias_experiment,
    run_mc_greeks_experiment,
    run_variance_reduction_experiment,
)

__all__ = [
    "run_convergence_experiment",
    "run_variance_reduction_experiment",
    "run_ci_coverage_experiment",
    "run_discretisation_bias_experiment",
    "run_mc_greeks_experiment",
    "aggregate_greeks_experiment",
]
