![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# Monte Carlo Methods for Derivative Pricing and Variance Reduction

## Overview

This repository contains a self-directed quantitative finance project implementing Monte Carlo option pricing under the Black-Scholes framework. It includes analytical benchmarks, stochastic simulation, confidence intervals, convergence analysis, variance reduction techniques, Greeks estimation and barrier option pricing.

The goal is to demonstrate practical skills relevant to quantitative finance roles: derivatives pricing, numerical methods, statistical validation, model risk analysis and Python-based quantitative tooling.

## Features

- Analytical Black-Scholes pricing for European calls and puts
- Monte Carlo pricing for European options
- Confidence intervals and convergence analysis
- Variance reduction with antithetic variates
- Greeks estimation using analytical formulas and finite differences — in progress
- Barrier option pricing — in progress
- Unit tests with pytest — in progress

## Why this project matters

This project demonstrates core building blocks of quantitative finance: stochastic simulation, derivatives pricing, numerical convergence, statistical confidence intervals, variance reduction and model validation against analytical benchmarks.

It is designed as an educational quantitative finance library, not as a production front-office pricing system.

## Research Questions

1. **Convergence** — How does the MC pricing error scale with simulation budget *N*?
   Does the observed *O(N^{-1/2})* rate hold in practice?
2. **Variance Reduction** — How much does antithetic sampling reduce estimator variance
   for European and barrier options, and what is the efficiency gain per unit of compute?
3. **Confidence Interval Coverage** — Do the 95 % asymptotic CIs based on the CLT achieve
   their nominal coverage across a realistic parameter grid?
4. **Discretisation Bias in Barrier Options** — How does path resolution (number of time
   steps) affect the knock-out probability and the resulting pricing bias?

## Methods Implemented

| Method | Description |
|---|---|
| Black-Scholes (analytical) | Closed-form price for European calls and puts |
| Standard Monte Carlo | i.i.d. GBM terminal-price simulation |
| Antithetic Variates | Paired ±Z draws; cuts variance roughly in half for smooth payoffs |
| Path Simulation | Full GBM path discretisation for path-dependent contracts |
| Barrier Options | Up-and-out / down-and-out knock-out payoffs |

## Tech Stack

- Python ≥ 3.10
- NumPy — vectorised simulation
- SciPy — normal CDF, statistical utilities
- Pandas — structured experiment outputs
- Matplotlib — convergence and bias plots
- pytest — reproducibility tests

## Project Structure

```text
monte-carlo-option-pricing/
├─ pyproject.toml
├─ README.md
├─ .gitignore
├─ src/
│  └─ option_pricing/
│     ├─ __init__.py
│     ├─ black_scholes.py      # Analytical BS prices and Greeks
│     ├─ monte_carlo.py        # Simulation engine (European + barrier)
│     ├─ experiments.py        # Reproducible research experiments
│     └─ utils.py              # Statistical helpers, convergence table
├─ tests/
│  └─ test_pricing.py
├─ notebooks/
│  └─ research_demo.ipynb
└─ reports/
   ├─ figures/
   └─ tables/
```

## Quickstart

```python
from option_pricing import mc_european_option_price, bs_call_price
from option_pricing.experiments import run_convergence_experiment

# Analytical benchmark
price = bs_call_price(S0=100, K=100, T=1, r=0.05, sigma=0.2)

# Monte Carlo estimate with antithetic variates
result = mc_european_option_price(
    S0=100, K=100, T=1, r=0.05, sigma=0.2,
    option_type="call", n_paths=100_000, antithetic=True, random_seed=42,
)
print(f"MC price: {result.price:.4f}  SE: {result.standard_error:.4f}")

# Convergence experiment
df = run_convergence_experiment()
print(df.to_string(index=False))
```

## Key Results (illustrative)

- Antithetic variates achieve roughly **2× variance reduction** for at-the-money European
  calls, consistent with the theoretical prediction for smooth payoffs.
- The empirical convergence rate is close to the theoretical *O(N^{-1/2})*, confirmed by
  log-log regression on absolute error vs. *N*.
- Barrier option prices exhibit a systematic **upward bias** when using few time steps,
  converging to the fine-grid estimate as *n_steps → ∞*.
