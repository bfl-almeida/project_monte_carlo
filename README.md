![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/bfl-almeida/project_monte_carlo/actions/workflows/test.yml/badge.svg)

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

## Key Results

All experiments are fully reproducible via `src/option_pricing/experiments.py`.
Base parameters unless noted: S₀ = K = 100, T = 1 yr, r = 5 %, σ = 20 %.
Analytical benchmark (Black-Scholes call): **10.4506**.

---

### 1 · Convergence of MC Pricing Error

To verify the theoretical *O(N^{−1/2})* convergence rate, both the standard and antithetic estimators
were run across nine simulation budgets spanning N = 500 to N = 250 000. For each budget, the absolute
pricing error |MC price − BS price| was recorded and a log-log OLS regression of error vs. N was
fitted to obtain an empirical slope β. Because a single draw sequence can produce an artificially steep
or shallow slope depending on the particular random numbers drawn, this procedure was repeated across
50 independent seeds (42 through 91) and the slopes were averaged.

The seed-averaged empirical convergence rates are **−0.46 ± 0.25** for standard MC and **−0.45 ± 0.23**
for the antithetic estimator (mean ± std of OLS slopes across seeds), both in close agreement with the
theoretical value of β = −0.50. The standard deviation of ≈ 0.24 across seeds confirms that
single-seed rate estimates carry substantial noise and should not be reported without averaging.

All values are mean ± std across the 50 seeds. The ± on MC Price reflects estimator variability across
seeds; the ± on |Error| reflects how much the absolute pricing error fluctuates from one draw sequence
to another — narrowing predictably as N grows.

| N | Method | MC Price (mean ± std) | \|Error\| (mean ± std) | Rel Error % | MC Std Error |
|--:|:-------|----------------------:|----------------------:|------------:|-------------:|
| 1 000 | Standard | 10.4547 ± 0.4274 | 0.3541 ± 0.2340 | 3.39 | 0.4655 |
| 1 000 | Antithetic | 10.3420 ± 0.3398 | 0.2762 ± 0.2229 | 2.64 | 0.4594 |
| 10 000 | Standard | 10.4578 ± 0.1486 | 0.1155 ± 0.0923 | 1.11 | 0.1471 |
| 10 000 | Antithetic | 10.4249 ± 0.0951 | 0.0738 ± 0.0644 | 0.71 | 0.1468 |
| 100 000 | Standard | 10.4557 ± 0.0531 | 0.0458 ± 0.0267 | 0.44 | 0.0466 |
| 100 000 | Antithetic | 10.4497 ± 0.0360 | 0.0299 ± 0.0195 | 0.29 | 0.0465 |
| 250 000 | Standard | 10.4491 ± 0.0295 | 0.0228 ± 0.0184 | 0.22 | 0.0294 |
| 250 000 | Antithetic | 10.4551 ± 0.0246 | 0.0205 ± 0.0141 | 0.20 | 0.0294 |

*MC Std Error is the within-simulation standard error (SE = sample std / √N), averaged across seeds.*

---

### 2 · Variance Reduction — Antithetic Variates

Seeds `0 … 49` (50 independent replications per N). VRF = Var(standard) / Var(antithetic). Efficiency ratio adjusts VRF for actual compute time.

| N | Var (Standard) | Var (Antithetic) | VRF | Efficiency Ratio |
|--:|---------------:|-----------------:|----:|-----------------:|
| 500 | 0.4866 | 0.1834 | 2.65× | 2.81× |
| 1 000 | 0.2371 | 0.0819 | 2.90× | 2.72× |
| 2 000 | 0.1139 | 0.0364 | 3.13× | 3.33× |
| 5 000 | 0.0568 | 0.0113 | 5.04× | 5.79× |
| 10 000 | 0.0240 | 0.0084 | 2.84× | 3.36× |
| 25 000 | 0.0086 | 0.0032 | 2.66× | 3.54× |
| 50 000 | 0.0036 | 0.0024 | 1.53× | 2.00× |
| 100 000 | 0.0016 | 0.0014 | 1.17× | 1.33× |
| 250 000 | 0.0008 | 0.0005 | 1.78× | 1.89× |

**Median VRF: 2.66×  ·  Median efficiency ratio: 2.92×**

---

### 3 · Confidence Interval Coverage

Seeds `0 … 199` (200 independent replications per scenario, N = 10 000 paths each). Nominal coverage: 95 %.

| Scenario | S₀ / K | σ | T | BS Price | Empirical Coverage |
|:---------|-------:|--:|--:|---------:|------------------:|
| ITM (K = 90) | 1.11 | 20 % | 1.00 yr | 16.6994 | 91.5 % |
| ATM (K = 100) | 1.00 | 20 % | 1.00 yr | 10.4506 | 91.5 % |
| OTM (K = 110) | 0.91 | 20 % | 1.00 yr | 6.0401 | 93.0 % |
| Short tenor | 1.00 | 20 % | 0.25 yr | 4.6150 | 92.5 % |
| Low vol (σ = 10 %) | 1.00 | 10 % | 1.00 yr | 6.8050 | 91.0 % |
| High vol (σ = 40 %) | 1.00 | 40 % | 1.00 yr | 18.0230 | 92.0 % |

*Empirical coverage of 91–93 % is slightly below the 95 % nominal. This is expected for asymptotic CLT-based CIs at N = 10 000 — coverage converges to 95 % as N grows.*

---

### 4 · Discretisation Bias — Up-and-Out Barrier Option

`seed=42`. S₀ = 100, K = 100, barrier B = 120, T = 1 yr, r = 5 %, σ = 20 %. N = 100 000 antithetic paths.
Bias is measured against the finest grid (n_steps = 504, proxy for continuous monitoring).
Coarse grids miss barrier crossings and overstate the price — by up to **+1.32** (+103 %) at n_steps = 2.

| n\_steps | dt | MC Price | 95 % CI | Bias vs. n = 504 |
|---------:|---:|--------:|:--------|----------------:|
| 2 | 0.500 | 2.6069 | [2.576, 2.637] | +1.3244 |
| 4 | 0.250 | 2.2629 | [2.234, 2.291] | +0.9804 |
| 8 | 0.125 | 1.9908 | [1.964, 2.017] | +0.7083 |
| 16 | 0.063 | 1.7651 | [1.740, 1.790] | +0.4826 |
| 32 | 0.031 | 1.6142 | [1.590, 1.638] | +0.3317 |
| 64 | 0.016 | 1.4915 | [1.469, 1.514] | +0.2090 |
| 128 | 0.008 | 1.3769 | [1.355, 1.398] | +0.0944 |
| 252 | 0.004 | 1.3239 | [1.303, 1.345] | +0.0414 |
| **504** | **0.002** | **1.2825** | **[1.262, 1.303]** | **—** |
