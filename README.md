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
- Unit tests with pytest

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

Antithetic variates reduce estimator variance by pairing each draw Z with its mirror −Z, producing
negatively correlated path pairs whose payoffs partially cancel each other's noise. For smooth,
monotone payoffs such as European calls, the theoretical variance reduction factor (VRF) approaches 2×
as the payoff-to-draw correlation approaches −1.

To measure this empirically, 50 independent replications (seeds 0 through 49) were run for both the
standard and antithetic estimators at each simulation budget N. The empirical VRF is the ratio of the
cross-replication variances of the two price estimators:

$$\text{VRF}(N) = \frac{\text{Var}_{\text{standard}}(N)}{\text{Var}_{\text{antithetic}}(N)}$$

Because antithetic paths require the same number of normal draws as standard paths but paired
differently, the compute overhead is negligible. The efficiency ratio adjusts the VRF for any
observed runtime difference, giving a work-normalised measure of gain per unit of wall-clock time.

The median VRF of **2.66×** and median efficiency ratio of **2.92×** confirm that antithetic sampling
consistently outperforms standard MC across all tested budgets. The VRF varies across N — peaking at
**5.04×** for N = 5 000 and compressing toward 1× at very large N where both estimators are already
highly precise — which is expected behaviour as the estimator variance becomes dominated by
systematic rather than random components.

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

The VRF varies across N because the gain depends on how much random noise remains to be cancelled. At moderate budgets (N = 1 000–25 000) both estimators are still far from convergence, so the negative correlation between antithetic pairs has a large noise pool to work with and the VRF is consistently above 2×, peaking at 5.04× for N = 5 000. At very large N (100 000+) both estimators have already converged close to the true price, the residual variance is tiny, and the two methods become nearly equally precise — compressing the VRF toward 1×. The reduction is most valuable at moderate N, indicating where the practical sweet spot is, since extremely large N is computationally expensive and delivers diminishing returns regardless of the estimator used.

*Variance estimates are cross-replication sample variances over 50 independent runs per (N, method) cell.*

---

### 3 · Confidence Interval Coverage

A Monte Carlo price estimate is only as useful as the uncertainty attached to it. The standard approach
is to accompany each estimate with an asymptotic 95 % confidence interval derived from the Central
Limit Theorem:

$$\hat{V} \pm 1.96 \times \frac{s}{\sqrt{N}}$$

where $s$ is the sample standard deviation of the discounted payoffs. This interval is valid
asymptotically — it relies on the CLT approximating the estimator distribution as normal, which holds
when N is large enough relative to the skewness of the payoff distribution.

To verify whether these intervals achieve their nominal 95 % coverage in practice, the experiment
constructs 200 independent CIs per scenario (seeds 0 through 199, N = 10 000 paths each) and counts
the fraction that contain the exact Black-Scholes price. A well-calibrated estimator should yield
empirical coverage close to 95 %; systematic deviations indicate either insufficient N for the CLT
approximation to hold, or payoff-distribution skewness that inflates the true variance beyond what the
normal approximation captures.

Empirical coverage across all six scenarios falls in the range **91.0 % – 93.0 %**, consistently below
the 95 % nominal. This undercoverage is statistically significant: with 200 replications, the standard
error of a coverage estimate is approximately 1.5 %, placing these readings 1–3 standard errors below
nominal. The root cause is the right-skew of call option payoffs — a large fraction of paths expire
out of the money with zero payoff, while the in-the-money paths produce a long right tail. This
asymmetry means the true estimator variance is slightly understated by the normal CLT approximation at
N = 10 000, causing the CI to be narrower than it should be. Coverage is expected to converge toward
95 % as N increases and the CLT approximation improves.

Notably, the undercoverage is most pronounced for in-the-money options (ITM, 91.5 %) and
low-volatility scenarios (91.0 %), where payoff distributions are more concentrated and the CLT
convergence is slower relative to the skewness. Out-of-the-money and high-volatility scenarios
approach 93 %, consistent with a more spread-out payoff distribution where the normal approximation
is somewhat better.

| Scenario | S₀ / K | σ | T | BS Price | Empirical Coverage |
|:---------|-------:|--:|--:|---------:|------------------:|
| ITM (K = 90) | 1.11 | 20 % | 1.00 yr | 16.6994 | 91.5 % |
| ATM (K = 100) | 1.00 | 20 % | 1.00 yr | 10.4506 | 91.5 % |
| OTM (K = 110) | 0.91 | 20 % | 1.00 yr | 6.0401 | 93.0 % |
| Short tenor | 1.00 | 20 % | 0.25 yr | 4.6150 | 92.5 % |
| Low vol (σ = 10 %) | 1.00 | 10 % | 1.00 yr | 6.8050 | 91.0 % |
| High vol (σ = 40 %) | 1.00 | 40 % | 1.00 yr | 18.0230 | 92.0 % |

*Coverage standard error ≈ 1.5 % per scenario (proportion SE over 200 replications).*

---

### 4 · Discretisation Bias — Up-and-Out Barrier Option

Path-dependent options such as barrier options cannot be priced from the terminal stock price alone —
the full trajectory must be simulated to check whether the barrier was crossed at any point during the
option's life. In practice, paths are simulated on a discrete time grid of n\_steps monitoring points.
The knock-out condition is then checked only at those grid points, meaning that crossings occurring
*between* two consecutive steps are invisible to the simulator. Sparse grids systematically under-count
knock-out events, leaving paths alive that should have been extinguished, and therefore **overstate**
the option price. As the grid becomes finer, the discrete-monitoring price converges to the
continuous-monitoring price.

This experiment prices an up-and-out call (S₀ = 100, K = 100, barrier B = 120, T = 1 yr, r = 5 %,
σ = 20 %) using 100 000 antithetic paths (seed = 42) across nine time-step resolutions ranging from
n\_steps = 2 (semi-annual monitoring) to n\_steps = 504 (twice-daily monitoring). The finest grid
serves as the proxy for the continuous price; bias at each coarser resolution is measured relative to it.

The results show a pronounced and monotonically decreasing bias: at n\_steps = 2 the price is
**2.6069**, more than double the finest-grid estimate of **1.2825** — an absolute overstatement of
**+1.32** (+103 %). The bias halves roughly every time the number of steps doubles, consistent with
the known $O(1/\sqrt{n\_\text{steps}})$ convergence rate for discrete barrier monitoring. By
n\_steps = 252 (daily monitoring) the bias has fallen to **+0.04** (3 %), and the 95 % confidence
intervals at n\_steps = 252 and n\_steps = 504 are nearly overlapping, indicating practical convergence
at daily resolution. This result has a direct operational implication: practitioners using weekly or
monthly monitoring grids for barrier products should expect a material upward pricing bias, and daily
or sub-daily grids are required for reliable estimates.

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
