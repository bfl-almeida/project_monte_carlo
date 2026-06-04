![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/bfl-almeida/project_monte_carlo/actions/workflows/test.yml/badge.svg)

# Monte Carlo Methods for Derivative Pricing, Greeks Estimation, and Variance Reduction

## Overview

A Python library for Monte Carlo option pricing, finite-difference Greeks, and variance reduction analysis under the Black-Scholes model — built as a quantitative research project with reproducible experiments, statistical validation, and a full pytest suite.

The goal is to demonstrate practical skills relevant to quantitative finance roles: derivatives pricing, numerical methods, statistical validation, model risk analysis and Python-based quantitative tooling.

## Features

- Analytical Black-Scholes pricing for European calls and puts
- Monte Carlo pricing for European options
- Finite-difference Monte Carlo Greeks (Delta, Gamma, Vega, Theta) with Common Random Numbers
- Confidence intervals and convergence analysis
- Variance reduction with antithetic variates
- Statistical metrics per experiment: absolute error, relative error, standard error, confidence intervals, runtime
- Analytical Black-Scholes Greeks (Delta, Gamma, Vega, Theta, Rho)
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
| Finite-Difference Greeks (CRN) | Bump-and-revalue Delta/Gamma/Vega/Theta via central differences with common random numbers |
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
│  ├─ option_pricing/           # Core library
│  │  ├─ __init__.py
│  │  ├─ black_scholes.py       # Analytical BS prices and Greeks (8 + 2 functions)
│  │  ├─ monte_carlo.py        # Simulation engine + finite-difference MC Greeks (CRN)
│  │  └─ utils.py              # Statistical helpers, convergence table
│  └─ research/                # Reproducible research experiments
│     ├─ __init__.py
│     └─ experiments.py        # Four main experiment functions
├─ tests/
│  ├─ test_black_scholes.py   # 30 tests: exact reference + properties + edge cases
│  └─ test_monte_carlo.py      # 18 tests: pricing, barrier, Greeks, experiments
├─ notebooks/
│  ├─ research_demo.ipynb      # Vanilla pricing experiments (convergence, VR, CI, bias)
│  └─ research_demo_greeks.ipynb # MC Greek estimation + P&L attribution
├─ foundations/                # Educational notebooks on theory
└─ reports/
   ├─ figures/
   └─ tables/
```

## Notebooks

**`research_demo.ipynb`** — Core experiments: convergence at O(N⁻¹/²), variance reduction effectiveness (antithetic VRF ≈ 2.66×), CI coverage (91–93 % empirical vs 95 % nominal), and discretisation bias in barrier options (O(1/√n_steps) convergence).

**`research_demo_greeks.ipynb`** — Seven experiments on finite-difference Greek estimation: Greek profiles vs spot and maturity, convergence to BS benchmarks under antithetic vs plain MC, log-log convergence rate, bump size sensitivity across multiple orders of magnitude, CRN effectiveness (VRF > 1000× for Gamma/Theta), and P&L attribution using Delta + ½Γ·ΔS² Taylor expansion.

## Quickstart

```python
from option_pricing import mc_european_option_price, bs_call_price, mc_european_option_greeks
from research.experiments import run_convergence_experiment

# Analytical benchmark
price = bs_call_price(S0=100, K=100, T=1, r=0.05, sigma=0.2)

# Monte Carlo estimate with antithetic variates
result = mc_european_option_price(
    S0=100, K=100, T=1, r=0.05, sigma=0.2,
    option_type="call", n_paths=100_000, antithetic=True, random_seed=42,
)
print(f"MC price: {result.price:.4f}  SE: {result.standard_error:.4f}")

# Finite-difference Greeks via Common Random Numbers
greeks = mc_european_option_greeks(
    S0=100, K=100, T=1, r=0.05, sigma=0.2,
    option_type="call", n_paths=200_000, random_seed=42,
)
print(f"Delta: {greeks.delta:.4f}, Gamma: {greeks.gamma:.6f}, Vega: {greeks.vega:.4f}, Theta: {greeks.theta:.4f}")

# Convergence experiment
df = run_convergence_experiment()
print(df.to_string(index=False))
```

## Key Results

All experiments are fully reproducible via `src/research/experiments.py`.
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

| N | Method | MC Price (mean ± std) | \|Error\| (mean ± std) | Rel Error % | MC Std Error | Runtime (s, mean) |
|--:|:-------|----------------------:|----------------------:|------------:|-------------:|------------------:|
| 1 000 | Standard | 10.4547 ± 0.4274 | 0.3541 ± 0.2340 | 3.39 | 0.4655 | see notebook |
| 1 000 | Antithetic | 10.3420 ± 0.3398 | 0.2762 ± 0.2229 | 2.64 | 0.4594 | see notebook |
| 10 000 | Standard | 10.4578 ± 0.1486 | 0.1155 ± 0.0923 | 1.11 | 0.1471 | see notebook |
| 10 000 | Antithetic | 10.4249 ± 0.0951 | 0.0738 ± 0.0644 | 0.71 | 0.1468 | see notebook |
| 100 000 | Standard | 10.4557 ± 0.0531 | 0.0458 ± 0.0267 | 0.44 | 0.0466 | see notebook |
| 100 000 | Antithetic | 10.4497 ± 0.0360 | 0.0299 ± 0.0195 | 0.29 | 0.0465 | see notebook |
| 250 000 | Standard | 10.4491 ± 0.0295 | 0.0228 ± 0.0184 | 0.22 | 0.0294 | see notebook |
| 250 000 | Antithetic | 10.4551 ± 0.0246 | 0.0205 ± 0.0141 | 0.20 | 0.0294 | see notebook |

*MC Std Error is the within-simulation standard error (SE = sample std / √N), averaged across seeds. Runtime is wall-clock time per simulation run, averaged across the 50 seeds; hardware-dependent — live values are shown in the research notebook.*

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

| N | RT Std (s) | RT Anti (s) | Var (Standard) | Var (Antithetic) | VRF | Efficiency Ratio |
|--:|-----------:|------------:|---------------:|-----------------:|----:|-----------------:|
| 500 | see nb | see nb | 0.4866 | 0.1834 | 2.65× | 2.81× |
| 1 000 | see nb | see nb | 0.2371 | 0.0819 | 2.90× | 2.72× |
| 2 000 | see nb | see nb | 0.1139 | 0.0364 | 3.13× | 3.33× |
| 5 000 | see nb | see nb | 0.0568 | 0.0113 | 5.04× | 5.79× |
| 10 000 | see nb | see nb | 0.0240 | 0.0084 | 2.84× | 3.36× |
| 25 000 | see nb | see nb | 0.0086 | 0.0032 | 2.66× | 3.54× |
| 50 000 | see nb | see nb | 0.0036 | 0.0024 | 1.53× | 2.00× |
| 100 000 | see nb | see nb | 0.0016 | 0.0014 | 1.17× | 1.33× |
| 250 000 | see nb | see nb | 0.0008 | 0.0005 | 1.78× | 1.89× |

**Median VRF: 2.66×  ·  Median efficiency ratio: 2.92×**

*RT Std / RT Anti: mean wall-clock runtime per replication for the standard and antithetic estimators respectively; hardware-dependent — live values are shown in the research notebook.*

The VRF varies across N because the gain depends on how much random noise remains to be cancelled. At moderate budgets (N = 1 000–25 000) both estimators are still far from convergence, so the negative correlation between antithetic pairs has a large noise pool to work with and the VRF is consistently above 2×, peaking at 5.04× for N = 5 000. At very large N (100 000+) both estimators have already converged close to the true price, the residual variance is tiny, and the two methods become nearly equally precise — compressing the VRF toward 1×. The reduction is most valuable at moderate N, indicating where the practical sweet spot is, since extremely large N is computationally expensive and delivers diminishing returns regardless of the estimator used.

*Variance estimates are cross-replication sample variances over 50 independent runs per (N, method) cell. Runtime columns show mean wall-clock time per replication; hardware-dependent — live values are shown in the research notebook.*

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

| Scenario | S₀ / K | σ | T | BS Price | Empirical Coverage | RT total (s) | RT / rep (s) |
|:---------|-------:|--:|--:|---------:|------------------:|-------------:|-------------:|
| ITM (K = 90) | 1.11 | 20 % | 1.00 yr | 16.6994 | 91.5 % | 0.082 | 0.000411 |
| ATM (K = 100) | 1.00 | 20 % | 1.00 yr | 10.4506 | 91.5 % | 0.070 | 0.000348 |
| OTM (K = 110) | 0.91 | 20 % | 1.00 yr | 6.0401 | 93.0 % | 0.070 | 0.000349 |
| Short tenor | 1.00 | 20 % | 0.25 yr | 4.6150 | 92.5 % | 0.064 | 0.000322 |
| Low vol (σ = 10 %) | 1.00 | 10 % | 1.00 yr | 6.8050 | 91.0 % | 0.073 | 0.000363 |
| High vol (σ = 40 %) | 1.00 | 40 % | 1.00 yr | 18.0230 | 92.0 % | 0.082 | 0.000411 |

*Coverage standard error ≈ 1.5 % per scenario (proportion SE over 200 replications). RT total: wall-clock time for all 200 replications per scenario; RT / rep: mean per replication. Hardware-dependent.*

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

| n\_steps | dt | MC Price | 95 % CI | Bias vs. n = 504 | Runtime (s) |
|---------:|---:|--------:|:--------|----------------:|------------:|
| 2 | 0.500 | 2.6069 | [2.576, 2.637] | +1.3244 | 0.015 |
| 4 | 0.250 | 2.2629 | [2.234, 2.291] | +0.9804 | 0.019 |
| 8 | 0.125 | 1.9908 | [1.964, 2.017] | +0.7083 | 0.037 |
| 16 | 0.063 | 1.7651 | [1.740, 1.790] | +0.4826 | 0.061 |
| 32 | 0.031 | 1.6142 | [1.590, 1.638] | +0.3317 | 0.093 |
| 64 | 0.016 | 1.4915 | [1.469, 1.514] | +0.2090 | 0.184 |
| 128 | 0.008 | 1.3769 | [1.355, 1.398] | +0.0944 | 0.412 |
| 252 | 0.004 | 1.3239 | [1.303, 1.345] | +0.0414 | 0.777 |
| **504** | **0.002** | **1.2825** | **[1.262, 1.303]** | **—** | **1.659** |

---

### 5 · Monte Carlo Greek Estimation via Bump-and-Revalue

Finite-difference Greeks are estimated using central differences with Common Random Numbers (CRN).
Without CRN, second derivatives (Gamma, Theta) would be drowned in Monte Carlo noise. CRN ensures
that parameter bumps generate differences driven purely by sensitivity, not sampling variation.

**Convergence to Black-Scholes**

All four MC Greeks converge to analytical BS benchmarks at the theoretical *O(N^{−1/2})* rate when
Gamma and Theta are estimated with CRN. At N = 500,000 paths all absolute errors fall below 10^{−3}.
The results validate both the bump-and-revalue technique and the CRN implementation:

| Greek | BS Value | MC (N=500k) | Error | Rel Error % | VRF (with CRN) |
|-------|-------:|--------:|-------:|----------:|---------------:|
| Delta | 0.6368 | 0.6369 | 0.0001 | 0.01 % | 1,031× |
| Gamma | 0.0188 | 0.0187 | 0.0001 | 0.53 % | 60,312× |
| Vega | 0.3752 | 0.3747 | 0.0005 | 0.13 % | 105× |
| Theta | -0.0176 | -0.0175 | 0.0001 | 0.57 % | 432,756× |

*VRF = Variance Reduction Factor: the variance of plain (non-CRN) estimates divided by CRN estimates.
Without CRN, Gamma and Theta estimates are purely noise; VRF quantifies the dramatic stabilization
that CRN provides.*

**Bump Size Optimization**

Bump size balances truncation error (too-large bumps) against floating-point cancellation (too-small bumps).
Testing across three orders of magnitude reveals distinct patterns:

- **First-order Greeks (Delta, Vega):** L-shaped mean absolute error (MAE). Optimal plateau spans
  h ∈ [0.005, 0.015] for Delta and dv ∈ [0.008, 0.012] for Vega. Market convention (h = 0.01,
  dv = 0.01) sits safely in the plateau.
- **Second-order Greels (Gamma):** U-shaped MAE with a sharper optimum near h ≈ 0.06. Theta shows
  similar structure. At the market-convention bump sizes, all four Greeks MAE remains below 10^{−3}.

**P&L Attribution: Delta + ½Γ·ΔS² vs. Actual Repricing**

The Taylor expansion P&L ≈ Δ·ΔS + ½Γ·ΔS² + V·Δσ + Θ·Δt is the foundation of intraday
P&L explain. Testing on ±$15 spot moves shows:

- **Delta-only model:** ~$2 residual (20 % of move size)
- **Delta + Gamma model:** ~$0.3 residual (3 % of move size)
- **Delta + Gamma + Vega model:** ~$0.2 residual (2 % of move size)

This empirical evidence underpins Delta-Gamma hedging: capturing the convexity (Gamma) term
reduces unexplained P&L by ~85 %, making intraday attribution tractable.

---

# Installation guidelines of the env

**Minimal usage (no Poetry):**

mamba env create -f environment.yml

mamba activate monte-carlo

jupyter lab

**Full dev workflow:**

mamba activate monte-carlo

poetry install --with dev

poetry run pytest
