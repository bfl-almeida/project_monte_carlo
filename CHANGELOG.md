# Changelog

All notable changes to this project are documented in this file.

---

## [Unreleased]

### Changed — Test Suite Structure
- **Renamed and split `tests/test_pricing.py`:**
  - `test_pricing.py` → `test_monte_carlo.py` (11 tests): Monte Carlo pricing, convergence, CI, experiments.
  - `test_pricing.py` → `test_black_scholes.py` (30 tests, new): Analytical prices and Greeks.
  - Mirrors source module structure (`black_scholes.py`, `monte_carlo.py`) for clarity.

### Added — Quantitative Greek Tests (`tests/test_black_scholes.py`)
- **Exact reference values** (10 tests): Each Greek and price locked to known-good values via `scipy.stats.norm`.
  - `test_bs_call_price_exact_reference`, `test_bs_put_price_exact_reference`
  - `test_bs_call_delta_exact_reference`, `test_bs_put_delta_exact_reference`
  - `test_bs_gamma_exact_reference`, `test_bs_vega_exact_reference`
  - `test_bs_call_theta_exact_reference`, `test_bs_put_theta_exact_reference`
  - `test_bs_call_rho_exact_reference`, `test_bs_put_rho_exact_reference`
- **Parametrized property tests** (15 tests): Qualitative invariants (sign, parity, monotonicity) across multiple input sets.
  - `test_bs_gamma_always_positive` (3 parameter sets: ATM/OTM/ITM)
  - `test_bs_vega_always_positive` (3 parameter sets)
  - `test_bs_call_theta_negative` (3 parameter sets)
  - `test_bs_put_theta_negative` (2 parameter sets: ATM and ITM; OTM put theta can be positive)
  - `test_bs_delta_parity` (3 diverse parameter sets)
- **Edge case tests** (5 tests): T ≤ 0, sigma ≤ 0, zero-vol step functions, ATM short-dated.

### Changed — Test Assertions
- All numeric comparisons now use `pytest.approx()` with explicit tolerances instead of manual `abs()` checks.
- Improves error messages on failure and makes tolerances explicit.

### Changed — Test Documentation
- Added docstrings to all test functions explaining intent.
- Organized test files into logical sections with clear headers.
- Reference parameters stored in module-level constant (`REF_PARAMS`) to reduce repetition.

---

## [0.3.1] — 2026-05-11


### Added — `src/option_pricing/black_scholes.py`
- `_norm_pdf(x)` — standard normal PDF helper, used by all new Greeks.
- `bs_call_delta(S0, K, T, r, sigma)` — analytical call Delta: N(d1). Handles T ≤ 0 (step function) and sigma ≤ 0 (step function on forward).
- `bs_put_delta(S0, K, T, r, sigma)` — analytical put Delta: N(d1) − 1. Same edge cases.
- `bs_gamma(S0, K, T, r, sigma)` — analytical Gamma (identical for call and put): N'(d1) / (S0 · σ · √T). Returns 0.0 for T ≤ 0 or sigma ≤ 0.
- `bs_vega(S0, K, T, r, sigma)` — analytical Vega (identical for call and put): S0 · N'(d1) · √T, expressed per 1% move in vol (divided by 100). Returns 0.0 for T ≤ 0 or sigma ≤ 0.
- `bs_call_theta(S0, K, T, r, sigma)` — analytical call Theta per calendar day: −(S0·N'(d1)·σ)/(2√T) − r·K·e^(−rT)·N(d2), divided by 365. Returns 0.0 for T ≤ 0.
- `bs_put_theta(S0, K, T, r, sigma)` — analytical put Theta per calendar day: −(S0·N'(d1)·σ)/(2√T) + r·K·e^(−rT)·N(−d2), divided by 365. Returns 0.0 for T ≤ 0.
- `bs_call_rho(S0, K, T, r, sigma)` — analytical call Rho per 1% move in rates: K·T·e^(−rT)·N(d2), divided by 100. Returns 0.0 for T ≤ 0.
- `bs_put_rho(S0, K, T, r, sigma)` — analytical put Rho per 1% move in rates: −K·T·e^(−rT)·N(−d2), divided by 100. Returns 0.0 for T ≤ 0.

### Changed — `src/option_pricing/black_scholes.py`
- `_norm_cdf` and `_norm_pdf` given one-line docstrings.
- `bs_call_delta` and `bs_put_delta` docstrings updated to include the formula (N(d1) / N(d1)−1).

### Changed — `pyproject.toml`
- Migrated from setuptools to **Poetry** (`poetry-core` build backend).
- `[tool.poetry.dependencies]` replaces `[project.dependencies]`.
- Dev group (`[tool.poetry.group.dev.dependencies]`) includes `pytest >=9.0.3,<10.0.0`, `pytest-cov`, and `ipykernel ^7.2.0`.

---

## [0.3.0] — 2026-05-10

### Removed
- **`[build-system].txt`** — Deleted. Stale duplicate of the build-system configuration that
  already lives in `pyproject.toml`. `pyproject.toml` remains the single source of truth for
  project metadata and build configuration.
- **`notebooks/generate_research_demo.py`** — Deleted. This was a code-generation script that
  programmatically wrote `research_demo.ipynb` from Python. The notebook is now maintained
  directly in its `.ipynb` format, which is more practical for iterative development, inline
  outputs and version-controlled diffs.

### Changed — `src/option_pricing/experiments.py`
- `run_convergence_experiment` gains a `n_seeds` parameter (default `1`, backwards compatible).
  When `n_seeds > 1`, the full N-grid is repeated for seeds `random_seed … random_seed + n_seeds − 1`.
  Each seed produces its own OLS slope; the reported `conv_rate` is the mean slope across seeds and
  `conv_rate_std` is its standard deviation, giving a robust, seed-averaged convergence rate estimate.
  A `seed` column is added to the output DataFrame to identify each replication.

### Changed — `README.md` — Key Results section (complete rewrite)
The previous "Key Results (illustrative)" placeholder was replaced with real, reproducible numbers
from the four experiments. Changes per subsection:

- **Section 1 · Convergence** — Single-seed rates (−0.77, −0.60) replaced with seed-averaged rates
  **−0.46 ± 0.25** (standard) and **−0.45 ± 0.23** (antithetic) from 50 seeds (42–91).
  Table now shows mean ± std of MC price and |error| across 50 seeds at selected N values.
  Prose explains the multi-seed methodology and why single-seed rates are unreliable.
- **Section 2 · Variance Reduction** — Full VRF and efficiency ratio table (9 N-values, seeds 0–49).
  Median VRF **2.66×**, median efficiency ratio **2.92×**. Prose includes the VRF formula,
  efficiency ratio formula, and interpretation of the N-dependent pattern (peak 5.04× at N = 5 000,
  compression toward 1× at large N). Seed information moved into the subsection header.
- **Section 3 · Confidence Interval Coverage** — CLT interval formula added. Full interpretation of
  systematic undercoverage (91.0–93.0 % vs. 95 % nominal): statistically significant at 1–3 SEs
  below nominal, root cause identified as right-skew of call payoffs. Per-scenario breakdown added.
- **Section 4 · Discretisation Bias** — Mechanism explained (missed crossings between grid points).
  Full 9-row table with 95 % CIs and signed bias. Key milestones cited (daily monitoring reduces
  bias to +3 %). Operational implication stated explicitly.

### Changed — `notebooks/research_demo.ipynb` (full overhaul)
- **Experiment 1:** Code updated to use `n_seeds=50`; output aggregated to mean ± std per (N, method).
  Plot redesigned with shaded ±1 std bands around price and error curves.
  Observations updated with actual seed-averaged rates.
- **Experiment 2:** Markdown adds VRF and efficiency ratio formulas in LaTeX. Code extended to full
  9-point path grid (500–250 000). Plot annotated with median VRF line and peak annotation with arrow.
  Observations explain the N-dependent pattern and flag the 50-replication limitation.
- **Experiment 3:** Markdown adds CLT formula and motivation for expected undercoverage. Bar chart
  redesigned with coverage values printed on bars, tighter y-axis, and 95 % CI error bars.
  Observations give per-scenario breakdown and root-cause explanation.
- **Experiment 4:** Markdown explains the discrete-monitoring mechanism in detail and cites the
  Broadie-Glasserman-Kou theoretical rate. Plot adds finest-grid reference line and annotated arrows
  for coarsest and daily bias. Observations list key milestones with exact prices and bias values,
  and state the operational implication.
- **Summary cell** updated with actual numbers for all four experiments.

### Added — Runtime metrics across all experiments

#### `src/option_pricing/experiments.py`
- `run_convergence_experiment` — each row now includes `runtime_s` (wall-clock seconds for that simulation).
- `run_variance_reduction_experiment` — output includes `mean_rt_standard` and `mean_rt_antithetic`
  (mean per-replication wall-clock time for each method).
- `run_ci_coverage_experiment` — output includes `runtime_total_s` (total wall-clock time for the
  200-replication loop) and `runtime_per_rep_s` (mean time per replication) per scenario.
- `run_discretisation_bias_experiment` — each row includes `runtime_s` (wall-clock seconds for that
  n_steps configuration).
- All timings measured with `time.perf_counter()`.

#### `notebooks/research_demo.ipynb`
- Experiment 1 results table: added `runtime (s)` column (mean across seeds).
- Experiment 2 results table: added `RT std (s)` and `RT anti (s)` columns.
- Experiment 3 results table: added `RT total (s)` and `RT / rep (s)` columns.
- Experiment 4 results table: added `runtime (s)` column.

#### `README.md`
- Features section: added "runtime" to the list of statistical metrics reported per experiment.
- All four key-results tables updated with the corresponding runtime column(s).

---

## [0.2.1] — 2026-05-10

### Added — CI/CD via GitHub Actions
- **`.github/workflows/test.yml`** — Automated test workflow triggered on every push and pull request.
  - Runs on `ubuntu-latest` with Python 3.11.
  - Installs dependencies via Poetry (`poetry install --no-root`).
  - Executes the full pytest suite with `PYTHONPATH=src poetry run pytest -v`.
- **`README.md`** — CI badge added (`![Tests](https://github.com/bfl-almeida/project_monte_carlo/actions/workflows/test.yml/badge.svg)`).

### Changed — `pyproject.toml`
- `pytest` added as a development dependency to support the CI workflow.

### Changed — `poetry.lock`
- Updated to reflect new `pytest` and transitive dependencies.

---

## [0.2.0] — 2026-04-21

### Repositioning
- Project reframed from a pricing library to a quantitative research study on Monte Carlo methods for derivative pricing.
- README rewritten with 4 explicit research questions, a method comparison table, quickstart example, and key results summary.

### Added — `src/option_pricing/experiments.py` (new module)
Four reproducible research experiments, each returning a tidy `pd.DataFrame`:
- `run_convergence_experiment` — MC pricing error vs. simulation budget N; attaches empirical convergence rate via OLS log-log regression (theoretical: -0.5).
- `run_variance_reduction_experiment` — Antithetic variates efficiency over repeated independent replications; computes variance reduction factor (VRF) and work-normalised efficiency ratio.
- `run_ci_coverage_experiment` — Empirical 95% CI coverage of CLT-based intervals across a moneyness/volatility parameter grid.
- `run_discretisation_bias_experiment` — Barrier option pricing bias as a function of time-step resolution (n_steps); quantifies discrete-monitoring overpricing.

### Added — `src/option_pricing/utils.py` (extended)
- `confidence_interval(result, alpha)` — Asymptotic CLT-based CI from a `MonteCarloResult`.
- `efficiency_ratio(baseline, improved, t_baseline, t_improved)` — Work-normalised variance-reduction efficiency ratio.
- `estimate_convergence_rate(n_grid, errors)` — OLS slope in log-log space; implemented with pure NumPy dot products to avoid LAPACK crashes on Windows.
- `convergence_table` extended with `rel_error`, `ci_lower`, `ci_upper`, `runtime_s` columns.

### Changed — `src/option_pricing/__init__.py`
- Fixed broken import (`from .analysis` -> `from .utils`).
- All new public symbols from `experiments.py` and `utils.py` added to `__all__`.

### Changed — `pyproject.toml`
- Version bumped `0.1.0` -> `0.2.0`.
- Description updated to reflect research positioning.
- `pandas` added as a core dependency.

### Changed — `tests/test_pricing.py`
- Test suite expanded from 3 to 12 tests.
- New tests: `confidence_interval`, `estimate_convergence_rate`, `efficiency_ratio`, `convergence_table` schema, and smoke tests for all four experiments.

---

## [0.1.0] — initial release

### Added
- `src/option_pricing/black_scholes.py` — Closed-form Black-Scholes prices for European calls and puts.
- `src/option_pricing/monte_carlo.py` — Monte Carlo simulation engine:
  - `simulate_terminal_price` — GBM terminal price simulation.
  - `simulate_price_paths` — Full GBM path discretisation.
  - `mc_european_option_price` — European call/put pricing with optional antithetic variates.
  - `mc_barrier_option_price` — Up-and-out / down-and-out knock-out barrier pricing.
  - `MonteCarloResult` dataclass (price, standard_error).
- `src/option_pricing/utils.py` — `convergence_table` comparing MC vs. BS prices.
- `tests/test_pricing.py` — 3 tests: put-call parity, MC vs. BS accuracy, barrier <= vanilla.
- `notebooks/demo.ipynb` — Demo notebook.

### Fixed (applied at 0.1.x)
- `utils.py` imported non-existent `mc_european_call_price`; corrected to `mc_european_option_price`.
- `utils.py` attempted tuple unpacking on a `MonteCarloResult` dataclass; corrected to use `.price` and `.standard_error` attributes.
