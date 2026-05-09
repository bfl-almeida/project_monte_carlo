# Changelog

All notable changes to this project are documented in this file.

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
