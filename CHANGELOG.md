# Changelog

All notable RewardLens changes are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-07-13

### Added

- Tested, reusable chart builders with one semantic visual language across the
  dashboard.
- Daily publisher and campaign health monitoring against leakage-safe prior
  seven-observed-day baselines.
- Action-zone distributions, source-priority bubbles, threshold workload
  curves, experiment-value waterfalls, and retention forest plots.
- Chart-level tests for policy selection, cutoff markers, sensitivity margins,
  experiment guardrails, and prior-only baselines.

### Changed

- The dashboard now derives its recommended and priority-review cutoffs from
  evaluated artifacts instead of hard-coded policy names or values.
- Overview charts use full-width, responsive layouts with shorter decision
  labels and mobile-safe legends.
- Experiment visuals now show the pre-specified retention non-inferiority
  boundary, overall effect, and exploratory country estimates explicitly.
- Traffic-source monitoring now distinguishes concentration, exposure, volume,
  and daily change instead of relying on a single aggregate score.

### Decision

- Balanced remains the value-maximizing offline review policy in every tested
  economic scenario, but the simulated experiment still supports a targeted
  pilot rather than global rollout.

## [1.0.0] — 2026-07-13

### Added

- Deterministic synthetic behavioural data for 10,000 users across eight event
  tables.
- Tested DuckDB and dbt feature pipeline with explicit leakage separation.
- Explainable ensemble of rules, robust deviations, Isolation Forest, and DBSCAN
  rarity.
- Cost-sensitive threshold comparison and 16-scenario policy sensitivity test.
- Country-stratified experiment simulation with retention non-inferiority and
  multiple-testing control.
- Beginner-to-expert learning guide and decision-oriented Streamlit dashboard.
- Public Streamlit Community Cloud deployment and verified dashboard screenshots.

### Decision

- The balanced offline threshold is the recommended review policy.
- The experiment recommendation remains a targeted pilot only because the
  retention non-inferiority guardrail did not pass.

[1.1.0]: https://github.com/chanderbhanu096/rewardlens-fraud-decision-lab/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/chanderbhanu096/rewardlens-fraud-decision-lab/releases/tag/v1.0.0
