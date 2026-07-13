# Changelog

All notable RewardLens changes are documented here. The project follows
[Semantic Versioning](https://semver.org/).

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

[1.0.0]: https://github.com/chanderbhanu096/rewardlens-fraud-decision-lab/releases/tag/v1.0.0
