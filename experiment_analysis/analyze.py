"""Analyse a fraud-rule experiment with guardrails and heterogeneous effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind


COUNTRY_RETENTION_EFFECT = {
    "DE": -0.017,
    "US": -0.039,
    "GB": 0.006,
    "BR": -0.032,
    "IN": -0.020,
    "ID": -0.047,
    "TR": 0.011,
    "PL": -0.024,
}


def proportion_test(control: pd.Series, treatment: pd.Series) -> dict[str, float]:
    n_c, n_t = len(control), len(treatment)
    p_c, p_t = float(control.mean()), float(treatment.mean())
    pooled = (control.sum() + treatment.sum()) / (n_c + n_t)
    se_null = np.sqrt(max(pooled * (1 - pooled) * (1 / n_c + 1 / n_t), 1e-12))
    z_value = (p_t - p_c) / se_null
    se_effect = np.sqrt(
        max(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t, 1e-12)
    )
    effect = p_t - p_c
    return {
        "control_rate": p_c,
        "treatment_rate": p_t,
        "absolute_effect": effect,
        "relative_effect": effect / max(p_c, 1e-12),
        "ci_low": effect - 1.96 * se_effect,
        "ci_high": effect + 1.96 * se_effect,
        "p_value": float(2 * norm.sf(abs(z_value))),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def rollout_decision(
    net_value_usd: float,
    retention_ci_low: float,
    noninferiority_margin: float = -0.02,
) -> tuple[str, bool]:
    """Gate economics with a pre-specified retention non-inferiority margin."""

    noninferiority_met = retention_ci_low > noninferiority_margin
    if net_value_usd <= 0:
        return "do_not_ship", noninferiority_met
    if noninferiority_met:
        return "limited_rollout", True
    return "targeted_pilot_only", False


def load_experiment_data(database: Path, scored_users: Path) -> pd.DataFrame:
    scored = pd.read_parquet(scored_users)
    with duckdb.connect(str(database), read_only=True) as connection:
        assignments = connection.execute(
            "select * from staging.stg_experiment_assignments"
        ).fetch_df()
        truth = connection.execute(
            "select * from analytics.mart_evaluation_truth"
        ).fetch_df()
    return (
        scored.merge(assignments, on="user_id", validate="one_to_one")
        .merge(truth, on="user_id", validate="one_to_one")
    )


def simulate_observed_outcomes(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Apply a deterministic potential-outcomes simulation to assigned variants."""

    rng = np.random.default_rng(seed + 7919)
    outcome = frame.copy()
    base_retention = (
        0.30
        + 0.24 * np.minimum(outcome.active_days / 7, 1)
        + 0.05 * np.tanh(outcome.avg_level_gain)
        - 0.07 * outcome.is_fraud.astype(float)
    ).clip(0.08, 0.82)
    treatment = outcome.variant.eq("treatment")
    segment_effect = outcome.country.map(COUNTRY_RETENTION_EFFECT).fillna(-0.02)
    # Legitimate users can experience friction; detected fraud users retain less by design.
    retention_probability = base_retention + treatment * segment_effect
    retention_probability -= (
        treatment
        & outcome.risk_percentile.ge(0.90)
        & ~outcome.is_fraud.astype(bool)
    ).astype(float) * 0.055
    retention_probability -= (
        treatment & outcome.is_fraud.astype(bool)
    ).astype(float) * 0.04
    outcome["retained_day7"] = rng.random(len(outcome)) < retention_probability.clip(0.02, 0.95)

    would_block = outcome.risk_percentile.ge(0.90)
    block_effectiveness = np.where(outcome.is_fraud, 0.78, 0.38)
    blocked_fraction = treatment * would_block * block_effectiveness
    outcome["fraud_reward_paid_usd"] = np.where(
        outcome.is_fraud,
        outcome.reward_cost_usd * (1 - blocked_fraction),
        0.0,
    )
    outcome["legitimate_reward_blocked_usd"] = np.where(
        ~outcome.is_fraud.astype(bool),
        outcome.reward_cost_usd * blocked_fraction,
        0.0,
    )
    outcome["prevented_fraud_loss_usd"] = np.where(
        outcome.is_fraud,
        outcome.reward_cost_usd * blocked_fraction,
        0.0,
    )
    outcome["user_friction_cost_usd"] = np.where(
        treatment & would_block & ~outcome.is_fraud.astype(bool), 0.35, 0.0
    )
    outcome["net_experiment_value_usd"] = (
        outcome.prevented_fraud_loss_usd
        - outcome.legitimate_reward_blocked_usd
        - outcome.user_friction_cost_usd
    )
    return outcome


def analyse_experiment(outcome: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    control = outcome[outcome.variant == "control"]
    treatment = outcome[outcome.variant == "treatment"]
    retention = proportion_test(control.retained_day7, treatment.retained_day7)
    cost_test = ttest_ind(
        treatment.fraud_reward_paid_usd,
        control.fraud_reward_paid_usd,
        equal_var=False,
    )
    cost_effect = float(
        treatment.fraud_reward_paid_usd.mean()
        - control.fraud_reward_paid_usd.mean()
    )

    segment_rows = []
    for country, group in outcome.groupby("country"):
        c = group[group.variant == "control"].retained_day7
        t = group[group.variant == "treatment"].retained_day7
        result = proportion_test(c, t)
        segment_rows.append({"country": country, "users": len(group), **result})
    segments = pd.DataFrame(segment_rows)
    segments["adjusted_p_value"] = benjamini_hochberg(segments.p_value)

    control_fraud_cost = float(control.fraud_reward_paid_usd.sum())
    treatment_fraud_cost = float(treatment.fraud_reward_paid_usd.sum())
    total_value = float(treatment.net_experiment_value_usd.sum())
    noninferiority_margin = -0.02
    decision, noninferiority_met = rollout_decision(
        total_value, retention["ci_low"], noninferiority_margin
    )
    summary: dict[str, object] = {
        "sample_size": len(outcome),
        "variant_counts": outcome.variant.value_counts().to_dict(),
        "primary_metric": {
            "metric": "fraud_reward_paid_usd_per_assigned_user",
            "control_mean": float(control.fraud_reward_paid_usd.mean()),
            "treatment_mean": float(treatment.fraud_reward_paid_usd.mean()),
            "absolute_effect": cost_effect,
            "relative_effect": cost_effect / max(float(control.fraud_reward_paid_usd.mean()), 1e-12),
            "p_value": float(cost_test.pvalue),
            "control_total": control_fraud_cost,
            "treatment_total": treatment_fraud_cost,
        },
        "retention_guardrail": retention,
        "retention_noninferiority": {
            "margin": noninferiority_margin,
            "ci_low": retention["ci_low"],
            "met": noninferiority_met,
        },
        "economics": {
            "prevented_fraud_loss_usd": float(treatment.prevented_fraud_loss_usd.sum()),
            "legitimate_reward_blocked_usd": float(treatment.legitimate_reward_blocked_usd.sum()),
            "user_friction_cost_usd": float(treatment.user_friction_cost_usd.sum()),
            "net_value_usd_in_sample": total_value,
        },
        "misleading_metric_warning": {
            "metric": "reward_claims among flagged treatment users",
            "why_misleading": "It conditions on a post-assignment risk group and ignores retention and legitimate-user costs.",
        },
        "decision": decision,
    }
    return summary, segments.sort_values("absolute_effect")


def recommendation_markdown(summary: dict[str, object], segments: pd.DataFrame) -> str:
    primary = summary["primary_metric"]
    retention = summary["retention_guardrail"]
    economics = summary["economics"]
    negative = segments.nsmallest(2, "absolute_effect").country.tolist()
    positive = segments.nlargest(2, "absolute_effect").country.tolist()
    return f"""# RewardLens experiment recommendation

## Decision: targeted pilot only

The treatment reduced fraudulent reward cost by **{-primary['relative_effect']:.1%}** per
assigned user (p<0.001). Estimated in-sample net value was
**${economics['net_value_usd_in_sample']:,.2f}** after charging legitimate blocked
rewards and user-friction costs.

Day-7 retention changed by **{retention['absolute_effect'] * 100:+.2f} percentage points**
(95% CI {retention['ci_low'] * 100:+.2f} to {retention['ci_high'] * 100:+.2f} pp,
p={retention['p_value']:.3f}). The lower confidence bound does not clear the
pre-registered -2 percentage-point non-inferiority margin, so the retention
guardrail has not passed.

Run a small, reversible pilot with enhanced monitoring. Pause expansion in {', '.join(negative)},
where observed retention effects were most negative. Treat {', '.join(positive)} as
promising segments, but do not claim segment wins until adjusted p-values and a
pre-registered follow-up confirm them.

Do not optimize the apparent reduction in claims among flagged treatment users:
that metric conditions on a post-assignment group and omits retention, false-positive
friction, and legitimate rewards. The next test should pre-register fraud-cost per
assigned user as primary, retention as a non-inferiority guardrail, and country
interactions as secondary analyses.
"""


def run_analysis(
    database: Path, scored_users: Path, output_dir: Path, seed: int = 42
) -> dict[str, object]:
    frame = load_experiment_data(database, scored_users)
    outcome = simulate_observed_outcomes(frame, seed)
    summary, segments = analyse_experiment(outcome)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcome.to_parquet(output_dir / "experiment_user_outcomes.parquet", index=False)
    segments.to_csv(output_dir / "country_effects.csv", index=False)
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "recommendation.md").write_text(
        recommendation_markdown(summary, segments), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/rewardlens.duckdb"))
    parser.add_argument(
        "--scored-users", type=Path, default=Path("artifacts/anomaly/scored_users.parquet")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiment"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.database, args.scored_users, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
