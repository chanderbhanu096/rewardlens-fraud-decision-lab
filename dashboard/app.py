"""RewardLens: post-install reward integrity decision dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.charts import (
    detector_contribution_chart,
    experiment_value_waterfall,
    feature_separation_chart,
    policy_sensitivity_chart,
    policy_tradeoff_chart,
    priority_review_policy,
    recommended_policy,
    retention_forest_chart,
    risk_distribution_chart,
    source_review_share_chart,
    source_trend_chart,
    threshold_workload_chart,
    traffic_priority_chart,
)


ANOMALY = ROOT / "artifacts" / "anomaly"
EXPERIMENT = ROOT / "artifacts" / "experiment"
DATA = ROOT / "data" / "generated"

SIGNAL_LABELS = {
    "shared_device": "≥8 accounts share this device",
    "emulator": "Emulator indicator",
    "rooted": "Rooted or jailbroken indicator",
    "instant_claims": "≥45% of claims occur in under 3 seconds",
    "high_frequency": "≥5 sessions per active day",
    "low_gameplay": "≤0.25 levels gained per session",
    "high_reward_volume": "Reward claims are in the top 10%",
    "peer_outlier": "≥2.5 SD from country × publisher peers",
    "no hard rule": "Unsupervised detectors only",
}

st.set_page_config(
    page_title="RewardLens · Reward integrity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_artifacts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    required = [
        ANOMALY / "scored_users.parquet",
        ANOMALY / "threshold_evaluation.csv",
        EXPERIMENT / "experiment_summary.json",
        EXPERIMENT / "country_effects.csv",
        ANOMALY / "feature_separation.csv",
        ANOMALY / "policy_sensitivity.csv",
        ANOMALY / "source_daily_health.parquet",
        DATA / "manifest.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run `python -m orchestration.pipeline` first. Missing: "
            + ", ".join(path.name for path in missing)
        )
    return (
        pd.read_parquet(required[0]),
        pd.read_csv(required[1]),
        json.loads(required[2].read_text(encoding="utf-8")),
        pd.read_csv(required[3]),
        pd.read_csv(required[4]),
        pd.read_csv(required[5]),
        pd.read_parquet(required[6]),
        json.loads(required[7].read_text(encoding="utf-8")),
    )


def page_intro(question: str, description: str) -> None:
    st.title(question)
    st.caption(description)


def explain_signals(raw_explanation: str) -> str:
    return " · ".join(
        SIGNAL_LABELS.get(signal.strip(), signal.strip().replace("_", " ").title())
        for signal in raw_explanation.split(",")
    )


def render_start_here(manifest: dict, recommendation_name: str) -> None:
    page_intro(
        "Start here: should CoinQuest stop this reward?",
        "One realistic story explains the complete RewardLens system before the technical details.",
    )
    st.info(
        "**The business problem:** stop automated reward abuse without treating unusual "
        "but honest players as fraudsters."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Maya · likely genuine player")
        st.markdown(
            """
- Uses one account on one phone
- Plays three 15-minute sessions
- Watches four ads during gameplay
- Claims each reward after the ad finishes
"""
        )
    with right:
        st.subheader("Emulator farm · likely abuse")
        st.markdown(
            """
- Creates 300 accounts on shared virtual devices
- Repeats sessions every few seconds
- Claims rewards almost immediately
- Produces nearly identical behaviour across accounts
"""
        )

    st.subheader("How one business question becomes a data product")
    steps = st.columns(4)
    steps[0].markdown("**1 · Observe**\n\nRecord installs, sessions, ads and reward claims.")
    steps[1].markdown("**2 · Summarize**\n\nTurn events into behavioural clues called features.")
    steps[2].markdown("**3 · Rank**\n\nCombine rules and anomaly detectors into a risk rank.")
    steps[3].markdown("**4 · Decide**\n\nChoose review or hold policies using value and user harm.")

    st.warning(
        f"**Reference decision: targeted pilot only.** The {recommendation_name} policy "
        "performs best offline, but the simulated retention guardrail does not justify "
        "a global launch."
    )

    with st.expander("Explain it like I am new to data", expanded=True):
        st.markdown(
            """
- An **event** is one recorded action, such as “user claimed a reward.”
- A **feature** is a useful summary, such as “claims per active day.”
- An **anomaly** is behaviour that differs strongly from a relevant comparison group.
- A **risk rank of 97%** means the account ranks above 97% of this batch. It is
  **not** a 97% fraud probability.
- A **false positive** is an honest user incorrectly flagged. That mistake has a
  customer and financial cost.
"""
        )

    with st.expander("Show the senior data-science design"):
        st.markdown(
            """
The score is an explicit ensemble of weighted rules, robust median/MAD deviations,
Isolation Forest and DBSCAN rarity. Planted fraud labels are held in a separate
evaluation mart and enter only after scoring. Threshold selection maximizes modeled
net value subject to operational workload and an experimental retention guardrail.

The current results are deterministic, in-sample measurements on synthetic data.
They demonstrate system design and reasoning—not production model performance.
"""
        )
        config = manifest["config"]
        st.caption(
            f"Reference run: {config['n_users']:,} users · {config['days']} days · "
            f"{config['fraud_rate']:.0%} planted fraud · random seed {config['seed']}"
        )


def render_overview(
    scored: pd.DataFrame, thresholds: pd.DataFrame, manifest: dict
) -> None:
    page_intro(
        "Where should RewardLens intervene?",
        "A decision view of post-install abuse, customer risk, and expected value.",
    )
    recommendation = recommended_policy(thresholds)
    priority = priority_review_policy(thresholds)
    review_cutoff = float(recommendation.risk_percentile_cutoff)
    priority_cutoff = float(priority.risk_percentile_cutoff)
    review_queue = scored.risk_percentile.ge(review_cutoff)

    st.success(
        f"Recommended offline operating point: **{recommendation.threshold.title()}** — "
        f"review the top-ranked {1 - review_cutoff:.0%} and route the top-ranked "
        f"{1 - priority_cutoff:.0%} to priority review."
    )
    metrics = st.columns(4)
    metrics[0].metric(
        "Users monitored",
        f"{len(scored):,}",
        help="Synthetic accounts scored in the current 30-day run.",
    )
    metrics[1].metric(
        f"Review queue ({review_queue.mean():.1%})",
        f"{int(review_queue.sum()):,}",
        help=f"Users at or above the {review_cutoff:.0%} risk-rank cutoff.",
    )
    metrics[2].metric(
        "Offline precision",
        f"{recommendation.precision:.1%}",
        help=(
            f"Recall {recommendation.recall:.1%}; false-positive rate "
            f"{recommendation.false_positive_rate:.1%}. In-sample evaluation against "
            "planted synthetic truth."
        ),
    )
    metrics[3].metric(
        "Scenario savings",
        f"${recommendation.estimated_net_savings_usd:,.0f}",
        help="Four-window projected fraud loss prevented minus modeled legitimate-user friction.",
    )

    st.plotly_chart(
        risk_distribution_chart(
            scored,
            review_cutoff=review_cutoff,
            priority_cutoff=priority_cutoff,
            policy_name=str(recommendation.threshold),
        ),
        width="stretch",
    )

    publisher = (
        scored.groupby("publisher_id", as_index=False)
        .agg(
            users=("user_id", "size"),
            review_queue_users=(
                "risk_percentile",
                lambda value: (value >= review_cutoff).sum(),
            ),
            review_queue_share=(
                "risk_percentile",
                lambda value: (value >= review_cutoff).mean(),
            ),
        )
        .nlargest(10, "review_queue_share")
    )
    st.plotly_chart(
        source_review_share_chart(
            publisher,
            source_column="publisher_id",
            overall_share=float(review_queue.mean()),
        ),
        width="stretch",
    )

    with st.expander("How to read these metrics"):
        st.markdown(
            """
- **Risk score** combines rules, robust distribution checks, Isolation Forest,
  and cluster rarity.
- **Risk rank** compares a user with the current scoring batch; it is not a
  probability of fraud.
- **Offline labels** are synthetic evaluation truth and never enter model features.
- **Net savings** depends on explicit cost assumptions, so the operating
  threshold should be recalibrated when those costs change.
"""
        )
        config = manifest["config"]
        st.caption(
            f"Current run: {config['n_users']:,} synthetic users · {config['days']}-day window · "
            f"{config['fraud_rate']:.0%} planted fraud prevalence · seed {config['seed']}"
        )


def render_investigation(
    scored: pd.DataFrame, separation: pd.DataFrame, review_cutoff: float
) -> None:
    page_intro(
        "Who should an analyst review first?",
        "Filter the review queue, inspect one case, and see which behaviours "
        "separate flagged users.",
    )
    filter_cols = st.columns([1, 1.2, 1.4])
    minimum = filter_cols[0].slider(
        "Minimum risk rank", 0.50, 1.00, review_cutoff, 0.01
    )
    country_filter = filter_cols[1].multiselect(
        "Country", sorted(scored.country.unique()), placeholder="All countries"
    )
    publisher_filter = filter_cols[2].multiselect(
        "Publisher", sorted(scored.publisher_id.unique()), placeholder="All publishers"
    )

    visible = scored[scored.risk_percentile.ge(minimum)].copy()
    if country_filter:
        visible = visible[visible.country.isin(country_filter)]
    if publisher_filter:
        visible = visible[visible.publisher_id.isin(publisher_filter)]
    visible = visible.sort_values("risk_percentile", ascending=False)

    summary = st.columns(3)
    summary[0].metric("Cases in queue", f"{len(visible):,}")
    summary[1].metric("Devices represented", f"{visible.device_id.nunique():,}")
    summary[2].metric("Rewards linked to cases", f"${visible.reward_cost_usd.sum():,.0f}")

    if visible.empty:
        st.info("No cases match these filters.")
        return

    selected_user = st.selectbox(
        "Inspect a case",
        visible.user_id.head(250).tolist(),
        help="The selector shows the 250 highest-risk matching cases.",
    )
    case = visible.loc[visible.user_id.eq(selected_user)].iloc[0]
    with st.container(border=True):
        st.subheader(f"Case {case.user_id}")
        case_metrics = st.columns(5)
        case_metrics[0].metric("Risk rank", f"{case.risk_percentile:.1%}")
        case_metrics[1].metric("Accounts on device", f"{int(case.users_on_device):,}")
        case_metrics[2].metric("Reward claims", f"{int(case.reward_claims):,}")
        case_metrics[3].metric("Median claim delay", f"{case.median_claim_delay_seconds:.1f}s")
        case_metrics[4].metric("Sessions / active day", f"{case.sessions_per_active_day:.1f}")
        st.markdown(f"**Why surfaced:** {explain_signals(case.anomaly_explanation)}")
        st.markdown(
            f"**Model view:** {case.model_explanation} "
            f"The largest robust deviation is {case.top_deviation_robust_z:.1f} MAD-scaled units."
        )
        st.caption(
            "This is an explanation of triggered signals, not proof of fraud. "
            "The decision policy determines review, verification, or hold actions."
        )

    contribution_data = pd.DataFrame(
        {
            "Detector": ["Rules", "Robust deviation", "Isolation Forest", "Cluster rarity"],
            "Contribution": [
                case.rule_contribution,
                case.robust_z_contribution,
                case.isolation_contribution,
                case.cluster_contribution,
            ],
        }
    ).sort_values("Contribution")
    st.plotly_chart(detector_contribution_chart(contribution_data), width="stretch")

    visible["signal_summary"] = visible.anomaly_explanation.map(explain_signals)
    table = visible[
        [
            "user_id",
            "device_id",
            "country",
            "publisher_id",
            "campaign_id",
            "risk_percentile",
            "users_on_device",
            "reward_claims",
            "median_claim_delay_seconds",
            "signal_summary",
        ]
    ].head(500)
    table.columns = [
        "User",
        "Device",
        "Country",
        "Publisher",
        "Campaign",
        "Risk rank",
        "Accounts on device",
        "Reward claims",
        "Median claim delay (s)",
        "Signals",
    ]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Risk rank": st.column_config.ProgressColumn(
                "Risk rank", min_value=0.0, max_value=1.0, format="percent"
            ),
            "Median claim delay (s)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.divider()
    st.subheader("What separates the recommended review queue?")
    st.caption(
        "Standardized mean differences compare flagged and unflagged users. "
        "They describe separation—not causal feature importance—and use the "
        "reference policy rather than the case filters above."
    )
    feature_labels = {
        "users_on_device": "Accounts on device",
        "sessions_per_active_day": "Sessions per active day",
        "avg_session_seconds": "Average session duration",
        "median_claim_delay_seconds": "Median reward-claim delay",
        "p10_claim_delay_seconds": "10th percentile claim delay",
        "instant_claim_rate": "Instant-claim rate",
        "reward_claim_rate": "Reward-claim rate",
        "claim_delay_peer_zscore": "Claim-delay peer z-score",
        "claim_rate_peer_zscore": "Claim-rate peer z-score",
        "sessions_peer_zscore": "Session-frequency peer z-score",
    }
    chart_data = separation.head(10).copy()
    chart_data["feature_label"] = chart_data.feature.map(feature_labels).fillna(
        chart_data.feature.str.replace("_", " ").str.title()
    )
    st.plotly_chart(feature_separation_chart(chart_data), width="stretch")


def render_traffic_health(
    scored: pd.DataFrame, daily_health: pd.DataFrame, review_cutoff: float
) -> None:
    page_intro(
        "Which traffic sources need attention?",
        "Compare source volume, review-queue concentration, and reward exposure "
        "before changing partner policy.",
    )
    group_label = st.radio(
        "Break down traffic by", ["Publisher", "Campaign"], horizontal=True
    )
    group_column = "publisher_id" if group_label == "Publisher" else "campaign_id"
    monitoring = scored.groupby(group_column, as_index=False).agg(
        users=("user_id", "size"),
        review_queue_users=(
            "risk_percentile", lambda value: (value >= review_cutoff).sum()
        ),
        review_queue_share=(
            "risk_percentile", lambda value: (value >= review_cutoff).mean()
        ),
        average_risk=("fraud_risk_score", "mean"),
        reward_cost_usd=("reward_cost_usd", "sum"),
    )
    overall_share = float(scored.risk_percentile.ge(review_cutoff).mean())
    st.plotly_chart(
        traffic_priority_chart(
            monitoring,
            source_column=group_column,
            source_label=group_label,
            overall_share=overall_share,
        ),
        width="stretch",
    )

    display = monitoring.sort_values("review_queue_share", ascending=False).rename(
        columns={
            group_column: group_label,
            "users": "Users",
            "review_queue_users": "Users above cutoff",
            "review_queue_share": "Review-queue share",
            "average_risk": "Average risk score",
            "reward_cost_usd": "Reward cost (USD)",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Review-queue share": st.column_config.NumberColumn(format="percent"),
            "Average risk score": st.column_config.NumberColumn(format="%.3f"),
            "Reward cost (USD)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.caption(
        "Source-level risk is a monitoring signal, not evidence that a publisher caused fraud. "
        "Investigate mix shifts and user-level explanations before taking partner action."
    )

    st.divider()
    st.subheader("Did suspicious reward timing spike?")
    source_type = group_label.lower()
    available_daily = daily_health[daily_health.source_type.eq(source_type)].copy()
    priority_order = (
        monitoring.assign(
            priority_score=lambda frame: (
                (frame.review_queue_share - overall_share).clip(lower=0)
                * frame.reward_cost_usd
            )
        )
        .sort_values("priority_score", ascending=False)[group_column]
        .tolist()
    )
    available_sources = set(available_daily.source_id)
    trend_options = [source for source in priority_order if source in available_sources]
    selected_source = st.selectbox(
        f"Inspect {source_type} trend",
        trend_options,
        help="The highest-priority source from the portfolio chart is selected first.",
    )
    st.plotly_chart(
        source_trend_chart(
            available_daily,
            source_id=selected_source,
            source_label=group_label,
        ),
        width="stretch",
    )
    st.caption(
        "An instant claim occurs less than three seconds after an ad. The baseline "
        "uses exactly the source's prior seven observed days, so the current day "
        "cannot influence its comparison."
    )


def render_decision_lab(
    scored: pd.DataFrame, thresholds: pd.DataFrame, sensitivity: pd.DataFrame
) -> None:
    page_intro(
        "What is the right intervention threshold?",
        "Explore operational load, then compare fixed policies using offline fraud "
        "labels and explicit cost assumptions.",
    )
    recommendation = recommended_policy(thresholds)
    recommended_cutoff = float(recommendation.risk_percentile_cutoff)
    cutoff = st.slider(
        "Risk-rank cutoff",
        0.70,
        0.99,
        recommended_cutoff,
        0.01,
        help="Lower cutoffs catch more users and create more customer friction.",
    )
    flagged = scored[scored.risk_percentile.ge(cutoff)]
    operations = st.columns(4)
    operations[0].metric("Users in review queue", f"{len(flagged):,}")
    operations[1].metric("Traffic in review queue", f"{len(flagged) / len(scored):.1%}")
    operations[2].metric("Rewards linked to alerts", f"${flagged.reward_cost_usd.sum():,.0f}")
    operations[3].metric("Distinct devices", f"{flagged.device_id.nunique():,}")
    st.caption(
        "Live operations should use these label-free workload measures. Precision, recall, "
        "and savings below are offline evaluations against synthetic truth."
    )
    st.plotly_chart(
        threshold_workload_chart(scored, selected_cutoff=cutoff), width="stretch"
    )

    st.subheader("Three pre-defined operating policies")
    comparison = thresholds.copy()
    comparison["Policy"] = comparison.threshold.str.title()
    comparison["Traffic affected"] = comparison.flagged_users / len(scored)
    comparison = comparison[
        [
            "Policy",
            "risk_percentile_cutoff",
            "Traffic affected",
            "precision",
            "recall",
            "false_positive_rate",
            "estimated_net_savings_usd",
        ]
    ].rename(
        columns={
            "risk_percentile_cutoff": "Risk-rank cutoff",
            "precision": "Precision",
            "recall": "Recall",
            "false_positive_rate": "False-positive rate",
            "estimated_net_savings_usd": "Estimated net savings",
        }
    )
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "Risk-rank cutoff": st.column_config.NumberColumn(format="percent"),
            "Traffic affected": st.column_config.NumberColumn(format="percent"),
            "Precision": st.column_config.NumberColumn(format="percent"),
            "Recall": st.column_config.NumberColumn(format="percent"),
            "False-positive rate": st.column_config.NumberColumn(format="percent"),
            "Estimated net savings": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    st.plotly_chart(
        policy_tradeoff_chart(thresholds, total_users=len(scored)), width="stretch"
    )
    with st.expander("Economic assumptions behind this comparison"):
        st.markdown(
            """
- Detected fraudulent reward cost is projected across four comparable future windows.
- Each false positive carries **$0.35** of user-friction cost plus **1.5×** observed ad revenue.
- These values are illustrative business assumptions, not measured production costs.
- Re-ranking these policies under alternative costs is a required sensitivity
  check before deployment.
"""
        )

    st.subheader(
        f"Does {recommendation.threshold.title()} remain best when assumptions change?"
    )
    sensitivity_figure, scenario_summary = policy_sensitivity_chart(sensitivity)
    winner_counts = scenario_summary.winner.value_counts()
    winning_policy = winner_counts.index[0]
    sensitivity_metrics = st.columns(3)
    sensitivity_metrics[0].metric(
        "Most frequent winner",
        winning_policy,
        help=f"Wins {winner_counts.iloc[0]} of {len(scenario_summary)} scenarios.",
    )
    sensitivity_metrics[1].metric(
        "Narrowest lead",
        f"${scenario_summary.winning_margin_usd.min():,.0f}",
        help="Smallest advantage over the second-best policy.",
    )
    sensitivity_metrics[2].metric(
        "Widest lead",
        f"${scenario_summary.winning_margin_usd.max():,.0f}",
        help="Largest advantage over the second-best policy.",
    )
    st.plotly_chart(sensitivity_figure, width="stretch")
    st.caption(
        "Sensitivity analysis varies the fraud-loss horizon and fixed false-positive cost. "
        "A recommendation that changes easily should be treated as assumption-sensitive."
    )


def render_experiment(experiment: dict, countries: pd.DataFrame) -> None:
    page_intro(
        "Did the new fraud rule create net value?",
        "Read the primary cost outcome together with retention, segment "
        "heterogeneity, and multiple-testing risk.",
    )
    primary = experiment["primary_metric"]
    retention = experiment["retention_guardrail"]
    economics = experiment["economics"]
    noninferiority = experiment["retention_noninferiority"]
    decision_label = {
        "targeted_pilot_only": "Targeted pilot only",
        "limited_rollout": "Limited rollout",
        "do_not_ship": "Do not ship",
    }[experiment["decision"]]
    st.warning(
        f"Decision: **{decision_label}** — fraud cost falls sharply, but the "
        f"{noninferiority['margin'] * 100:.0f} pp retention non-inferiority guardrail did not pass."
    )
    result_metrics = st.columns(4)
    result_metrics[0].metric(
        "Fraud cost / assigned user",
        f"${primary['control_mean']:.3f} → ${primary['treatment_mean']:.3f}",
        help="Primary intent-to-treat outcome.",
    )
    result_metrics[1].metric(
        "Fraud-cost reduction",
        f"{-primary['relative_effect']:.1%}",
        help=f"Primary-metric p-value: {primary['p_value']:.3g}.",
    )
    result_metrics[2].metric(
        "Day-7 retention effect",
        f"{retention['absolute_effect'] * 100:+.2f} pp",
        help=(
            f"95% CI {retention['ci_low'] * 100:+.2f} to "
            f"{retention['ci_high'] * 100:+.2f} pp; p={retention['p_value']:.3f}. "
            "The result is statistically inconclusive."
        ),
    )
    result_metrics[3].metric(
        f"Net value · {experiment['variant_counts']['treatment']:,} treated",
        f"${economics['net_value_usd_in_sample']:,.0f}",
        help="In-sample value after modeled blocked legitimate rewards and user friction.",
    )
    st.caption(
        f"Primary effect p < 0.001 · Retention 95% CI "
        f"{retention['ci_low'] * 100:+.2f} to {retention['ci_high'] * 100:+.2f} pp · "
        f"retention p={retention['p_value']:.3f} (inconclusive)"
    )
    st.plotly_chart(experiment_value_waterfall(economics), width="stretch")

    st.subheader("Is the retention risk acceptably small?")
    st.plotly_chart(
        retention_forest_chart(
            countries,
            retention=retention,
            noninferiority_margin=float(noninferiority["margin"]),
            sample_size=int(experiment["sample_size"]),
        ),
        width="stretch",
    )
    confirmed_segments = int(countries.adjusted_p_value.lt(0.05).sum())
    st.caption(
        f"{confirmed_segments} of {len(countries)} country effects remain significant "
        "after Benjamini–Hochberg correction. Hollow country markers emphasize that "
        "these segment estimates are exploratory."
    )

    st.info(
        "Do not optimize the apparent claim reduction among flagged treatment users. "
        "That is a post-assignment subgroup and omits retention, false positives, "
        "and legitimate rewards."
    )
    with st.expander("Why this is not a global-launch decision", expanded=True):
        st.markdown(
            """
1. The retention interval crosses zero by a very small margin and its lower bound
   fails the -2 percentage-point non-inferiority margin.
2. Country estimates point in different directions, and only adjusted p-values
   should guide segment claims.
3. The next test should pre-register fraud cost per assigned user as primary and
   retention as a non-inferiority guardrail.
"""
        )


try:
    (
        scored,
        thresholds,
        experiment,
        countries,
        separation,
        sensitivity,
        daily_health,
        manifest,
    ) = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.sidebar.title("🔎 RewardLens")
st.sidebar.caption("Post-install reward integrity decision lab")
page = st.sidebar.radio(
    "Navigate",
    [
        "0 · Start here",
        "1 · Decision overview",
        "2 · Investigation queue",
        "3 · Traffic health",
        "4 · Threshold decision lab",
        "5 · Experiment readout",
    ],
)
st.sidebar.divider()
st.sidebar.markdown("**Optimize the decision, not the score.**")
config = manifest["config"]
recommended = recommended_policy(thresholds)
review_cutoff = float(recommended.risk_percentile_cutoff)
st.sidebar.caption(
    f"{config['n_users']:,} synthetic users · {config['days']}-day window\n\n"
    "Evaluation labels are isolated from scoring features."
)

if page == "0 · Start here":
    render_start_here(manifest, str(recommended.threshold))
elif page == "1 · Decision overview":
    render_overview(scored, thresholds, manifest)
elif page == "2 · Investigation queue":
    render_investigation(scored, separation, review_cutoff)
elif page == "3 · Traffic health":
    render_traffic_health(scored, daily_health, review_cutoff)
elif page == "4 · Threshold decision lab":
    render_decision_lab(scored, thresholds, sensitivity)
else:
    render_experiment(experiment, countries)
