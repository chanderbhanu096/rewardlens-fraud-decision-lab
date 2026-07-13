"""RewardLens: post-install reward integrity decision dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
ANOMALY = ROOT / "artifacts" / "anomaly"
EXPERIMENT = ROOT / "artifacts" / "experiment"
DATA = ROOT / "data" / "generated"

BALANCED_CUTOFF = 0.90
CRITICAL_CUTOFF = 0.95

SIGNAL_LABELS = {
    "shared_device": "≥8 accounts share this device",
    "emulator": "Emulator indicator",
    "rooted": "Rooted or jailbroken indicator",
    "instant_claims": "≥45% of claims occur within 3 seconds",
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
    dict,
]:
    required = [
        ANOMALY / "scored_users.parquet",
        ANOMALY / "threshold_evaluation.csv",
        EXPERIMENT / "experiment_summary.json",
        EXPERIMENT / "country_effects.csv",
        ANOMALY / "feature_separation.csv",
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
        json.loads(required[5].read_text(encoding="utf-8")),
    )


def clean_chart(figure: go.Figure, *, height: int = 390) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        legend_title_text="",
        hoverlabel=dict(namelength=-1),
    )
    return figure


def page_intro(question: str, description: str) -> None:
    st.title(question)
    st.caption(description)


def explain_signals(raw_explanation: str) -> str:
    return " · ".join(
        SIGNAL_LABELS.get(signal.strip(), signal.strip().replace("_", " ").title())
        for signal in raw_explanation.split(",")
    )


def render_overview(
    scored: pd.DataFrame, thresholds: pd.DataFrame, manifest: dict
) -> None:
    page_intro(
        "Where should RewardLens intervene?",
        "A decision view of post-install abuse, customer risk, and expected value.",
    )
    balanced = thresholds.loc[thresholds.threshold.eq("balanced")].iloc[0]
    review_queue = scored.risk_percentile.ge(BALANCED_CUTOFF)

    st.success(
        "Recommended operating point: **Balanced** — review the top-ranked 10% "
        "and reserve automatic holds for the top-ranked 5%."
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
        help="Users at or above the 90th risk percentile.",
    )
    metrics[2].metric(
        "Offline precision",
        f"{balanced.precision:.1%}",
        help=(
            f"Recall {balanced.recall:.1%}; false-positive rate "
            f"{balanced.false_positive_rate:.1%}. In-sample evaluation against "
            "planted synthetic truth."
        ),
    )
    metrics[3].metric(
        "Scenario savings",
        f"${balanced.estimated_net_savings_usd:,.0f}",
        help="Four-window projected fraud loss prevented minus modeled legitimate-user friction.",
    )

    left, right = st.columns([1.2, 1])
    score_cutoff = scored.loc[review_queue, "fraud_risk_score"].min()
    distribution = px.histogram(
        scored,
        x="fraud_risk_score",
        nbins=45,
        title="Risk-score distribution",
        labels={"fraud_risk_score": "Combined fraud-risk score", "count": "Users"},
    )
    distribution.update_traces(marker_color="#0F766E")
    distribution.update_yaxes(title_text="Users")
    distribution.add_vline(
        x=score_cutoff,
        line_dash="dash",
        line_color="#C2410C",
        annotation_text="Balanced cutoff",
        annotation_position="top right",
    )
    left.plotly_chart(clean_chart(distribution), width="stretch")

    publisher = (
        scored.groupby("publisher_id", as_index=False)
        .agg(
            users=("user_id", "size"),
            high_risk_share=("risk_percentile", lambda value: (value >= BALANCED_CUTOFF).mean()),
        )
        .nlargest(10, "high_risk_share")
        .sort_values("high_risk_share")
    )
    source_chart = px.bar(
        publisher,
        x="high_risk_share",
        y="publisher_id",
        orientation="h",
        title="Publishers with the highest review-queue share",
        labels={"high_risk_share": "Users above balanced cutoff", "publisher_id": "Publisher"},
        hover_data={"users": ":,"},
    )
    source_chart.update_traces(marker_color="#2563EB")
    source_chart.update_xaxes(tickformat=".0%")
    right.plotly_chart(clean_chart(source_chart), width="stretch")

    with st.expander("How to read these metrics"):
        st.markdown(
            """
- **Risk score** combines rules, robust distribution checks, Isolation Forest, and cluster rarity.
- **Risk rank** compares a user with the current scoring batch; it is not a probability of fraud.
- **Offline labels** are synthetic evaluation truth and never enter model features.
- **Net savings** depends on explicit cost assumptions, so the operating threshold should be recalibrated when those costs change.
"""
        )
        config = manifest["config"]
        st.caption(
            f"Current run: {config['n_users']:,} synthetic users · {config['days']}-day window · "
            f"{config['fraud_rate']:.0%} planted fraud prevalence · seed {config['seed']}"
        )


def render_investigation(scored: pd.DataFrame, separation: pd.DataFrame) -> None:
    page_intro(
        "Who should an analyst review first?",
        "Filter the review queue, inspect one case, and see which behaviours separate flagged users.",
    )
    filter_cols = st.columns([1, 1.2, 1.4])
    minimum = filter_cols[0].slider(
        "Minimum risk rank", 0.50, 1.00, BALANCED_CUTOFF, 0.01
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
        st.caption(
            "This is an explanation of triggered signals, not proof of fraud. "
            "The decision policy determines review, verification, or hold actions."
        )

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
    st.subheader("What separates the balanced review queue?")
    st.caption(
        "Standardized mean differences compare flagged and unflagged users. "
        "They describe separation; they are not causal feature importance."
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
    chart_data = chart_data.sort_values("flagged_standardized_difference")
    feature_chart = px.bar(
        chart_data,
        x="flagged_standardized_difference",
        y="feature_label",
        orientation="h",
        labels={
            "flagged_standardized_difference": "Standardized difference: flagged minus unflagged",
            "feature_label": "Behaviour",
        },
    )
    feature_chart.update_traces(marker_color="#7C3AED")
    st.plotly_chart(clean_chart(feature_chart, height=420), width="stretch")


def render_traffic_health(scored: pd.DataFrame) -> None:
    page_intro(
        "Which traffic sources need attention?",
        "Compare source volume, review-queue concentration, and reward exposure before changing partner policy.",
    )
    group_label = st.radio("Break down traffic by", ["Publisher", "Campaign"], horizontal=True)
    group_column = "publisher_id" if group_label == "Publisher" else "campaign_id"
    monitoring = scored.groupby(group_column, as_index=False).agg(
        users=("user_id", "size"),
        above_cutoff_users=("risk_percentile", lambda value: (value >= BALANCED_CUTOFF).sum()),
        review_queue_share=("risk_percentile", lambda value: (value >= BALANCED_CUTOFF).mean()),
        average_risk=("fraud_risk_score", "mean"),
        reward_cost_usd=("reward_cost_usd", "sum"),
    )
    overall_share = scored.risk_percentile.ge(BALANCED_CUTOFF).mean()
    source_plot = px.scatter(
        monitoring,
        x="users",
        y="review_queue_share",
        size="reward_cost_usd",
        color="average_risk",
        hover_name=group_column,
        title=f"{group_label} volume vs. review-queue share",
        labels={
            "users": "Attributed users",
            "review_queue_share": "Users above balanced cutoff",
            "reward_cost_usd": "Reward cost (USD)",
            "average_risk": "Average risk score",
        },
        color_continuous_scale="Tealgrn",
    )
    source_plot.update_yaxes(tickformat=".0%")
    source_plot.add_hline(
        y=overall_share,
        line_dash="dash",
        line_color="#6B7280",
        annotation_text="Portfolio average",
        annotation_position="bottom right",
    )
    st.plotly_chart(clean_chart(source_plot, height=470), width="stretch")

    display = monitoring.sort_values("review_queue_share", ascending=False).rename(
        columns={
            group_column: group_label,
            "users": "Users",
            "above_cutoff_users": "Users above cutoff",
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


def render_decision_lab(scored: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    page_intro(
        "What is the right intervention threshold?",
        "Explore operational load, then compare fixed policies using offline fraud labels and explicit cost assumptions.",
    )
    cutoff = st.slider(
        "Risk-rank cutoff",
        0.70,
        0.99,
        BALANCED_CUTOFF,
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

    economics = thresholds.copy()
    economics["Policy"] = economics.threshold.str.title()
    economics_chart = px.bar(
        economics,
        x="Policy",
        y="estimated_net_savings_usd",
        color="Policy",
        title="Balanced maximizes modeled net savings",
        labels={"estimated_net_savings_usd": "Estimated net savings (USD)"},
        text_auto="$.3s",
    )
    economics_chart.update_layout(showlegend=False)
    st.plotly_chart(clean_chart(economics_chart), width="stretch")
    with st.expander("Economic assumptions behind this comparison"):
        st.markdown(
            """
- Detected fraudulent reward cost is projected across four comparable future windows.
- Each false positive carries **$0.35** of user-friction cost plus **1.5×** observed ad revenue.
- These values are illustrative business assumptions, not measured production costs.
- Re-ranking these policies under alternative costs is a required sensitivity check before deployment.
"""
        )


def render_experiment(experiment: dict, countries: pd.DataFrame) -> None:
    page_intro(
        "Did the new fraud rule create net value?",
        "Read the primary cost outcome together with retention, segment heterogeneity, and multiple-testing risk.",
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
        f"USD {primary['control_mean']:.3f} → USD {primary['treatment_mean']:.3f}",
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

    ordered = countries.sort_values("absolute_effect")
    forest = go.Figure(
        go.Scatter(
            x=ordered.absolute_effect,
            y=ordered.country,
            mode="markers",
            marker=dict(size=10, color="#0F766E"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=ordered.ci_high - ordered.absolute_effect,
                arrayminus=ordered.absolute_effect - ordered.ci_low,
                color="#475569",
            ),
            customdata=ordered[["users", "p_value", "adjusted_p_value"]],
            hovertemplate=(
                "Country %{y}<br>Retention effect %{x:.2%}<br>Users %{customdata[0]:,}"
                "<br>p=%{customdata[1]:.3f}<br>adjusted p=%{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    forest.add_vline(x=0, line_dash="dash", line_color="#6B7280")
    forest.update_layout(
        title="Country-level day-7 retention effects (95% confidence intervals)",
        xaxis_title="Treatment effect on retention",
        yaxis_title="Country",
    )
    forest.update_xaxes(tickformat=".1%")
    st.plotly_chart(clean_chart(forest, height=430), width="stretch")

    st.info(
        "Do not optimize the apparent claim reduction among flagged treatment users. "
        "That is a post-assignment subgroup and omits retention, false positives, and legitimate rewards."
    )
    with st.expander("Why this is not a global-launch decision", expanded=True):
        st.markdown(
            """
1. The retention interval crosses zero by a very small margin and its lower bound fails the -2 percentage-point non-inferiority margin.
2. Country estimates point in different directions, and only adjusted p-values should guide segment claims.
3. The next test should pre-register fraud cost per assigned user as primary and retention as a non-inferiority guardrail.
"""
        )


try:
    scored, thresholds, experiment, countries, separation, manifest = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.sidebar.title("🔎 RewardLens")
st.sidebar.caption("Post-install reward integrity decision lab")
page = st.sidebar.radio(
    "Navigate",
    [
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
st.sidebar.caption(
    f"{config['n_users']:,} synthetic users · {config['days']}-day window\n\n"
    "Evaluation labels are isolated from scoring features."
)

if page == "1 · Decision overview":
    render_overview(scored, thresholds, manifest)
elif page == "2 · Investigation queue":
    render_investigation(scored, separation)
elif page == "3 · Traffic health":
    render_traffic_health(scored)
elif page == "4 · Threshold decision lab":
    render_decision_lab(scored, thresholds)
else:
    render_experiment(experiment, countries)
