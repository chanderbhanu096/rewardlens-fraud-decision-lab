"""Decision-focused Plotly charts for the RewardLens dashboard.

The chart builders live outside the Streamlit page so their decision semantics,
reference lines, and labels can be tested without starting the application.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "primary": "#0F766E",
    "primary_soft": "#99D5CF",
    "blue": "#2563EB",
    "blue_soft": "#BFDBFE",
    "warning": "#C2410C",
    "warning_soft": "#FED7AA",
    "danger": "#B91C1C",
    "danger_soft": "#FECACA",
    "neutral": "#64748B",
    "neutral_soft": "#CBD5E1",
    "grid": "rgba(100, 116, 139, 0.16)",
}

POLICY_COLORS = {
    "Conservative": COLORS["blue"],
    "Balanced": COLORS["primary"],
    "Aggressive": COLORS["warning"],
}

DETECTOR_COLORS = {
    "Rules": "#0F766E",
    "Robust deviation": "#2563EB",
    "Isolation Forest": "#7C3AED",
    "Cluster rarity": "#C2410C",
}

COUNTRY_NAMES = {
    "BR": "Brazil",
    "DE": "Germany",
    "GB": "United Kingdom",
    "ID": "Indonesia",
    "IN": "India",
    "PL": "Poland",
    "TR": "Türkiye",
    "US": "United States",
}


def clean_chart(figure: go.Figure, *, height: int = 390) -> go.Figure:
    """Apply one accessible, low-noise visual grammar to every chart."""

    legend_traces = [
        trace
        for trace in figure.data
        if trace.showlegend is not False and getattr(trace, "name", None)
    ]
    has_legend = figure.layout.showlegend is not False and len(legend_traces) > 1
    hovermode = figure.layout.hovermode or "closest"
    figure.update_layout(
        height=height,
        margin=dict(l=16, r=24, t=58, b=84 if has_legend else 28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        title=dict(x=0.01, xanchor="left", font=dict(size=17)),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0.0,
            title_text="",
        ),
        hoverlabel=dict(namelength=-1),
        hovermode=hovermode,
        showlegend=has_legend,
    )
    figure.update_xaxes(
        automargin=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        title_standoff=12,
    )
    figure.update_yaxes(
        automargin=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        title_standoff=12,
    )
    return figure


def recommended_policy(thresholds: pd.DataFrame) -> pd.Series:
    """Return the value-maximizing policy from the evaluated artifact."""

    if thresholds.empty:
        raise ValueError("At least one evaluated policy is required")
    return thresholds.loc[thresholds.estimated_net_savings_usd.idxmax()]


def priority_review_policy(thresholds: pd.DataFrame) -> pd.Series:
    """Return the narrowest evaluated queue for higher-priority review."""

    if thresholds.empty:
        raise ValueError("At least one evaluated policy is required")
    return thresholds.loc[thresholds.risk_percentile_cutoff.idxmax()]


def risk_distribution_chart(
    scored: pd.DataFrame,
    *,
    review_cutoff: float,
    priority_cutoff: float,
    policy_name: str,
) -> go.Figure:
    """Show the score distribution with action zones defined by risk rank."""

    data = scored[["fraud_risk_score", "risk_percentile"]].copy()
    data["action"] = np.select(
        [
            data.risk_percentile >= priority_cutoff,
            data.risk_percentile >= review_cutoff,
        ],
        ["Priority review", "Review"],
        default="Monitor",
    )
    score_min = float(data.fraud_risk_score.min())
    score_max = float(data.fraud_risk_score.max())
    bin_size = (score_max - score_min) / 44
    review_score = float(
        data.loc[data.risk_percentile >= review_cutoff, "fraud_risk_score"].min()
    )
    priority_score = float(
        data.loc[data.risk_percentile >= priority_cutoff, "fraud_risk_score"].min()
    )
    colors = {
        "Monitor": COLORS["neutral_soft"],
        "Review": COLORS["warning_soft"],
        "Priority review": COLORS["warning"],
    }
    figure = go.Figure()
    for action in ("Monitor", "Review", "Priority review"):
        subset = data.loc[data.action.eq(action), "fraud_risk_score"]
        figure.add_histogram(
            x=subset,
            name=action,
            marker_color=colors[action],
            opacity=0.96,
            xbins=dict(start=score_min, end=score_max + bin_size, size=bin_size),
            hovertemplate=f"{action}<br>Risk score %{{x:.3f}}<br>Users %{{y:,}}<extra></extra>",
        )
    figure.update_layout(
        barmode="stack",
        bargap=0.04,
        title=f"Risk-score action zones · {policy_name.title()} policy",
        xaxis_title="Risk score (rank, not probability)",
        yaxis_title="Users",
    )
    figure.add_vline(
        x=review_score,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"Review · top {1 - review_cutoff:.0%}",
        annotation_position="top left",
    )
    if not np.isclose(review_cutoff, priority_cutoff):
        figure.add_vline(
            x=priority_score,
            line_dash="dot",
            line_color=COLORS["danger"],
            annotation_text=f"Priority · top {1 - priority_cutoff:.0%}",
            annotation_position="top right",
        )
    return clean_chart(figure, height=430)


def source_review_share_chart(
    sources: pd.DataFrame,
    *,
    source_column: str,
    overall_share: float,
) -> go.Figure:
    """Rank sources against the portfolio queue rate without hiding denominators."""

    data = sources.copy().sort_values("review_queue_share")
    data["status"] = np.select(
        [
            data.review_queue_share >= 2 * overall_share,
            data.review_queue_share >= overall_share,
        ],
        ["At least 2× portfolio", "Above portfolio"],
        default="At or below portfolio",
    )
    status_colors = {
        "At least 2× portfolio": COLORS["warning"],
        "Above portfolio": COLORS["blue"],
        "At or below portfolio": COLORS["neutral_soft"],
    }
    figure = go.Figure()
    for status in status_colors:
        subset = data[data.status.eq(status)]
        figure.add_bar(
            x=subset.review_queue_share,
            y=subset[source_column],
            orientation="h",
            name=status,
            marker_color=status_colors[status],
            text=[f"{value:.1%}" for value in subset.review_queue_share],
            textposition="outside",
            cliponaxis=False,
            customdata=subset[["review_queue_users", "users"]],
            hovertemplate=(
                "%{y}<br>Queue share %{x:.1%}"
                "<br>Review users %{customdata[0]:,} of %{customdata[1]:,}"
                "<extra></extra>"
            ),
        )
    figure.add_vline(
        x=overall_share,
        line_dash="dash",
        line_color=COLORS["neutral"],
    )
    figure.add_annotation(
        x=overall_share,
        y=1.01,
        yref="paper",
        text=f"Portfolio · {overall_share:.1%}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color=COLORS["neutral"], size=11),
    )
    figure.update_layout(
        title="Publishers vs. portfolio queue rate",
        xaxis_title="Share sent to review",
        yaxis_title="Publisher",
        barmode="overlay",
    )
    figure.update_xaxes(
        tickformat=".0%",
        range=[0, max(float(data.review_queue_share.max()) * 1.23, overall_share * 2.4)],
    )
    figure.update_yaxes(
        categoryorder="array", categoryarray=data[source_column].tolist()
    )
    return clean_chart(figure, height=430)


def detector_contribution_chart(contributions: pd.DataFrame) -> go.Figure:
    """Explain one score as additive, weighted detector contributions."""

    data = contributions.copy()
    total = float(data.Contribution.sum())
    data["share"] = data.Contribution / total if total else 0.0
    data = data.sort_values("Contribution")
    dominant = data.iloc[-1]
    figure = go.Figure(
        go.Bar(
            x=data.Contribution,
            y=data.Detector,
            orientation="h",
            marker_color=[DETECTOR_COLORS[name] for name in data.Detector],
            text=[f"{share:.0%}" for share in data.share],
            textposition="outside",
            cliponaxis=False,
            customdata=data[["share"]],
            hovertemplate=(
                "%{y}<br>Weighted contribution %{x:.3f}"
                "<br>Share of combined score %{customdata[0]:.1%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=(
            f"{dominant.Detector} supplies {dominant.share:.0%} of the "
            f"{total:.3f} combined score"
        ),
        xaxis_title="Weighted score contribution",
        yaxis_title="",
        showlegend=False,
    )
    figure.update_xaxes(range=[0, max(float(data.Contribution.max()) * 1.28, 0.01)])
    figure.update_yaxes(showgrid=False)
    return clean_chart(figure, height=320)


def feature_separation_chart(separation: pd.DataFrame) -> go.Figure:
    """Render standardized differences as a signed, zero-centred comparison."""

    data = separation.copy().sort_values("flagged_standardized_difference")
    data["direction"] = np.where(
        data.flagged_standardized_difference >= 0,
        "Higher among reviewed",
        "Lower among reviewed",
    )
    colors = {
        "Higher among reviewed": COLORS["primary"],
        "Lower among reviewed": COLORS["blue"],
    }
    figure = go.Figure()
    for direction in colors:
        subset = data[data.direction.eq(direction)]
        figure.add_bar(
            x=subset.flagged_standardized_difference,
            y=subset.feature_label,
            orientation="h",
            name=direction,
            marker_color=colors[direction],
            text=[f"{value:+.1f}" for value in subset.flagged_standardized_difference],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>Standardized difference %{x:+.2f}"
                f"<br>{direction}<extra></extra>"
            ),
        )
    limit = max(float(data.flagged_standardized_difference.abs().max()) * 1.22, 1.0)
    figure.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
    figure.update_layout(
        title="Reviewed accounts act faster, more often, and with less gameplay",
        xaxis_title="Standardized difference",
        yaxis_title="Behaviour",
        barmode="overlay",
    )
    figure.update_xaxes(range=[-limit, limit])
    figure.update_yaxes(
        categoryorder="array", categoryarray=data.feature_label.tolist(), showgrid=False
    )
    return clean_chart(figure, height=455)


def traffic_priority_chart(
    monitoring: pd.DataFrame,
    *,
    source_column: str,
    source_label: str,
    overall_share: float,
) -> go.Figure:
    """Prioritize sources by queue concentration and financial exposure."""

    data = monitoring.copy()
    median_exposure = float(data.reward_cost_usd.median())
    data["status"] = np.select(
        [
            (data.review_queue_share >= overall_share)
            & (data.reward_cost_usd >= median_exposure),
            data.review_queue_share >= overall_share,
        ],
        ["Investigate first", "Elevated concentration"],
        default="Portfolio range",
    )
    data["priority_score"] = (
        (data.review_queue_share - overall_share).clip(lower=0)
        * data.reward_cost_usd
    )
    labelled_sources = data.nlargest(min(4, len(data)), "priority_score")[
        source_column
    ].tolist()
    labelled = set(labelled_sources)
    label_positions = dict(
        zip(
            labelled_sources,
            ["top center", "middle left", "top left", "bottom right"],
        )
    )
    max_users = max(float(data.users.max()), 1.0)
    size_reference = 2 * max_users / 34**2
    colors = {
        "Investigate first": COLORS["warning"],
        "Elevated concentration": COLORS["blue"],
        "Portfolio range": COLORS["neutral_soft"],
    }
    figure = go.Figure()
    for status in colors:
        subset = data[data.status.eq(status)]
        figure.add_scatter(
            x=subset.reward_cost_usd,
            y=subset.review_queue_share,
            mode="markers+text",
            name=status,
            marker=dict(
                size=subset.users,
                sizemode="area",
                sizeref=size_reference,
                sizemin=10,
                color=colors[status],
                opacity=0.88,
                line=dict(color="rgba(255,255,255,0.85)", width=1),
            ),
            text=[value if value in labelled else "" for value in subset[source_column]],
            textposition=[
                label_positions.get(value, "top center")
                for value in subset[source_column]
            ],
            customdata=subset[
                [source_column, "users", "review_queue_users", "average_risk"]
            ],
            hovertemplate=(
                "%{customdata[0]}<br>Reward exposure $%{x:,.0f}"
                "<br>Queue share %{y:.1%}"
                "<br>Review users %{customdata[2]:,} of %{customdata[1]:,}"
                "<br>Average risk score %{customdata[3]:.3f}<extra></extra>"
            ),
        )
    figure.add_hline(
        y=overall_share,
        line_dash="dash",
        line_color=COLORS["neutral"],
        annotation_text=f"Portfolio queue rate · {overall_share:.1%}",
        annotation_position="bottom right",
    )
    figure.add_vline(
        x=median_exposure,
        line_dash="dot",
        line_color=COLORS["neutral"],
        annotation_text="Median exposure",
        annotation_position="top left",
    )
    figure.add_annotation(
        x=0.99,
        y=0.98,
        xref="paper",
        yref="paper",
        text="High concentration + high exposure ↗",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(color=COLORS["warning"]),
    )
    figure.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text="Bubble size = attributed users",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color=COLORS["neutral"], size=11),
    )
    figure.update_layout(
        title=f"{source_label}s: prioritize concentration and reward exposure",
        xaxis_title="Reward exposure (USD)",
        yaxis_title="Review-queue share",
    )
    figure.update_xaxes(tickprefix="$", tickformat=",.0f")
    figure.update_yaxes(tickformat=".0%")
    return clean_chart(figure, height=500)


def source_trend_chart(
    daily: pd.DataFrame,
    *,
    source_id: str,
    source_label: str,
) -> go.Figure:
    """Compare a daily abuse signal with its prior-seven-day source baseline."""

    data = daily[daily.source_id.eq(source_id)].sort_values("activity_date").copy()
    if data.empty:
        raise ValueError(f"No daily monitoring data found for {source_id}")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.72, 0.28],
    )
    figure.add_scatter(
        x=data.activity_date,
        y=data.instant_claim_rate,
        mode="lines+markers",
        name="Daily instant-claim rate",
        line=dict(color=COLORS["warning"], width=2.5),
        marker=dict(size=6),
        hovertemplate="%{x|%b %d}<br>Instant claims %{y:.1%}<extra></extra>",
        row=1,
        col=1,
    )
    figure.add_scatter(
        x=data.activity_date,
        y=data.prior_7d_instant_claim_rate,
        mode="lines",
        name="Prior 7-day baseline",
        line=dict(color=COLORS["blue"], width=2, dash="dash"),
        hovertemplate="%{x|%b %d}<br>Prior 7-day baseline %{y:.1%}<extra></extra>",
        row=1,
        col=1,
    )
    figure.add_bar(
        x=data.activity_date,
        y=data.reward_claims,
        name="Reward claims",
        marker_color=COLORS["neutral_soft"],
        hovertemplate="%{x|%b %d}<br>Reward claims %{y:,}<extra></extra>",
        row=2,
        col=1,
    )
    eligible = data.dropna(subset=["instant_claim_rate_lift_pp"])
    if not eligible.empty:
        spike = eligible.loc[eligible.instant_claim_rate_lift_pp.idxmax()]
        figure.add_annotation(
            x=spike.activity_date,
            y=spike.instant_claim_rate,
            text=f"Largest lift · {spike.instant_claim_rate_lift_pp:+.1f} pp",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-42,
            font=dict(color=COLORS["warning"]),
            row=1,
            col=1,
        )
    figure.update_layout(
        title=(
            f"{source_label} {source_id}: daily instant claims vs its prior 7-day baseline"
        ),
        hovermode="x unified",
    )
    figure.update_yaxes(title_text="Instant-claim rate", tickformat=".0%", row=1, col=1)
    figure.update_yaxes(title_text="Claims", tickformat=",d", row=2, col=1)
    figure.update_xaxes(title_text="Activity date", row=2, col=1)
    return clean_chart(figure, height=520)


def threshold_workload_chart(
    scored: pd.DataFrame, *, selected_cutoff: float
) -> go.Figure:
    """Make the label-free operational effect of the threshold slider visible."""

    cutoffs = np.round(np.arange(0.70, 0.991, 0.01), 2)
    total_reward = float(scored.reward_cost_usd.sum())
    rows = []
    for cutoff in cutoffs:
        flagged = scored.risk_percentile.ge(cutoff)
        rows.append(
            {
                "cutoff": cutoff,
                "traffic_share": float(flagged.mean()),
                "reward_share": float(
                    scored.loc[flagged, "reward_cost_usd"].sum() / total_reward
                ),
            }
        )
    curve = pd.DataFrame(rows)
    selected = curve.iloc[(curve.cutoff - selected_cutoff).abs().argsort()[:1]].iloc[0]
    figure = go.Figure()
    series = (
        ("Traffic sent to review", "traffic_share", COLORS["primary"]),
        ("Reward value linked to queue", "reward_share", COLORS["blue"]),
    )
    for label, column, color in series:
        figure.add_scatter(
            x=curve.cutoff,
            y=curve[column],
            mode="lines",
            name=label,
            line=dict(color=color, width=2.5),
            hovertemplate=f"Cutoff %{{x:.0%}}<br>{label} %{{y:.1%}}<extra></extra>",
        )
        figure.add_scatter(
            x=[selected.cutoff],
            y=[selected[column]],
            mode="markers",
            name=f"Selected · {label}",
            showlegend=False,
            marker=dict(color=color, size=11, line=dict(color="white", width=2)),
            hovertemplate=f"Selected cutoff %{{x:.0%}}<br>{label} %{{y:.1%}}<extra></extra>",
        )
    figure.add_vline(
        x=selected.cutoff,
        line_dash="dot",
        line_color=COLORS["neutral"],
        annotation_text=(
            f"Selected · {selected.traffic_share:.1%} traffic / "
            f"{selected.reward_share:.1%} reward value"
        ),
        annotation_position="top left",
    )
    figure.update_layout(
        title="Higher cutoffs shrink workload and concentrate reward exposure",
        xaxis_title="Risk-rank cutoff",
        yaxis_title="Share of total",
        hovermode="x unified",
    )
    figure.update_xaxes(tickformat=".0%")
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    return clean_chart(figure, height=410)


def policy_tradeoff_chart(thresholds: pd.DataFrame, *, total_users: int) -> go.Figure:
    """Compare modeled value and operational load without a dual-axis chart."""

    data = thresholds.copy()
    data["Policy"] = data.threshold.str.title()
    data["traffic_share"] = data.flagged_users / total_users
    winner = data.loc[data.estimated_net_savings_usd.idxmax(), "Policy"]
    order = [
        name
        for name in ("Conservative", "Balanced", "Aggressive")
        if name in set(data.Policy)
    ]
    remaining = [name for name in data.Policy if name not in order]
    order.extend(remaining)
    data = data.set_index("Policy").loc[order].reset_index()
    colors = [POLICY_COLORS.get(policy, COLORS["neutral"]) for policy in data.Policy]
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.13,
        subplot_titles=("Modeled net savings", "Traffic sent to review"),
    )
    figure.add_bar(
        x=data.estimated_net_savings_usd,
        y=data.Policy,
        orientation="h",
        marker_color=colors,
        text=[f"${value:,.0f}" for value in data.estimated_net_savings_usd],
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
        hovertemplate="%{y}<br>Net savings $%{x:,.0f}<extra></extra>",
        row=1,
        col=1,
    )
    figure.add_bar(
        x=data.traffic_share,
        y=data.Policy,
        orientation="h",
        marker_color=colors,
        text=[f"{value:.1%}" for value in data.traffic_share],
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
        customdata=data[["flagged_users"]],
        hovertemplate=(
            "%{y}<br>Traffic affected %{x:.1%}"
            "<br>Users %{customdata[0]:,}<extra></extra>"
        ),
        row=1,
        col=2,
    )
    figure.update_layout(title=f"{winner} creates the most value at a manageable queue size")
    figure.update_xaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    figure.update_xaxes(tickformat=".0%", row=1, col=2)
    figure.update_yaxes(
        categoryorder="array", categoryarray=order[::-1], showgrid=False, row=1, col=1
    )
    return clean_chart(figure, height=350)


def summarize_policy_sensitivity(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Return the winner and runner-up margin for every economic scenario."""

    rows: list[dict[str, float | str]] = []
    keys = ["false_positive_fixed_cost_usd", "fraud_loss_horizon_multiplier"]
    for (fixed_cost, horizon), scenario in sensitivity.groupby(keys, sort=True):
        ranked = scenario.sort_values("estimated_net_savings_usd", ascending=False)
        winner = ranked.iloc[0]
        runner_up = ranked.iloc[1]
        rows.append(
            {
                "false_positive_fixed_cost_usd": float(fixed_cost),
                "fraud_loss_horizon_multiplier": float(horizon),
                "winner": str(winner.threshold).title(),
                "runner_up": str(runner_up.threshold).title(),
                "winning_margin_usd": float(
                    winner.estimated_net_savings_usd
                    - runner_up.estimated_net_savings_usd
                ),
            }
        )
    return pd.DataFrame(rows)


def policy_sensitivity_chart(sensitivity: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    """Show whether the recommendation wins narrowly or decisively."""

    summary = summarize_policy_sensitivity(sensitivity)
    matrix = summary.pivot(
        index="false_positive_fixed_cost_usd",
        columns="fraud_loss_horizon_multiplier",
        values="winning_margin_usd",
    )
    winner_matrix = summary.pivot(
        index="false_positive_fixed_cost_usd",
        columns="fraud_loss_horizon_multiplier",
        values="winner",
    ).loc[matrix.index, matrix.columns]
    runner_matrix = summary.pivot(
        index="false_positive_fixed_cost_usd",
        columns="fraud_loss_horizon_multiplier",
        values="runner_up",
    ).loc[matrix.index, matrix.columns]
    customdata = np.dstack([winner_matrix.to_numpy(), runner_matrix.to_numpy()])
    unique_winners = summary.winner.unique().tolist()
    title = (
        f"{unique_winners[0]} wins every scenario; color shows its lead"
        if len(unique_winners) == 1
        else "Winning-policy advantage across economic assumptions"
    )
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            customdata=customdata,
            colorscale=[
                [0.0, "#D9F0ED"],
                [0.45, "#77C5BC"],
                [1.0, COLORS["primary"]],
            ],
            colorbar=dict(title="Lead over<br>runner-up", tickprefix="$", tickformat=",.0f"),
            text=[[f"${value:,.0f}" for value in row] for row in matrix.to_numpy()],
            texttemplate="%{text}",
            hovertemplate=(
                "Fraud horizon %{x:g}×<br>False-positive cost $%{y:.2f}"
                "<br>Winner %{customdata[0]}"
                "<br>Runner-up %{customdata[1]}"
                "<br>Winning margin $%{z:,.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Fraud-loss projection horizon (multiples)",
        yaxis_title="Fixed cost per false positive (USD)",
    )
    figure.update_xaxes(ticksuffix="×")
    figure.update_yaxes(tickprefix="$", tickformat=".2f")
    return clean_chart(figure, height=410), summary


def experiment_value_waterfall(economics: dict[str, float]) -> go.Figure:
    """Decompose experiment value into prevented loss and modeled harm."""

    prevented = float(economics["prevented_fraud_loss_usd"])
    blocked = -float(economics["legitimate_reward_blocked_usd"])
    friction = -float(economics["user_friction_cost_usd"])
    net = float(economics["net_value_usd_in_sample"])
    labels = [
        "Fraud loss prevented",
        "Legitimate rewards blocked",
        "User friction",
        "Net value",
    ]
    values = [prevented, blocked, friction, net]
    display_values = [
        (f"+${value:,.0f}" if value >= 0 else f"-${abs(value):,.0f}")
        if index < 3
        else f"${value:,.0f}"
        for index, value in enumerate(values)
    ]
    figure = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=["relative", "relative", "relative", "total"],
            text=display_values,
            textposition="outside",
            connector=dict(line=dict(color=COLORS["neutral_soft"])),
            increasing=dict(marker=dict(color=COLORS["primary"])),
            decreasing=dict(marker=dict(color=COLORS["warning"])),
            totals=dict(marker=dict(color=COLORS["blue"])),
            hovertemplate="%{x}<br>%{y:$,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Fraud savings outweigh modeled customer costs in this sample",
        yaxis_title="Experiment value (USD)",
        showlegend=False,
    )
    figure.update_yaxes(tickprefix="$", tickformat=",.0f")
    return clean_chart(figure, height=390)


def retention_forest_chart(
    countries: pd.DataFrame,
    *,
    retention: dict[str, float],
    noninferiority_margin: float,
    sample_size: int,
) -> go.Figure:
    """Show overall and country effects against the retention decision boundary."""

    segments = countries.copy()
    segments["label"] = segments.country.map(COUNTRY_NAMES).fillna(segments.country)
    segments = segments.sort_values("absolute_effect")
    overall_label = "All users"
    figure = go.Figure()
    figure.add_scatter(
        x=segments.absolute_effect,
        y=segments.label,
        mode="markers",
        name="Country estimate",
        marker=dict(
            size=10,
            color="rgba(37, 99, 235, 0.12)",
            line=dict(color=COLORS["blue"], width=2),
            symbol="circle-open",
        ),
        error_x=dict(
            type="data",
            symmetric=False,
            array=segments.ci_high - segments.absolute_effect,
            arrayminus=segments.absolute_effect - segments.ci_low,
            color=COLORS["neutral"],
        ),
        customdata=segments[["users", "p_value", "adjusted_p_value"]],
        hovertemplate=(
            "%{y}<br>Retention effect %{x:.2%}<br>Users %{customdata[0]:,}"
            "<br>p=%{customdata[1]:.3f}<br>BH-adjusted p=%{customdata[2]:.3f}"
            "<extra></extra>"
        ),
    )
    overall_effect = float(retention["absolute_effect"])
    figure.add_scatter(
        x=[overall_effect],
        y=[overall_label],
        mode="markers+text",
        name="Overall estimate",
        marker=dict(size=15, color=COLORS["primary"], symbol="diamond"),
        text=[f"{overall_effect * 100:+.2f} pp"],
        textposition="middle right",
        error_x=dict(
            type="data",
            symmetric=False,
            array=[float(retention["ci_high"]) - overall_effect],
            arrayminus=[overall_effect - float(retention["ci_low"])],
            color=COLORS["primary"],
            thickness=2.5,
        ),
        customdata=[[sample_size, float(retention["p_value"])]],
        hovertemplate=(
            "All users<br>Retention effect %{x:.2%}<br>Users %{customdata[0]:,}"
            "<br>p=%{customdata[1]:.3f}<extra></extra>"
        ),
    )
    lower = min(
        float(segments.ci_low.min()),
        float(retention["ci_low"]),
        noninferiority_margin,
    ) - 0.012
    upper = max(float(segments.ci_high.max()), float(retention["ci_high"]), 0.0) + 0.018
    figure.add_vrect(
        x0=lower,
        x1=noninferiority_margin,
        fillcolor="rgba(185, 28, 28, 0.07)",
        line_width=0,
        layer="below",
    )
    figure.add_vline(
        x=noninferiority_margin,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text=f"Guardrail · {noninferiority_margin * 100:.0f} pp",
        annotation_position="top left",
    )
    figure.add_vline(
        x=0,
        line_color=COLORS["neutral"],
        annotation_text="No effect",
        annotation_position="top right",
    )
    order = [overall_label] + segments.label.tolist()
    noninferiority_passed = float(retention["ci_low"]) >= noninferiority_margin
    title = (
        "Retention clears the pre-specified non-inferiority guardrail"
        if noninferiority_passed
        else "Retention non-inferiority is not established"
    )
    figure.update_layout(
        title=title,
        xaxis_title="Day-7 retention effect (percentage points)",
        yaxis_title="",
    )
    figure.update_xaxes(tickformat=".1%", range=[lower, upper])
    figure.update_yaxes(
        categoryorder="array", categoryarray=order[::-1], showgrid=False
    )
    return clean_chart(figure, height=520)
