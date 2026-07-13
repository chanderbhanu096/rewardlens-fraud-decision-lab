import json
from pathlib import Path

import pandas as pd
import pytest

from dashboard.charts import (
    policy_sensitivity_chart,
    priority_review_policy,
    recommended_policy,
    retention_forest_chart,
    risk_distribution_chart,
    source_trend_chart,
    threshold_workload_chart,
    traffic_priority_chart,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "artifacts/anomaly/scored_users.parquet")


@pytest.fixture(scope="module")
def thresholds() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts/anomaly/threshold_evaluation.csv")


def test_policy_choice_is_derived_from_artifact(thresholds: pd.DataFrame) -> None:
    changed = thresholds.copy()
    changed.loc[changed.threshold.eq("conservative"), "estimated_net_savings_usd"] = 1e9

    assert recommended_policy(changed).threshold == "conservative"
    assert priority_review_policy(changed).threshold == "conservative"


def test_risk_distribution_has_review_and_priority_boundaries(
    scored: pd.DataFrame, thresholds: pd.DataFrame
) -> None:
    recommendation = recommended_policy(thresholds)
    priority = priority_review_policy(thresholds)
    figure = risk_distribution_chart(
        scored,
        review_cutoff=float(recommendation.risk_percentile_cutoff),
        priority_cutoff=float(priority.risk_percentile_cutoff),
        policy_name=str(recommendation.threshold),
    )

    assert [trace.name for trace in figure.data] == [
        "Monitor",
        "Review",
        "Priority review",
    ]
    assert len(figure.layout.shapes) == 2
    annotations = " ".join(annotation.text for annotation in figure.layout.annotations)
    assert "Review · top 10%" in annotations
    assert "Priority · top 5%" in annotations


def test_threshold_workload_marker_tracks_slider(scored: pd.DataFrame) -> None:
    figure = threshold_workload_chart(scored, selected_cutoff=0.83)

    selected_markers = [trace for trace in figure.data if trace.mode == "markers"]
    assert len(selected_markers) == 2
    assert all(float(trace.x[0]) == pytest.approx(0.83) for trace in selected_markers)
    assert float(figure.layout.shapes[0].x0) == pytest.approx(0.83)
    assert "Selected" in figure.layout.annotations[0].text
    assert figure.layout.hovermode == "x unified"


def test_sensitivity_chart_quantifies_winning_margin() -> None:
    sensitivity = pd.read_csv(ROOT / "artifacts/anomaly/policy_sensitivity.csv")
    figure, summary = policy_sensitivity_chart(sensitivity)

    assert len(summary) == 16
    assert summary.winning_margin_usd.gt(0).all()
    assert set(summary.winner) == {"Balanced"}
    assert figure.data[0].type == "heatmap"
    assert "lead" in figure.layout.title.text.lower()


def test_retention_forest_includes_overall_effect_and_guardrail() -> None:
    countries = pd.read_csv(ROOT / "artifacts/experiment/country_effects.csv")
    experiment = json.loads(
        (ROOT / "artifacts/experiment/experiment_summary.json").read_text()
    )
    margin = float(experiment["retention_noninferiority"]["margin"])
    figure = retention_forest_chart(
        countries,
        retention=experiment["retention_guardrail"],
        noninferiority_margin=margin,
        sample_size=int(experiment["sample_size"]),
    )

    assert {trace.name for trace in figure.data} == {
        "Country estimate",
        "Overall estimate",
    }
    vertical_lines = [
        shape
        for shape in figure.layout.shapes
        if shape.type == "line" and shape.x0 == shape.x1
    ]
    assert any(float(shape.x0) == pytest.approx(margin) for shape in vertical_lines)
    assert any(float(shape.x0) == pytest.approx(0.0) for shape in vertical_lines)
    assert figure.data[1].y[0] == "All users"
    assert "not established" in figure.layout.title.text.lower()


def test_source_trend_exposes_prior_only_baseline() -> None:
    daily = pd.read_parquet(ROOT / "artifacts/anomaly/source_daily_health.parquet")
    publisher = daily.loc[daily.source_type.eq("publisher"), "source_id"].iloc[0]
    figure = source_trend_chart(
        daily[daily.source_type.eq("publisher")],
        source_id=publisher,
        source_label="Publisher",
    )

    assert [trace.name for trace in figure.data] == [
        "Daily instant-claim rate",
        "Prior 7-day baseline",
        "Reward claims",
    ]
    baseline = figure.data[1].y
    assert all(pd.isna(value) for value in baseline[:7])
    assert not pd.isna(baseline[7])
    assert figure.layout.hovermode == "x unified"


def test_traffic_bubbles_encode_users_by_area(scored: pd.DataFrame) -> None:
    monitoring = scored.groupby("publisher_id", as_index=False).agg(
        users=("user_id", "size"),
        review_queue_users=(
            "risk_percentile", lambda value: value.ge(0.90).sum()
        ),
        review_queue_share=(
            "risk_percentile", lambda value: value.ge(0.90).mean()
        ),
        average_risk=("fraud_risk_score", "mean"),
        reward_cost_usd=("reward_cost_usd", "sum"),
    )
    figure = traffic_priority_chart(
        monitoring,
        source_column="publisher_id",
        source_label="Publisher",
        overall_share=float(scored.risk_percentile.ge(0.90).mean()),
    )

    assert all(trace.marker.sizemode == "area" for trace in figure.data)
    assert all(float(trace.marker.sizeref) > 0 for trace in figure.data)


def test_dashboard_does_not_claim_automatic_holds() -> None:
    dashboard_source = (ROOT / "dashboard/app.py").read_text(encoding="utf-8").lower()
    assert "automatic hold" not in dashboard_source
