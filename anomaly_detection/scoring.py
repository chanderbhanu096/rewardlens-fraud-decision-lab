"""Score RewardLens users with interpretable and unsupervised detectors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


MODEL_FEATURES = [
    "users_on_device",
    "account_creation_span_hours",
    "session_count",
    "active_days",
    "sessions_per_active_day",
    "avg_session_seconds",
    "stddev_session_seconds",
    "total_level_gain",
    "avg_level_gain",
    "distinct_session_hours",
    "ad_views",
    "ad_completion_rate",
    "reward_claims",
    "reward_claim_rate",
    "median_claim_delay_seconds",
    "p10_claim_delay_seconds",
    "instant_claim_rate",
    "net_reward_loss_usd",
    "sessions_peer_zscore",
    "claim_rate_peer_zscore",
    "claim_delay_peer_zscore",
]

FORBIDDEN_SCORING_COLUMNS = {"is_fraud", "fraud_type"}

RULE_WEIGHTS = {
    "shared_device": 0.22,
    "emulator": 0.15,
    "rooted": 0.07,
    "instant_claims": 0.18,
    "high_frequency": 0.14,
    "low_gameplay": 0.08,
    "high_reward_volume": 0.10,
    "peer_outlier": 0.06,
}

DETECTOR_WEIGHTS = {
    "rule_score": 0.34,
    "robust_z_score": 0.24,
    "isolation_score": 0.32,
    "cluster_score": 0.10,
}

THRESHOLDS = {
    "conservative": 0.95,
    "balanced": 0.90,
    "aggressive": 0.82,
}


@dataclass(frozen=True)
class ScoringConfig:
    seed: int = 42
    isolation_contamination: float = 0.10
    dbscan_eps: float = 2.8
    dbscan_min_samples: int = 12


@dataclass(frozen=True)
class EconomicConfig:
    """Illustrative policy economics; production values require owner sign-off."""

    fraud_loss_horizon_multiplier: float = 4.0
    false_positive_fixed_cost_usd: float = 0.35
    false_positive_revenue_multiplier: float = 1.5


def validate_scoring_frame(frame: pd.DataFrame) -> None:
    """Fail fast when the scoring contract or leakage boundary is violated."""

    missing = sorted(set(MODEL_FEATURES + ["user_id"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Scoring frame is missing required columns: {missing}")
    leaked = sorted(FORBIDDEN_SCORING_COLUMNS & set(frame.columns))
    if leaked:
        raise ValueError(f"Offline truth must not enter scoring: {leaked}")
    if frame.empty:
        raise ValueError("Scoring frame must contain at least one user")
    if frame.user_id.isna().any() or frame.user_id.duplicated().any():
        raise ValueError("user_id must be present and unique at scoring time")
    if not np.isclose(sum(RULE_WEIGHTS.values()), 1.0):
        raise ValueError("Rule weights must sum to 1.0")
    if not np.isclose(sum(DETECTOR_WEIGHTS.values()), 1.0):
        raise ValueError("Detector weights must sum to 1.0")


def load_features(database: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(database), read_only=True) as connection:
        features = connection.execute(
            "select * from analytics.mart_user_population_comparison"
        ).fetch_df()
        truth = connection.execute(
            "select * from analytics.mart_evaluation_truth"
        ).fetch_df()
    return features, truth


def percentile_score(values: np.ndarray | pd.Series) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).fillna(0.5)


def rule_scores(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rules = pd.DataFrame(index=frame.index)
    rules["shared_device"] = (frame.users_on_device >= 8).astype(float)
    rules["emulator"] = frame.is_emulator.astype(float)
    rules["rooted"] = frame.is_rooted.astype(float)
    rules["instant_claims"] = (frame.instant_claim_rate >= 0.45).astype(float)
    rules["high_frequency"] = (frame.sessions_per_active_day >= 5).astype(float)
    rules["low_gameplay"] = (frame.avg_level_gain <= 0.25).astype(float)
    rules["high_reward_volume"] = (
        frame.reward_claims >= frame.reward_claims.quantile(0.90)
    ).astype(float)
    rules["peer_outlier"] = (
        frame[["sessions_peer_zscore", "claim_rate_peer_zscore"]]
        .abs()
        .max(axis=1)
        .ge(2.5)
        .astype(float)
    )
    score = rules.mul(pd.Series(RULE_WEIGHTS)).sum(axis=1)
    explanations = rules.apply(
        lambda row: ", ".join(name for name, active in row.items() if active) or "no hard rule",
        axis=1,
    )
    return score, explanations


def robust_z_details(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return cohort-relative score plus the most unusual feature per user."""

    values = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    median = values.median()
    mad = (values - median).abs().median().replace(0, 1.0)
    absolute_z = ((values - median) / (1.4826 * mad)).abs().clip(upper=20).fillna(0)
    top_count = min(5, absolute_z.shape[1])
    raw = np.sort(absolute_z.to_numpy(), axis=1)[:, -top_count:].mean(axis=1)
    top_feature = absolute_z.idxmax(axis=1)
    top_value = absolute_z.max(axis=1)
    return percentile_score(raw), top_feature, top_value


def robust_z_scores(frame: pd.DataFrame) -> pd.Series:
    """Backward-compatible convenience wrapper used by analytical notebooks."""

    return robust_z_details(frame)[0]


def score_users(
    frame: pd.DataFrame, config: ScoringConfig = ScoringConfig()
) -> pd.DataFrame:
    validate_scoring_frame(frame)
    scored = frame.copy()
    matrix = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    rule_score, explanations = rule_scores(frame)
    scored["rule_score"] = rule_score
    robust_score, top_feature, top_robust_z = robust_z_details(frame)
    scored["robust_z_score"] = robust_score.to_numpy()
    scored["top_deviation_feature"] = top_feature.to_numpy()
    scored["top_deviation_robust_z"] = top_robust_z.to_numpy()

    isolation = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        IsolationForest(
            n_estimators=250,
            contamination=config.isolation_contamination,
            random_state=config.seed,
            n_jobs=-1,
        ),
    )
    isolation.fit(matrix)
    isolation_raw = -isolation.decision_function(matrix)
    scored["isolation_score"] = percentile_score(isolation_raw).to_numpy()

    cluster_columns = [
        "users_on_device",
        "sessions_per_active_day",
        "avg_session_seconds",
        "reward_claim_rate",
        "median_claim_delay_seconds",
        "instant_claim_rate",
    ]
    cluster_matrix = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    ).fit_transform(frame[cluster_columns])
    labels = DBSCAN(
        eps=config.dbscan_eps, min_samples=config.dbscan_min_samples, n_jobs=-1
    ).fit_predict(cluster_matrix)
    cluster_sizes = pd.Series(labels).value_counts()
    rarity = pd.Series(labels).map(
        lambda label: 1.0 if label == -1 else 1.0 / np.sqrt(cluster_sizes[label])
    )
    scored["cluster_id"] = labels
    scored["cluster_score"] = percentile_score(rarity).to_numpy()

    contribution_columns = []
    for detector, weight in DETECTOR_WEIGHTS.items():
        contribution = detector.replace("_score", "_contribution")
        scored[contribution] = scored[detector] * weight
        contribution_columns.append(contribution)
    scored["fraud_risk_score"] = scored[contribution_columns].sum(axis=1).clip(0, 1)
    detector_labels = {
        "rule_contribution": "known behavioural rules",
        "robust_z_contribution": "unusual feature values",
        "isolation_contribution": "rare multivariate behaviour",
        "cluster_contribution": "cluster rarity",
    }
    scored["dominant_detector"] = (
        scored[contribution_columns].idxmax(axis=1).map(detector_labels)
    )
    scored["risk_percentile"] = percentile_score(scored.fraud_risk_score).to_numpy()
    scored["risk_tier"] = pd.cut(
        scored.risk_percentile,
        bins=[0, 0.82, 0.90, 0.95, 1.0000001],
        labels=["low", "review", "high", "critical"],
        include_lowest=True,
        right=False,
    ).astype(str)
    scored["anomaly_explanation"] = explanations
    scored["model_explanation"] = (
        "Main model signal: "
        + scored.dominant_detector
        + "; largest robust deviation: "
        + scored.top_deviation_feature.str.replace("_", " ")
        + "."
    )
    return scored


def evaluate_thresholds(
    scored: pd.DataFrame,
    truth: pd.DataFrame,
    economics: EconomicConfig = EconomicConfig(),
) -> pd.DataFrame:
    evaluated = scored.merge(truth, on="user_id", validate="one_to_one")
    rows: list[dict[str, float | str | int]] = []
    for name, cutoff in THRESHOLDS.items():
        predicted = evaluated.risk_percentile >= cutoff
        actual = evaluated.is_fraud.astype(bool)
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        tn = int((~predicted & ~actual).sum())
        prevented_loss = float(
            evaluated.loc[predicted & actual, "reward_cost_usd"].sum()
            * economics.fraud_loss_horizon_multiplier
        )
        false_positive_cost = float(
            (
                economics.false_positive_fixed_cost_usd
                + evaluated.loc[predicted & ~actual, "ad_revenue_usd"]
                * economics.false_positive_revenue_multiplier
            ).sum()
        )
        flagged = max(tp + fp, 1)
        rows.append(
            {
                "threshold": name,
                "risk_percentile_cutoff": cutoff,
                "flagged_users": int(predicted.sum()),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "false_positive_rate": fp / max(fp + tn, 1),
                "specificity": tn / max(tn + fp, 1),
                "f1_score": 2 * tp / max(2 * tp + fp + fn, 1),
                "alerts_per_1000_users": predicted.mean() * 1000,
                "prevented_loss_usd": prevented_loss,
                "false_positive_cost_usd": false_positive_cost,
                "estimated_net_savings_usd": prevented_loss - false_positive_cost,
                "net_savings_per_flagged_user_usd": (
                    prevented_loss - false_positive_cost
                )
                / flagged,
            }
        )
    return pd.DataFrame(rows)


def policy_sensitivity(scored: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Stress-test the recommended policy across plausible cost assumptions."""

    rows: list[pd.DataFrame] = []
    for horizon in (1.0, 2.0, 4.0, 8.0):
        for fixed_cost in (0.10, 0.35, 1.00, 2.00):
            economics = EconomicConfig(
                fraud_loss_horizon_multiplier=horizon,
                false_positive_fixed_cost_usd=fixed_cost,
            )
            evaluated = evaluate_thresholds(scored, truth, economics)
            evaluated["fraud_loss_horizon_multiplier"] = horizon
            evaluated["false_positive_fixed_cost_usd"] = fixed_cost
            best_index = evaluated.estimated_net_savings_usd.idxmax()
            evaluated["is_optimal_in_scenario"] = evaluated.index == best_index
            rows.append(evaluated)
    return pd.concat(rows, ignore_index=True)


def feature_separation(scored: pd.DataFrame) -> pd.DataFrame:
    flagged = scored.risk_percentile >= THRESHOLDS["balanced"]
    rows = []
    for feature in MODEL_FEATURES:
        overall_std = scored[feature].std()
        effect = 0.0 if not overall_std else (
            scored.loc[flagged, feature].mean()
            - scored.loc[~flagged, feature].mean()
        ) / overall_std
        rows.append({"feature": feature, "flagged_standardized_difference": effect})
    return pd.DataFrame(rows).sort_values(
        "flagged_standardized_difference", key=abs, ascending=False
    )


def run_scoring(database: Path, output_dir: Path, seed: int = 42) -> dict[str, object]:
    features, truth = load_features(database)
    scored = score_users(features, ScoringConfig(seed=seed))
    evaluation = evaluate_thresholds(scored, truth)
    sensitivity = policy_sensitivity(scored, truth)
    separation = feature_separation(scored)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_dir / "scored_users.parquet", index=False)
    evaluation.to_csv(output_dir / "threshold_evaluation.csv", index=False)
    sensitivity.to_csv(output_dir / "policy_sensitivity.csv", index=False)
    separation.to_csv(output_dir / "feature_separation.csv", index=False)

    best = evaluation.loc[evaluation.estimated_net_savings_usd.idxmax()].to_dict()
    metrics: dict[str, object] = {
        "users_scored": len(scored),
        "model_features": MODEL_FEATURES,
        "rule_weights": RULE_WEIGHTS,
        "detector_weights": DETECTOR_WEIGHTS,
        "economic_config": EconomicConfig().__dict__,
        "recommended_threshold": best["threshold"],
        "recommended_threshold_metrics": best,
    }
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/rewardlens.duckdb"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/anomaly"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_scoring(args.database, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
