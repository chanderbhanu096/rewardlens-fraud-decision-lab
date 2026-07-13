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
    weights = pd.Series(
        {
            "shared_device": 0.22,
            "emulator": 0.15,
            "rooted": 0.07,
            "instant_claims": 0.18,
            "high_frequency": 0.14,
            "low_gameplay": 0.08,
            "high_reward_volume": 0.10,
            "peer_outlier": 0.06,
        }
    )
    score = rules.mul(weights).sum(axis=1)
    explanations = rules.apply(
        lambda row: ", ".join(name for name, active in row.items() if active) or "no hard rule",
        axis=1,
    )
    return score, explanations


def robust_z_scores(frame: pd.DataFrame) -> pd.Series:
    values = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    median = values.median()
    mad = (values - median).abs().median().replace(0, 1.0)
    absolute_z = ((values - median) / (1.4826 * mad)).abs().clip(upper=20)
    top_count = min(5, absolute_z.shape[1])
    raw = np.sort(absolute_z.to_numpy(), axis=1)[:, -top_count:].mean(axis=1)
    return percentile_score(raw)


def score_users(
    frame: pd.DataFrame, config: ScoringConfig = ScoringConfig()
) -> pd.DataFrame:
    scored = frame.copy()
    matrix = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    rule_score, explanations = rule_scores(frame)
    scored["rule_score"] = rule_score
    scored["robust_z_score"] = robust_z_scores(frame).to_numpy()

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

    scored["fraud_risk_score"] = (
        0.34 * scored.rule_score
        + 0.24 * scored.robust_z_score
        + 0.32 * scored.isolation_score
        + 0.10 * scored.cluster_score
    ).clip(0, 1)
    scored["risk_percentile"] = percentile_score(scored.fraud_risk_score).to_numpy()
    scored["risk_tier"] = pd.cut(
        scored.risk_percentile,
        bins=[0, 0.82, 0.90, 0.95, 1.0000001],
        labels=["low", "review", "high", "critical"],
        include_lowest=True,
        right=False,
    ).astype(str)
    scored["anomaly_explanation"] = explanations
    return scored


def evaluate_thresholds(scored: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
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
            evaluated.loc[predicted & actual, "reward_cost_usd"].sum() * 4.0
        )
        false_positive_cost = float(
            (
                0.35
                + evaluated.loc[predicted & ~actual, "ad_revenue_usd"] * 1.5
            ).sum()
        )
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
                "prevented_loss_usd": prevented_loss,
                "false_positive_cost_usd": false_positive_cost,
                "estimated_net_savings_usd": prevented_loss - false_positive_cost,
            }
        )
    return pd.DataFrame(rows)


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
    separation = feature_separation(scored)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_dir / "scored_users.parquet", index=False)
    evaluation.to_csv(output_dir / "threshold_evaluation.csv", index=False)
    separation.to_csv(output_dir / "feature_separation.csv", index=False)

    best = evaluation.loc[evaluation.estimated_net_savings_usd.idxmax()].to_dict()
    metrics: dict[str, object] = {
        "users_scored": len(scored),
        "model_features": MODEL_FEATURES,
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
