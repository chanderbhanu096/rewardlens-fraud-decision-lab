import numpy as np
import pandas as pd

from anomaly_detection.scoring import MODEL_FEATURES, evaluate_thresholds, score_users


def mock_features(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({feature: rng.lognormal(0, 0.5, n) for feature in MODEL_FEATURES})
    frame["user_id"] = [f"u{i}" for i in range(n)]
    frame["device_id"] = [f"d{i}" for i in range(n)]
    frame["is_emulator"] = False
    frame["is_rooted"] = False
    frame["ad_revenue_usd"] = rng.uniform(0, 2, n)
    frame["reward_cost_usd"] = rng.uniform(0, 2, n)
    frame.loc[n - 10 :, "users_on_device"] = 40
    frame.loc[n - 10 :, "instant_claim_rate"] = 0.95
    return frame


def test_scores_are_bounded_and_complete() -> None:
    scored = score_users(mock_features())
    assert scored.fraud_risk_score.between(0, 1).all()
    assert scored.risk_percentile.between(0, 1).all()
    assert scored.anomaly_explanation.notna().all()
    assert (scored.loc[scored.risk_percentile >= 0.95, "risk_tier"] == "critical").all()
    assert scored.loc[
        scored.risk_percentile.between(0.90, 0.95, inclusive="left"), "risk_tier"
    ].eq("high").all()


def test_threshold_evaluation_has_consistent_confusion_matrix() -> None:
    features = mock_features()
    scored = score_users(features)
    truth = pd.DataFrame(
        {"user_id": features.user_id, "fraud_type": "normal", "is_fraud": False}
    )
    truth.loc[truth.index[-10:], ["fraud_type", "is_fraud"]] = ["reward_abuse", True]
    evaluation = evaluate_thresholds(scored, truth)
    totals = evaluation[["true_positives", "false_positives", "false_negatives", "true_negatives"]].sum(axis=1)
    assert (totals == len(features)).all()
    assert evaluation.estimated_net_savings_usd.notna().all()
