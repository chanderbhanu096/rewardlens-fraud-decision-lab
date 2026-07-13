import pandas as pd
import pytest

from experiment_analysis.analyze import (
    benjamini_hochberg,
    proportion_test,
    rollout_decision,
)


def test_proportion_test_direction() -> None:
    control = pd.Series([True] * 60 + [False] * 40)
    treatment = pd.Series([True] * 50 + [False] * 50)
    result = proportion_test(control, treatment)
    assert result["absolute_effect"] == pytest.approx(-0.10)
    assert result["ci_low"] < result["absolute_effect"] < result["ci_high"]
    assert 0 <= result["p_value"] <= 1


def test_benjamini_hochberg_is_valid() -> None:
    raw = pd.Series([0.01, 0.04, 0.03, 0.20])
    adjusted = benjamini_hochberg(raw)
    assert adjusted.between(0, 1).all()
    assert (adjusted >= raw).all()


def test_rollout_decision_requires_economics_and_retention_guardrail() -> None:
    assert rollout_decision(100, -0.01) == ("limited_rollout", True)
    assert rollout_decision(100, -0.03) == ("targeted_pilot_only", False)
    assert rollout_decision(-1, -0.01) == ("do_not_ship", True)
