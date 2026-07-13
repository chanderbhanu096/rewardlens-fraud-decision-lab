import pandas as pd

from data_generator.generate import GeneratorConfig, RewardLensGenerator


def generated_tables(n_users: int = 500) -> dict[str, pd.DataFrame]:
    config = GeneratorConfig(n_users=n_users, days=7, seed=123, fraud_rate=0.12)
    return RewardLensGenerator(config).generate()


def test_generator_is_deterministic() -> None:
    first = generated_tables(200)
    second = generated_tables(200)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_primary_keys_and_relationships() -> None:
    tables = generated_tables()
    primary_keys = {
        "users": "user_id",
        "devices": "device_id",
        "installs": "install_id",
        "sessions": "session_id",
        "ad_events": "ad_event_id",
        "reward_claims": "reward_claim_id",
    }
    for table, key in primary_keys.items():
        assert tables[table][key].notna().all()
        assert tables[table][key].is_unique

    assert set(tables["installs"].user_id) <= set(tables["users"].user_id)
    assert set(tables["sessions"].user_id) <= set(tables["users"].user_id)
    assert set(tables["ad_events"].session_id) <= set(tables["sessions"].session_id)
    assert set(tables["reward_claims"].ad_event_id) <= set(tables["ad_events"].ad_event_id)


def test_temporal_constraints() -> None:
    tables = generated_tables()
    sessions = tables["sessions"]
    ads = tables["ad_events"]
    rewards = tables["reward_claims"]
    assert (sessions.ended_at >= sessions.started_at).all()
    ad_session = ads.merge(sessions[["session_id", "started_at"]], on="session_id")
    assert (ad_session.viewed_at >= ad_session.started_at).all()
    reward_ads = rewards.merge(ads[["ad_event_id", "viewed_at"]], on="ad_event_id")
    assert (reward_ads.claimed_at >= reward_ads.viewed_at).all()


def test_planted_fraud_patterns_are_detectable_but_not_perfect() -> None:
    tables = generated_tables(1_000)
    users = tables["users"]
    rewards = tables["reward_claims"].merge(
        users[["user_id", "fraud_type"]], on="user_id"
    )
    delay = rewards.groupby("fraud_type").seconds_after_ad.median()
    assert delay["reward_abuse"] < delay["normal"]
    assert delay["emulator_farm"] < delay["normal"]

    device_sizes = tables["user_devices"].groupby("device_id").size()
    assert device_sizes.max() >= 18
    assert (device_sizes == 1).mean() > 0.8


def test_experiment_is_balanced_within_country() -> None:
    tables = generated_tables()
    assigned = tables["experiment_assignments"].merge(
        tables["users"][["user_id", "country"]], on="user_id"
    )
    counts = assigned.groupby(["country", "variant"]).size().unstack(fill_value=0)
    assert ((counts.control - counts.treatment).abs() <= 1).all()
