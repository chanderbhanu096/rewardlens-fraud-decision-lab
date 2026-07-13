"""Generate a relational, labelled post-install behavioural fraud dataset.

The labels are ground truth for offline evaluation. Feature-building code should
exclude ``fraud_type`` and ``is_fraud`` to avoid target leakage.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FRAUD_TYPES = ("normal", "reward_abuse", "emulator_farm", "click_farm")
COUNTRIES = ("DE", "US", "GB", "BR", "IN", "ID", "TR", "PL")
COUNTRY_WEIGHTS = np.array([0.18, 0.18, 0.10, 0.13, 0.15, 0.10, 0.08, 0.08])
PUBLISHERS = tuple(f"pub_{i:02d}" for i in range(1, 13))
CAMPAIGNS = tuple(f"cmp_{i:03d}" for i in range(1, 25))
APPS = ("castle_quest", "word_dash", "idle_tycoon", "puzzle_bloom")


@dataclass(frozen=True)
class GeneratorConfig:
    """Runtime controls for a reproducible dataset."""

    n_users: int = 10_000
    days: int = 30
    seed: int = 42
    start_date: str = "2025-01-01"
    fraud_rate: float = 0.09

    def validate(self) -> None:
        if self.n_users < 100:
            raise ValueError("n_users must be at least 100")
        if not 1 <= self.days <= 365:
            raise ValueError("days must be between 1 and 365")
        if not 0 <= self.fraud_rate <= 0.5:
            raise ValueError("fraud_rate must be between 0 and 0.5")
        pd.Timestamp(self.start_date)


class RewardLensGenerator:
    """Create internally consistent user, device, install, and event tables."""

    def __init__(self, config: GeneratorConfig):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.start = pd.Timestamp(config.start_date, tz="UTC")
        self.end = self.start + pd.Timedelta(days=config.days)

    def generate(self) -> dict[str, pd.DataFrame]:
        """Generate all tables in dependency order."""

        users = self._generate_users()
        devices, user_devices = self._generate_devices(users)
        installs = self._generate_installs(users, user_devices)
        sessions, ad_events, rewards = self._generate_activity(users, installs)
        assignments = self._generate_experiment_assignments(users, installs)
        return {
            "users": users,
            "devices": devices,
            "user_devices": user_devices,
            "installs": installs,
            "sessions": sessions,
            "ad_events": ad_events,
            "reward_claims": rewards,
            "experiment_assignments": assignments,
        }

    def _generate_users(self) -> pd.DataFrame:
        n = self.config.n_users
        fraud_total = int(round(n * self.config.fraud_rate))
        fraud_mix = self.rng.multinomial(fraud_total, [0.38, 0.34, 0.28])
        labels = np.array(
            ["normal"] * (n - fraud_total)
            + ["reward_abuse"] * fraud_mix[0]
            + ["emulator_farm"] * fraud_mix[1]
            + ["click_farm"] * fraud_mix[2],
            dtype=object,
        )
        self.rng.shuffle(labels)

        countries = self.rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
        # Fraud rings have stronger geographic concentration than normal traffic.
        emulator_mask = labels == "emulator_farm"
        click_mask = labels == "click_farm"
        countries[emulator_mask] = self.rng.choice(
            ["BR", "IN", "ID"], emulator_mask.sum(), p=[0.25, 0.45, 0.30]
        )
        countries[click_mask] = self.rng.choice(
            ["IN", "ID", "TR"], click_mask.sum(), p=[0.45, 0.30, 0.25]
        )

        created_offsets = self.rng.uniform(0, self.config.days * 0.72, n)
        created_at = self.start + pd.to_timedelta(created_offsets, unit="D")
        return pd.DataFrame(
            {
                "user_id": [f"usr_{i:07d}" for i in range(1, n + 1)],
                "country": countries,
                "created_at": created_at.floor("s"),
                "fraud_type": labels,
                "is_fraud": labels != "normal",
            }
        )

    def _generate_devices(
        self, users: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        device_rows: list[dict[str, Any]] = []
        link_rows: list[dict[str, Any]] = []
        device_no = 0

        # Normal/reward-abuse accounts mostly own a device; a small fraction share.
        ordinary = users[users.fraud_type.isin(["normal", "reward_abuse"])]
        shared_device: str | None = None
        for row in ordinary.itertuples(index=False):
            should_share = shared_device is not None and self.rng.random() < 0.035
            if not should_share:
                device_no += 1
                shared_device = f"dev_{device_no:07d}"
                device_rows.append(
                    self._device_row(shared_device, row.fraud_type, row.created_at)
                )
            link_rows.append(
                {"user_id": row.user_id, "device_id": shared_device, "is_primary": True}
            )

        # Farm accounts are allocated into conspicuously large device clusters.
        for fraud_type, low, high in (
            ("emulator_farm", 18, 75),
            ("click_farm", 8, 32),
        ):
            group = users[users.fraud_type == fraud_type]
            cursor = 0
            ids = group.user_id.tolist()
            while cursor < len(ids):
                cluster_size = int(self.rng.integers(low, high + 1))
                cluster = ids[cursor : cursor + cluster_size]
                device_no += 1
                device_id = f"dev_{device_no:07d}"
                created = group[group.user_id.isin(cluster)].created_at.min()
                device_rows.append(self._device_row(device_id, fraud_type, created))
                link_rows.extend(
                    {"user_id": user_id, "device_id": device_id, "is_primary": True}
                    for user_id in cluster
                )
                cursor += cluster_size

        devices = pd.DataFrame(device_rows).sort_values("device_id").reset_index(drop=True)
        links = pd.DataFrame(link_rows).sort_values("user_id").reset_index(drop=True)
        return devices, links

    def _device_row(
        self, device_id: str, fraud_type: str, first_seen_at: pd.Timestamp
    ) -> dict[str, Any]:
        is_emulator = fraud_type == "emulator_farm" and self.rng.random() < 0.86
        is_rooted = (
            self.rng.random()
            < {"normal": 0.015, "reward_abuse": 0.13, "emulator_farm": 0.68, "click_farm": 0.20}[fraud_type]
        )
        os_name = self.rng.choice(["android", "ios"], p=[0.77, 0.23])
        if is_emulator:
            os_name = "android"
        return {
            "device_id": device_id,
            "os_name": os_name,
            "os_version": str(self.rng.choice(["12", "13", "14", "15", "16", "17"])),
            "is_emulator": bool(is_emulator),
            "is_rooted": bool(is_rooted),
            "first_seen_at": pd.Timestamp(first_seen_at).floor("s"),
        }

    def _generate_installs(
        self, users: pd.DataFrame, user_devices: pd.DataFrame
    ) -> pd.DataFrame:
        merged = users.merge(user_devices, on="user_id", validate="one_to_one")
        rows: list[dict[str, Any]] = []
        for row in merged.itertuples(index=False):
            install_time = row.created_at + pd.Timedelta(
                minutes=float(self.rng.uniform(0, 180))
            )
            publisher = str(self.rng.choice(PUBLISHERS))
            campaign = str(self.rng.choice(CAMPAIGNS))
            if row.fraud_type == "emulator_farm":
                publisher = str(self.rng.choice(["pub_09", "pub_11"]))
                campaign = str(self.rng.choice(["cmp_019", "cmp_022", "cmp_024"]))
            elif row.fraud_type == "click_farm":
                publisher = str(self.rng.choice(["pub_07", "pub_11", "pub_12"]))
            rows.append(
                {
                    "install_id": f"ins_{len(rows) + 1:08d}",
                    "user_id": row.user_id,
                    "device_id": row.device_id,
                    "installed_at": install_time.floor("s"),
                    "app_id": str(self.rng.choice(APPS)),
                    "country": row.country,
                    "publisher_id": publisher,
                    "campaign_id": campaign,
                    "attribution_cost_usd": round(float(self.rng.lognormal(-0.45, 0.5)), 4),
                }
            )
        return pd.DataFrame(rows)

    def _activity_profile(self, fraud_type: str) -> dict[str, float]:
        return {
            "normal": {"daily_sessions": 0.85, "session_minutes": 8.5, "ads_per_session": 1.2, "claim_rate": 0.78, "claim_delay": 38.0},
            "reward_abuse": {"daily_sessions": 4.2, "session_minutes": 3.2, "ads_per_session": 3.8, "claim_rate": 0.98, "claim_delay": 2.8},
            "emulator_farm": {"daily_sessions": 7.5, "session_minutes": 2.1, "ads_per_session": 2.7, "claim_rate": 0.99, "claim_delay": 1.4},
            "click_farm": {"daily_sessions": 5.4, "session_minutes": 1.6, "ads_per_session": 0.55, "claim_rate": 0.60, "claim_delay": 8.0},
        }[fraud_type]

    def _generate_activity(
        self, users: pd.DataFrame, installs: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        base = installs.merge(users[["user_id", "fraud_type"]], on="user_id")
        session_rows: list[dict[str, Any]] = []
        ad_rows: list[dict[str, Any]] = []
        reward_rows: list[dict[str, Any]] = []

        for row in base.itertuples(index=False):
            profile = self._activity_profile(row.fraud_type)
            active_days = max(1, math.ceil((self.end - row.installed_at).total_seconds() / 86400))
            user_tempo = float(self.rng.lognormal(0, 0.26))
            session_count = max(
                1,
                int(self.rng.poisson(profile["daily_sessions"] * active_days * user_tempo)),
            )
            max_seconds = max(1.0, (self.end - row.installed_at).total_seconds())
            offsets = np.sort(self.rng.uniform(0, max_seconds, session_count))

            # Farms replay a small set of schedules, yielding near-identical timing.
            if row.fraud_type in ("emulator_farm", "click_farm"):
                offsets = np.round(offsets / 300) * 300

            for offset in offsets:
                session_id = f"ses_{len(session_rows) + 1:010d}"
                started_at = row.installed_at + pd.Timedelta(seconds=float(offset))
                duration = float(
                    np.clip(
                        self.rng.lognormal(math.log(profile["session_minutes"]), 0.5),
                        0.25,
                        120,
                    )
                )
                level_gain = int(
                    self.rng.poisson(max(0.1, duration / 3.5))
                    if row.fraud_type == "normal"
                    else self.rng.poisson(0.35)
                )
                session_rows.append(
                    {
                        "session_id": session_id,
                        "user_id": row.user_id,
                        "device_id": row.device_id,
                        "started_at": started_at.floor("s"),
                        "ended_at": (started_at + pd.Timedelta(minutes=duration)).floor("s"),
                        "duration_seconds": int(duration * 60),
                        "level_gain": level_gain,
                    }
                )
                ad_count = int(self.rng.poisson(profile["ads_per_session"]))
                for _ in range(ad_count):
                    ad_id = f"adv_{len(ad_rows) + 1:010d}"
                    ad_offset = float(self.rng.uniform(5, max(6, duration * 60)))
                    viewed_at = started_at + pd.Timedelta(seconds=ad_offset)
                    completed = bool(self.rng.random() < (0.92 if row.fraud_type == "normal" else 0.99))
                    revenue = float(self.rng.lognormal(-4.0, 0.55)) if completed else 0.0
                    ad_rows.append(
                        {
                            "ad_event_id": ad_id,
                            "session_id": session_id,
                            "user_id": row.user_id,
                            "device_id": row.device_id,
                            "viewed_at": viewed_at.floor("s"),
                            "ad_network": str(self.rng.choice(["unity", "applovin", "ironsource"])),
                            "placement": str(self.rng.choice(["level_end", "store", "revive"])),
                            "completed": completed,
                            "revenue_usd": round(revenue, 5),
                        }
                    )
                    if completed and self.rng.random() < profile["claim_rate"]:
                        delay = float(self.rng.exponential(profile["claim_delay"]))
                        # Legitimate claims have a client/UI floor most of the time.
                        if row.fraud_type == "normal":
                            delay += float(self.rng.uniform(4, 18))
                        claimed_at = viewed_at + pd.Timedelta(seconds=delay)
                        reward_rows.append(
                            {
                                "reward_claim_id": f"rwd_{len(reward_rows) + 1:010d}",
                                "ad_event_id": ad_id,
                                "user_id": row.user_id,
                                "device_id": row.device_id,
                                "claimed_at": claimed_at.floor("s"),
                                "seconds_after_ad": round(delay, 3),
                                "reward_type": str(self.rng.choice(["coins", "energy", "gems"], p=[0.65, 0.25, 0.10])),
                                "reward_value_usd": round(float(self.rng.choice([0.01, 0.02, 0.05])), 2),
                            }
                        )

        return (
            pd.DataFrame(session_rows),
            pd.DataFrame(ad_rows),
            pd.DataFrame(reward_rows),
        )

    def _generate_experiment_assignments(
        self, users: pd.DataFrame, installs: pd.DataFrame
    ) -> pd.DataFrame:
        """Deterministic 50/50 assignment, stratified by country."""

        base = users[["user_id", "country"]].merge(
            installs[["user_id", "installed_at"]], on="user_id"
        )
        rows: list[dict[str, Any]] = []
        for _, group in base.groupby("country", sort=True):
            shuffled = group.sample(frac=1, random_state=self.config.seed)
            for position, row in enumerate(shuffled.itertuples(index=False)):
                rows.append(
                    {
                        "experiment_id": "exp_fraud_rule_v1",
                        "user_id": row.user_id,
                        "variant": "treatment" if position % 2 else "control",
                        "assigned_at": row.installed_at,
                    }
                )
        return pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)


def write_dataset(
    tables: dict[str, pd.DataFrame], output_dir: Path, output_format: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if output_format in ("csv", "both"):
            frame.to_csv(output_dir / f"{name}.csv", index=False)
        if output_format in ("parquet", "both"):
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)


def dataset_summary(
    tables: dict[str, pd.DataFrame], config: GeneratorConfig
) -> dict[str, Any]:
    users = tables["users"]
    links = tables["user_devices"]
    rewards = tables["reward_claims"]
    users_per_device = links.groupby("device_id").size()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "row_counts": {name: len(frame) for name, frame in tables.items()},
        "fraud_prevalence": round(float(users.is_fraud.mean()), 6),
        "fraud_type_counts": users.fraud_type.value_counts().to_dict(),
        "max_users_on_one_device": int(users_per_device.max()),
        "median_claim_delay_seconds": round(float(rewards.seconds_after_ad.median()), 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--fraud-rate", type=float, default=0.09)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GeneratorConfig(
        n_users=args.users,
        days=args.days,
        seed=args.seed,
        start_date=args.start_date,
        fraud_rate=args.fraud_rate,
    )
    tables = RewardLensGenerator(config).generate()
    write_dataset(tables, args.output_dir, args.format)
    summary = dataset_summary(tables, config)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
