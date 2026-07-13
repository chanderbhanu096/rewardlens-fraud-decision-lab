"""Load generated Parquet files into the raw schema of DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


TABLES = (
    "users",
    "devices",
    "user_devices",
    "installs",
    "sessions",
    "ad_events",
    "reward_claims",
    "experiment_assignments",
)


def load_raw(data_dir: Path, database: Path) -> dict[str, int]:
    missing = [name for name in TABLES if not (data_dir / f"{name}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing generated tables: {', '.join(missing)}")

    database.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with duckdb.connect(str(database)) as connection:
        connection.execute("create schema if not exists raw")
        for name in TABLES:
            path = (data_dir / f"{name}.parquet").resolve().as_posix()
            connection.execute(
                f'create or replace table raw."{name}" as '
                f"select * from read_parquet('{path}')"
            )
            counts[name] = connection.execute(
                f'select count(*) from raw."{name}"'
            ).fetchone()[0]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--database", type=Path, default=Path("data/rewardlens.duckdb"))
    args = parser.parse_args()
    for table, count in load_raw(args.data_dir, args.database).items():
        print(f"loaded raw.{table}: {count:,} rows")


if __name__ == "__main__":
    main()
