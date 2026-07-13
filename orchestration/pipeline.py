"""Run generation, ingestion, dbt, scoring, and experiment analysis as one flow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prefect import flow, task

from anomaly_detection.scoring import run_scoring
from data_generator.generate import (
    GeneratorConfig,
    RewardLensGenerator,
    dataset_summary,
    write_dataset,
)
from experiment_analysis.analyze import run_analysis
from sql.load_raw import load_raw


ROOT = Path(__file__).resolve().parents[1]


@task(retries=1, retry_delay_seconds=2)
def generate_events(config: GeneratorConfig) -> dict[str, Any]:
    tables = RewardLensGenerator(config).generate()
    output_dir = ROOT / "data" / "generated"
    write_dataset(tables, output_dir, "parquet")
    summary = dataset_summary(tables, config)
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


@task
def ingest_to_duckdb() -> dict[str, int]:
    return load_raw(ROOT / "data" / "generated", ROOT / "data" / "rewardlens.duckdb")


@task
def build_analytics() -> str:
    dbt = Path(sys.executable).parent / "dbt"
    result = subprocess.run(
        [str(dbt), "build", "--profiles-dir", "."],
        cwd=ROOT / "dbt_project",
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@task
def detect_anomalies(seed: int) -> dict[str, object]:
    return run_scoring(
        ROOT / "data" / "rewardlens.duckdb",
        ROOT / "artifacts" / "anomaly",
        seed,
    )


@task
def analyse_rule_experiment(seed: int) -> dict[str, object]:
    return run_analysis(
        ROOT / "data" / "rewardlens.duckdb",
        ROOT / "artifacts" / "anomaly" / "scored_users.parquet",
        ROOT / "artifacts" / "experiment",
        seed,
    )


@flow(name="rewardlens-end-to-end", log_prints=True)
def rewardlens_flow(
    n_users: int = 10_000,
    days: int = 30,
    seed: int = 42,
    fraud_rate: float = 0.09,
    skip_generate: bool = False,
) -> dict[str, Any]:
    config = GeneratorConfig(
        n_users=n_users, days=days, seed=seed, fraud_rate=fraud_rate
    )
    generation = (
        {"status": "reused existing generated files"}
        if skip_generate
        else generate_events(config)
    )
    ingestion = ingest_to_duckdb()
    dbt_output = build_analytics()
    model = detect_anomalies(seed)
    experiment = analyse_rule_experiment(seed)
    manifest = {
        "completed_at": datetime.now(UTC).isoformat(),
        "generation": generation,
        "ingestion": ingestion,
        "model": model,
        "experiment": experiment,
    }
    output = ROOT / "artifacts" / "pipeline_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraud-rate", type=float, default=0.09)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    rewardlens_flow(
        n_users=args.users,
        days=args.days,
        seed=args.seed,
        fraud_rate=args.fraud_rate,
        skip_generate=args.skip_generate,
    )


if __name__ == "__main__":
    main()
