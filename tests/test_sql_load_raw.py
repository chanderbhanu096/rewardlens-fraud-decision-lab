import duckdb
import pandas as pd
import pytest

from sql.load_raw import TABLES, load_raw


def test_load_raw_creates_all_raw_tables(tmp_path) -> None:
    data_dir = tmp_path / "generated"
    data_dir.mkdir()
    for index, name in enumerate(TABLES):
        pd.DataFrame({"value": [index]}).to_parquet(data_dir / f"{name}.parquet", index=False)

    database = tmp_path / "rewardlens.duckdb"
    counts = load_raw(data_dir, database)

    assert counts == {name: 1 for name in TABLES}
    with duckdb.connect(str(database), read_only=True) as connection:
        loaded_counts = {
            name: connection.execute(
                f'select count(*) from raw."{name}"'
            ).fetchone()[0]
            for name in TABLES
        }
    assert loaded_counts == counts


def test_load_raw_lists_missing_generated_tables(tmp_path) -> None:
    data_dir = tmp_path / "generated"
    data_dir.mkdir()
    pd.DataFrame({"user_id": [1]}).to_parquet(data_dir / "users.parquet", index=False)

    with pytest.raises(FileNotFoundError) as error:
        load_raw(data_dir, tmp_path / "rewardlens.duckdb")

    message = str(error.value)
    assert "users" not in message
    for name in TABLES[1:]:
        assert name in message
