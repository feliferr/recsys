import pandas as pd
import pytest

from recsys.ingestion.sources.synthetic import SyntheticSource
from recsys.ingestion.validation import (
    DataValidationError,
    validate_interactions,
    validate_items,
)


def test_valid_synthetic_tables_pass() -> None:
    src = SyntheticSource()
    items = validate_items(src.items())
    validate_interactions(src.interactions(), set(items["item_id"]))


def test_duplicate_item_id_rejected() -> None:
    items = SyntheticSource().items()
    items.loc[1, "item_id"] = items.loc[0, "item_id"]
    with pytest.raises(DataValidationError, match="duplicated"):
        validate_items(items)


def test_missing_column_rejected() -> None:
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_items(SyntheticSource().items().drop(columns=["genres"]))


def test_invalid_year_rejected() -> None:
    items = SyntheticSource().items()
    items.loc[0, "release_year"] = 5
    with pytest.raises(DataValidationError, match="invalid row"):
        validate_items(items)


def test_null_genres_become_empty_list() -> None:
    items = SyntheticSource().items().astype({"genres": object})
    items.at[0, "genres"] = None
    assert validate_items(items).at[0, "genres"] == []


def test_unknown_item_in_interactions_rejected() -> None:
    src = SyntheticSource()
    inter = src.interactions()
    inter.loc[0, "item_id"] = "does-not-exist"
    with pytest.raises(DataValidationError, match="not in items"):
        validate_interactions(inter, set(src.items()["item_id"]))


def test_null_user_rejected() -> None:
    inter = SyntheticSource().interactions()
    inter.loc[0, "user_id"] = None
    with pytest.raises(DataValidationError, match="null values"):
        validate_interactions(inter)


def test_interactions_sorted_per_user_by_time() -> None:
    inter = SyntheticSource().interactions().sample(frac=1.0, random_state=1)
    out = validate_interactions(inter)
    for _, g in out.groupby("user_id"):
        assert g["timestamp"].is_monotonic_increasing
    assert isinstance(out["timestamp"].iloc[0], pd.Timestamp)
