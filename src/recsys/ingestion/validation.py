"""Validation of item / interaction tables against the contracts in :mod:`recsys.core.schemas`."""

from typing import Any, cast

import pandas as pd

from recsys.core.schemas import (
    INTERACTION_COLUMNS,
    ITEM_COLUMNS,
    ITEM_LIST_COLUMNS,
    ItemMetadata,
)


class DataValidationError(ValueError):
    """Raised when a table violates its data contract."""


def _require_columns(df: pd.DataFrame, expected: tuple[str, ...], table: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"{table}: missing columns {missing}")


def _normalize_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet round-trips lists as numpy arrays and nulls as None/NaN; make them plain lists."""
    out = df.copy()
    for col in ITEM_LIST_COLUMNS:
        out[col] = [[] if v is None else list(v) for v in out[col]]
    return out


def validate_items(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the items table. Returns a copy with the canonical column order."""
    _require_columns(df, ITEM_COLUMNS, "items")
    out = _normalize_lists(df.loc[:, list(ITEM_COLUMNS)])
    if out["item_id"].duplicated().any():
        dup = out.loc[out["item_id"].duplicated(), "item_id"].iloc[0]
        raise DataValidationError(f"items: duplicated item_id {dup!r}")
    for rec in out.to_dict(orient="records"):
        clean = {k: (None if _is_null(v) else v) for k, v in rec.items()}
        try:
            ItemMetadata.model_validate(clean)
        except ValueError as exc:
            raise DataValidationError(
                f"items: invalid row {clean.get('item_id')!r}: {exc}"
            ) from exc
    return out.reset_index(drop=True)


def validate_interactions(df: pd.DataFrame, item_ids: set[str] | None = None) -> pd.DataFrame:
    """Validate the interactions table (vectorized). Optionally check referential integrity."""
    _require_columns(df, INTERACTION_COLUMNS, "interactions")
    out = df.loc[:, list(INTERACTION_COLUMNS)].copy()
    for col in ("user_id", "item_id", "timestamp"):
        if out[col].isna().any():
            raise DataValidationError(f"interactions: null values in {col!r}")
    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    if (out["user_id"] == "").any() or (out["item_id"] == "").any():
        raise DataValidationError("interactions: empty user_id/item_id")
    if item_ids is not None:
        unknown = set(out["item_id"]) - item_ids
        if unknown:
            raise DataValidationError(
                f"interactions: {len(unknown)} item_id(s) not in items, e.g. {sorted(unknown)[0]!r}"
            )
    return out.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)


def _is_null(value: object) -> bool:
    if isinstance(value, list):
        return False
    return bool(pd.isna(cast("Any", value)))
