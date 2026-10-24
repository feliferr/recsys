"""Leakage-free temporal splits. Every split orders by timestamp; ties broken by input order."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def leave_last_out(interactions: pd.DataFrame, n_test: int = 1) -> Split:
    """Per user, the last ``n_test`` interactions go to test; users with <= ``n_test`` events
    stay entirely in train (they cannot be evaluated without history)."""
    if n_test < 1:
        raise ValueError("n_test must be >= 1")
    df = interactions.sort_values(["user_id", "timestamp"], kind="stable")
    rank_from_end = df.groupby("user_id", sort=False).cumcount(ascending=False)
    size = df.groupby("user_id", sort=False)["item_id"].transform("size")
    is_test = (rank_from_end < n_test) & (size > n_test)
    return Split(df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True))


def global_temporal(interactions: pd.DataFrame, test_frac: float = 0.2) -> Split:
    """Everything after the ``1 - test_frac`` time quantile is test (no future data in train)."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    df = interactions.sort_values("timestamp", kind="stable")
    cutoff = df["timestamp"].quantile(1.0 - test_frac)
    is_test = df["timestamp"] > cutoff
    return Split(df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True))
