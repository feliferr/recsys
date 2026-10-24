"""Dense index of users/items and matrices derived from the canonical tables."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CatalogIndex:
    """Maps item ids to dense positions (the catalog is the full ``items`` table)."""

    item_ids: tuple[str, ...]
    item_pos: dict[str, int]

    @classmethod
    def from_items(cls, items: pd.DataFrame) -> "CatalogIndex":
        ids = tuple(str(i) for i in items["item_id"])
        return cls(ids, {i: p for p, i in enumerate(ids)})

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def interaction_matrix(
    interactions: pd.DataFrame, catalog: CatalogIndex, users: list[str]
) -> BoolArray:
    """Boolean ``[len(users), n_items]`` matrix; users without interactions are all-False."""
    user_pos = {u: p for p, u in enumerate(users)}
    mat = np.zeros((len(users), catalog.n_items), dtype=np.bool_)
    sub = interactions[interactions["user_id"].isin(user_pos)]
    rows = sub["user_id"].map(user_pos).to_numpy(dtype=np.int64)
    cols = sub["item_id"].map(catalog.item_pos).to_numpy(dtype=np.int64)
    mat[rows, cols] = True
    return mat


def multi_hot(values: list[list[str]]) -> tuple[FloatArray, list[str]]:
    """Multi-hot encode a list-valued column. Returns ``(matrix, vocabulary)``."""
    vocab = sorted({v for vs in values for v in vs})
    pos = {v: i for i, v in enumerate(vocab)}
    mat = np.zeros((len(values), len(vocab)), dtype=np.float64)
    for r, vs in enumerate(values):
        for v in vs:
            mat[r, pos[v]] = 1.0
    return mat, vocab
