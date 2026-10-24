"""Cheap, strong baselines that every learned model must beat (see RecSys 2026 capability probes).

All recommenders score the full catalog: ``score(users) -> [len(users), n_items]``.
Seen-item masking is the evaluator's job, not the recommender's.
"""

from typing import Protocol

import numpy as np
import pandas as pd

from recsys.evaluation.index import (
    CatalogIndex,
    FloatArray,
    interaction_matrix,
    multi_hot,
)


class Recommender(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, catalog: CatalogIndex) -> None: ...

    def score(self, users: list[str]) -> FloatArray: ...


def _counts(train: pd.DataFrame, catalog: CatalogIndex) -> FloatArray:
    mapped = train["item_id"].map(catalog.item_pos).to_numpy(dtype=np.int64)
    return np.bincount(mapped, minlength=catalog.n_items).astype(np.float64)


class RandomRecommender:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_items = 0

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, catalog: CatalogIndex) -> None:
        self._n_items = catalog.n_items

    def score(self, users: list[str]) -> FloatArray:
        return self._rng.random((len(users), self._n_items))


class PopularityRecommender:
    name = "popularity"

    def __init__(self) -> None:
        self._counts: FloatArray = np.zeros(0)

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, catalog: CatalogIndex) -> None:
        self._counts = _counts(train, catalog)

    def score(self, users: list[str]) -> FloatArray:
        return np.tile(self._counts, (len(users), 1))


def genre_tfidf(items: pd.DataFrame, column: str = "genres") -> FloatArray:
    """L2-normalized TF-IDF of a list-valued item column (rows follow ``items`` order)."""
    mat, _ = multi_hot([list(v) for v in items[column]])
    df = mat.sum(axis=0)
    idf = np.log((1.0 + len(mat)) / (1.0 + df)) + 1.0
    weighted = mat * idf
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    return np.asarray(weighted / np.maximum(norms, 1e-12))


class ContentProfileRecommender:
    """Cosine between the mean of a user's consumed item vectors and every catalog item.

    ``item_vectors`` is pluggable: TF-IDF of metadata now, learned/pretrained embeddings later
    (rows must follow the catalog order).
    """

    def __init__(self, name: str, item_vectors: FloatArray) -> None:
        self.name = name
        norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
        self._vectors: FloatArray = item_vectors / np.maximum(norms, 1e-12)
        self._train = pd.DataFrame()
        self._catalog: CatalogIndex | None = None

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, catalog: CatalogIndex) -> None:
        if len(self._vectors) != catalog.n_items:
            raise ValueError("item_vectors rows must match the catalog size")
        self._train, self._catalog = train, catalog

    def score(self, users: list[str]) -> FloatArray:
        assert self._catalog is not None, "call fit() first"
        seen = interaction_matrix(self._train, self._catalog, users).astype(np.float64)
        profile = seen @ self._vectors
        norms = np.linalg.norm(profile, axis=1, keepdims=True)
        return np.asarray((profile / np.maximum(norms, 1e-12)) @ self._vectors.T)


class TransitionRecommender:
    """Recency-weighted pairwise transitions ``i -> j`` (the probe that beat eSASRec on
    several benchmarks in RecSys 2026). Popularity breaks ties/cold histories.
    """

    name = "transition"

    def __init__(self, window: int = 3, decay: float = 0.5, history: int = 3) -> None:
        self.window, self.decay, self.history = window, decay, history
        self._catalog: CatalogIndex | None = None
        self._trans: FloatArray = np.zeros((0, 0))
        self._pop: FloatArray = np.zeros(0)
        self._last: dict[str, list[int]] = {}

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, catalog: CatalogIndex) -> None:
        n = catalog.n_items
        trans = np.zeros((n, n))
        last: dict[str, list[int]] = {}
        ordered = train.sort_values(["user_id", "timestamp"], kind="stable")
        for user, group in ordered.groupby("user_id", sort=False):
            seq = group["item_id"].map(catalog.item_pos).to_numpy(dtype=np.int64)
            for gap in range(1, self.window + 1):
                if len(seq) > gap:
                    np.add.at(trans, (seq[:-gap], seq[gap:]), self.decay ** (gap - 1))
            last[str(user)] = seq[-self.history :].tolist()
        row_sums = trans.sum(axis=1, keepdims=True)
        self._trans = trans / np.maximum(row_sums, 1e-12)
        counts = _counts(train, catalog)
        self._pop = counts / max(counts.sum(), 1.0)
        self._last, self._catalog = last, catalog

    def score(self, users: list[str]) -> FloatArray:
        out = np.zeros((len(users), len(self._pop)))
        for r, user in enumerate(users):
            recent = self._last.get(user, [])
            for age, item in enumerate(reversed(recent)):
                out[r] += (self.decay**age) * self._trans[item]
        return out + 1e-6 * self._pop[None, :]
