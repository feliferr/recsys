"""Run recommenders against a split and compute the metric suite."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from recsys.evaluation import metrics
from recsys.evaluation.baselines import Recommender
from recsys.evaluation.index import (
    CatalogIndex,
    FloatArray,
    interaction_matrix,
    multi_hot,
)
from recsys.evaluation.splits import Split

Results = dict[str, dict[str, float]]


def evaluate(
    recommender: Recommender,
    split: Split,
    items: pd.DataFrame,
    ks: Sequence[int] = (10, 20),
    diversity_column: str = "genres",
) -> dict[str, float]:
    """Full-ranking evaluation. Items already in a user's train history are excluded.

    Only users with at least one test item *and* one train item are evaluated; the count is
    reported as ``n_users``.
    """
    catalog = CatalogIndex.from_items(items)
    train_users = set(split.train["user_id"])
    users = sorted(u for u in set(split.test["user_id"]) if u in train_users)
    if not users:
        raise ValueError("no evaluable users (need both train and test interactions)")

    recommender.fit(split.train, items, catalog)
    scores = np.asarray(recommender.score(users), dtype=np.float64).copy()
    seen = interaction_matrix(split.train, catalog, users)
    scores[seen] = -np.inf
    relevant = interaction_matrix(split.test, catalog, users) & ~seen

    keep = relevant.any(axis=1)
    if not keep.any():
        raise ValueError("test items are all already present in train history")
    scores, relevant = scores[keep], relevant[keep]

    item_features, _ = multi_hot([list(v) for v in items[diversity_column]])
    counts = _train_counts(split.train, catalog)
    out: dict[str, float] = {"n_users": float(keep.sum())}
    for k in ks:
        ranked = metrics.top_k(scores, k)
        out[f"recall@{k}"] = metrics.recall_at_k(relevant, ranked)
        out[f"hit_rate@{k}"] = metrics.hit_rate_at_k(relevant, ranked)
        out[f"ndcg@{k}"] = metrics.ndcg_at_k(relevant, ranked)
        out[f"map@{k}"] = metrics.map_at_k(relevant, ranked)
        out[f"coverage@{k}"] = metrics.catalog_coverage(ranked, catalog.n_items)
        out[f"diversity@{k}"] = metrics.intra_list_diversity(ranked, item_features)
        out[f"novelty@{k}"] = metrics.mean_novelty(ranked, counts)
        out[f"avg_popularity@{k}"] = metrics.mean_popularity(ranked, counts)
    return out


def _train_counts(train: pd.DataFrame, catalog: CatalogIndex) -> FloatArray:
    mapped = train["item_id"].map(catalog.item_pos).to_numpy(dtype=np.int64)
    return np.bincount(mapped, minlength=catalog.n_items).astype(np.float64)


def evaluate_all(
    recommenders: Sequence[Recommender],
    split: Split,
    items: pd.DataFrame,
    ks: Sequence[int] = (10, 20),
) -> Results:
    return {rec.name: evaluate(rec, split, items, ks) for rec in recommenders}
