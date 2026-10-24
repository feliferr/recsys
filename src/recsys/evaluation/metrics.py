"""Full-ranking top-k metrics (no sampled negatives) plus beyond-accuracy metrics."""

import numpy as np

from recsys.evaluation.index import BoolArray, FloatArray, IntArray


def top_k(scores: FloatArray, k: int) -> IntArray:
    """Indices of the ``k`` highest scores per row, best first. Ties break by lower index."""
    k = min(k, scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :k].astype(np.int64)


def _hits(relevant: BoolArray, ranked: IntArray) -> BoolArray:
    rows = np.arange(len(ranked))[:, None]
    return relevant[rows, ranked]


def recall_at_k(relevant: BoolArray, ranked: IntArray) -> float:
    n_rel = relevant.sum(axis=1)
    hits = _hits(relevant, ranked).sum(axis=1)
    return float(np.mean(hits / np.maximum(n_rel, 1)))


def hit_rate_at_k(relevant: BoolArray, ranked: IntArray) -> float:
    return float(np.mean(_hits(relevant, ranked).any(axis=1)))


def ndcg_at_k(relevant: BoolArray, ranked: IntArray) -> float:
    k = ranked.shape[1]
    discounts = 1.0 / np.log2(np.arange(k) + 2.0)
    dcg = (_hits(relevant, ranked) * discounts).sum(axis=1)
    n_rel = np.minimum(relevant.sum(axis=1), k)
    cum = np.concatenate([[0.0], np.cumsum(discounts)])
    idcg = cum[n_rel]
    return float(np.mean(np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0)))


def map_at_k(relevant: BoolArray, ranked: IntArray) -> float:
    k = ranked.shape[1]
    hits = _hits(relevant, ranked).astype(np.float64)
    precision_at_i = np.cumsum(hits, axis=1) / np.arange(1, k + 1)
    n_rel = np.minimum(relevant.sum(axis=1), k)
    ap = (precision_at_i * hits).sum(axis=1) / np.maximum(n_rel, 1)
    return float(np.mean(ap))


def catalog_coverage(ranked: IntArray, n_items: int) -> float:
    return float(len(np.unique(ranked)) / n_items)


def intra_list_diversity(ranked: IntArray, item_features: FloatArray) -> float:
    """Mean pairwise ``1 - Jaccard`` of binary item features within each list.
    Items with no features are treated as having zero similarity to everything."""
    feats = item_features > 0
    k = ranked.shape[1]
    if k < 2:
        return 0.0
    off_diag = ~np.eye(k, dtype=np.bool_)
    out = np.empty(len(ranked))
    for r, row in enumerate(ranked):
        f = feats[row].astype(np.float64)
        inter = f @ f.T
        sizes = f.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - inter
        sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        out[r] = 1.0 - sim[off_diag].mean()
    return float(out.mean())


def mean_novelty(ranked: IntArray, train_counts: FloatArray) -> float:
    """Mean self-information ``-log2 p(item)`` of recommended items (higher = less popular)."""
    p = (train_counts + 1.0) / (train_counts.sum() + len(train_counts))
    return float(np.mean(-np.log2(p[ranked])))


def mean_popularity(ranked: IntArray, train_counts: FloatArray) -> float:
    """Mean training-set interaction count of recommended items."""
    return float(np.mean(train_counts[ranked]))
