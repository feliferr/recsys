"""Item-to-item ("similar items") evaluation with weak labels.

Relevance of a neighbor = shares at least one value of ``label_column`` with the query.
Caveat: if the embeddings were built from the same field, the metric is circular; evaluate
against a field the encoder did not see (cast, director, co-consumption).
"""

import numpy as np
import pandas as pd

from recsys.evaluation.index import FloatArray, multi_hot
from recsys.evaluation.metrics import top_k


def weak_label_precision_at_k(
    embeddings: FloatArray, items: pd.DataFrame, label_column: str, k: int = 10
) -> float:
    """Mean precision@k over items that have at least one label; cosine neighbors, self excluded."""
    if len(embeddings) != len(items):
        raise ValueError("embeddings rows must match items rows")
    labels, _ = multi_hot([list(v) for v in items[label_column]])
    has_label = labels.any(axis=1)
    norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    sims = norm @ norm.T
    np.fill_diagonal(sims, -np.inf)
    ranked = top_k(sims, k)
    shared = (labels @ labels.T) > 0
    hits = shared[np.arange(len(items))[:, None], ranked]
    return float(hits.mean(axis=1)[has_label].mean())
