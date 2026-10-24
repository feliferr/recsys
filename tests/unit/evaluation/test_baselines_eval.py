import numpy as np
import pandas as pd
import pytest

from recsys.evaluation.baselines import (
    ContentProfileRecommender,
    PopularityRecommender,
    RandomRecommender,
    TransitionRecommender,
    genre_tfidf,
)
from recsys.evaluation.evaluator import evaluate, evaluate_all
from recsys.evaluation.item_similarity import weak_label_precision_at_k
from recsys.evaluation.splits import leave_last_out
from recsys.ingestion.sources.synthetic import SyntheticConfig, SyntheticSource


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    src = SyntheticSource(SyntheticConfig(n_items=200, n_users=150, interactions_per_user=20))
    return src.items(), src.interactions()


def test_content_beats_random_and_popularity_on_genre_driven_data(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    items, inter = data
    res = evaluate_all(
        [
            RandomRecommender(seed=1),
            PopularityRecommender(),
            ContentProfileRecommender("content", genre_tfidf(items)),
        ],
        leave_last_out(inter),
        items,
        ks=(10,),
    )
    assert res["content"]["ndcg@10"] > 2 * res["random"]["ndcg@10"]
    assert res["content"]["ndcg@10"] > res["popularity"]["ndcg@10"]


def test_seen_items_are_never_recommended() -> None:
    """A recommender that scores seen items highest must still score zero after masking."""
    items = pd.DataFrame(
        {"item_id": ["a", "b", "c"], "genres": [["x"], ["x"], ["y"]]},
    )
    inter = pd.DataFrame(
        {
            "user_id": ["u", "u", "u"],
            "item_id": ["a", "b", "c"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    res = evaluate(PopularityRecommender(), leave_last_out(inter), items, ks=(2,))
    # train = {a, b}; test = {c}; popularity ties -> lowest index first, but a,b are masked
    assert res["hit_rate@2"] == 1.0


def test_transition_recommender_learns_sequences() -> None:
    n = 30
    items = pd.DataFrame({"item_id": [f"i{j}" for j in range(n)], "genres": [["g"]] * n})
    rows = []
    for u in range(40):  # everybody walks the same chain i0 -> i1 -> ... -> i5
        for step in range(6):
            rows.append(
                (f"u{u}", f"i{step}", pd.Timestamp("2024-01-01") + pd.Timedelta(hours=step))
            )
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"]).assign(rating=None)
    res = evaluate(TransitionRecommender(), leave_last_out(inter), items, ks=(1,))
    assert res["hit_rate@1"] == 1.0  # last step i5 follows i4


def test_item_similarity_precision_perfect_and_random() -> None:
    items = pd.DataFrame({"genres": [["a"], ["a"], ["b"], ["b"]]})
    emb = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
    assert weak_label_precision_at_k(emb, items, "genres", k=1) == 1.0
    crossed = emb[[0, 2, 1, 3]]  # neighbors now always carry the other label
    assert weak_label_precision_at_k(crossed, items, "genres", k=1) == 0.0


def test_no_evaluable_users_raises(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    items, inter = data
    from recsys.evaluation.splits import Split

    with pytest.raises(ValueError, match="no evaluable users"):
        evaluate(PopularityRecommender(), Split(inter, inter.iloc[0:0]), items)
