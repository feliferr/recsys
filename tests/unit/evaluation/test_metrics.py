import numpy as np
import pytest

from recsys.evaluation import metrics

# 2 users, 5 items. user0 relevant {1}, user1 relevant {0, 4}
REL = np.array(
    [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1]],
    dtype=np.bool_,
)
SCORES = np.array(
    [[0.1, 0.9, 0.5, 0.3, 0.2], [0.2, 0.9, 0.8, 0.1, 0.7]],
)


def test_top_k_orders_best_first_and_breaks_ties_by_index() -> None:
    ranked = metrics.top_k(np.array([[1.0, 1.0, 2.0]]), 3)
    assert ranked.tolist() == [[2, 0, 1]]


def test_recall_hit_ndcg_map_by_hand() -> None:
    ranked = metrics.top_k(SCORES, 3)  # u0: [1,2,3]  u1: [1,2,4]
    assert metrics.hit_rate_at_k(REL, ranked) == pytest.approx(1.0)
    assert metrics.recall_at_k(REL, ranked) == pytest.approx((1.0 + 0.5) / 2)
    # u0: hit at rank1 -> 1. u1: hit at rank3 -> 0.5 / (1 + 1/log2(3)) normalised
    idcg1 = 1.0 + 1.0 / np.log2(3)
    expected = (1.0 + (1.0 / np.log2(4)) / idcg1) / 2
    assert metrics.ndcg_at_k(REL, ranked) == pytest.approx(expected)
    # u0 AP = 1. u1 AP = (precision@3 = 1/3) / min(2,3)
    assert metrics.map_at_k(REL, ranked) == pytest.approx((1.0 + (1 / 3) / 2) / 2)


def test_perfect_ranking_scores_one() -> None:
    ranked = metrics.top_k(REL.astype(float), 2)
    assert metrics.ndcg_at_k(REL, ranked) == pytest.approx(1.0)
    assert metrics.map_at_k(REL, ranked) == pytest.approx(1.0)
    assert metrics.recall_at_k(REL, ranked) == pytest.approx(1.0)


def test_coverage_and_popularity_metrics() -> None:
    ranked = np.array([[0, 1], [0, 2]])
    assert metrics.catalog_coverage(ranked, 4) == pytest.approx(0.75)
    counts = np.array([10.0, 0.0, 0.0, 0.0])
    assert metrics.mean_popularity(ranked, counts) == pytest.approx(5.0)
    assert metrics.mean_novelty(ranked[:, :1], counts) < metrics.mean_novelty(
        np.array([[1], [2]]), counts
    )


def test_diversity_identical_vs_disjoint() -> None:
    feats = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float64)
    assert metrics.intra_list_diversity(np.array([[0, 1]]), feats) == pytest.approx(0.0)
    assert metrics.intra_list_diversity(np.array([[0, 2]]), feats) == pytest.approx(1.0)
