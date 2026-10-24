from pathlib import Path

import pandas as pd

from recsys.ingestion.sources.base import DataSource
from recsys.ingestion.sources.local import LocalParquetSource, write_dataset
from recsys.ingestion.sources.movielens import MovieLensSource
from recsys.ingestion.sources.synthetic import SyntheticConfig, SyntheticSource
from recsys.ingestion.validation import validate_interactions, validate_items


def test_sources_satisfy_protocol(tmp_path: Path) -> None:
    assert isinstance(SyntheticSource(), DataSource)
    assert isinstance(LocalParquetSource(tmp_path), DataSource)
    assert isinstance(MovieLensSource(tmp_path), DataSource)


def test_synthetic_is_deterministic() -> None:
    a, b = SyntheticSource(SyntheticConfig(seed=7)), SyntheticSource(SyntheticConfig(seed=7))
    pd.testing.assert_frame_equal(a.interactions(), b.interactions())
    c = SyntheticSource(SyntheticConfig(seed=8))
    assert not a.interactions()["item_id"].equals(c.interactions()["item_id"])


def test_synthetic_has_genre_signal() -> None:
    """Users should interact mostly with items sharing genres (needed for later evaluation)."""
    src = SyntheticSource(SyntheticConfig(n_items=300, n_users=50, seed=1))
    items = src.items().set_index("item_id")
    inter = src.interactions()
    purity = []
    for _, g in inter.groupby("user_id"):
        counts = pd.Series([x for gs in items.loc[g["item_id"], "genres"] for x in gs])
        purity.append(counts.value_counts(normalize=True).iloc[:2].sum())
    assert sum(purity) / len(purity) > 0.5  # uniform over 8 genres would give ~0.25-0.35


def test_parquet_roundtrip(tmp_path: Path) -> None:
    src = SyntheticSource()
    write_dataset(tmp_path, src.items(), src.interactions())
    loaded = LocalParquetSource(tmp_path)
    pd.testing.assert_frame_equal(loaded.items(), validate_items(src.items()))
    pd.testing.assert_frame_equal(loaded.interactions(), validate_interactions(src.interactions()))


def test_movielens_parsing(tmp_path: Path) -> None:
    (tmp_path / "movies.csv").write_text(
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation\n"
        '2,"Movie, The (2001)",(no genres listed)\n'
        "3,No Year,Drama\n"
    )
    (tmp_path / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n1,1,4.0,964982703\n1,3,3.5,964981247\n"
    )
    src = MovieLensSource(tmp_path)
    items = validate_items(src.items())
    assert items["name"].tolist() == ["Toy Story", "Movie, The", "No Year"]
    assert items["release_year"].tolist()[:2] == [1995, 2001]
    assert pd.isna(items["release_year"].iloc[2])
    assert items["genres"].tolist() == [["Adventure", "Animation"], [], ["Drama"]]
    inter = validate_interactions(src.interactions(), set(items["item_id"]))
    assert inter["item_id"].tolist() == ["3", "1"]  # sorted by time
