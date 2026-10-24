"""MovieLens loader (ml-latest-small / ml-1m CSV layout: ``movies.csv`` + ``ratings.csv``).

Download manually from https://grouplens.org/datasets/movielens/ and point ``root`` at the
extracted directory. Optional ``links``-style enrichment (TMDB) is a later phase.
"""

import re
from pathlib import Path

import pandas as pd

_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")
_NO_GENRES = "(no genres listed)"


def _split_title(raw: str) -> tuple[str, int | None]:
    match = _YEAR_RE.search(raw)
    if not match:
        return raw.strip(), None
    return raw[: match.start()].strip(), int(match.group(1))


class MovieLensSource:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def items(self) -> pd.DataFrame:
        movies = pd.read_csv(self.root / "movies.csv", dtype={"movieId": str})
        parsed = [_split_title(t) for t in movies["title"]]
        n = len(movies)
        return pd.DataFrame(
            {
                "item_id": movies["movieId"].astype(str),
                "name": [p[0] for p in parsed],
                "description": [None] * n,
                "cover_url": [None] * n,
                "item_type": ["movie"] * n,
                "content_rating": [None] * n,
                "genres": [[] if g == _NO_GENRES else str(g).split("|") for g in movies["genres"]],
                "cast": [[] for _ in range(n)],
                "directors": [[] for _ in range(n)],
                "countries": [[] for _ in range(n)],
                "release_year": pd.array([p[1] for p in parsed], dtype="Int64"),
            }
        )

    def interactions(self) -> pd.DataFrame:
        ratings = pd.read_csv(self.root / "ratings.csv", dtype={"userId": str, "movieId": str})
        return pd.DataFrame(
            {
                "user_id": ratings["userId"],
                "item_id": ratings["movieId"],
                "timestamp": pd.to_datetime(ratings["timestamp"], unit="s"),
                "rating": ratings["rating"].astype("float64"),
            }
        )
