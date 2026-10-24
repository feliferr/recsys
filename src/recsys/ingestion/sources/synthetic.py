"""Deterministic synthetic catalog with learnable structure (genre-driven user taste).

Used by unit/integration tests and for offline experimentation without real data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

GENRES = ("drama", "comedy", "action", "scifi", "horror", "romance", "documentary", "animation")


@dataclass(frozen=True)
class SyntheticConfig:
    n_items: int = 200
    n_users: int = 100
    interactions_per_user: int = 20
    seed: int = 0


class SyntheticSource:
    """Items belong to 1-2 genres; each user prefers 1-2 genres and samples items accordingly."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self._items, self._interactions = self._generate()

    def items(self) -> pd.DataFrame:
        return self._items.copy()

    def interactions(self) -> pd.DataFrame:
        return self._interactions.copy()

    def _generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n_genres = len(GENRES)

        item_genres: list[list[int]] = [
            sorted(rng.choice(n_genres, size=int(rng.integers(1, 3)), replace=False).tolist())
            for _ in range(cfg.n_items)
        ]
        items = pd.DataFrame(
            {
                "item_id": [f"i{idx:04d}" for idx in range(cfg.n_items)],
                "name": [f"Title {idx}" for idx in range(cfg.n_items)],
                "description": [f"Synthetic description {idx}" for idx in range(cfg.n_items)],
                "cover_url": [None] * cfg.n_items,
                "item_type": ["movie"] * cfg.n_items,
                "content_rating": [None] * cfg.n_items,
                "genres": [[GENRES[g] for g in gs] for gs in item_genres],
                "cast": [[] for _ in range(cfg.n_items)],
                "directors": [[] for _ in range(cfg.n_items)],
                "countries": [[] for _ in range(cfg.n_items)],
                "release_year": rng.integers(1980, 2025, size=cfg.n_items).astype("int64"),
            }
        )

        membership = np.zeros((cfg.n_items, n_genres))
        for idx, gs in enumerate(item_genres):
            membership[idx, gs] = 1.0

        base_ts = pd.Timestamp("2024-01-01")
        rows: list[dict[str, object]] = []
        for u in range(cfg.n_users):
            liked = rng.choice(n_genres, size=int(rng.integers(1, 3)), replace=False)
            affinity = 0.05 + membership[:, liked].sum(axis=1)
            probs = affinity / affinity.sum()
            n = min(cfg.interactions_per_user, cfg.n_items)
            chosen = rng.choice(cfg.n_items, size=n, replace=False, p=probs)
            for step, item_idx in enumerate(chosen):
                rows.append(
                    {
                        "user_id": f"u{u:04d}",
                        "item_id": f"i{int(item_idx):04d}",
                        "timestamp": base_ts + pd.Timedelta(hours=u * 3 + step),
                        "rating": None,
                    }
                )
        return items, pd.DataFrame(rows)
