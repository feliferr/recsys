"""Local Parquet source/sink: the canonical on-disk layout used by every pipeline stage."""

from pathlib import Path

import pandas as pd

from recsys.ingestion.validation import validate_interactions, validate_items

ITEMS_FILE = "items.parquet"
INTERACTIONS_FILE = "interactions.parquet"


class LocalParquetSource:
    """Reads ``items.parquet`` and ``interactions.parquet`` from a directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def items(self) -> pd.DataFrame:
        return validate_items(pd.read_parquet(self.root / ITEMS_FILE))

    def interactions(self) -> pd.DataFrame:
        df = pd.read_parquet(self.root / INTERACTIONS_FILE)
        return validate_interactions(df, set(self.items()["item_id"]))


def write_dataset(root: Path, items: pd.DataFrame, interactions: pd.DataFrame) -> None:
    """Validate and persist a dataset in the canonical layout."""
    valid_items = validate_items(items)
    valid_inter = validate_interactions(interactions, set(valid_items["item_id"]))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    valid_items.to_parquet(root / ITEMS_FILE, index=False)
    valid_inter.to_parquet(root / INTERACTIONS_FILE, index=False)
