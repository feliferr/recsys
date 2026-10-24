"""DataSource contract: every ingestion backend returns validated item/interaction tables."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    """Read-only source of catalog metadata and (optional) user interactions."""

    def items(self) -> pd.DataFrame:
        """Items table following :class:`recsys.core.schemas.ItemMetadata`."""
        ...

    def interactions(self) -> pd.DataFrame:
        """Interactions table following :class:`recsys.core.schemas.Interaction`."""
        ...
