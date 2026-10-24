"""Data contracts shared across pipeline stages (ingestion -> feature store -> training -> serving).

Row-level contracts are Pydantic models; their column layout is mirrored in
:data:`ITEM_COLUMNS` / :data:`INTERACTION_COLUMNS` for the Parquet/DataFrame form.
"""

from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class ItemMetadata(BaseModel):
    """A catalog item. Single, typed source of truth for item metadata."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    cover_url: str | None = None
    item_type: str | None = None
    content_rating: str | None = None
    genres: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    release_year: int | None = Field(default=None, ge=1850, le=2100)


class Interaction(BaseModel):
    """A user-item event (view, rating, click)."""

    model_config = ConfigDict(frozen=True)

    user_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    timestamp: datetime
    rating: float | None = None


ITEM_COLUMNS: Final[tuple[str, ...]] = tuple(ItemMetadata.model_fields)
INTERACTION_COLUMNS: Final[tuple[str, ...]] = tuple(Interaction.model_fields)
ITEM_LIST_COLUMNS: Final[tuple[str, ...]] = ("genres", "cast", "directors", "countries")
