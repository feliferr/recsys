"""Command-line entry point."""

from pathlib import Path
from typing import Annotated

import typer

from recsys import __version__
from recsys.evaluation.baselines import (
    ContentProfileRecommender,
    PopularityRecommender,
    RandomRecommender,
    Recommender,
    TransitionRecommender,
    genre_tfidf,
)
from recsys.evaluation.evaluator import evaluate_all
from recsys.evaluation.report import to_markdown, write_report
from recsys.evaluation.splits import Split, global_temporal, leave_last_out
from recsys.ingestion.sources.base import DataSource
from recsys.ingestion.sources.local import LocalParquetSource, write_dataset
from recsys.ingestion.sources.movielens import MovieLensSource
from recsys.ingestion.sources.synthetic import SyntheticConfig, SyntheticSource

app = typer.Typer(help="recsys command line interface", no_args_is_help=True)
ingest_app = typer.Typer(help="Build the canonical Parquet dataset.", no_args_is_help=True)
eval_app = typer.Typer(help="Offline evaluation.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(eval_app, name="eval")

OutDir = Annotated[Path, typer.Option("--out", help="Output directory for the Parquet dataset.")]


def _persist(source: DataSource, out: Path) -> None:
    items, interactions = source.items(), source.interactions()
    write_dataset(out, items, interactions)
    typer.echo(f"wrote {len(items)} items and {len(interactions)} interactions to {out}")


@app.callback()
def main() -> None:
    """recsys command line interface."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@ingest_app.command("synthetic")
def ingest_synthetic(
    out: OutDir = Path("data/synthetic"),
    n_items: int = 200,
    n_users: int = 100,
    seed: int = 0,
) -> None:
    """Generate a deterministic synthetic dataset."""
    config = SyntheticConfig(n_items=n_items, n_users=n_users, seed=seed)
    _persist(SyntheticSource(config), out)


@ingest_app.command("movielens")
def ingest_movielens(
    src: Annotated[Path, typer.Option("--src", help="Extracted MovieLens directory.")],
    out: OutDir = Path("data/movielens"),
) -> None:
    """Convert a local MovieLens download (movies.csv + ratings.csv) to the canonical layout."""
    _persist(MovieLensSource(src), out)


@eval_app.command("baselines")
def eval_baselines(
    data: Annotated[Path, typer.Option("--data", help="Canonical Parquet dataset directory.")],
    out: OutDir = Path("reports/baselines"),
    split: Annotated[str, typer.Option(help="leave-last-out | temporal")] = "leave-last-out",
    k: Annotated[list[int] | None, typer.Option("--k", help="Cutoffs (repeatable).")] = None,
) -> None:
    """Evaluate non-learned baselines (random, popularity, content, transition)."""
    source = LocalParquetSource(data)
    items, interactions = source.items(), source.interactions()
    if split == "leave-last-out":
        parts: Split = leave_last_out(interactions)
    elif split == "temporal":
        parts = global_temporal(interactions)
    else:
        raise typer.BadParameter(f"unknown split {split!r}")
    ks = tuple(k) if k else (10, 20)
    recommenders: list[Recommender] = [
        RandomRecommender(),
        PopularityRecommender(),
        ContentProfileRecommender("content_genre_tfidf", genre_tfidf(items)),
        TransitionRecommender(),
    ]
    results = evaluate_all(recommenders, parts, items, ks)
    write_report(results, out, {"data": str(data), "split": split, "ks": list(ks)})
    typer.echo(to_markdown(results))
