from pathlib import Path

from typer.testing import CliRunner

from recsys.cli import app
from recsys.ingestion.sources.local import LocalParquetSource


def test_ingest_synthetic_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    result = CliRunner().invoke(
        app, ["ingest", "synthetic", "--out", str(out), "--n-items", "50", "--n-users", "10"]
    )
    assert result.exit_code == 0, result.output
    src = LocalParquetSource(out)
    assert len(src.items()) == 50
    assert src.interactions()["user_id"].nunique() == 10


def test_ingest_movielens_end_to_end(tmp_path: Path) -> None:
    ml = tmp_path / "ml"
    ml.mkdir()
    (ml / "movies.csv").write_text("movieId,title,genres\n1,A (1999),Drama\n2,B (2005),Comedy\n")
    (ml / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n1,1,5.0,1000000000\n2,2,3.0,1000000100\n"
    )
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["ingest", "movielens", "--src", str(ml), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert len(LocalParquetSource(out).interactions()) == 2
