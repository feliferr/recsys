import json
from pathlib import Path

from typer.testing import CliRunner

from recsys.cli import app


def test_ingest_then_eval_baselines(tmp_path: Path) -> None:
    runner = CliRunner()
    data, out = tmp_path / "ds", tmp_path / "rep"
    assert runner.invoke(app, ["ingest", "synthetic", "--out", str(data)]).exit_code == 0
    for split in ("leave-last-out", "temporal"):
        result = runner.invoke(
            app, ["eval", "baselines", "--data", str(data), "--out", str(out), "--split", split]
        )
        assert result.exit_code == 0, result.output
    payload = json.loads((out / "results.json").read_text())
    assert set(payload["results"]) == {
        "random",
        "popularity",
        "content_genre_tfidf",
        "transition",
    }
    assert "| model |" in (out / "results.md").read_text()
