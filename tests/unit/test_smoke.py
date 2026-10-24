from typer.testing import CliRunner

import recsys
from recsys.cli import app


def test_version_is_exposed() -> None:
    assert recsys.__version__ == "0.1.0"


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == recsys.__version__
