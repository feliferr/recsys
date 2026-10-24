"""Serialize evaluation results as JSON + a Markdown table (reproducible, diff-friendly)."""

import json
from pathlib import Path

from recsys.evaluation.evaluator import Results


def to_markdown(
    results: Results,
    headline: tuple[str, ...] = (
        "ndcg@10",
        "recall@10",
        "coverage@10",
        "diversity@10",
        "novelty@10",
    ),
) -> str:
    cols = [c for c in headline if all(c in m for m in results.values())]
    lines = ["| model | n_users | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, m in results.items():
        cells = " | ".join(f"{m[c]:.4f}" for c in cols)
        lines.append(f"| {name} | {int(m['n_users'])} | {cells} |")
    return "\n".join(lines) + "\n"


def write_report(results: Results, out_dir: Path, meta: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "results": results}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out_dir / "results.md").write_text(to_markdown(results))
