"""CLI: prepares a frozen, reproducible WebWalkerQA manifest. Does NOT
crawl URLs, does NOT call an LLM, does NOT run any evaluation -- see this
package's own `__init__.py` docstring for the full scope statement.

Usage (matches the repository's existing `python -m` module-execution
convention for evaluation-suite scripts, and `typer` for the CLI layer
itself, the same framework `ant.cli`'s own commands use):

    python -m ant.evaluation_suite.webwalkerqa.prepare \\
        --split main --language English --output <manifest.json>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ant.evaluation_suite.webwalkerqa.loader import (
    DEFAULT_SPLIT,
    filter_by_difficulty,
    filter_by_domain,
    filter_by_hop_type,
    filter_english,
    inference_view,
    load_webwalkerqa_records,
    sample_deterministic,
)
from ant.evaluation_suite.webwalkerqa.manifest import build_prep_manifest, save_prep_manifest

app = typer.Typer(no_args_is_help=True, add_completion=False)

# Module-level Option singletons -- same convention as ant.cli's own
# INDEX_OPTION/MAX_ROUNDS_OPTION etc. (avoids constructing typer.Option()
# inline in the function signature's own defaults).
SPLIT_OPTION = typer.Option(DEFAULT_SPLIT, "--split")
LANGUAGE_OPTION = typer.Option(None, "--language", help="e.g. English")
DOMAIN_OPTION = typer.Option(None, "--domain")
DIFFICULTY_OPTION = typer.Option(None, "--difficulty")
HOP_TYPE_OPTION = typer.Option(None, "--hop-type")
LIMIT_OPTION = typer.Option(
    None, "--limit", help="Cap the SAMPLED set to this many examples (after filtering)."
)
SEED_OPTION = typer.Option(
    None, "--seed", help="Seeded deterministic sample instead of first-N when --limit is set."
)
OUTPUT_OPTION = typer.Option(..., "--output", help="Manifest output path.")


@app.command()
def prepare(
    split: str = SPLIT_OPTION,
    language: str | None = LANGUAGE_OPTION,
    domain: str | None = DOMAIN_OPTION,
    difficulty: str | None = DIFFICULTY_OPTION,
    hop_type: str | None = HOP_TYPE_OPTION,
    limit: int | None = LIMIT_OPTION,
    seed: int | None = SEED_OPTION,
    output: Path = OUTPUT_OPTION,
) -> None:
    records = load_webwalkerqa_records(split=split)
    total_before = len(records)

    filtered = records
    filters_applied: dict[str, str] = {}
    if language:
        # Real data uses ISO-ish codes ("en"/"zh"), not full names -- see
        # loader.py's own schema-correction note. "english"/"en" both
        # route to filter_english(); anything else falls back to a
        # generic exact (case-insensitive) match against the raw value.
        filtered = (
            filter_english(filtered)
            if language.strip().lower() in {"english", "en"}
            else [r for r in filtered if r.language.strip().lower() == language.strip().lower()]
        )
        filters_applied["language"] = language
    if domain:
        filtered = filter_by_domain(filtered, domain)
        filters_applied["domain"] = domain
    if difficulty:
        filtered = filter_by_difficulty(filtered, difficulty)
        filters_applied["difficulty"] = difficulty
    if hop_type:
        filtered = filter_by_hop_type(filtered, hop_type)
        filters_applied["hop_type"] = hop_type

    selected = filtered
    if limit is not None:
        selected = sample_deterministic(filtered, limit, seed=seed)

    manifest = build_prep_manifest(
        split=split,
        filters=filters_applied,
        seed=seed,
        total_rows_before_filter=total_before,
        selected_records=selected,
    )
    save_prep_manifest(manifest, output)

    report = {
        "successful_dataset_load": True,
        "total_rows": total_before,
        "rows_after_filters": len(filtered),
        "rows_selected": len(selected),
        "manifest_path": str(output),
        "sanitized_example": inference_view(selected[0]) if selected else None,
    }
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
