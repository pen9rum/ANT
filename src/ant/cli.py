from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from ant.coordinator import LocalCoordinator
from ant.environment import RepoEnvironment
from ant.evaluation import (
    build_report,
    fetch_repositories,
    load_examples,
    load_repo_specs,
    run_batch,
)
from ant.evolution import evolve_workers
from ant.generation import generate_worker_cards
from ant.git_refresh import refresh_changed_workers
from ant.indexing import discover_territories
from ant.memory import IndexStore
from ant.memory.colony import ColonyMemoryStore
from ant.providers import OpenAIProvider

app = typer.Typer(no_args_is_help=True)
INDEX_OPTION = typer.Option(Path(".ant"), "--index")
MAX_ROUNDS_OPTION = typer.Option(2, "--max-rounds", min=1)
SAVE_TRACE_OPTION = typer.Option(True, "--save-trace/--no-save-trace")
LLM_CARDS_OPTION = typer.Option(False, "--llm-cards/--heuristic-cards")
SYNTHESIZE_OPTION = typer.Option("none", "--synthesize")


@app.command()
def index(repo: Path, out: Path = Path(".ant"), llm_cards: bool = LLM_CARDS_OPTION) -> None:
    """Build worker cards for a repository."""
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    generator = OpenAIProvider() if llm_cards else None
    workers = generate_worker_cards(environment.root, territories, generator=generator)
    IndexStore(out).save(territories, workers)
    typer.echo(f"Indexed {len(territories)} territories and {len(workers)} workers into {out}.")


@app.command()
def ask(
    question: str,
    repo: Path = Path("."),
    index_path: Path = INDEX_OPTION,
    max_rounds: int = MAX_ROUNDS_OPTION,
    save_trace: bool = SAVE_TRACE_OPTION,
    synthesize: str = SYNTHESIZE_OPTION,
) -> None:
    """Ask a local evidence question using saved worker cards."""
    store = IndexStore(index_path)
    workers = store.load_workers()
    provider = OpenAIProvider() if synthesize == "openai" else None
    state = LocalCoordinator(repo.resolve(), workers, synthesizer=provider).ask(
        question,
        max_rounds=max_rounds,
    )
    if save_trace:
        store.save_trace(state)
    typer.echo(json.dumps(state.model_dump(), indent=2))


@app.command("openai-smoke")
def openai_smoke() -> None:
    """Run a minimal Responses API call using .env settings."""
    provider = OpenAIProvider()
    typer.echo(provider.smoke_test())


@app.command("eval")
def eval_command(
    dataset: str,
    repo: Path = Path("."),
    index_path: Path = INDEX_OPTION,
    out: Path | None = None,
    run_id: str | None = None,
    run_dir: Path | None = None,
    split: str = "test",
    limit: int | None = None,
    max_rounds: int = MAX_ROUNDS_OPTION,
    synthesize: str = SYNTHESIZE_OPTION,
    judge: str = "heuristic",
    report: Path | None = None,
) -> None:
    """Run a small batch evaluation from JSONL or hf://dataset_name."""
    resolved_run_dir = _resolve_run_dir(run_id=run_id, run_dir=run_dir)
    out = out or resolved_run_dir / "results.jsonl"
    examples = load_examples(dataset, split=split, limit=limit)
    results = run_batch(
        examples=examples,
        repo_root=repo.resolve(),
        index_path=index_path,
        out_path=out,
        max_rounds=max_rounds,
        synthesize=synthesize,
        judge=judge,
    )
    report_path = report or resolved_run_dir / "summary.json"
    summary = build_report(out, report_path)
    manifest = {
        "dataset": dataset,
        "repo": str(repo),
        "index": str(index_path),
        "split": split,
        "limit": limit,
        "max_rounds": max_rounds,
        "synthesize": synthesize,
        "judge": judge,
        "results": str(out),
        "summary": str(report_path),
    }
    (resolved_run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"Wrote {len(results)} results to {out}.")
    typer.echo(f"Wrote summary to {report_path}: {summary.model_dump_json()}")


@app.command("report")
def report_command(results: Path, out: Path | None = None) -> None:
    """Aggregate an eval JSONL file into a summary report."""
    summary = build_report(results, out)
    typer.echo(summary.model_dump_json(indent=2))


@app.command()
def refresh(
    repo: Path = Path("."),
    index_path: Path = INDEX_OPTION,
    base: str = "HEAD",
    llm_cards: bool = LLM_CARDS_OPTION,
) -> None:
    """Refresh only worker cards affected by git diff."""
    generator = OpenAIProvider() if llm_cards else None
    result = refresh_changed_workers(
        repo_root=repo.resolve(),
        index_path=index_path,
        base=base,
        generator=generator,
    )
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def evolve(
    index_path: Path = INDEX_OPTION,
    min_coalition_count: int = 2,
    retire_empty: bool = True,
    merge_overlap: float = 0.9,
) -> None:
    """Apply worker population evolution from recurring coalition memory."""
    result = evolve_workers(
        index_path,
        min_coalition_count=min_coalition_count,
        retire_empty=retire_empty,
        merge_overlap=merge_overlap,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def revalidate(repo: Path = Path("."), index_path: Path = INDEX_OPTION) -> None:
    """Revalidate stale memory records after repository changes."""
    result = ColonyMemoryStore(index_path).revalidate_stale(repo.resolve())
    typer.echo(json.dumps(result, indent=2))


@app.command("sweqa-fetch-repos")
def sweqa_fetch_repos(
    target_dir: Path = Path("repos"),
    source: str = "https://raw.githubusercontent.com/TIGER-AI-Lab/SWE-QA-Pro/main/eval/repos.txt",
    limit: int | None = None,
    repo_filter: str | None = typer.Option(None, "--repo"),
) -> None:
    """Clone and checkout SWE-QA-Pro repositories from repos.txt."""
    specs = load_repo_specs(source)
    paths = fetch_repositories(
        target_dir=target_dir,
        specs=specs,
        limit=limit,
        repo_filter=repo_filter,
    )
    typer.echo(json.dumps([str(path) for path in paths], indent=2))


def _resolve_run_dir(run_id: str | None, run_dir: Path | None) -> Path:
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path("output") / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    app()
