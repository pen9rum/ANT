from __future__ import annotations

import json
from pathlib import Path

import typer

from ant.coordinator import LocalCoordinator
from ant.environment import RepoEnvironment
from ant.evaluation import load_examples, run_batch
from ant.generation import generate_worker_cards
from ant.git_refresh import refresh_changed_workers
from ant.indexing import discover_territories
from ant.memory import IndexStore
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
    out: Path = Path("output/eval_results.jsonl"),
    split: str = "test",
    limit: int | None = None,
    max_rounds: int = MAX_ROUNDS_OPTION,
    synthesize: str = SYNTHESIZE_OPTION,
) -> None:
    """Run a small batch evaluation from JSONL or hf://dataset_name."""
    examples = load_examples(dataset, split=split, limit=limit)
    results = run_batch(
        examples=examples,
        repo_root=repo.resolve(),
        index_path=index_path,
        out_path=out,
        max_rounds=max_rounds,
        synthesize=synthesize,
    )
    typer.echo(f"Wrote {len(results)} results to {out}.")


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


if __name__ == "__main__":
    app()
