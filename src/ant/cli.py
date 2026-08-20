from __future__ import annotations

import json
from pathlib import Path

import typer

from ant.coordinator import LocalCoordinator
from ant.environment import RepoEnvironment
from ant.indexing import build_worker_cards, discover_territories
from ant.memory import IndexStore

app = typer.Typer(no_args_is_help=True)
INDEX_OPTION = typer.Option(Path(".ant"), "--index")
MAX_ROUNDS_OPTION = typer.Option(2, "--max-rounds", min=1)
SAVE_TRACE_OPTION = typer.Option(True, "--save-trace/--no-save-trace")


@app.command()
def index(repo: Path, out: Path = Path(".ant")) -> None:
    """Build worker cards for a repository."""
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    IndexStore(out).save(territories, workers)
    typer.echo(f"Indexed {len(territories)} territories and {len(workers)} workers into {out}.")


@app.command()
def ask(
    question: str,
    repo: Path = Path("."),
    index_path: Path = INDEX_OPTION,
    max_rounds: int = MAX_ROUNDS_OPTION,
    save_trace: bool = SAVE_TRACE_OPTION,
) -> None:
    """Ask a local evidence question using saved worker cards."""
    store = IndexStore(index_path)
    workers = store.load_workers()
    state = LocalCoordinator(repo.resolve(), workers).ask(question, max_rounds=max_rounds)
    if save_trace:
        store.save_trace(state)
    typer.echo(json.dumps(state.model_dump(), indent=2))


if __name__ == "__main__":
    app()
