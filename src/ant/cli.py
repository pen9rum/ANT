from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from ant.coordinator import LocalCoordinator
from ant.domain import EvidenceState
from ant.environment import RepoEnvironment
from ant.evaluation import (
    build_report,
    fetch_repositories,
    load_examples,
    load_repo_specs,
    run_batch,
    run_gen_compare,
)
from ant.evolution import evolve_workers
from ant.generation import generate_worker_cards
from ant.git_refresh import refresh_changed_workers
from ant.indexing import discover_territories
from ant.memory import (
    GlobalMemoryStore,
    IndexStore,
    record_global_experience_safe,
    retrieve_cross_repo_experience_safe,
)
from ant.memory.colony import ColonyMemoryStore, record_task_memory
from ant.providers import OpenAIProvider
from ant.retrieval.dense import WORKER_CARDS_KEY, DenseEmbedder, build_worker_card_index

app = typer.Typer(no_args_is_help=True)
INDEX_OPTION = typer.Option(Path(".ant"), "--index")
MAX_ROUNDS_OPTION = typer.Option(6, "--max-rounds", min=1)
SAVE_TRACE_OPTION = typer.Option(True, "--save-trace/--no-save-trace")
LLM_CARDS_OPTION = typer.Option(False, "--llm-cards/--heuristic-cards")
DENSE_OPTION = typer.Option(False, "--dense/--no-dense")
SYNTHESIZE_OPTION = typer.Option("none", "--synthesize")
REASONER_OPTION = typer.Option("none", "--reasoner")


@app.command()
def index(
    repo: Path,
    out: Path = Path(".ant"),
    llm_cards: bool = LLM_CARDS_OPTION,
    dense: bool = DENSE_OPTION,
) -> None:
    """Build worker cards for a repository."""
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    generator = OpenAIProvider() if llm_cards else None
    workers = generate_worker_cards(environment.root, territories, generator=generator)
    store = IndexStore(out)
    store.save(territories, workers)
    if dense:
        # Only the worker-card level index is built eagerly here: one
        # embedding per worker (dozens), not one per code chunk (thousands).
        # Per-worker chunk embeddings for dense_search are built lazily, on
        # first actual use, and cached under out/dense/ -- see
        # LocalSearchTool.dense_search.
        card_index = build_worker_card_index(workers, DenseEmbedder())
        store.save_embedding_index(WORKER_CARDS_KEY, card_index)
        typer.echo(f"Built a dense worker-card index over {len(workers)} workers.")
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
    colony_memory = ColonyMemoryStore(index_path)
    global_memory = GlobalMemoryStore()
    workers = store.load_workers()
    provider = OpenAIProvider() if synthesize == "openai" else None
    memory_routes = colony_memory.matching_routes(question.split())
    # Only when a real reasoner is in play: MockLLMProvider's plan_round
    # ignores cross_repo_experience entirely and its summarize_task_experience
    # always returns "", so retrieving/recording anything here would just be
    # a real embedding-model load spent on a result nothing will ever use.
    resolved_repo_name = repo.resolve().name
    cross_repo_experience = (
        retrieve_cross_repo_experience_safe(
            global_memory, question, exclude_repo=resolved_repo_name
        )
        if provider
        else []
    )
    coordinator = LocalCoordinator(
        repo.resolve(),
        workers,
        synthesizer=provider,
        memory_routes=memory_routes,
        index_path=index_path,
        cross_repo_experience=cross_repo_experience,
    )
    state = coordinator.ask(
        question,
        max_rounds=max_rounds,
    )
    if save_trace:
        store.save_trace(state)
    record_task_memory(colony_memory, question, state)
    if provider:
        record_global_experience_safe(
            global_memory, coordinator.reasoner, question, state, repo=resolved_repo_name
        )
    typer.echo(json.dumps(state.model_dump(), indent=2))


@app.command("symbols")
def symbols(index_path: Path = INDEX_OPTION, query: str = "") -> None:
    """Inspect the persisted repository symbol manifest."""
    store = IndexStore(index_path)
    symbol_items = store.load_symbol_manifest()
    if query:
        lowered = query.lower()
        symbol_items = [
            entry
            for entry in symbol_items
            if isinstance((symbol := entry.get("symbol")), dict)
            and (
                lowered in str(symbol.get("name", "")).lower()
                or lowered in str(symbol.get("qualname", "")).lower()
                or any(lowered in str(base).lower() for base in symbol.get("bases", []))
            )
        ]
    typer.echo(json.dumps(symbol_items, indent=2))


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


@app.command("gen-compare")
def gen_compare(
    dataset: str,
    repo: Path = Path("."),
    index_path: Path = INDEX_OPTION,
    run_id: str | None = None,
    run_dir: Path | None = None,
    split: str = "test",
    limit: int | None = None,
    max_rounds: int = MAX_ROUNDS_OPTION,
    fast_max_rounds: int | None = None,
    judge: str = "heuristic",
    gen0: bool = typer.Option(True, "--gen0/--no-gen0"),
    slow_gen1: bool = typer.Option(True, "--slow-gen1/--no-slow-gen1"),
    fast_gen1: bool = typer.Option(True, "--fast-gen1/--no-fast-gen1"),
) -> None:
    """Unified gen0 -> slow-gen1 (colony evolution) -> fast-gen1
    (task-conditioned retry) comparison over one dataset. Every stage's
    per-question EvidenceState JSON is saved under the run directory
    (gen0-<id>.json, slow-gen1-<id>.json, fast-gen1-<id>.json, plus a
    *-results.jsonl per stage) and one summary.json aggregates each stage's
    per-question EvalScore -- the same on-disk shape every prior gen0/gen1
    comparison run of this project's own ad-hoc scripts produced by hand,
    now from one committed, reusable command.

    Each of --gen0/--slow-gen1/--fast-gen1 can be switched off independently
    (all four combinations that make sense: everything, gen0 only, slow-gen1
    only, fast-gen1 only) -- e.g. `--no-gen0 --no-slow-gen1` re-runs only
    fast-gen1 against this same --run-dir's already-saved gen0 traces, so a
    fix that only touches fast-gen1's own code path doesn't require paying
    for gen0/slow-gen1 again. A skipped stage's numbers are still read back
    from this run_dir's own earlier *-results.jsonl into the summary, if one
    is already there.
    """
    resolved_run_dir = _resolve_run_dir(run_id=run_id, run_dir=run_dir)
    examples = load_examples(dataset, split=split, limit=limit)
    result = run_gen_compare(
        examples=examples,
        repo_root=repo.resolve(),
        index_path=index_path,
        run_dir=resolved_run_dir,
        max_rounds=max_rounds,
        fast_max_rounds=fast_max_rounds,
        judge=judge,
        run_gen0=gen0,
        run_slow_gen1=slow_gen1,
        run_fast_gen1=fast_gen1,
    )
    summary_path = resolved_run_dir / "summary.json"
    summary_path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    typer.echo(f"Wrote summary to {summary_path}")


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
    repo: Path = Path("."),
    index_path: Path = INDEX_OPTION,
    reasoner: str = REASONER_OPTION,
    min_coalition_count: int = 2,
    retire_empty: bool = True,
    merge_overlap: float = 0.9,
    min_specialization_routes: int = 4,
    min_specialization_group_routes: int = 2,
    min_episode_count: int = 3,
) -> None:
    """Apply worker population evolution from recurring coalition/route memory.

    Without --reasoner openai, only the structural gates run (file-overlap
    merge, route-count specialize, empty-file retire, raw coalition-count
    birth) with no LLM veto on merge/specialize, and the richer episode-driven
    strengthen_route/birth_bridge/merge path (EvolutionReasoner.decide_episode_action,
    which also weighs whether a recurring strategy actually worked, not just
    how often it recurred) is skipped entirely -- ant.evolution.evolve_workers
    only runs it when a reasoner is passed in at all.
    """
    llm_reasoner = OpenAIProvider() if reasoner == "openai" else None
    result = evolve_workers(
        index_path,
        repo_root=repo.resolve(),
        reasoner=llm_reasoner,
        min_coalition_count=min_coalition_count,
        retire_empty=retire_empty,
        merge_overlap=merge_overlap,
        min_specialization_routes=min_specialization_routes,
        min_specialization_group_routes=min_specialization_group_routes,
        min_episode_count=min_episode_count,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def retry(
    trace_path: Path,
    repo: Path = Path("."),
    index_path: Path = INDEX_OPTION,
    max_rounds: int = MAX_ROUNDS_OPTION,
    synthesize: str = SYNTHESIZE_OPTION,
) -> None:
    """Task-conditioned ("fast") retry of one finished task, from its own
    trajectory -- not colony reorganization. `trace_path` is a saved
    EvidenceState JSON (e.g. `ant ask ... > trace.json`). Repairs an
    ephemeral, task-local copy of the prior attempt's Need Graph from what
    it actually tried and found (never a reference answer or judge score)
    and retries the same question. Unlike `ask`, deliberately never calls
    record_task_memory/record_global_experience_safe/store.save_trace --
    nothing about invoking this touches persistent colony/global memory,
    even by omission-prone accident.
    """
    prior_state = EvidenceState.model_validate_json(trace_path.read_text(encoding="utf-8"))
    if synthesize != "openai":
        raise typer.BadParameter(
            "retry requires --synthesize openai: FastEvolutionReasoner.propose_repair "
            "has no heuristic/mock fallback."
        )
    workers = IndexStore(index_path).load_workers()
    provider = OpenAIProvider()
    coordinator = LocalCoordinator(
        repo.resolve(), workers, synthesizer=provider, index_path=index_path
    )
    state = coordinator.retry_from_trajectory(
        prior_state, fast_reasoner=provider, max_rounds=max_rounds
    )
    typer.echo(json.dumps(state.model_dump(), indent=2))


@app.command()
def revalidate(index_path: Path = INDEX_OPTION) -> None:
    """Revalidate stale memory records against the colony's current workers."""
    current_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    result = ColonyMemoryStore(index_path).revalidate_stale(current_worker_ids)
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
