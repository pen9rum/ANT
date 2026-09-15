"""ANTMAN over the Track B (web-navigation) substrate: wires the frozen,
UNMODIFIED `ant.coordinator.local.LocalCoordinator` onto WebWalkerQA.

ARCHITECTURAL NOTE (confirmed by direct source reading before writing
this file, not assumed): `LocalCoordinator.ask()` constructs its own
`LocalSearchTool` internally (hardcoded by class name, not caller-
injectable), which only ever reads real bytes from local disk -- there is
no HTTP fetch anywhere inside core coordination code. And `self.workers`
(the territory/worker roster) is provably immutable for the whole
`ask()` call: `evolve_workers` (ANTMAN's only worker-growth mechanism) is
never invoked from `ask()` -- confirmed by grep across `src/ant`, its two
call sites are the separate `ant evolve` CLI command and `ant
gen-compare`. **This is not a web-specific limitation.** Track A's own
`AntAgent` has the identical shape: `discover_territories()` enumerates
the WHOLE repo BEFORE `ask()` starts, and the worker set stays fixed for
that call too. ANTMAN's "dynamic" story has always been about ROUTING
(which worker handles which need, each round), never about the worker
set growing mid-task.

So the substrate-appropriate mapping here is a bounded, gold-blind
NAVIGATION-BASED BOOTSTRAP (root page + up to `nav_budget` real
`navigate()` calls over the root page's own discovered links, using the
exact same `EvalWebEnvironment`/`PageCache` the ReAct baseline uses --
same-site restriction, meta-refresh following, deterministic cache, all
unmodified) that builds the initial `WorkerCard` roster BEFORE calling
the completely unmodified `LocalCoordinator.ask()` -- analogous in role
to `discover_territories()`, not a violation of "runtime-discovered, not
pre-enumerated": nothing here ever reads `golden_path`/`source_website`/
the gold answer, and it does not enumerate "the site" (bounded to root +
one hop, at most `nav_budget` pages -- versus a repo's own COMPLETE file
tree, which Track A's own bootstrap genuinely does enumerate in full).

Each of the root page's own links becomes ONE single-page territory/
worker, in the order the links were extracted (never sorted/prioritized
by any relevance heuristic, which would risk gold-adjacent bias) -- up
to `nav_budget` workers, fewer if the root has fewer links or some
navigations fail (skipped gracefully, matching the ReAct baseline's own
degradation). The root page itself is ALSO its own worker, so root-page-
only content stays directly reachable.

Per this task's own explicit instruction: `nav_budget=15` (matching
WebWalkerQA's paper-stated Explorer cap, identical to
`MatchedReActWebAgent`'s own budget) and `max_rounds=15` (a deliberate
task-specific override of ANTMAN's own canonical `max_rounds=6` default
-- `max_rounds` is a caller-supplied runtime parameter throughout this
project, e.g. the long-context multineedle-scaling experiment's own
`max_rounds=10`, never a core-code change).

GOLD-LEAKAGE NOTE: bootstrap and `run()` read only `example.question`
and `example.metadata["root_url"]` -- never `example.reference` or
`example.metadata["_audit_only"]`.
"""

from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.coordinator import LocalCoordinator
from ant.domain import WorkerCard
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats
from ant.evaluation_suite.web_fetch import FetchedPage, PageCache, urllib_fetcher
from ant.evaluation_suite.web_scope import EvalWebEnvironment

# WebWalkerQA's own paper-stated Explorer-agent step ceiling -- the SAME
# navigation budget `MatchedReActWebAgent` uses (fairness requirement).
DEFAULT_NAV_BUDGET = 15
# Task-specific override of ANTMAN's canonical max_rounds=6 default --
# see this module's own docstring for why.
DEFAULT_MAX_ROUNDS = 15


def _bootstrap_territories(
    example: TaskExample, environment_root: Path, nav_budget: int, timeout_seconds: float
) -> tuple[list[WorkerCard], Path, int, int, bool]:
    """Builds the initial WorkerCard roster via a bounded, gold-blind
    navigation pass -- see this module's own docstring. Returns
    (workers, materialized_root, navigation_steps_used,
    n_inaccessible_pages, nav_budget_exhausted).
    """
    root_url = example.metadata["root_url"]
    page_cache_dir = environment_root / "pages"
    materialized_dir = environment_root / "materialized"
    materialized_dir.mkdir(parents=True, exist_ok=True)

    cache = PageCache(
        cache_dir=page_cache_dir,
        fetcher=urllib_fetcher(timeout_seconds=timeout_seconds),
        root_url=root_url,
    )
    env = EvalWebEnvironment(root_url, cache, max_steps=nav_budget)

    root_page = env.root_page()
    workers: list[WorkerCard] = []
    n_inaccessible = 0

    def _materialize(page: FetchedPage, filename: str, worker_id: str, label: str) -> None:
        (materialized_dir / filename).write_text(page.text, encoding="utf-8")
        workers.append(
            WorkerCard(
                id=worker_id,
                territory_id=worker_id,
                name=label,
                root=".",
                files=[filename],
                responsibilities=[label] if label else [],
            )
        )

    if root_page.status == "ok":
        _materialize(
            root_page, "page_root.txt", "worker-root", "The website's own root/landing page."
        )
    else:
        n_inaccessible += 1

    for i, link in enumerate(root_page.links):
        if env.step_count() >= nav_budget:
            break
        try:
            page = env.navigate(root_page, link.url)
        except (ValueError, RuntimeError):
            continue
        if page.status != "ok":
            n_inaccessible += 1
            continue
        label = link.text.strip() or f"page {i}"
        _materialize(page, f"page_{i:04d}.txt", f"worker-{i}", label)

    nav_budget_exhausted = env.step_count() >= nav_budget
    return workers, materialized_dir, env.step_count(), n_inaccessible, nav_budget_exhausted


class AntWebAgent:
    """See this module's own docstring for the full design/architecture
    note. Preserves ANTMAN's canonical coordination method (Need Graph,
    runtime Need revision, dynamic worker routing, progress/resolution
    tracking, rerouting/recovery, evidence selection, synthesis) via a
    completely unmodified `LocalCoordinator.ask()` call -- only the
    substrate-specific territory-BUILDING step (this file's own
    `_bootstrap_territories`) differs from Track A's `AntAgent`.
    """

    name = "ant_web"

    def __init__(
        self,
        model: str = "gpt-4.1",
        nav_budget: int = DEFAULT_NAV_BUDGET,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.model = model
        self.nav_budget = nav_budget
        self.max_rounds = max_rounds
        self.timeout_seconds = timeout_seconds

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        started = time.time()
        workers, materialized_dir, n_nav_steps, n_inaccessible, nav_budget_exhausted = (
            _bootstrap_territories(example, environment_root, self.nav_budget, self.timeout_seconds)
        )

        provider = CountingOpenAIProvider(model=self.model)
        coordinator = LocalCoordinator(
            materialized_dir, workers, reasoner=provider, synthesizer=provider, index_path=None
        )
        state = coordinator.ask(example.question, max_rounds=self.max_rounds)

        active_worker_ids = {
            wid for r in state.rounds for ne in r.node_executions for wid in ne.worker_ids
        }
        n_reroutes = sum(
            1
            for r in state.rounds
            for ne in r.node_executions
            if ne.special_tactic in ("temporary_bridge", "global_fallback")
        )
        n_need_revisions = sum(
            len(r.graph_delta.created_nodes)
            + len(r.graph_delta.dependency_changes)
            + len(r.graph_delta.created_children)
            for r in state.rounds
        )

        # LocalCoordinator.ask() already drains self.synthesizer.drain_usage()
        # once internally (local.py's own final-synthesis block) and returns
        # it as state.usage -- since reasoner and synthesizer are the SAME
        # provider instance here, a second provider.drain_usage() call would
        # see nothing (already reset to empty). Only drain_call_count() is
        # safe to read again: ask() never calls it (not part of the
        # UsageReporter protocol it drains).
        llm_calls = provider.drain_call_count()
        token_usage = state.usage
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=state.answer,
            trajectory=[r.model_dump() for r in state.rounds],
            evidence=[e.model_dump() for e in state.evidence],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=n_nav_steps,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len(active_worker_ids),
            ),
            termination_reason=(
                "unresolved_needs_remain" if state.unresolved_needs else "all_needs_resolved"
            ),
            metadata={
                "generation_model": self.model,
                "nav_budget": self.nav_budget,
                "max_rounds": self.max_rounds,
                "navigation_steps": n_nav_steps,
                "territories_discovered": len(workers),
                "active_workers": len(active_worker_ids),
                "n_rounds": len(state.rounds),
                "n_reroutes": n_reroutes,
                "n_need_revisions": n_need_revisions,
                "n_inaccessible_pages_in_bootstrap": n_inaccessible,
                "nav_budget_exhausted": nav_budget_exhausted,
                "final_need_graph_size": len(state.final_need_graph),
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(AntWebAgent())
