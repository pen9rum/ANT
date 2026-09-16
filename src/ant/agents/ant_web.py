"""ANTMAN over the Track B (web-navigation) substrate: wires the frozen
`ant.coordinator.local.LocalCoordinator` onto WebWalkerQA, with LIVE
multi-hop web navigation happening inside worker execution.

HISTORY / WHY THIS SHAPE: an earlier version of this file built a frozen,
root+one-hop worker roster entirely at bootstrap time (before `ask()`
ever started) and handed LocalCoordinator a static file set. That shipped
and was confirmed live to be fundamentally broken: a 24-task paid run
scored 0/24, and a direct trajectory comparison against the corrected
ReAct baseline's own runs for those same 24 tasks showed EVERY one of
them needed >=2 real navigate() hops (several needed 10-14) to reach the
answer page -- a one-hop-only bootstrap can never materialize that page
into any worker's evidence, regardless of model quality. See
`output/runs/_invalid_debug/webwalkerqa-antman-full60-INVALID-onehop-bootstrap/`
for the preserved, never-to-be-merged evidence of that failure.

THE FIX: `ant.agents.web_navigation_tool.WebSearchTool` (see its own
module docstring for the full mechanism) is a duck-typed substitute for
`ant.tools.local.LocalSearchTool`, injected into a COMPLETELY UNMODIFIED
`LocalCoordinator.ask()` call via the one small, additive,
default-preserving `search_tool_factory` seam added to
`LocalCoordinator.__init__` for exactly this purpose (omitting it
reproduces Track A's exact prior behavior byte-for-byte -- see that
parameter's own docstring in local.py). Its `search()` method -- called
UNCONDITIONALLY by `AutonomousWorker.run()` at the start of every worker
execution -- is where the real, live, budget-checked, link-restricted
navigation happens: a worker's first hop is its own deterministic
assigned first-level link (preserving per-worker territory
specialization); further hops, if the need still isn't answered, are
chosen by a small dedicated LLM call over the (anchor_text, url) pairs
actually present on whatever page the worker is currently on. Navigation
is enforced by a SINGLE, QUERY-LEVEL SHARED `EvalWebEnvironment`
instance (the same primitive `MatchedReActWebAgent` uses) -- so the
15-step ceiling this task specifies is a shared budget across every
worker combined, never a per-worker allowance, and "only a link actually
present on the current page" is enforced by that same primitive, not
re-implemented here.

Everything genuinely ANTMAN-canonical -- Need Graph construction, runtime
Need revision, dynamic worker routing, progress/resolution tracking,
rerouting/recovery, evidence selection, synthesis -- is untouched
`LocalCoordinator.ask()` behavior; this file and web_navigation_tool.py
only supply the substrate-specific tool implementation and the initial
(root-page-derived, gold-blind) territory roster.

nav_budget=15 (WebWalkerQA's own paper-stated Explorer cap, identical to
`MatchedReActWebAgent`'s own budget -- a fairness requirement) and
max_rounds=10 (an explicit, deliberate coordination-budget choice, kept
INDEPENDENT of nav_budget: a coordination round may consume zero, one,
or several navigation actions depending on worker activity, so the two
are different dimensions and neither one bounds the other).

GOLD-LEAKAGE NOTE: bootstrap, run(), and WebSearchTool's own link-choice
prompt read only `example.question` and `example.metadata["root_url"]`
plus pages actually reached through real navigation -- never
`example.reference` or `example.metadata["_audit_only"]`.
"""

from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.agents.web_navigation_tool import WebSearchTool, WorkerFrontier
from ant.benchmarks.base import TaskExample
from ant.coordinator import LocalCoordinator
from ant.domain import WorkerCard
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats
from ant.evaluation_suite.web_fetch import PageCache, urllib_fetcher
from ant.evaluation_suite.web_scope import EvalWebEnvironment

DEFAULT_NAV_BUDGET = 15
# Independent of nav_budget -- see this module's own docstring. Matches
# the project's own precedent for an explicit task-specific max_rounds
# override (e.g. the long-context multineedle-scaling experiment's own
# max_rounds=10), not ANTMAN's canonical max_rounds=6 default.
DEFAULT_MAX_ROUNDS = 10
# Upper bound on how many first-level-link CANDIDATE workers bootstrap
# creates. Deliberately NOT tied to nav_budget: creating a candidate
# WorkerCard costs no navigation budget at all (a worker's assigned link
# is only actually fetched lazily, the first time that worker's search()
# is actually called during real coordination -- see WebSearchTool), so
# there is no reason to cap candidate-worker count at the navigation
# ceiling. This is a generous safety bound against a pathologically
# link-heavy root page, not a scarce resource -- "many territories may
# exist; only the few an unresolved Need actually selects ever consume
# navigation."
DEFAULT_MAX_CANDIDATE_WORKERS = 40


def _bootstrap_territories(
    example: TaskExample,
    environment_root: Path,
    nav_budget: int,
    max_candidate_workers: int,
    timeout_seconds: float,
) -> tuple[list[WorkerCard], dict[str, WorkerFrontier], Path, EvalWebEnvironment, int]:
    """Fetches ONLY the root page (no navigation budget spent yet -- see
    EvalWebEnvironment.root_page()'s own docstring: the root is given, not
    discovered) and builds one provisional WorkerCard per first-level link
    found on it, in extraction order (never sorted/prioritized by
    relevance, which would risk gold-adjacent bias), up to
    max_candidate_workers -- a generous safety bound, not a navigation
    budget concern (creating a candidate worker costs no navigation at
    all). Does NOT fetch any of those linked pages -- that happens lazily,
    per-worker, only if and when that worker's search() is actually
    called during coordination (see WebSearchTool.search()). Also creates
    one worker representing root-page-only content, so root text itself
    stays directly reachable even if it has no useful outgoing links for
    the current question. Returns (workers, frontiers, materialized_root,
    shared_web_environment, n_root_inaccessible).
    """
    root_url = example.metadata["root_url"]
    materialized_dir = environment_root / "materialized"
    materialized_dir.mkdir(parents=True, exist_ok=True)

    cache = PageCache(
        cache_dir=environment_root / "pages",
        fetcher=urllib_fetcher(timeout_seconds=timeout_seconds),
        root_url=root_url,
    )
    env = EvalWebEnvironment(root_url, cache, max_steps=nav_budget)
    root_page = env.root_page()

    workers: list[WorkerCard] = []
    frontiers: dict[str, WorkerFrontier] = {}
    if root_page.status != "ok":
        return workers, frontiers, materialized_dir, env, 1

    def _add_worker(worker_id: str, label: str, assigned_link) -> None:
        filename = f"{worker_id}__root.txt"
        (materialized_dir / filename).write_text(root_page.text, encoding="utf-8")
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
        frontiers[worker_id] = WorkerFrontier(
            current_page=root_page,
            assigned_link=assigned_link,
            visited_urls={root_page.url},
            taken_first_hop=assigned_link is None,
        )

    _add_worker("worker-root", "The website's own root/landing page.", None)
    for i, link in enumerate(root_page.links[:max_candidate_workers]):
        label = link.text.strip() or f"page {i}"
        _add_worker(f"worker-{i}", label, link)

    return workers, frontiers, materialized_dir, env, 0


class AntWebAgent:
    """See this module's own docstring for the full design/architecture
    note. Preserves ANTMAN's canonical coordination method via a
    completely unmodified `LocalCoordinator.ask()` call -- only the
    substrate-specific tool implementation (web_navigation_tool.py) and
    the initial territory roster (this file's own `_bootstrap_territories`)
    differ from Track A's `AntAgent`.
    """

    name = "ant_web"

    def __init__(
        self,
        model: str = "gpt-4.1",
        nav_budget: int = DEFAULT_NAV_BUDGET,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        max_candidate_workers: int = DEFAULT_MAX_CANDIDATE_WORKERS,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.model = model
        self.nav_budget = nav_budget
        self.max_rounds = max_rounds
        self.max_candidate_workers = max_candidate_workers
        self.timeout_seconds = timeout_seconds

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        started = time.time()
        workers, frontiers, materialized_dir, env, n_root_inaccessible = _bootstrap_territories(
            example,
            environment_root,
            self.nav_budget,
            self.max_candidate_workers,
            self.timeout_seconds,
        )

        provider = CountingOpenAIProvider(model=self.model)
        tool = WebSearchTool(materialized_dir, env, provider, frontiers)
        coordinator = LocalCoordinator(
            materialized_dir,
            workers,
            reasoner=provider,
            synthesizer=provider,
            index_path=None,
            search_tool_factory=lambda root, idx: tool,
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

        n_nav_steps = env.step_count()
        assert n_nav_steps <= self.nav_budget, (
            f"shared navigation budget violated: {n_nav_steps} > {self.nav_budget}"
        )
        nav_by_worker: dict[str, int] = {}
        for entry in tool.nav_log:
            worker_id = entry["worker"]
            nav_by_worker[worker_id] = nav_by_worker.get(worker_id, 0) + 1

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
                "navigation_steps_by_worker": nav_by_worker,
                "nav_log": tool.nav_log,
                "territories_discovered": len(workers),
                "pages_fetched": len(env.discovered_pages()),
                "active_workers": len(active_worker_ids),
                "n_rounds": len(state.rounds),
                "n_reroutes": n_reroutes,
                "n_need_revisions": n_need_revisions,
                "n_root_inaccessible": n_root_inaccessible,
                "nav_budget_exhausted": n_nav_steps >= self.nav_budget,
                "final_need_graph_size": len(state.final_need_graph),
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(AntWebAgent())
