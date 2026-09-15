"""Compatible interface/config STUBS for the Track B (WebWalkerQA)
baseline methods -- Direct, Retrieval, Matched ReAct, and ANTMAN, each
wired to accept the same substrate primitives
(`ant.evaluation_suite.web_scope.EvalWebEnvironment`/
`ant.evaluation_suite.web_fetch.PageCache`) so a later implementation
pass has the right shape to fill in.

Per the governing spec: "Do not launch any inference" -- every `run()`
below raises `NotImplementedError` immediately, before touching the
network or an LLM. This file exists so the constructor signatures,
config knobs, and step-budget parity between ReAct and ANTMAN are decided
and reviewable NOW, not improvised later.

FAIRNESS NOTE (per the governing spec's explicit requirement): "the most
important fairness requirement is that ReAct and ANTMAN receive the same
runtime page/navigation primitives." Both `MatchedReActWebAgent` and
`AntWebAgent` below share the identical `max_steps=DEFAULT_MAX_STEPS`
default (15, WebWalkerQA's own paper-stated Explorer step cap -- see
`docs/webwalkerqa_ant_mapping.md` section 2) and are constructed against
the SAME `EvalWebEnvironment` primitives (`root_page()`/`navigate()`/
`inspect()`) -- neither gets a wider navigation budget or an extra
primitive the other lacks. `RepoGraph` is explicitly out of scope here
(repository-specific, per the governing spec).
"""

from __future__ import annotations

from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample

# WebWalkerQA's own paper-stated Explorer-agent step ceiling
# (`docs/webwalkerqa_ant_mapping.md` section 2) -- the shared navigation
# budget for both ReAct and ANTMAN's web substrate, per this file's own
# fairness note above.
DEFAULT_MAX_STEPS = 15


class DirectWebAgent:
    """Root-page-only reference: answers using ONLY the root URL's own
    page content, no navigation at all. The web-substrate analogue of
    Track A's `DirectAgent` (whole-repo-blind, single-call baseline).
    """

    name = "direct_web"

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        raise NotImplementedError(
            "DirectWebAgent is an environment-only interface stub (see the governing "
            "spec's Step 7: 'Do not launch any inference'). Constructor/config shape is "
            "frozen; run() is intentionally unimplemented."
        )


class RetrievalWebAgent:
    """Simple retrieval baseline: fetches up to `max_pages` pages reachable
    from the root (breadth-first over discovered links, no LLM-decided
    navigation), then answers from the combined fetched text -- the web
    analogue of Track A's `RetrievalAgent` (bounded, non-agentic evidence
    gathering before one synthesis call).
    """

    name = "retrieval_web"

    def __init__(self, model: str = "gpt-4.1", max_pages: int = 8) -> None:
        self.model = model
        self.max_pages = max_pages

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        raise NotImplementedError(
            "RetrievalWebAgent is an environment-only interface stub (see the governing "
            "spec's Step 7: 'Do not launch any inference'). Constructor/config shape is "
            "frozen; run() is intentionally unimplemented."
        )


class MatchedReActWebAgent:
    """Matched ReAct web navigation: a single-agent Thought-Action-
    Observation loop over `EvalWebEnvironment.navigate()`/`.inspect()`,
    capped at `max_steps` (shared with `AntWebAgent` -- see this module's
    own fairness note). No Need Graph, no worker coordination, no
    RepoGraph-equivalent -- the web-substrate analogue of Track A's
    `MatchedReActAgent`.
    """

    name = "matched_react_web"

    def __init__(self, model: str = "gpt-4.1", max_steps: int = DEFAULT_MAX_STEPS) -> None:
        self.model = model
        self.max_steps = max_steps

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        raise NotImplementedError(
            "MatchedReActWebAgent is an environment-only interface stub (see the "
            "governing spec's Step 7: 'Do not launch any inference'). Constructor/config "
            "shape is frozen; run() is intentionally unimplemented."
        )


class AntWebAgent:
    """ANTMAN over the web substrate: reuses ANT's core coordination
    (Need Graph semantics, runtime Need revision, progress tracking,
    rerouting/recovery, worker coordination) UNCHANGED, per the governing
    spec's Step 4 -- only the substrate-specific information-access layer
    differs (territories/workers/navigation/inspection from
    `ant.evaluation_suite.web_scope`, in place of Track A's
    `EvalRepoEnvironment`/`LocalSearchTool`). Capped at `max_steps`,
    shared with `MatchedReActWebAgent` -- see this module's own fairness
    note.

    NOT YET WIRED to `LocalCoordinator`: doing so is explicitly out of
    scope for this environment-validation pass (the governing spec's
    "STOP after environment validation"). This stub exists to fix the
    constructor/config shape (model, max_steps, and -- once a future pass
    designs it -- however territory/worker construction from
    `ant.evaluation_suite.web_scope.build_territory_from_discovered` gets
    threaded into `LocalCoordinator`) ahead of that implementation work.
    """

    name = "ant_web"

    def __init__(self, model: str = "gpt-4.1", max_steps: int = DEFAULT_MAX_STEPS) -> None:
        self.model = model
        self.max_steps = max_steps

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        raise NotImplementedError(
            "AntWebAgent is an environment-only interface stub (see the governing spec's "
            "Step 7: 'Do not launch any inference'). Constructor/config shape is frozen; "
            "run() is intentionally unimplemented -- LocalCoordinator wiring is deferred "
            "to a future implementation pass per the governing spec's explicit scope stop."
        )
