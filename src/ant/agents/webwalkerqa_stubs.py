"""Compatible interface/config STUBS for the remaining Track B
(WebWalkerQA) baseline methods -- Direct and Retrieval -- wired to accept
the same substrate primitives (`ant.evaluation_suite.web_scope.
EvalWebEnvironment`/`ant.evaluation_suite.web_fetch.PageCache`) so a
later implementation pass has the right shape to fill in.

Matched ReAct and ANTMAN have since been implemented for real -- see
`ant.agents.matched_react_web.MatchedReActWebAgent` and
`ant.agents.ant_web.AntWebAgent` -- and are NOT defined in this file
anymore; `DEFAULT_MAX_STEPS` below is kept in sync with both modules'
own copies (all equal 15) rather than imported cross-module, to avoid a
real dependency between "the implemented baselines" and "the still-
stubbed ones."

Per the governing spec: "Do not launch any inference" for everything
still in this file -- every `run()` below raises `NotImplementedError`
immediately, before touching the network or an LLM.
"""

from __future__ import annotations

from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample

# WebWalkerQA's own paper-stated Explorer-agent step ceiling
# (`docs/webwalkerqa_ant_mapping.md` section 2) -- the shared navigation
# budget across every Track B method (ReAct, ANTMAN, and these stubs).
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
