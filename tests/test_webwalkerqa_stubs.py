"""Tests for the remaining Track B baseline interface stubs
(ant.agents.webwalkerqa_stubs, ant.external_wrappers.webwalker_native_agent).
Every run() must raise before touching network or an LLM -- these are
environment-validation-only interface stubs, per the governing spec's
"Do not launch any inference." Matched ReAct is implemented for real now
-- see tests/test_matched_react_web.py -- and is intentionally excluded
from this file's "raises NotImplementedError" checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents.matched_react_web import DEFAULT_MAX_STEPS as REACT_DEFAULT_MAX_STEPS
from ant.agents.matched_react_web import MatchedReActWebAgent
from ant.agents.webwalkerqa_stubs import (
    DEFAULT_MAX_STEPS,
    AntWebAgent,
    DirectWebAgent,
    RetrievalWebAgent,
)
from ant.benchmarks.base import TaskExample
from ant.external_wrappers.webwalker_native_agent import (
    WebWalkerNativeAgent,
    WebWalkerNativeAgentNotCheckedOut,
)

_EXAMPLE = TaskExample(
    benchmark="webwalkerqa",
    task_id="webwalkerqa-abc123",
    question="What is the keynote speaker's affiliation?",
    reference="",
    metadata={"root_url": "http://conf.example.com/"},
)


@pytest.mark.parametrize("agent_cls", [DirectWebAgent, RetrievalWebAgent, AntWebAgent])
def test_stub_run_raises_not_implemented(agent_cls, tmp_path: Path) -> None:
    agent = agent_cls()
    with pytest.raises(NotImplementedError):
        agent.run(_EXAMPLE, tmp_path)


def test_official_webwalker_stub_raises_not_checked_out(tmp_path: Path) -> None:
    agent = WebWalkerNativeAgent()
    with pytest.raises(WebWalkerNativeAgentNotCheckedOut):
        agent.run(_EXAMPLE, tmp_path)


def test_react_and_ant_share_the_same_default_step_budget() -> None:
    # The governing spec's explicit fairness requirement: ReAct and ANTMAN
    # must receive the same runtime navigation primitives/budget. The two
    # DEFAULT_MAX_STEPS constants live in different modules now that
    # MatchedReActWebAgent is implemented for real (see that module's own
    # docstring for why they are kept manually in sync, not imported
    # cross-module) -- assert they still agree.
    assert DEFAULT_MAX_STEPS == REACT_DEFAULT_MAX_STEPS == 15
    assert MatchedReActWebAgent().max_steps == AntWebAgent().max_steps == 15


def test_default_max_steps_matches_the_paper_stated_explorer_cap() -> None:
    # docs/webwalkerqa_ant_mapping.md section 2: "Hard ceiling: at most 15
    # exploration steps."
    assert DEFAULT_MAX_STEPS == 15


def test_stub_names_are_distinct() -> None:
    names = {
        DirectWebAgent.name,
        RetrievalWebAgent.name,
        MatchedReActWebAgent.name,
        AntWebAgent.name,
        WebWalkerNativeAgent.name,
    }
    assert len(names) == 5


def test_custom_max_steps_is_honored_in_constructor() -> None:
    agent = AntWebAgent(max_steps=5)
    assert agent.max_steps == 5
