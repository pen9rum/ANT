"""Tests for the remaining Track B baseline interface stubs
(ant.agents.webwalkerqa_stubs, ant.external_wrappers.webwalker_native_agent).
Every run() must raise before touching network or an LLM -- these are
environment-validation-only interface stubs, per the governing spec's
"Do not launch any inference." Matched ReAct and ANTMAN are implemented
for real now -- see tests/test_matched_react_web.py and
tests/test_ant_web.py -- and are intentionally excluded from this file's
"raises NotImplementedError" checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents.webwalkerqa_stubs import DEFAULT_MAX_STEPS, DirectWebAgent, RetrievalWebAgent
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


@pytest.mark.parametrize("agent_cls", [DirectWebAgent, RetrievalWebAgent])
def test_stub_run_raises_not_implemented(agent_cls, tmp_path: Path) -> None:
    agent = agent_cls()
    with pytest.raises(NotImplementedError):
        agent.run(_EXAMPLE, tmp_path)


def test_official_webwalker_stub_raises_not_checked_out(tmp_path: Path) -> None:
    agent = WebWalkerNativeAgent()
    with pytest.raises(WebWalkerNativeAgentNotCheckedOut):
        agent.run(_EXAMPLE, tmp_path)


def test_default_max_steps_matches_the_paper_stated_explorer_cap() -> None:
    # docs/webwalkerqa_ant_mapping.md section 2: "Hard ceiling: at most 15
    # exploration steps."
    assert DEFAULT_MAX_STEPS == 15
