from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents.ablation_agents import AblationAntAgent, AblationAntDocumentAgent
from ant.coordinator import LocalCoordinator
from ant.coordinator.ablations import (
    PROFILES,
    active_coordination_profile,
    resolve_profile,
    use_coordination_profile,
)
from ant.domain import (
    GraphConsolidationDecision,
    GraphConsolidationPlan,
    NeedNode,
    NeedResolution,
    RoundPlan,
    UnresolvedNeed,
    WorkerCard,
    WorkerObservation,
)
from ant.providers import MockLLMProvider


def _workers() -> list[WorkerCard]:
    return [
        WorkerCard(
            id="worker-a",
            territory_id="a",
            name="worker a",
            root="src",
            searchable_terms=["target"],
            files=["src/a.py"],
        ),
        WorkerCard(
            id="worker-b",
            territory_id="b",
            name="worker b",
            root="src",
            searchable_terms=["target"],
            files=["src/b.py"],
        ),
    ]


class _ScriptedReasoner(MockLLMProvider):
    def __init__(self, plans: list[RoundPlan], resolution: NeedResolution | None = None) -> None:
        self._plans = plans
        self._resolution = resolution or NeedResolution(status="unresolved")
        self.consolidation_calls = 0

    def plan_round(self, **_kwargs) -> RoundPlan:
        return self._plans.pop(0) if self._plans else RoundPlan()

    def observe(self, **_kwargs) -> WorkerObservation:
        return WorkerObservation(worker_id="scripted", territory_id="scripted")

    def check_need_resolution(self, **_kwargs) -> NeedResolution:
        return self._resolution

    def consolidate_graph(self, *, proposals, **_kwargs) -> GraphConsolidationPlan:
        self.consolidation_calls += 1
        return GraphConsolidationPlan(
            decisions=[
                GraphConsolidationDecision(proposal_id=proposal.proposal_id, action="create")
                for proposal in proposals
            ]
        )

    def select_evidence(self, **_kwargs) -> tuple[list[str], list[str]]:
        return [], []

    def assess_facet_completeness(self, **_kwargs):
        from ant.domain import FacetRescuePlan

        return FacetRescuePlan()


def _coordinator(
    tmp_path: Path, reasoner: _ScriptedReasoner, monkeypatch: pytest.MonkeyPatch
) -> LocalCoordinator:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def target(): pass\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("def target(): pass\n", encoding="utf-8")
    coordinator = LocalCoordinator(tmp_path, _workers(), reasoner=reasoner)
    monkeypatch.setattr(coordinator, "_run_selected_workers", lambda *args, **kwargs: ([], []))
    return coordinator


def _revision_plan() -> RoundPlan:
    return RoundPlan(
        graph_updates={
            "root": NeedNode(
                need_id="root",
                need="rewritten root",
                detail=UnresolvedNeed(description="rewritten root"),
            ),
            "child": NeedNode(
                need_id="child",
                need="new child",
                detail=UnresolvedNeed(description="new child"),
            ),
        },
        assignments={"root": ["worker-a"]},
    )


def test_no_need_revision_blocks_graph_and_formulation_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refined = NeedResolution(
        status="partial", refined_need=UnresolvedNeed(description="resolution refinement")
    )
    reasoner = _ScriptedReasoner([_revision_plan()], refined)
    state = _coordinator(tmp_path, reasoner, monkeypatch).ask(
        "original question", max_rounds=1, coordination_profile="no_need_revision"
    )

    assert set(state.final_need_graph) == {"root"}
    assert state.final_need_graph["root"].need == "original question"
    assert state.rounds[0].graph_delta.created_nodes == []
    assert reasoner.consolidation_calls == 0


def test_full_profile_keeps_the_same_revision_path_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reasoner = _ScriptedReasoner([_revision_plan()])
    state = _coordinator(tmp_path, reasoner, monkeypatch).ask("original question", max_rounds=1)

    assert "child" in state.final_need_graph
    assert state.final_need_graph["root"].need == "rewritten root"
    assert reasoner.consolidation_calls == 1


@pytest.mark.parametrize("profile", ["no_adaptive_rerouting", "static"])
def test_fixed_routing_profiles_reuse_the_first_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    reasoner = _ScriptedReasoner(
        [
            RoundPlan(assignments={"root": ["worker-a"]}),
            RoundPlan(assignments={"root": ["worker-b"]}),
        ]
    )
    state = _coordinator(tmp_path, reasoner, monkeypatch).ask(
        "target question", max_rounds=2, coordination_profile=profile
    )

    assert [round_.node_executions[0].worker_ids for round_ in state.rounds] == [
        ["worker-a"],
        ["worker-a"],
    ]


def test_no_recovery_disables_stuck_state_attempt_memory_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reasoner = _ScriptedReasoner(
        [
            RoundPlan(assignments={"root": ["worker-a"]}),
            RoundPlan(
                assignments={"root": ["worker-b"]},
                special_tactics={"root": "global_fallback"},
            ),
        ]
    )
    state = _coordinator(tmp_path, reasoner, monkeypatch).ask(
        "target question", max_rounds=2, coordination_profile="no_recovery"
    )

    assert [round_.node_executions[0].worker_ids for round_ in state.rounds] == [
        ["worker-a"],
        ["worker-b"],
    ]
    assert state.final_need_graph["root"].progress == "not_stuck"
    assert state.final_recovery_state.stuck_episodes == []
    assert state.final_recovery_state.tried_workers_by_node == {}
    assert not any(
        trace.special_tactic
        for round_ in state.rounds
        for trace in round_.node_executions
    )


def test_static_profile_blocks_need_revision_and_recovery_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reasoner = _ScriptedReasoner(
        [
            _revision_plan(),
            RoundPlan(assignments={"root": ["worker-b"]}),
        ]
    )
    state = _coordinator(tmp_path, reasoner, monkeypatch).ask(
        "original question", max_rounds=2, coordination_profile="static"
    )

    assert set(state.final_need_graph) == {"root"}
    assert [round_.node_executions[0].worker_ids for round_ in state.rounds] == [
        ["worker-a"],
        ["worker-a"],
    ]
    assert state.final_recovery_state.tried_workers_by_node == {}


def test_profiles_are_named_and_exposed_as_distinct_agent_conditions() -> None:
    assert set(PROFILES) == {
        "full",
        "static",
        "no_need_revision",
        "no_adaptive_rerouting",
        "no_recovery",
    }
    assert AblationAntAgent("static").name == "ant_static"
    assert AblationAntDocumentAgent("no_recovery").name == "ant_document_no_recovery"
    with pytest.raises(ValueError, match="Unknown adaptive-coordination profile"):
        resolve_profile("not-a-profile")


def test_profile_scope_is_reset_after_one_ablation_task() -> None:
    assert active_coordination_profile().key == "full"
    with use_coordination_profile("static"):
        assert active_coordination_profile().key == "static"
    assert active_coordination_profile().key == "full"
