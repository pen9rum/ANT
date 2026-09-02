import os
from pathlib import Path

from ant.domain import (
    AbsenceProof,
    AnswerObligation,
    Evidence,
    FrontierResult,
    NeedGraph,
    NeedNode,
    ObligationCoverage,
    ProposedNode,
    StuckNodeSummary,
    TaskTrajectoryPackage,
    TokenUsage,
    UnresolvedNeed,
    WorkerCard,
)
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import (
    _FULL_TERMS_WORKER_LIMIT,
    ResponseResult,
    _completeness_notes,
    _extract_output_text,
    _extract_usage,
    _loads_json_object,
)


def test_openai_provider_loads_org_and_project_from_dotenv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
    monkeypatch.delenv("ANT_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test",
                "OPENAI_ORG_ID=org-test",
                "OPENAI_PROJECT_ID=proj_test",
                "ANT_MODEL=gpt-5.4-nano",
            ]
        ),
        encoding="utf-8",
    )

    provider = OpenAIProvider()

    assert provider.is_configured()
    assert provider.settings.organization == "org-test"
    assert provider.settings.project == "proj_test"
    assert provider.model == "gpt-5.4-nano"
    assert os.environ["OPENAI_API_KEY"] == "sk-test"


def test_openai_provider_prefers_dotenvs_own_credential_trio_over_a_stale_os_env_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Regression test: load_dotenv() defaulting to override=False (fixed
    # elsewhere) is not sufficient on its own -- if OPENAI_API_KEY happens
    # to already be set at the OS level (e.g. a stale value from some
    # unrelated earlier context) while OPENAI_ORG_ID/OPENAI_PROJECT_ID are
    # not, override=False alone would source the key from the OS and the
    # org/project from .env, pairing a key with an organization it doesn't
    # belong to -- confirmed live to fail every request with OpenAI's 401
    # "mismatched_organization". OPENAI_API_KEY/OPENAI_ORG_ID/
    # OPENAI_PROJECT_ID must come from .env together, atomically, whenever
    # .env defines them.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stale-os-level-key")
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-dotenvs-own-key",
                "OPENAI_ORG_ID=org-dotenv",
                "OPENAI_PROJECT_ID=proj_dotenv",
            ]
        ),
        encoding="utf-8",
    )

    provider = OpenAIProvider()

    assert provider.settings.api_key == "sk-dotenvs-own-key"
    assert provider.settings.organization == "org-dotenv"
    assert provider.settings.project == "proj_dotenv"


def test_extract_output_text_from_responses_payload() -> None:
    assert (
        _extract_output_text(
            {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "OK"},
                        ]
                    }
                ]
            }
        )
        == "OK"
    )


def test_extract_usage_and_json_object() -> None:
    usage = _extract_usage(
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        }
    )

    assert usage.total_tokens == 15
    assert _loads_json_object('```json\n{"ok": true}\n```') == {"ok": True}


def test_responses_kwargs_include_reasoning_effort_only_when_configured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ANT_MODEL", raising=False)

    provider = OpenAIProvider(model="gpt-5-2025-08-07", reasoning_effort="low")
    default_provider = OpenAIProvider(model="gpt-4.1")

    assert provider._responses_kwargs("judge", 128)["reasoning"] == {"effort": "low"}
    assert "reasoning" not in default_provider._responses_kwargs("synthesize", 128)


def test_responses_text_records_usage_on_sdk_path(monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd() / "tests")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)

    class Response:
        output_text = "OK"

        def model_dump(self) -> dict:
            return {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}

    class Responses:
        def create(self, **kwargs):
            return Response()

    class Client:
        responses = Responses()

    provider = OpenAIProvider(model="gpt-4.1")
    provider.client = lambda: Client()  # type: ignore[method-assign]

    assert provider.responses_text("hello").text == "OK"
    assert provider.drain_usage().total_tokens == 5


def test_generate_card_includes_typed_symbols(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "models.py").write_text(
        "class QAOA:\n    pass\n\nclass FALQON(QAOA):\n    pass\n",
        encoding="utf-8",
    )

    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result",
        (),
        {
            "text": (
                '{"name": "models", "responsibilities": ["models"], '
                '"searchable_terms": ["QAOA"]}'
            )
        },
    )()

    card = provider.generate_card(
        repo_root=str(tmp_path),
        territory_root="src",
        files=["src/models.py"],
    )

    symbols = {symbol.name: symbol for symbol in card.symbols}
    assert symbols["FALQON"].bases == ["QAOA"]


def test_plan_round_parses_graph_updates_assignments_and_special_tactics() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result",
        (),
        {
            "text": (
                '{"graph_updates": ['
                '{"need_id": "n2", "need": "locate timeout definition", '
                '"depends_on": ["n1"], "need_type": "implementation_location", '
                '"scope": "local"}'
                '], "assignments": {"n1": ["worker-a", "worker-b"]}, '
                '"special_tactics": {"n3": "temporary_bridge", "n4": "not_a_real_tactic"}}'
            )
        },
    )()
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a"),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b"),
    ]

    plan = provider.plan_round(
        question="q",
        graph=NeedGraph(nodes={}),
        resolution_results={},
        evidence=[],
        workers=workers,
        memory_hints={},
        frontier=FrontierResult(ready=["n1"], blocked=[], stuck_subgraphs=[]),
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert plan.graph_updates["n2"].need_id == "n2"
    assert plan.graph_updates["n2"].depends_on == ["n1"]
    assert plan.graph_updates["n2"].detail.need_type == "implementation_location"
    assert plan.assignments == {"n1": ["worker-a", "worker-b"]}
    # A worker id not among the supplied candidates is dropped, and a
    # tactic string outside the two allowed values is dropped too --
    # tolerating a malformed field without discarding the whole plan.
    assert plan.special_tactics == {"n3": "temporary_bridge"}


def test_plan_round_shows_worker_searchable_terms_not_just_routing_summary() -> None:
    # Regression test for a real qibo trace: a worker's searchable_terms
    # contained exact lexical hits for the question ("bloch", "sphere",
    # "paint_world_map") that its own LLM-compressed routing_summary
    # dropped entirely -- the Orchestrator never assigned that worker
    # directly for 5 of 6 rounds because it had no way to see those terms.
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_json(prompt: str, max_output_tokens: int = 512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "{}"})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]
    workers = [
        WorkerCard(
            id="worker-examples",
            territory_id="examples",
            name="examples",
            root="examples",
            routing_summary="clarifies classifier concepts, usage examples, and test or "
            "algorithm gaps",
            searchable_terms=["classify", "bloch", "sphere", "paint_world_map"],
        )
    ]

    provider.plan_round(
        question="How does Qibo render Bloch sphere visualizations?",
        graph=NeedGraph(nodes={}),
        resolution_results={},
        evidence=[],
        workers=workers,
        memory_hints={},
        frontier=FrontierResult(ready=[], blocked=[], stuck_subgraphs=[]),
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert "bloch" in captured["prompt"]
    assert "sphere" in captured["prompt"]
    assert "paint_world_map" in captured["prompt"]


def test_plan_round_shows_all_searchable_terms_when_candidates_are_few() -> None:
    # Regression test for the seaborn/pennylane failure mode: a worker's
    # own answering symbol (e.g. EstimateAggregator) sat past position 12
    # of its card's full term list -- the old fixed `[:12]` slice hid it
    # from the Orchestrator even after two-stage routing had already
    # correctly narrowed the candidate set down to this worker. At a
    # narrowed width (well under _FULL_TERMS_WORKER_LIMIT), every term on
    # the card should now reach the prompt.
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_json(prompt: str, max_output_tokens: int = 512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "{}"})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]
    terms = [f"term{i}" for i in range(30)] + ["EstimateAggregator"]
    workers = [
        WorkerCard(
            id="worker-seaborn",
            territory_id="seaborn",
            name="seaborn",
            root="seaborn",
            routing_summary="owns seaborn's top-level modules",
            searchable_terms=terms,
        )
    ]

    provider.plan_round(
        question="How does seaborn aggregate estimates for confidence intervals?",
        graph=NeedGraph(nodes={}),
        resolution_results={},
        evidence=[],
        workers=workers,
        memory_hints={},
        frontier=FrontierResult(ready=[], blocked=[], stuck_subgraphs=[]),
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert "EstimateAggregator" in captured["prompt"]


def test_plan_round_keeps_the_narrow_term_slice_for_a_large_unnarrowed_worker_list() -> None:
    # The full-terms behavior above is only safe at the narrowed
    # (two-stage-routed) candidate width. The unnarrowed fallback path --
    # every worker, no candidate limit, used when a need is stuck -- can
    # legitimately be 20-30+ workers; showing every worker's full term
    # list there would blow the prompt up quadratically. Confirms the
    # `_FULL_TERMS_WORKER_LIMIT` guard actually gates on worker count.
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_json(prompt: str, max_output_tokens: int = 512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "{}"})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]
    terms = [f"term{i}" for i in range(30)] + ["EstimateAggregator"]
    workers = [
        WorkerCard(
            id=f"worker-{index}",
            territory_id=f"territory-{index}",
            name=f"territory-{index}",
            root=f"territory-{index}",
            routing_summary=f"owns territory {index}",
            searchable_terms=terms if index == 0 else [f"other{index}"],
        )
        for index in range(_FULL_TERMS_WORKER_LIMIT + 1)
    ]

    provider.plan_round(
        question="How does seaborn aggregate estimates for confidence intervals?",
        graph=NeedGraph(nodes={}),
        resolution_results={},
        evidence=[],
        workers=workers,
        memory_hints={},
        frontier=FrontierResult(ready=[], blocked=[], stuck_subgraphs=[]),
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert "EstimateAggregator" not in captured["prompt"]
    assert "term0" in captured["prompt"]


def test_plan_round_shows_probe_anchors_and_orders_candidates_by_them() -> None:
    # candidate_probes comes from LocalCoordinator._probe_need_candidates --
    # a cheap search()/dense_search() look each candidate takes into its
    # own territory before the Orchestrator commits to one. The prompt
    # should show what was actually found (or "no anchors found"),
    # grouped by need_id, and put the candidate with the strongest probe
    # signal first -- replacing the old retrieval-rank annotation entirely
    # (confirmed live that a rank number alone was not reliable enough:
    # a "gates" question still pulled assignment toward a gates-named
    # worker over a better-ranked one with no actual gates-drawing
    # content). Never exclusionary -- worker-b (no anchors) still listed.
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_json(prompt: str, max_output_tokens: int = 512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "{}"})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a"),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b"),
        WorkerCard(id="worker-c", territory_id="c", name="c", root="c"),
    ]

    provider.plan_round(
        question="q",
        graph=NeedGraph(nodes={}),
        resolution_results={},
        evidence=[],
        workers=workers,
        memory_hints={},
        frontier=FrontierResult(ready=[], blocked=[], stuck_subgraphs=[]),
        incomplete_parents=[],
        cross_repo_experience=[],
        candidate_probes={
            "root": {
                "worker-c": [
                    Evidence(
                        path="src/c.py",
                        line_start=10,
                        line_end=12,
                        quote="def target_function():",
                        reason="probe",
                    )
                ],
                "worker-a": [],
                "worker-b": [],
            }
        },
    )

    prompt = captured["prompt"]
    # Every worker still listed -- worker-a/worker-b found nothing but
    # must not be excluded.
    assert "worker-a" in prompt
    assert "worker-b" in prompt
    assert "worker-c" in prompt
    assert "retrieval rank" not in prompt
    assert "src/c.py:10" in prompt
    assert "target_function" in prompt
    assert "no anchors found" in prompt
    # Strongest probe signal first: worker-c (1 anchor) precedes the
    # zero-anchor workers.
    assert prompt.index("worker-c") < prompt.index("worker-a")
    assert prompt.index("worker-c") < prompt.index("worker-b")


def test_plan_round_keeps_original_worker_order_with_no_candidate_probes() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_json(prompt: str, max_output_tokens: int = 512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "{}"})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]
    workers = [
        WorkerCard(id="worker-z", territory_id="z", name="z", root="z"),
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a"),
    ]

    provider.plan_round(
        question="q",
        graph=NeedGraph(nodes={}),
        resolution_results={},
        evidence=[],
        workers=workers,
        memory_hints={},
        frontier=FrontierResult(ready=[], blocked=[], stuck_subgraphs=[]),
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    prompt = captured["prompt"]
    assert "retrieval rank" not in prompt
    assert prompt.index("worker-z") < prompt.index("worker-a")


def test_plan_round_drops_an_assignment_to_a_worker_id_not_in_the_candidate_list() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result",
        (),
        {"text": '{"graph_updates": [], "assignments": {"n1": ["worker-ghost"]}, '
                 '"special_tactics": {}}'},
    )()

    plan = provider.plan_round(
        question="q",
        graph=NeedGraph(nodes={}),
        resolution_results={},
        evidence=[],
        workers=[WorkerCard(id="worker-a", territory_id="a", name="a", root="a")],
        memory_hints={},
        frontier=FrontierResult(ready=["n1"], blocked=[], stuck_subgraphs=[]),
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert plan.assignments == {}


def test_consolidate_graph_returns_empty_plan_with_no_proposals() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result", (), {"text": "should never be called"}
    )()

    plan = provider.consolidate_graph(
        question="q", active_nodes={}, proposals=[], candidate_hints={}
    )

    assert plan.decisions == []


def test_consolidate_graph_shows_active_nodes_and_candidate_hints_in_the_prompt() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_json(prompt: str, max_output_tokens: int = 512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "{}"})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]
    active_nodes = {
        "existing-gap": NeedNode(
            need_id="existing-gap",
            need="the existing gap",
            detail=UnresolvedNeed(description="the existing gap"),
        )
    }
    proposals = [
        ProposedNode(
            proposal_id="proposal-1",
            need="a reworded version of the existing gap",
            detail=UnresolvedNeed(description="a reworded version of the existing gap"),
            source="worker_observed",
        )
    ]

    provider.consolidate_graph(
        question="q",
        active_nodes=active_nodes,
        proposals=proposals,
        candidate_hints={"proposal-1": ["existing-gap"]},
    )

    prompt = captured["prompt"]
    assert "existing-gap" in prompt
    assert "the existing gap" in prompt
    assert "proposal-1" in prompt
    assert "a reworded version of the existing gap" in prompt
    assert "nearby existing nodes: existing-gap" in prompt


def test_consolidate_graph_parses_create_merge_and_drop_decisions() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=2048: type(  # type: ignore[method-assign]
        "Result",
        (),
        {
            "text": (
                '{"decisions": ['
                '{"proposal_id": "p1", "action": "create"}, '
                '{"proposal_id": "p2", "action": "merge", "target_node_id": "existing-gap"}, '
                '{"proposal_id": "p3", "action": "drop"}'
                "]}"
            )
        },
    )()
    proposals = [
        ProposedNode(proposal_id="p1", need="a", detail=UnresolvedNeed(description="a")),
        ProposedNode(proposal_id="p2", need="b", detail=UnresolvedNeed(description="b")),
        ProposedNode(proposal_id="p3", need="c", detail=UnresolvedNeed(description="c")),
    ]

    plan = provider.consolidate_graph(
        question="q", active_nodes={}, proposals=proposals, candidate_hints={}
    )

    decisions = {d.proposal_id: d for d in plan.decisions}
    assert decisions["p1"].action == "create"
    assert decisions["p2"].action == "merge"
    assert decisions["p2"].target_node_id == "existing-gap"
    assert decisions["p3"].action == "drop"


def test_consolidate_graph_drops_a_merge_decision_missing_its_required_target() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=2048: type(  # type: ignore[method-assign]
        "Result",
        (),
        {"text": '{"decisions": [{"proposal_id": "p1", "action": "merge"}]}'},
    )()
    proposals = [ProposedNode(proposal_id="p1", need="a", detail=UnresolvedNeed(description="a"))]

    plan = provider.consolidate_graph(
        question="q", active_nodes={}, proposals=proposals, candidate_hints={}
    )

    # No target_node_id for an action that requires one -- dropped rather
    # than trusted, same tolerant-of-malformed-entries posture as
    # _parse_round_plan. LocalCoordinator defaults an undecided proposal to
    # "create", so this is a safe degrade, not a lost proposal.
    assert plan.decisions == []


def test_consolidate_graph_drops_a_decision_for_an_unknown_proposal_id() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=2048: type(  # type: ignore[method-assign]
        "Result",
        (),
        {
            "text": (
                '{"decisions": [{"proposal_id": "not-a-real-proposal", "action": "create"}]}'
            )
        },
    )()
    proposals = [ProposedNode(proposal_id="p1", need="a", detail=UnresolvedNeed(description="a"))]

    plan = provider.consolidate_graph(
        question="q", active_nodes={}, proposals=proposals, candidate_hints={}
    )

    assert plan.decisions == []


def test_consolidate_graph_adds_the_alignment_instruction_only_when_enforced() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_json(prompt: str, max_output_tokens: int = 2048):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "{}"})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]
    proposals = [ProposedNode(proposal_id="p1", need="a", detail=UnresolvedNeed(description="a"))]

    provider.consolidate_graph(
        question="original question",
        active_nodes={},
        proposals=proposals,
        candidate_hints={},
        enforce_alignment=False,
    )
    assert "Grounded Fast Repair" not in captured["prompt"]

    provider.consolidate_graph(
        question="original question",
        active_nodes={},
        proposals=proposals,
        candidate_hints={},
        enforce_alignment=True,
    )
    assert "Grounded Fast Repair" in captured["prompt"]
    assert "DIRECTLY help answer the original question" in captured["prompt"]


def test_assess_need_alignment_returns_empty_plan_with_no_stuck_nodes() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=1536: type(  # type: ignore[method-assign]
        "Result", (), {"text": "should never be called"}
    )()

    plan = provider.assess_need_alignment(
        question="q", package=TaskTrajectoryPackage(question="q")
    )

    assert plan.verdicts == []


def test_assess_need_alignment_shows_epistemic_state_as_read_only_context() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_json(prompt: str, max_output_tokens: int = 1536):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "{}"})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]
    package = TaskTrajectoryPackage(
        question="original question",
        stuck_nodes=[
            StuckNodeSummary(
                need_id="n1",
                need="role resolution inheritance",
                resolution="unresolved",
                epistemic_state="absence_supported",
            )
        ],
    )

    provider.assess_need_alignment(question="original question", package=package)

    prompt = captured["prompt"]
    assert "n1" in prompt
    assert "role resolution inheritance" in prompt
    assert "absence_supported" in prompt
    assert "read-only" in prompt
    # The prompt must never ask the model to SET epistemic_state -- only
    # verdict/reframed_need/rationale are requested outputs.
    assert "verdict (one of keep/reframe/drop)" in prompt


def test_assess_need_alignment_parses_keep_reframe_and_drop_verdicts() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=1536: type(  # type: ignore[method-assign]
        "Result",
        (),
        {
            "text": (
                '{"verdicts": ['
                '{"need_id": "n1", "verdict": "keep"}, '
                '{"need_id": "n2", "verdict": "reframe", "reframed_need": "TLS/auth inheritance"}, '
                '{"need_id": "n3", "verdict": "drop"}'
                "]}"
            )
        },
    )()
    package = TaskTrajectoryPackage(
        question="q",
        stuck_nodes=[
            StuckNodeSummary(need_id="n1", need="a", resolution="unresolved"),
            StuckNodeSummary(need_id="n2", need="b", resolution="unresolved"),
            StuckNodeSummary(need_id="n3", need="c", resolution="unresolved"),
        ],
    )

    plan = provider.assess_need_alignment(question="q", package=package)

    verdicts = {v.need_id: v for v in plan.verdicts}
    assert verdicts["n1"].verdict == "keep"
    assert verdicts["n2"].verdict == "reframe"
    assert verdicts["n2"].reframed_need == "TLS/auth inheritance"
    assert verdicts["n3"].verdict == "drop"


def test_assess_need_alignment_drops_a_reframe_with_no_reframed_need_text() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=1536: type(  # type: ignore[method-assign]
        "Result",
        (),
        {"text": '{"verdicts": [{"need_id": "n1", "verdict": "reframe"}]}'},
    )()
    package = TaskTrajectoryPackage(
        question="q",
        stuck_nodes=[StuckNodeSummary(need_id="n1", need="a", resolution="unresolved")],
    )

    plan = provider.assess_need_alignment(question="q", package=package)

    assert plan.verdicts == []


def test_assess_need_alignment_drops_a_verdict_for_an_unknown_need_id() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=1536: type(  # type: ignore[method-assign]
        "Result",
        (),
        {"text": '{"verdicts": [{"need_id": "not-a-real-need", "verdict": "drop"}]}'},
    )()
    package = TaskTrajectoryPackage(
        question="q",
        stuck_nodes=[StuckNodeSummary(need_id="n1", need="a", resolution="unresolved")],
    )

    plan = provider.assess_need_alignment(question="q", package=package)

    assert plan.verdicts == []


def test_extract_answer_obligations_parses_a_list_of_strings() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result",
        (),
        {"text": '{"obligations": ["which classes subclass X", "what each one overrides"]}'},
    )()

    obligations = provider.extract_answer_obligations(question="q")

    assert [item.description for item in obligations] == [
        "which classes subclass X",
        "what each one overrides",
    ]
    # Ids are assigned locally, not trusted from the model -- downstream
    # code never depends on the LLM producing unique identifiers.
    assert len({item.obligation_id for item in obligations}) == 2


def test_extract_answer_obligations_degrades_to_empty_on_malformed_response() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result", (), {"text": "{}"}
    )()

    assert provider.extract_answer_obligations(question="q") == []


def test_check_obligation_coverage_parses_covered_and_uncovered() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=768: type(  # type: ignore[method-assign]
        "Result",
        (),
        {
            "text": (
                '{"coverage": ['
                '{"obligation_id": "obligation-0", "covered": true, "rationale": "found it"}, '
                '{"obligation_id": "obligation-1", "covered": false, "rationale": "not shown"}'
                "]}"
            )
        },
    )()
    obligations = [
        AnswerObligation(obligation_id="obligation-0", description="a"),
        AnswerObligation(obligation_id="obligation-1", description="b"),
    ]

    coverage = provider.check_obligation_coverage(
        question="q", obligations=obligations, evidence=[]
    )

    by_id = {item.obligation_id: item for item in coverage}
    assert by_id["obligation-0"].covered is True
    assert by_id["obligation-1"].covered is False


def test_check_obligation_coverage_defaults_an_omitted_obligation_to_uncovered() -> None:
    # "an obligation_id you omit defaults to covered=false" -- a malformed
    # or incomplete response must never look like confirmed coverage.
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=768: type(  # type: ignore[method-assign]
        "Result", (), {"text": '{"coverage": []}'}
    )()
    obligations = [AnswerObligation(obligation_id="obligation-0", description="a")]

    coverage = provider.check_obligation_coverage(
        question="q", obligations=obligations, evidence=[]
    )

    assert coverage == [ObligationCoverage(obligation_id="obligation-0", covered=False)]


def test_check_obligation_coverage_drops_an_unknown_obligation_id() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=768: type(  # type: ignore[method-assign]
        "Result",
        (),
        {"text": '{"coverage": [{"obligation_id": "not-real", "covered": true}]}'},
    )()
    obligations = [AnswerObligation(obligation_id="obligation-0", description="a")]

    coverage = provider.check_obligation_coverage(
        question="q", obligations=obligations, evidence=[]
    )

    assert coverage == [ObligationCoverage(obligation_id="obligation-0", covered=False)]


def test_verify_evidence_upgrade_parses_an_approved_verdict() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result",
        (),
        {
            "text": (
                '{"approved": true, "supported_claim": "X directly implements Y", '
                '"evidence_ids": ["0", "1"]}'
            )
        },
    )()

    verdict = provider.verify_evidence_upgrade(
        need=UnresolvedNeed(description="need"),
        epistemic_state="absence_supported",
        new_evidence=[
            Evidence(path="a.py", line_start=1, line_end=2, quote="x", reason="r")
        ],
        question="q",
    )

    assert verdict.approved is True
    assert verdict.supported_claim == "X directly implements Y"
    assert verdict.evidence_ids == ["0", "1"]


def test_verify_evidence_upgrade_rejects_an_unapproved_or_malformed_response() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    evidence = [Evidence(path="a.py", line_start=1, line_end=2, quote="x", reason="r")]

    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result", (), {"text": '{"approved": false}'}
    )()
    verdict = provider.verify_evidence_upgrade(
        need=UnresolvedNeed(description="need"),
        epistemic_state="open",
        new_evidence=evidence,
        question="q",
    )
    assert verdict.approved is False
    assert verdict.supported_claim == ""
    assert verdict.evidence_ids == []

    # A non-bool "approved" value must never look like an approval either.
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result", (), {"text": '{"approved": "yes"}'}
    )()
    verdict = provider.verify_evidence_upgrade(
        need=UnresolvedNeed(description="need"),
        epistemic_state="open",
        new_evidence=evidence,
        question="q",
    )
    assert verdict.approved is False


def test_verify_evidence_upgrade_rejects_approved_true_with_no_supported_claim() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result", (), {"text": '{"approved": true}'}
    )()

    verdict = provider.verify_evidence_upgrade(
        need=UnresolvedNeed(description="need"),
        epistemic_state="open",
        new_evidence=[Evidence(path="a.py", line_start=1, line_end=2, quote="x", reason="r")],
        question="q",
    )

    # "approved" with nothing concrete to point at is not a real grounded
    # upgrade -- degrades to unapproved rather than producing a
    # claim-less GroundedUpdate downstream.
    assert verdict.approved is False


def test_select_lookups_ranks_by_relevance_and_diversifies_by_path() -> None:
    # Budget-critical (K=6 kept), but arrival order carries no epistemic
    # meaning: a relevant item arriving late must still be visible, and a
    # single relevant item from an otherwise-underrepresented path must
    # not be crowded out by a same-path glut that happens to arrive first.
    provider = OpenAIProvider(model="gpt-4.1")
    captured = {}

    def fake_responses_json(prompt, max_output_tokens=256):
        captured["prompt"] = prompt
        return type("Result", (), {"text": '{"selected": []}'})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]

    evidence = [
        Evidence(
            path="src/samepath.py",
            line_start=index,
            line_end=index + 1,
            quote=f"target_symbol usage {index}",
            reason="r",
        )
        for index in range(9)
    ] + [
        Evidence(
            path="src/other.py",
            line_start=1,
            line_end=2,
            quote="target_symbol lonely usage",
            reason="r",
        )
    ]

    provider.select_lookups(need="target_symbol", evidence=evidence, candidates=["target_symbol"])

    assert "src/other.py" in captured["prompt"]


def test_plan_worker_actions_ranks_by_relevance_and_diversifies_by_path() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    captured = {}

    def fake_responses_json(prompt, max_output_tokens=768):
        captured["prompt"] = prompt
        return type("Result", (), {"text": '{"actions": []}'})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]

    evidence = [
        Evidence(
            path="src/samepath.py",
            line_start=index,
            line_end=index + 1,
            quote=f"target_symbol usage {index}",
            reason="r",
        )
        for index in range(10)
    ] + [
        Evidence(
            path="src/other.py",
            line_start=1,
            line_end=2,
            quote="target_symbol lonely usage",
            reason="r",
        )
    ]

    provider.plan_worker_actions(
        need="target_symbol",
        evidence=evidence,
        candidate_symbols=["target_symbol"],
        available_tools=["navigate"],
        hints=[],
        max_actions=4,
    )

    assert "src/other.py" in captured["prompt"]


def test_verify_evidence_upgrade_shows_every_new_evidence_item_no_count_cap() -> None:
    # Regression test for a real sphinx trace: a round's own new_evidence
    # can easily exceed 8 items once multiple workers or a coalition
    # contribute to the same round, and the need's own exact-match
    # answer (the literal import statement its description asked for by
    # name) landed at position 24 -- a fixed [:8] slice silently never
    # showed it to this verifier at all, not rejected, invisible. Must
    # show every item, same "zero relevance-based truncation" precedent
    # as select_evidence's own docstring.
    provider = OpenAIProvider(model="gpt-4.1")
    captured = {}

    def fake_responses_json(prompt, max_output_tokens=512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": '{"approved": false}'})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]

    new_evidence = [
        Evidence(path=f"src/noise_{index}.py", line_start=1, line_end=2, quote="noise", reason="r")
        for index in range(23)
    ] + [
        Evidence(
            path="src/decisive.py",
            line_start=10,
            line_end=12,
            quote="from sphinx.util import logging",
            reason="r",
        )
    ]

    provider.verify_evidence_upgrade(
        need=UnresolvedNeed(description="need"),
        epistemic_state="open",
        new_evidence=new_evidence,
        question="q",
    )

    assert "[23] src/decisive.py" in captured["prompt"]
    assert "from sphinx.util import logging" in captured["prompt"]


def test_observe_shows_every_evidence_item_no_count_cap() -> None:
    # Correctness-critical: observe() decides whether a real gap exists at
    # all. A fixed [:8] slice by arrival order (no epistemic meaning)
    # could hide the one item that would have shown the gap is already
    # closed, or hide the one item that reveals it -- same visibility
    # bug confirmed live in verify_evidence_upgrade.
    provider = OpenAIProvider(model="gpt-4.1")
    captured = {}

    def fake_responses_json(prompt, max_output_tokens=512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": '{"unresolved_needs": []}'})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]

    evidence = [
        Evidence(path=f"src/noise_{index}.py", line_start=1, line_end=2, quote="noise", reason="r")
        for index in range(23)
    ] + [
        Evidence(
            path="src/decisive.py",
            line_start=10,
            line_end=12,
            quote="from sphinx.util import logging",
            reason="r",
        )
    ]

    provider.observe(
        question="q", worker_id="worker-1", territory_id="territory-1", evidence=evidence
    )

    assert "[23] src/decisive.py" in captured["prompt"]
    assert "from sphinx.util import logging" in captured["prompt"]


def test_check_need_resolution_shows_every_new_evidence_item_no_count_cap() -> None:
    # Correctness-critical: resolved/partial/unresolved is the decision
    # this whole method exists for. A fixed [:8] slice by arrival order
    # (no epistemic meaning) could silently hide the one item that
    # actually resolves the need -- same visibility bug confirmed live
    # in verify_evidence_upgrade (a decisive item at position 24 of a
    # round's own findings never reached that verifier at all).
    provider = OpenAIProvider(model="gpt-4.1")
    captured = {}

    def fake_responses_json(prompt, max_output_tokens=512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": '{"status": "unresolved"}'})()

    provider.responses_json = fake_responses_json  # type: ignore[method-assign]

    new_evidence = [
        Evidence(path=f"src/noise_{index}.py", line_start=1, line_end=2, quote="noise", reason="r")
        for index in range(23)
    ] + [
        Evidence(
            path="src/decisive.py",
            line_start=10,
            line_end=12,
            quote="from sphinx.util import logging",
            reason="r",
        )
    ]

    provider.check_need_resolution(
        need=UnresolvedNeed(description="need"), new_evidence=new_evidence, question="q"
    )

    assert "[23] src/decisive.py" in captured["prompt"]
    assert "from sphinx.util import logging" in captured["prompt"]


def test_verify_evidence_upgrade_returns_unapproved_with_no_new_evidence() -> None:
    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result", (), {"text": "should never be called"}
    )()

    verdict = provider.verify_evidence_upgrade(
        need=UnresolvedNeed(description="need"),
        epistemic_state="open",
        new_evidence=[],
        question="q",
    )

    assert verdict.approved is False


def test_completeness_notes_only_includes_exhaustive_proofs() -> None:
    exhaustive = AbsenceProof(
        query="What subclasses inherit from QAOA?",
        relevant_symbols=["QAOA"],
        searched_worker_ids=["worker-a", "worker-b"],
        searched_territories=["a", "b"],
        searched_paths=["a/mod.py", "b/mod.py"],
        tools=["subclasses"],
        exhaustive=True,
        conclusion="found_1_subclass",
    )
    partial = AbsenceProof(query="...", exhaustive=False, conclusion="inconclusive")

    text = _completeness_notes([exhaustive, partial])

    assert "Exhaustive search for QAOA" in text
    assert "found_1_subclass" in text
    assert "inconclusive" not in text
    assert _completeness_notes([]) == ""
    assert _completeness_notes(None) == ""
    assert _completeness_notes([partial]) == ""


def test_synthesize_includes_completeness_notes_in_prompt(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_text(prompt: str, max_output_tokens: int = 512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "answer"})()

    provider.responses_text = fake_responses_text  # type: ignore[method-assign]

    provider.synthesize(
        question="What subclasses inherit from QAOA?",
        evidence=[
            Evidence(
                path="src/models/variational.py",
                line_start=549,
                line_end=575,
                quote="class FALQON(QAOA):\n    pass",
                reason="Subclass lookup for base symbol QAOA.",
            )
        ],
        absence_proofs=[
            AbsenceProof(
                query="What subclasses inherit from QAOA?",
                relevant_symbols=["QAOA"],
                searched_worker_ids=["worker-src"],
                searched_territories=["src"],
                searched_paths=["src/models/variational.py"],
                tools=["subclasses"],
                exhaustive=True,
                conclusion="found_1_subclass",
            )
        ],
    )

    assert "Completeness notes" in captured["prompt"]
    assert "Exhaustive search for QAOA" in captured["prompt"]
    assert "found_1_subclass" in captured["prompt"]
    assert "Subclass lookup for base symbol QAOA" in captured["prompt"]


def test_synthesize_patch_mode_instructs_against_narrating_the_revision(monkeypatch) -> None:
    # Regression test for a real, confirmed failure: patch-mode answers
    # (fast-repair retries, prior_answer set) opened with lines like
    # "Revision: Cross-checked, epistemic commitments retained..." and
    # "Here is a revised, evidence-rooted analysis..." instead of a clean
    # direct answer -- the model was narrating this prompt's own framing
    # ("this is a revision pass") back into its output, confirmed
    # identical across 3 independent sphinx questions. The prompt must
    # explicitly say not to.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(model="gpt-4.1")
    captured: dict[str, str] = {}

    def fake_responses_text(prompt: str, max_output_tokens: int = 512):
        captured["prompt"] = prompt
        return type("Result", (), {"text": "answer"})()

    provider.responses_text = fake_responses_text  # type: ignore[method-assign]

    provider.synthesize(
        question="q",
        evidence=[],
        prior_answer="gen0's own prior answer",
    )

    assert "never mention that this is a revision" in captured["prompt"]
    # A gen0/slow-gen1 call (prior_answer=="") must stay byte-identical to
    # before this instruction existed -- it's patch-mode-only.
    captured.clear()
    provider.synthesize(question="q", evidence=[])
    assert "never mention that this is a revision" not in captured["prompt"]


def test_responses_json_falls_back_to_empty_object_when_repair_also_fails(
    monkeypatch,
) -> None:
    # Regression test: a coalition cross-check (or any observe() call) used
    # to crash the whole batch if the model's JSON came back malformed and
    # the one repair attempt was *also* malformed (e.g. truncated by
    # max_output_tokens) -- responses_json returned that unvalidated text and
    # the caller's unguarded json.loads blew up. It must degrade to "{}"
    # instead of ever returning unparseable text.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(model="gpt-4.1")
    calls: list[str] = []

    def fake_responses_text(prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        calls.append(prompt)
        return ResponseResult(text="{not valid json", usage=TokenUsage(), raw={})

    provider.responses_text = fake_responses_text  # type: ignore[method-assign]

    result = provider.responses_json("some prompt")

    assert result.text == "{}"
    assert len(calls) == 2  # the original attempt plus exactly one repair attempt
    assert _loads_json_object(result.text) == {}


def test_responses_json_returns_repaired_result_when_repair_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(model="gpt-4.1")
    responses = iter(["{broken", '{"ok": true}'])

    def fake_responses_text(prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        return ResponseResult(text=next(responses), usage=TokenUsage(), raw={})

    provider.responses_text = fake_responses_text  # type: ignore[method-assign]

    result = provider.responses_json("some prompt")

    assert _loads_json_object(result.text) == {"ok": True}
