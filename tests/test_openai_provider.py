import os
from pathlib import Path

from ant.domain import AbsenceProof, Evidence, FrontierResult, NeedGraph, TokenUsage, WorkerCard
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import (
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
        observed_needs=[],
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
        observed_needs=[],
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert "bloch" in captured["prompt"]
    assert "sphere" in captured["prompt"]
    assert "paint_world_map" in captured["prompt"]


def test_plan_round_reorders_and_annotates_workers_by_relevance_rank() -> None:
    # worker_relevance_rank comes from ant.coordinator.worker_retrieval's
    # retrieval-based ranking (BM25 + exact-symbol + dense over
    # WorkerCard.symbols) -- when the coordinator supplies one, the prompt
    # should surface it as a rank annotation and put the best match first,
    # WITHOUT dropping any worker from the list (see rank_workers' own
    # docstring: never exclusionary, only reordering/annotation).
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
        observed_needs=[],
        incomplete_parents=[],
        cross_repo_experience=[],
        worker_relevance_rank={"worker-c": 1, "worker-a": 2},
    )

    prompt = captured["prompt"]
    # Every worker still listed -- worker-b has no rank entry (no
    # retrieval signal) but must not be excluded.
    assert "worker-a" in prompt
    assert "worker-b" in prompt
    assert "worker-c" in prompt
    assert "(retrieval rank 1/3)" in prompt
    assert "(retrieval rank 2/3)" in prompt
    # Best match first: worker-c's line precedes worker-a's, which
    # precedes worker-b's (unranked, falls back to original order last).
    assert prompt.index("worker-c") < prompt.index("worker-a") < prompt.index("worker-b")


def test_plan_round_keeps_original_worker_order_when_no_relevance_rank_given() -> None:
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
        observed_needs=[],
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
        observed_needs=[],
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert plan.assignments == {}


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
