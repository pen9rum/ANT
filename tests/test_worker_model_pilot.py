"""Tests for the worker-model bake-off's own additive plumbing:
- `LocalCoordinator`'s new `worker_reasoner` seam defaults to reusing
  `self.reasoner` (byte-for-byte unaffected when unset) and, when set,
  routes ONLY `AutonomousWorker`'s local tool-call loop to it -- every
  orchestrator-level call (`observe`, `plan_round`,
  `check_need_resolution`, `consolidate_graph`, `verify_evidence_upgrade`,
  `select_evidence`) stays on `self.reasoner`.
- `AntAgent`'s new `worker_model`/`worker_base_url` constructor
  parameters: unset reproduces today's single-provider construction
  exactly; set constructs a genuinely separate `VLLMChatCompletionsProvider`
  and threads it through as `worker_reasoner=`.
- `VLLMChatCompletionsProvider` talks Chat Completions (not the Responses
  API) and reports zero cost for an unrecognized local model name.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ant.coordinator.local import LocalCoordinator
from ant.domain import Evidence, WorkerCard, WorkerObservation
from ant.evaluation_suite.vllm_provider import VLLMChatCompletionsProvider


class _StubWorkerReasoner:
    """Minimal WorkerReasoner stub recording which of its methods were
    actually called, without needing a real LLM."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def observe(self, **kwargs):
        self.calls.append("observe")
        return WorkerObservation(
            worker_id=kwargs["worker_id"],
            territory_id=kwargs["territory_id"],
            evidence=[],
            unresolved_needs=[],
            actions=[],
        )

    def select_lookups(self, **kwargs):
        self.calls.append("select_lookups")
        return []

    def plan_worker_actions(self, **kwargs):
        self.calls.append("plan_worker_actions")
        return []

    def __getattr__(self, name):
        # Any other WorkerReasoner method: record the call, return a
        # harmless default. Only used for methods this test never asserts
        # on directly (plan_round/check_need_resolution/etc. go through
        # the real orchestrator reasoner in these tests, never this stub).
        def _unexpected(*args, **kwargs):
            raise AssertionError(f"unexpected call to worker-scoped stub: {name}")

        return _unexpected


class TestLocalCoordinatorWorkerReasonerSeam:
    def test_worker_reasoner_defaults_to_reasoner(self, tmp_path: Path) -> None:
        reasoner = MagicMock()
        coordinator = LocalCoordinator(tmp_path, [], reasoner=reasoner)
        assert coordinator.worker_reasoner is coordinator.reasoner
        assert coordinator.worker_reasoner is reasoner

    def test_worker_reasoner_defaults_via_synthesizer_only(self, tmp_path: Path) -> None:
        # LocalCoordinator's existing fallback: reasoner=None, synthesizer
        # set -> self.reasoner becomes the synthesizer. worker_reasoner
        # must follow that same resolved object, not stay None.
        synthesizer = MagicMock()
        coordinator = LocalCoordinator(tmp_path, [], synthesizer=synthesizer)
        assert coordinator.reasoner is synthesizer
        assert coordinator.worker_reasoner is synthesizer

    def test_explicit_worker_reasoner_is_kept_separate_from_reasoner(
        self, tmp_path: Path
    ) -> None:
        reasoner = MagicMock(name="orchestrator_reasoner")
        worker_reasoner = MagicMock(name="worker_reasoner")
        coordinator = LocalCoordinator(
            tmp_path, [], reasoner=reasoner, worker_reasoner=worker_reasoner
        )
        assert coordinator.reasoner is reasoner
        assert coordinator.worker_reasoner is worker_reasoner
        assert coordinator.worker_reasoner is not coordinator.reasoner

    def test_run_selected_workers_routes_only_worker_loop_to_worker_reasoner(
        self, tmp_path: Path
    ) -> None:
        orchestrator_reasoner = MagicMock(name="orchestrator_reasoner")
        orchestrator_reasoner.observe.return_value = WorkerObservation(
            worker_id="worker-a",
            territory_id="root",
            evidence=[],
            unresolved_needs=[],
            actions=[],
        )
        worker_reasoner = _StubWorkerReasoner()

        coordinator = LocalCoordinator(
            tmp_path, [], reasoner=orchestrator_reasoner, worker_reasoner=worker_reasoner
        )
        worker = WorkerCard(
            id="worker-a",
            territory_id="root",
            name="worker-a",
            root="",
            responsibilities=[],
            searchable_terms=[],
            routing_summary="",
        )
        search = MagicMock()
        search.search.return_value = []
        search.dense_search.return_value = []

        with patch(
            "ant.coordinator.local.AutonomousWorker"
        ) as MockAutonomousWorker:
            MockAutonomousWorker.return_value.run.return_value = WorkerObservation(
                worker_id="worker-a",
                territory_id="root",
                evidence=[],
                unresolved_needs=[],
                actions=[],
            )
            coordinator._run_selected_workers(
                [worker],
                query="q",
                question="q",
                search=search,
                worker_config=MagicMock(),
                evidence=[],
                seen_worker_ids=set(),
            )
            # AutonomousWorker must be constructed with worker_reasoner,
            # NOT orchestrator_reasoner.
            _, kwargs = MockAutonomousWorker.call_args
            assert kwargs["reasoner"] is worker_reasoner

        # observe() -- progress interpretation -- must stay on the
        # orchestrator reasoner, never the worker_reasoner.
        orchestrator_reasoner.observe.assert_called_once()
        assert "observe" not in worker_reasoner.calls


class TestAntAgentWorkerModelSplit:
    def test_default_construction_has_no_worker_model(self) -> None:
        from ant.agents.ant_adapter import AntAgent

        agent = AntAgent()
        assert agent.worker_model is None
        assert agent.worker_base_url is None
        assert agent.worker_max_context_tokens is None

    def test_setting_only_worker_model_without_base_url_asserts(
        self, tmp_path: Path
    ) -> None:
        from ant.agents.ant_adapter import AntAgent
        from ant.benchmarks.base import TaskExample

        agent = AntAgent(worker_model="qwen3-8b")  # worker_base_url left unset
        example = TaskExample(
            benchmark="repoprobe", task_id="t1", question="q", reference=""
        )
        (tmp_path / "workers.json")  # not created; _ensure_indexed will index
        with pytest.raises(AssertionError):
            agent.run(example, tmp_path)


class TestVLLMChatCompletionsProvider:
    def test_uses_chat_completions_not_responses_api(self) -> None:
        provider = VLLMChatCompletionsProvider(
            model="qwen3-8b", base_url="http://localhost:8000/v1"
        )
        fake_response = MagicMock()
        fake_response.model_dump.return_value = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        fake_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(provider, "client", return_value=fake_client):
            result = provider.responses_text("prompt", max_output_tokens=64)

        fake_client.chat.completions.create.assert_called_once_with(
            model="qwen3-8b",
            messages=[{"role": "user", "content": "prompt"}],
            max_tokens=64,
        )
        assert result.text == "hello"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_local_model_cost_is_always_zero(self) -> None:
        provider = VLLMChatCompletionsProvider(
            model="qwen3.5-9b-not-in-pricing-table", base_url="http://localhost:8000/v1"
        )
        fake_response = MagicMock()
        fake_response.model_dump.return_value = {
            "usage": {"prompt_tokens": 100000, "completion_tokens": 50000, "total_tokens": 150000}
        }
        fake_response.choices = [MagicMock(message=MagicMock(content="x"))]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(provider, "client", return_value=fake_client):
            result = provider.responses_text("prompt")

        assert result.usage.estimated_cost_usd == 0.0

    def test_is_configured_without_real_api_key(self) -> None:
        provider = VLLMChatCompletionsProvider(
            model="qwen3-8b", base_url="http://localhost:8000/v1"
        )
        assert provider.is_configured() is True
        provider.require_configured()  # must not raise

    def test_call_counting_inherited_from_counting_provider(self) -> None:
        provider = VLLMChatCompletionsProvider(
            model="qwen3-8b", base_url="http://localhost:8000/v1"
        )
        fake_response = MagicMock()
        fake_response.model_dump.return_value = {"usage": {}}
        fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(provider, "client", return_value=fake_client):
            provider.responses_text("p1")
            provider.responses_text("p2")

        assert provider.drain_call_count() == 2
