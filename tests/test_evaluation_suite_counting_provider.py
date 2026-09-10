from __future__ import annotations

from dataclasses import replace

from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.providers import OpenAIProvider


class _FakeResponse:
    def __init__(self, output_text: str, input_tokens: int, output_tokens: int) -> None:
        self.output_text = output_text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def model_dump(self) -> dict:
        return {
            "usage": {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }
        }


class _FakeResponsesEndpoint:
    def __init__(self, replies: list[_FakeResponse]) -> None:
        self._replies = list(replies)
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        if not self._replies:
            raise AssertionError("no more fake replies configured")
        return self._replies.pop(0)


class _FakeClient:
    def __init__(self, replies: list[_FakeResponse]) -> None:
        self.responses = _FakeResponsesEndpoint(replies)


def _provider_with_fake_client(monkeypatch, replies: list[_FakeResponse]) -> CountingOpenAIProvider:
    provider = CountingOpenAIProvider(model="gpt-4.1")
    fake_client = _FakeClient(replies)
    # Real responses_text() branches on organization/project: when BOTH
    # are configured (as they are via this machine's own .env, outside
    # test control) it takes the _responses_create_raw path instead of
    # client().responses.create -- which would silently bypass a
    # client()-only monkeypatch and make a REAL, paid API call. Force the
    # settings this instance holds to have neither set, so the branch this
    # test actually exercises is deterministic regardless of the real
    # environment's own .env/os.environ configuration.
    provider.settings = replace(provider.settings, organization=None, project=None)
    # Real responses_text()/_record_result bodies run unmodified here --
    # only the network call itself is replaced -- so this test exercises
    # this suite's actual usage-accounting logic, not a reimplementation
    # of it.
    monkeypatch.setattr(CountingOpenAIProvider, "client", lambda self: fake_client)
    return provider


def test_no_repair_path_counts_exactly_one_physical_call(monkeypatch) -> None:
    provider = _provider_with_fake_client(
        monkeypatch, [_FakeResponse('{"tool": "search", "query": "x"}', 100, 20)]
    )
    result = provider.responses_json("some prompt")
    assert result.text == '{"tool": "search", "query": "x"}'
    assert provider.drain_call_count() == 1
    usage = provider.drain_usage()
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.total_tokens == 120


def test_repair_path_counts_exactly_two_physical_calls_and_sums_usage(monkeypatch) -> None:
    provider = _provider_with_fake_client(
        monkeypatch,
        [
            _FakeResponse("not json at all", 80, 15),
            _FakeResponse('{"tool": "navigate", "query": "Foo"}', 60, 10),
        ],
    )
    result = provider.responses_json("some prompt")
    assert result.text == '{"tool": "navigate", "query": "Foo"}'
    # Physical call count: exactly 2 -- the malformed first attempt AND
    # the repair attempt, never conflated with "1 decision" the way a
    # caller's own "+1 per loop iteration" counter would undercount it.
    assert provider.drain_call_count() == 2
    usage = provider.drain_usage()
    # Tokens/cost are the SUM of both physical calls, not just the last
    # one and not double-counted.
    assert usage.input_tokens == 80 + 60
    assert usage.output_tokens == 15 + 10
    assert usage.total_tokens == 95 + 70


def test_repair_path_that_still_fails_still_counts_two_physical_calls(monkeypatch) -> None:
    """Even when the repair attempt ALSO comes back unparseable (responses_json
    degrades to text="{}" rather than raising), both physical calls that
    actually reached the model must still be counted -- the call count is
    about what happened on the wire, not about whether the caller's own
    parsing ultimately succeeded."""
    provider = _provider_with_fake_client(
        monkeypatch,
        [
            _FakeResponse("still not json", 50, 5),
            _FakeResponse("still not json after repair either", 40, 5),
        ],
    )
    result = provider.responses_json("some prompt")
    assert result.text == "{}"
    assert provider.drain_call_count() == 2
    usage = provider.drain_usage()
    assert usage.input_tokens == 50 + 40
    assert usage.output_tokens == 5 + 5


def test_drain_call_count_resets_to_zero(monkeypatch) -> None:
    provider = _provider_with_fake_client(
        monkeypatch, [_FakeResponse('{"a": 1}', 10, 2), _FakeResponse('{"b": 2}', 10, 2)]
    )
    provider.responses_text("first")
    assert provider.drain_call_count() == 1
    provider.responses_text("second")
    assert provider.drain_call_count() == 1
    assert provider.drain_call_count() == 0


def test_matched_react_agent_usage_llm_calls_reflects_a_real_repair_pass(
    monkeypatch, tmp_path
) -> None:
    """End-to-end: not just that CountingOpenAIProvider counts correctly
    in isolation, but that MatchedReActAgent.run() actually wires
    provider.drain_call_count() into the returned AgentResult.usage.
    llm_calls -- the previous manual "+1 per loop iteration" counter would
    have reported 1 here; the correct physical count is 2 (the malformed
    first attempt, then the repair call that produced a valid `finish`)."""
    from ant.agents.matched_react import MatchedReActAgent

    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    fake_client = _FakeClient(
        [
            _FakeResponse("not valid json, rambling instead", 90, 20),
            _FakeResponse('{"thought": "done", "finish": "the answer"}', 70, 15),
        ]
    )

    def fake_init(self, *args, **kwargs):
        # Bypass CountingOpenAIProvider.__init__ itself (monkeypatching it
        # below means referring to it here would re-invoke the patched
        # version and recurse) -- call the grandparent's __init__ directly
        # and replicate the lines CountingOpenAIProvider.__init__ adds.
        OpenAIProvider.__init__(self, model="gpt-4.1")
        self._physical_call_count = 0
        self._retry_log = []
        self.settings = replace(self.settings, organization=None, project=None)

    monkeypatch.setattr(CountingOpenAIProvider, "__init__", fake_init)
    monkeypatch.setattr(CountingOpenAIProvider, "client", lambda self: fake_client)

    agent = MatchedReActAgent(tool_call_budget=1)
    from ant.benchmarks.base import TaskExample

    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    result = agent.run(example, tmp_path)

    assert result.final_answer == "the answer"
    assert result.usage.llm_calls == 2
    # Tokens are the sum of both physical calls, not just the last one.
    assert result.usage.input_tokens == 90 + 70
    assert result.usage.output_tokens == 20 + 15


def test_synthesize_counts_as_one_physical_call_with_no_repair_logic(monkeypatch) -> None:
    """synthesize() calls responses_text() directly (free text, no JSON
    parsing/repair) -- confirms the counting override at responses_text()
    also correctly covers callers that never go through responses_json()."""
    provider = _provider_with_fake_client(
        monkeypatch, [_FakeResponse("a synthesized answer", 200, 50)]
    )
    answer = provider.synthesize(question="Where is X?", evidence=[])
    assert answer == "a synthesized answer"
    assert provider.drain_call_count() == 1
    usage = provider.drain_usage()
    assert usage.input_tokens == 200
    assert usage.output_tokens == 50
