from __future__ import annotations

from typing import Any

from ant.providers import OpenAIProvider
from ant.providers.openai_provider import ResponseResult


class CountingOpenAIProvider(OpenAIProvider):
    """Wraps `OpenAIProvider` (frozen ANT core, unmodified -- this is a
    plain subclass, not a monkeypatch) to accurately count PHYSICAL
    model/API invocations. `llm_calls` in this evaluation suite's usage
    reporting should equal exactly the number of times a real HTTP call
    reached the model -- not the number of times a caller's own loop
    asked for "a decision".

    Read directly from openai_provider.py: `responses_text()` is the SOLE
    choke point every physical call goes through -- `self.client().
    responses.create(...)` and `self._responses_create_raw(...)` are
    invoked from nowhere else in that file. `responses_json()`'s own
    internal JSON-repair pass calls `self.responses_text(...)` a second
    time when the first response isn't valid JSON; `synthesize()` and
    every other reasoner method (`plan_round`, `consolidate_graph`, ...)
    also funnel through `responses_text()`. Overriding only this one
    method therefore counts every physical call made by ANY provider
    method -- including calls this evaluation suite's own baseline agents
    never call directly, such as every call LocalCoordinator/
    AutonomousWorker make through a shared provider instance passed in as
    `reasoner=`/`synthesizer=` -- without needing to read or duplicate a
    single line of prompt-building logic.

    Used by every one of this suite's four baseline agents (Direct,
    Retrieval, Matched ReAct, ANT) in place of constructing `OpenAIProvider`
    directly, so `llm_calls` is accurate and comparable across all four,
    not just the ones whose call-counting loop happened to be written
    carefully.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._physical_call_count = 0

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        self._physical_call_count += 1
        return super().responses_text(prompt, max_output_tokens=max_output_tokens)

    def drain_call_count(self) -> int:
        """Read-and-reset, mirroring `drain_usage()`'s own convention."""
        count = self._physical_call_count
        self._physical_call_count = 0
        return count
