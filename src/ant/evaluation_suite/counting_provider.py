from __future__ import annotations

from typing import Any

from openai import OpenAI

from ant.evaluation_suite.retry_policy import call_with_transient_retry
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import ResponseResult


class CountingOpenAIProvider(OpenAIProvider):
    """Wraps `OpenAIProvider` (frozen ANT core, unmodified -- this is a
    plain subclass, not a monkeypatch) to accurately count PHYSICAL
    model/API invocations and to apply this evaluation suite's frozen
    transient-provider retry policy (see `retry_policy.py`). `llm_calls`
    in this evaluation suite's usage reporting should equal exactly the
    number of times a real HTTP call reached the model -- not the number
    of times a caller's own loop asked for "a decision", and INCLUDING
    every retried attempt (a retried attempt is still a real physical
    call that reached the network, whether or not it ultimately
    succeeded).

    Read directly from openai_provider.py: `responses_text()` is the SOLE
    choke point every physical call goes through -- `self.client().
    responses.create(...)` and `self._responses_create_raw(...)` are
    invoked from nowhere else in that file. `responses_json()`'s own
    internal JSON-repair pass calls `self.responses_text(...)` a second
    time when the first response isn't valid JSON; `synthesize()` and
    every other reasoner method (`plan_round`, `consolidate_graph`, ...)
    also funnel through `responses_text()`. Overriding only this one
    method therefore counts every physical call made by ANY provider
    method, AND retries every one of them uniformly -- including calls
    this evaluation suite's own baseline agents never call directly, such
    as every call LocalCoordinator/AutonomousWorker make through a shared
    provider instance passed in as `reasoner=`/`synthesizer=` -- without
    needing to read or duplicate a single line of prompt-building logic.

    Used by every one of this suite's five baseline agents (Direct,
    Retrieval, Matched ReAct, Matched ReAct+RepoGraph, ANT) in place of
    constructing `OpenAIProvider` directly, so `llm_calls` and the retry
    policy are both accurate and comparable across all five, not just the
    ones whose call-counting loop happened to be written carefully.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._physical_call_count = 0
        self._retry_log: list[dict] = []

    def client(self) -> OpenAI:
        """Disables the SDK's own default internal retry behavior
        (`max_retries=2` otherwise) so this evaluation suite's own
        explicit, observable `call_with_transient_retry` policy below is
        the SOLE source of retry behavior for every provider call.
        Without this override, a transient error the SDK already retried
        and silently recovered from internally would never reach
        `responses_text()`'s own except-block at all -- making
        attempt_count/retry_count accounting for this run undercount the
        real number of physical HTTP requests made."""
        self.require_configured()
        return OpenAI(
            api_key=self.settings.api_key,
            organization=self.settings.organization,
            project=self.settings.project,
            max_retries=0,
        )

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        result, outcome = call_with_transient_retry(
            lambda: super(CountingOpenAIProvider, self).responses_text(
                prompt, max_output_tokens=max_output_tokens
            )
        )
        self._physical_call_count += outcome.attempt_count
        self._retry_log.append(outcome.to_dict())
        if result is None:
            assert outcome.final_error is not None
            raise outcome.final_error
        return result

    def drain_call_count(self) -> int:
        """Read-and-reset, mirroring `drain_usage()`'s own convention."""
        count = self._physical_call_count
        self._physical_call_count = 0
        return count

    def drain_retry_log(self) -> list[dict]:
        """Read-and-reset the per-physical-call retry record (one entry
        per `responses_text()` invocation, regardless of whether it
        needed any retries) -- mirrors `drain_usage()`/`drain_call_count()`'s
        own read-and-reset convention. Each entry is
        `RetryOutcome.to_dict()`: attempt_count, retry_count, succeeded,
        total_wall_clock_seconds, final_error_type, and a per-attempt
        breakdown (error_type/status_code/wall_clock_seconds per
        attempt -- token/cost usage per FAILED attempt is included only
        when the provider itself returned it; never fabricated)."""
        log = self._retry_log
        self._retry_log = []
        return log
