from __future__ import annotations

import time

from openai import OpenAI

from ant.domain import TokenUsage
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.retry_policy import call_with_transient_retry
from ant.providers.openai_provider import OpenAISettings, ResponseResult, _with_latency_and_cost


class VLLMChatCompletionsProvider(CountingOpenAIProvider):
    """`CountingOpenAIProvider` (evaluation-suite subclass of frozen core
    `OpenAIProvider`) pointed at a local vLLM OpenAI-compatible server
    instead of the real OpenAI API -- the worker-model bake-off's own
    provider (see `AntAgent`'s `worker_model`/`worker_base_url`
    parameters and `LocalCoordinator`'s `worker_reasoner` seam).

    `OpenAIProvider.responses_text()` is the sole physical-call chokepoint
    every provider method funnels through (see `CountingOpenAIProvider`'s
    own docstring); the frozen implementation calls `client.responses.
    create(...)` -- the OpenAI **Responses** API. vLLM's OpenAI-compatible
    server does not implement `/v1/responses` as of current vLLM
    versions, only `/v1/chat/completions`. Every other method on
    `WorkerReasoner` (`select_lookups`, `plan_worker_actions`, prompt
    construction, `responses_json`'s JSON-repair reprompt) is inherited
    unchanged -- they all call `self.responses_text(...)`, which is the
    only thing that needs a different transport. Inherits from
    `CountingOpenAIProvider` rather than `OpenAIProvider` directly so this
    pilot's own call-counting/retry-policy accounting
    (`drain_call_count`/`drain_retry_log`) is identical to every other
    baseline's, not a second, divergent implementation.

    `estimate_cost_usd` (providers/pricing.py) already returns 0.0 for any
    model name it doesn't recognize -- a local vLLM model name is never
    added to that pricing table, so cost is correctly always $0 for calls
    made through this provider, with no special-casing needed here.

    Context cap: deliberately NOT enforced client-side by truncating the
    prompt here. The worker-model bake-off's whole point is testing
    whether a small model can do useful LOCAL, tool-scoped reasoning over
    what the substrate's own search/evidence-selection logic already
    narrowed things down to -- not whether it can swallow a huge blob of
    raw repo text just because the model's own native context window
    (e.g. Qwen3.5's 262K) happens to fit it. Capping context is done
    authoritatively at the vLLM server's own `--max-model-len` launch
    flag, which makes an oversized request fail loudly at the server
    rather than being silently truncated by a second, client-side
    heuristic that could disagree with the server's own tokenizer.
    `max_context_tokens` here is recorded for pilot-harness metadata only
    (so a result row can report what cap the run was launched under).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_context_tokens: int | None = None,
    ) -> None:
        # Deliberately not calling CountingOpenAIProvider.__init__ /
        # OpenAIProvider.__init__ -- both read OPENAI_API_KEY/OPENAI_ORG_ID/
        # OPENAI_PROJECT_ID from .env/os.environ and require a real key via
        # require_configured(); a local vLLM server needs none of that.
        # Sets the same attributes those __init__s would, so every
        # inherited method that reads self.model/self.settings/self._usage
        # still works unmodified.
        self.settings = OpenAISettings(api_key=api_key, model=model, reasoning_effort=None)
        self.model = model
        self.reasoning_effort = None
        self._usage = TokenUsage()
        self._physical_call_count = 0
        self._retry_log: list[dict] = []
        self.base_url = base_url
        self.max_context_tokens = max_context_tokens

    def is_configured(self) -> bool:
        return True

    def require_configured(self) -> None:
        return None

    def client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.settings.api_key, max_retries=0)

    def _chat_completions_call(self, prompt: str, max_output_tokens: int) -> ResponseResult:
        start = time.perf_counter()
        response = self.client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            # Qwen3's chat template defaults to thinking mode (a <think>...
            # </think> preamble before any real content), which the ANTMAN
            # worker prompts' tight budgets (select_lookups=256,
            # plan_worker_actions=768 max_output_tokens) never survive --
            # confirmed live: even 256 tokens against a trivial "reply with
            # this exact JSON" prompt burned the ENTIRE budget on thinking
            # and never reached the answer (finish_reason="length", content
            # is pure <think> text, no JSON at all). responses_json's own
            # repair pass then re-asks the SAME model, which thinks again
            # and fails the same way, degrading every worker call to "{}" --
            # not a measurement of whether an 8B model can do the delegated
            # reasoning, just a measurement of its thinking preamble length.
            # vLLM honors Qwen's own chat-template kwarg for this (confirmed
            # live: the identical prompt above returns a clean, correct
            # answer in 6 tokens with this set).
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.model_dump()
        text = response.choices[0].message.content or ""
        usage_obj = raw.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_obj.get("prompt_tokens") or 0),
            output_tokens=int(usage_obj.get("completion_tokens") or 0),
            total_tokens=int(usage_obj.get("total_tokens") or 0),
        )
        return self._record_result(
            _with_latency_and_cost(
                ResponseResult(text=text, usage=usage, raw=raw), self.model, start
            )
        )

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        result, outcome = call_with_transient_retry(
            lambda: self._chat_completions_call(prompt, max_output_tokens)
        )
        self._physical_call_count += outcome.attempt_count
        self._retry_log.append(outcome.to_dict())
        if result is None:
            assert outcome.final_error is not None
            raise outcome.final_error
        return result
