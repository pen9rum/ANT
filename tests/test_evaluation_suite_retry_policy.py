from __future__ import annotations

from urllib.error import HTTPError, URLError

# openai's own exception classes type-check their `response`/`request`
# constructor arguments against ITS vendored httpx2, not the top-level
# `httpx` package -- importing httpx2 directly (it is a real, separately
# installed package openai depends on, not a private alias) keeps these
# synthetic test exceptions exactly typed, not just structurally similar.
import httpx2
import openai
import pytest

from ant.domain import TokenUsage
from ant.evaluation_suite import retry_policy
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.retry_policy import (
    TRANSIENT_HTTP_STATUS_CODES,
    call_with_transient_retry,
    is_transient_provider_error,
)
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import OpenAISettings, ResponseResult

_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/responses")


def _status_error(status_code: int) -> openai.APIStatusError:
    return openai.APIStatusError(
        f"status {status_code}", response=httpx2.Response(status_code, request=_REQUEST), body=None
    )


def _http_error(code: int) -> HTTPError:
    return HTTPError("https://api.openai.com/v1/responses", code, "boom", {}, None)  # type: ignore[arg-type]


def _fake_result() -> ResponseResult:
    return ResponseResult(text="ok", usage=TokenUsage(), raw={})


# ---------------------------------------------------------------------------
# is_transient_provider_error classification
# ---------------------------------------------------------------------------


def test_classifies_http_500_as_transient() -> None:
    assert is_transient_provider_error(_status_error(500)) is True


def test_classifies_http_429_as_transient() -> None:
    assert is_transient_provider_error(_status_error(429)) is True


def test_classifies_http_502_and_503_as_transient() -> None:
    assert is_transient_provider_error(_status_error(502)) is True
    assert is_transient_provider_error(_status_error(503)) is True


def test_classifies_non_transient_http_status_as_not_transient() -> None:
    # 400 (bad request) and 401 (auth) are real, permanent problems --
    # retrying them wastes budget and never succeeds.
    assert is_transient_provider_error(_status_error(400)) is False
    assert is_transient_provider_error(_status_error(401)) is False


def test_classifies_api_timeout_as_transient() -> None:
    assert is_transient_provider_error(openai.APITimeoutError(request=_REQUEST)) is True


def test_classifies_api_connection_error_as_transient() -> None:
    assert (
        is_transient_provider_error(openai.APIConnectionError(message="reset", request=_REQUEST))
        is True
    )


def test_classifies_hard_deadline_timeout_error_as_transient() -> None:
    # openai_provider.py's own _call_with_hard_timeout raises a plain
    # TimeoutError on a double stall -- socket.timeout is also an alias
    # of TimeoutError since Python 3.10, so covered by the same check.
    assert is_transient_provider_error(TimeoutError("hard deadline")) is True


def test_classifies_urllib_http_error_by_status_code() -> None:
    assert is_transient_provider_error(_http_error(500)) is True
    assert is_transient_provider_error(_http_error(400)) is False


def test_classifies_urllib_url_error_as_transient() -> None:
    # Connection refused / DNS failure / non-HTTP transport failure.
    assert is_transient_provider_error(URLError("connection refused")) is True


def test_classifies_connection_reset_as_transient() -> None:
    assert is_transient_provider_error(ConnectionResetError("reset by peer")) is True


def test_classifies_unrelated_exception_as_not_transient() -> None:
    assert is_transient_provider_error(ValueError("malformed decision")) is False
    assert is_transient_provider_error(KeyError("missing field")) is False
    assert is_transient_provider_error(RuntimeError("method-level logic failure")) is False


def test_transient_status_code_set_is_exactly_429_500_502_503() -> None:
    assert TRANSIENT_HTTP_STATUS_CODES == frozenset({429, 500, 502, 503})


# ---------------------------------------------------------------------------
# call_with_transient_retry loop behavior
# ---------------------------------------------------------------------------


def test_http_500_triggers_retry_and_eventually_succeeds() -> None:
    calls = [_status_error(500), _status_error(500), "ok"]

    def fn():
        outcome = calls.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    sleeps: list[float] = []
    result, outcome = call_with_transient_retry(fn, sleep=sleeps.append)

    assert result == "ok"
    assert outcome.succeeded is True
    assert outcome.attempt_count == 3
    assert outcome.retry_count == 2
    assert outcome.final_error is None
    assert [a.error_type for a in outcome.attempts] == ["APIStatusError", "APIStatusError", None]
    assert [a.status_code for a in outcome.attempts] == [500, 500, None]


def test_http_429_triggers_retry_and_eventually_succeeds() -> None:
    calls = [_status_error(429), "ok"]

    def fn():
        outcome = calls.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    result, outcome = call_with_transient_retry(fn, sleep=lambda s: None)

    assert result == "ok"
    assert outcome.attempt_count == 2
    assert outcome.retry_count == 1


def test_timeout_triggers_retry_and_eventually_succeeds() -> None:
    calls = [openai.APITimeoutError(request=_REQUEST), "ok"]

    def fn():
        outcome = calls.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    result, outcome = call_with_transient_retry(fn, sleep=lambda s: None)

    assert result == "ok"
    assert outcome.attempt_count == 2
    assert outcome.attempts[0].error_type == "APITimeoutError"


def test_successful_normal_run_does_not_retry() -> None:
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    sleeps: list[float] = []
    result, outcome = call_with_transient_retry(fn, sleep=sleeps.append)

    assert result == "ok"
    assert call_count == 1
    assert outcome.attempt_count == 1
    assert outcome.retry_count == 0
    assert sleeps == []


def test_method_logic_failure_does_not_retry() -> None:
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("malformed-but-handled model decision")

    sleeps: list[float] = []
    result, outcome = call_with_transient_retry(fn, sleep=sleeps.append)

    assert result is None
    assert call_count == 1  # never retried
    assert outcome.attempt_count == 1
    assert outcome.retry_count == 0
    assert isinstance(outcome.final_error, ValueError)
    assert sleeps == []


def test_scoring_and_budget_style_exceptions_are_never_retried() -> None:
    # These are stand-ins for the categories the spec explicitly excludes
    # from retry: a retry decision must be a pure function of the
    # exception's transport classification, never of what a caller's own
    # logic decided about an answer.
    for exc in [
        RuntimeError("budget exhausted"),
        KeyError("unexpected tool name"),
        LookupError("deterministic parsing failure"),
    ]:

        def fn(exc=exc):
            raise exc

        result, outcome = call_with_transient_retry(fn, sleep=lambda s: None)
        assert result is None
        assert outcome.attempt_count == 1


def test_retries_stop_after_configured_maximum() -> None:
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        raise _status_error(503)

    sleeps: list[float] = []
    result, outcome = call_with_transient_retry(fn, max_retries=2, sleep=sleeps.append)

    assert result is None
    assert call_count == 3  # 1 initial + 2 retries, never a 4th
    assert outcome.attempt_count == 3
    assert outcome.retry_count == 2
    assert isinstance(outcome.final_error, openai.APIStatusError)
    assert outcome.final_error.status_code == 503


def test_backoff_is_bounded_exponential_between_attempts_only() -> None:
    def fn():
        raise _status_error(500)

    sleeps: list[float] = []
    call_with_transient_retry(fn, max_retries=2, base_backoff_seconds=1.0, sleep=sleeps.append)

    # Exactly 2 sleeps (between the 3 attempts), doubling each time --
    # never a sleep after the final, un-retried attempt.
    assert sleeps == [1.0, 2.0]


def test_retry_budget_is_configurable() -> None:
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        raise _status_error(500)

    call_with_transient_retry(fn, max_retries=0, sleep=lambda s: None)
    assert call_count == 1  # no retries at all when max_retries=0


# ---------------------------------------------------------------------------
# CountingOpenAIProvider integration: retries applied at the real physical
# call boundary, attempts counted as physical calls, retry log recorded.
# ---------------------------------------------------------------------------


def _fake_settings() -> OpenAISettings:
    return OpenAISettings(api_key="sk-test", model="gpt-4.1")


def _provider(monkeypatch: pytest.MonkeyPatch) -> CountingOpenAIProvider:
    monkeypatch.setattr(OpenAIProvider, "require_configured", lambda self: None)
    monkeypatch.setattr(retry_policy.time, "sleep", lambda s: None)
    provider = CountingOpenAIProvider.__new__(CountingOpenAIProvider)
    provider.settings = _fake_settings()
    provider.model = "gpt-4.1"
    provider.reasoning_effort = None
    provider._usage = TokenUsage()
    provider._physical_call_count = 0
    provider._retry_log = []
    return provider


def test_counting_provider_retries_transient_errors_and_counts_physical_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    calls = [_status_error(500), _status_error(500), _fake_result()]

    def fake_super_responses_text(prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        outcome = calls.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def fake_responses_text(self, p, max_output_tokens=512):
        return fake_super_responses_text(p, max_output_tokens)

    monkeypatch.setattr(OpenAIProvider, "responses_text", fake_responses_text)

    result = provider.responses_text("hello")

    assert result.text == "ok"
    assert provider.drain_call_count() == 3
    log = provider.drain_retry_log()
    assert len(log) == 1
    assert log[0]["attempt_count"] == 3
    assert log[0]["retry_count"] == 2
    assert log[0]["succeeded"] is True


def test_counting_provider_does_not_retry_non_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)

    def always_fail(self, p, max_output_tokens=512):
        raise ValueError("malformed-but-handled model decision")

    monkeypatch.setattr(OpenAIProvider, "responses_text", always_fail)

    with pytest.raises(ValueError):
        provider.responses_text("hello")

    assert provider.drain_call_count() == 1
    log = provider.drain_retry_log()
    assert len(log) == 1
    assert log[0]["attempt_count"] == 1
    assert log[0]["succeeded"] is False
    assert log[0]["final_error_type"] == "ValueError"


def test_counting_provider_successful_run_needs_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch)
    monkeypatch.setattr(
        OpenAIProvider, "responses_text", lambda self, p, max_output_tokens=512: _fake_result()
    )

    result = provider.responses_text("hello")

    assert result.text == "ok"
    assert provider.drain_call_count() == 1
    log = provider.drain_retry_log()
    assert log[0]["retry_count"] == 0


def test_counting_provider_client_disables_sdk_internal_retries() -> None:
    provider = CountingOpenAIProvider.__new__(CountingOpenAIProvider)
    provider.settings = _fake_settings()
    client = provider.client()
    assert client.max_retries == 0
