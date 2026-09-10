from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError

import openai

#: HTTP status codes this policy treats as transient provider/network
#: infrastructure failures, never as a judgment about the response's
#: content.
TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503})

#: Retries AFTER the initial attempt (so 3 attempts total by default).
DEFAULT_MAX_RETRIES = 2
DEFAULT_BASE_BACKOFF_SECONDS = 1.0


def _status_code_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    if isinstance(exc, HTTPError):
        return exc.code
    return None


def is_transient_provider_error(exc: BaseException) -> bool:
    """True ONLY for failures attributable to transient provider/network
    infrastructure -- HTTP 429/500/502/503, a provider-side timeout, or a
    connection-level transport failure (reset, refused, DNS blip, socket
    timeout). This function is only ever reached from inside the single
    physical-call choke point (`responses_text`'s override) when that
    call itself raised -- a low-scoring answer, an agent-declared finish,
    budget exhaustion, a malformed-but-handled model decision, a tool
    returning no evidence, or a benchmark scoring outcome never raises an
    exception through this boundary in the first place, so none of those
    are ever classified here at all, transient or not. The classification
    below never inspects a response's own content -- only its transport
    status/type.
    """
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in TRANSIENT_HTTP_STATUS_CODES
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, HTTPError):
        return exc.code in TRANSIENT_HTTP_STATUS_CODES
    if isinstance(exc, URLError):
        # Any URLError that is not the more specific HTTPError above is a
        # transport-level failure (DNS, connection refused, socket
        # timeout, ...) -- never a well-formed error response.
        return True
    if isinstance(exc, TimeoutError):
        # Covers both socket.timeout (an alias of TimeoutError since
        # Python 3.10) and openai_provider.py's own
        # _call_with_hard_timeout's hard-deadline TimeoutError.
        return True
    if isinstance(exc, (ConnectionResetError, ConnectionRefusedError, ConnectionAbortedError)):
        return True
    return False


@dataclass
class RetryAttempt:
    attempt_number: int
    succeeded: bool
    error_type: str | None
    status_code: int | None
    wall_clock_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class RetryOutcome:
    attempts: list[RetryAttempt] = field(default_factory=list)
    final_error: BaseException | None = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def retry_count(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def succeeded(self) -> bool:
        return self.final_error is None and bool(self.attempts) and self.attempts[-1].succeeded

    @property
    def total_wall_clock_seconds(self) -> float:
        return round(sum(a.wall_clock_seconds for a in self.attempts), 3)

    def to_dict(self) -> dict:
        return {
            "attempt_count": self.attempt_count,
            "retry_count": self.retry_count,
            "succeeded": self.succeeded,
            "total_wall_clock_seconds": self.total_wall_clock_seconds,
            "final_error_type": type(self.final_error).__name__ if self.final_error else None,
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "succeeded": a.succeeded,
                    "error_type": a.error_type,
                    "status_code": a.status_code,
                    "wall_clock_seconds": a.wall_clock_seconds,
                    "input_tokens": a.input_tokens,
                    "output_tokens": a.output_tokens,
                }
                for a in self.attempts
            ],
        }


def call_with_transient_retry[T](
    fn: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T | None, RetryOutcome]:
    """Calls fn() up to `1 + max_retries` times total (default: 1 initial
    attempt + 2 retries = 3 attempts). Retries ONLY when the raised
    exception is `is_transient_provider_error` -- any other exception
    (a method/logic failure, a deterministic parsing failure, ...)
    propagates immediately after its first, only attempt. This function
    never inspects fn()'s own return value -- a retry decision is a pure
    function of the exception's transport classification, never of
    answer quality or a gold/reference answer, both of which this
    function has no access to in the first place.

    Backoff is bounded exponential: base_backoff_seconds * 2**(attempt
    index), applied only between attempts (never after the final one).

    Returns (result, outcome). On success, result is fn()'s return value
    and outcome.final_error is None. On exhaustion (either the retry
    budget ran out, or the failure was not transient), result is None and
    outcome.final_error is the last raised exception -- callers must
    re-raise it themselves; this function never swallows a failure.
    """
    outcome = RetryOutcome()
    for attempt_number in range(1, max_retries + 2):
        started = time.perf_counter()
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 - classified below, always re-raised by the caller
            elapsed = time.perf_counter() - started
            outcome.attempts.append(
                RetryAttempt(
                    attempt_number=attempt_number,
                    succeeded=False,
                    error_type=type(exc).__name__,
                    status_code=_status_code_of(exc),
                    wall_clock_seconds=round(elapsed, 3),
                )
            )
            is_last_attempt = attempt_number == max_retries + 1
            if not is_transient_provider_error(exc) or is_last_attempt:
                outcome.final_error = exc
                return None, outcome
            sleep(base_backoff_seconds * (2 ** (attempt_number - 1)))
            continue
        else:
            elapsed = time.perf_counter() - started
            outcome.attempts.append(
                RetryAttempt(
                    attempt_number=attempt_number,
                    succeeded=True,
                    error_type=None,
                    status_code=None,
                    wall_clock_seconds=round(elapsed, 3),
                )
            )
            return result, outcome
    raise AssertionError("unreachable: loop always returns on its final iteration")
