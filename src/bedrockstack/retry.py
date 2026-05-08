"""Bedrock-aware retry policy.

Captures the retry semantics that every team running Bedrock at scale ends up
re-implementing on top of `boto3.client('bedrock-runtime')`:

  - ThrottlingException — backoff and retry, the most common failure
  - ModelNotReadyException — retry after a longer pause (model warming)
  - ServiceUnavailableException, InternalServerException, ModelTimeoutException
  - ModelStreamErrorException — retryable for non-streaming calls
  - ValidationException, AccessDeniedException, ResourceNotFoundException — never retry
  - read timeouts (botocore) — retry

This module does NOT depend on boto3 at runtime. It exposes pure-Python
predicates and a backoff iterator. The caller wraps their own client.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, TypeVar

T = TypeVar("T")


_BEDROCK_RETRYABLE = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ModelStreamErrorException",
    "ServiceQuotaExceededException",
})

_BEDROCK_NEVER_RETRY = frozenset({
    "ValidationException",
    "AccessDeniedException",
    "ResourceNotFoundException",
    "ModelErrorException",  # bad model output, retrying won't help
    "InvalidRequestException",
})


@dataclass(frozen=True)
class RetryPolicy:
    """Bedrock retry policy with exponential backoff + decorrelated jitter.

    `base_delay` and `max_delay` bracket the backoff in seconds. `multiplier`
    is the exponential growth factor. `max_attempts` is the total attempt
    count including the first one.
    """

    max_attempts: int = 6
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    model_not_ready_initial_delay: float = 10.0

    def is_retryable(self, exc: BaseException) -> bool:
        return _is_retryable(exc)

    def delays(self) -> Iterable[float]:
        """Yield (max_attempts - 1) backoff durations.

        Decorrelated jitter formula (AWS Architecture Blog, "Exponential
        Backoff and Jitter"):
            sleep = min(max, random_between(base, prev_sleep * 3))
        """
        prev = self.base_delay
        for _ in range(self.max_attempts - 1):
            high = min(self.max_delay, prev * self.multiplier * (1 + random.random()))
            sleep = max(self.base_delay, random.uniform(self.base_delay, max(self.base_delay, high)))
            yield sleep
            prev = sleep

    def call(self, fn: Callable[[], T], on_retry: Callable[[BaseException, int, float], None] | None = None) -> T:
        """Run `fn`, retrying on retryable Bedrock errors per the policy."""
        attempt = 0
        delays = self.delays()
        last_exc: BaseException | None = None
        while attempt < self.max_attempts:
            try:
                return fn()
            except BaseException as exc:
                last_exc = exc
                if not self.is_retryable(exc):
                    raise
                attempt += 1
                if attempt >= self.max_attempts:
                    raise
                try:
                    delay = next(delays)
                except StopIteration:
                    raise
                if _is_model_not_ready(exc):
                    delay = max(delay, self.model_not_ready_initial_delay)
                if on_retry is not None:
                    on_retry(exc, attempt, delay)
                time.sleep(delay)
        # Defensive — loop above always either returns or raises.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("retry policy exhausted with no exception captured")


def bedrock_default() -> RetryPolicy:
    """The default policy most teams want: 6 attempts, 0.5s..30s exponential."""
    return RetryPolicy()


# ---------- internals ----------


def _exc_code(exc: BaseException) -> str | None:
    """Extract the AWS error code from a botocore ClientError without
    importing botocore. Falls back to the exception class name."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("Error")
        if isinstance(err, dict):
            code = err.get("Code")
            if isinstance(code, str):
                return code
    return type(exc).__name__


def _is_retryable(exc: BaseException) -> bool:
    code = _exc_code(exc)
    if code in _BEDROCK_NEVER_RETRY:
        return False
    if code in _BEDROCK_RETRYABLE:
        return True
    # Connection / timeout errors from botocore + urllib3 + httpx — match by
    # class name to avoid the dependency.
    cls = type(exc).__name__
    if cls in {"ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError",
               "ConnectionError", "ConnectionResetError", "Timeout"}:
        return True
    return False


def _is_model_not_ready(exc: BaseException) -> bool:
    return _exc_code(exc) == "ModelNotReadyException"
