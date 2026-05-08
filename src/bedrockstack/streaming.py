"""Normalize AnthropicBedrock streaming-error semantics.

Documented gap (anthropic-sdk-python issue #866): the AnthropicBedrock client
surfaces streaming errors in a different shape than the native Anthropic
client, which means error-handling code that works against one breaks against
the other.

This module normalizes the two into a single set of exception classes the
caller can rely on. Use `wrap_stream(stream)` around either client's stream
iterator to get a consistent error surface.
"""

from __future__ import annotations

from typing import Iterator


class StreamingError(Exception):
    """Base class for normalized streaming errors."""


class StreamOverloadedError(StreamingError):
    """The provider returned 529 / overloaded mid-stream."""


class StreamRateLimitError(StreamingError):
    """The provider returned 429 mid-stream."""


class StreamModelError(StreamingError):
    """The model itself errored mid-stream (bad output, content policy)."""


class StreamServerError(StreamingError):
    """5xx mid-stream that isn't an overload."""


class StreamConnectionError(StreamingError):
    """Underlying transport dropped mid-stream."""


def normalize_exception(exc: BaseException) -> StreamingError | None:
    """Map a backend exception to a normalized StreamingError, or None.

    Returns None if `exc` is not a streaming error this normalizer recognizes.
    """
    code = _exc_code(exc)
    status = _exc_status(exc)
    msg = str(exc)

    if code == "overloaded_error" or status == 529 or "overloaded" in msg.lower():
        return StreamOverloadedError(msg)
    if code == "rate_limit_error" or status == 429:
        return StreamRateLimitError(msg)
    if code in {"api_error", "internal_server_error"} or (status and 500 <= status < 600):
        return StreamServerError(msg)
    if code in {"invalid_request_error", "model_error"}:
        return StreamModelError(msg)

    cls = type(exc).__name__
    if cls in {"ReadTimeoutError", "ConnectTimeoutError", "ConnectionError",
               "EndpointConnectionError", "ConnectionResetError"}:
        return StreamConnectionError(msg)

    return None


def wrap_stream(stream: Iterator) -> Iterator:
    """Iterate a stream, re-raising backend errors as StreamingError subclasses."""
    try:
        yield from stream
    except StreamingError:
        raise
    except BaseException as exc:
        normalized = normalize_exception(exc)
        if normalized is None:
            raise
        raise normalized from exc


# ---------- internals ----------


def _exc_code(exc: BaseException) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            t = err.get("type")
            if isinstance(t, str):
                return t

    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("Error")
        if isinstance(err, dict):
            code = err.get("Code")
            if isinstance(code, str):
                return code
    return None


def _exc_status(exc: BaseException) -> int | None:
    s = getattr(exc, "status_code", None)
    if isinstance(s, int):
        return s
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        meta = response.get("ResponseMetadata")
        if isinstance(meta, dict):
            code = meta.get("HTTPStatusCode")
            if isinstance(code, int):
                return code
    return None
