"""Tests for streaming-error normalization."""

import pytest

from bedrockstack import (
    StreamConnectionError,
    StreamModelError,
    StreamOverloadedError,
    StreamRateLimitError,
    StreamServerError,
    normalize_exception,
    wrap_stream,
)


def _anthropic_style(status: int, error_type: str) -> Exception:
    e = Exception(f"anthropic {error_type}")
    e.status_code = status  # type: ignore[attr-defined]
    e.body = {"error": {"type": error_type, "message": "x"}}  # type: ignore[attr-defined]
    return e


def _botocore_style(code: str, status: int) -> Exception:
    e = Exception(f"bedrock {code}")
    e.response = {  # type: ignore[attr-defined]
        "Error": {"Code": code, "Message": "x"},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return e


def test_anthropic_overloaded_normalizes():
    n = normalize_exception(_anthropic_style(529, "overloaded_error"))
    assert isinstance(n, StreamOverloadedError)


def test_anthropic_rate_limit_normalizes():
    n = normalize_exception(_anthropic_style(429, "rate_limit_error"))
    assert isinstance(n, StreamRateLimitError)


def test_anthropic_server_error_normalizes():
    n = normalize_exception(_anthropic_style(500, "api_error"))
    assert isinstance(n, StreamServerError)


def test_anthropic_invalid_request_is_model_error():
    n = normalize_exception(_anthropic_style(400, "invalid_request_error"))
    assert isinstance(n, StreamModelError)


def test_bedrock_overload_via_message_normalizes():
    e = Exception("Service is overloaded, please retry")
    n = normalize_exception(e)
    assert isinstance(n, StreamOverloadedError)


def test_bedrock_5xx_status_normalizes():
    n = normalize_exception(_botocore_style("InternalServerError", 503))
    assert isinstance(n, StreamServerError)


def test_unrelated_exception_returns_none():
    assert normalize_exception(KeyError("nothing to see here")) is None


def test_connection_error_normalized():
    class ReadTimeoutError(Exception): pass
    n = normalize_exception(ReadTimeoutError("timed out"))
    assert isinstance(n, StreamConnectionError)


def test_wrap_stream_passes_normal_iteration():
    def gen():
        yield 1
        yield 2
        yield 3
    assert list(wrap_stream(gen())) == [1, 2, 3]


def test_wrap_stream_translates_anthropic_overload():
    def gen():
        yield 1
        raise _anthropic_style(529, "overloaded_error")
    out = []
    with pytest.raises(StreamOverloadedError):
        for x in wrap_stream(gen()):
            out.append(x)
    assert out == [1]


def test_wrap_stream_passes_through_unrecognized():
    def gen():
        yield 1
        raise ValueError("totally unrelated")
    with pytest.raises(ValueError):
        for _ in wrap_stream(gen()):
            pass


def test_wrap_stream_does_not_double_wrap():
    def gen():
        raise StreamRateLimitError("already normalized")
    with pytest.raises(StreamRateLimitError):
        list(wrap_stream(gen()))
