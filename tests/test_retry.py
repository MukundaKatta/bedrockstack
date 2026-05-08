"""Tests for RetryPolicy."""

import time

import pytest

from bedrockstack import RetryPolicy, bedrock_default


def _client_error(code: str) -> Exception:
    """Construct an exception that mimics botocore.exceptions.ClientError."""
    e = Exception(f"{code}: simulated")
    e.response = {"Error": {"Code": code, "Message": "simulated"}}  # type: ignore[attr-defined]
    return e


# ---------- predicates ----------


@pytest.mark.parametrize("code", [
    "ThrottlingException", "ServiceUnavailableException", "InternalServerException",
    "ModelNotReadyException", "ModelTimeoutException", "ModelStreamErrorException",
])
def test_retryable_codes(code: str):
    assert RetryPolicy().is_retryable(_client_error(code)) is True


@pytest.mark.parametrize("code", [
    "ValidationException", "AccessDeniedException", "ResourceNotFoundException",
    "InvalidRequestException",
])
def test_never_retry_codes(code: str):
    assert RetryPolicy().is_retryable(_client_error(code)) is False


def test_unrelated_exception_not_retryable():
    assert RetryPolicy().is_retryable(KeyError("foo")) is False


def test_connection_class_names_are_retryable():
    class ReadTimeoutError(Exception): pass
    class ConnectTimeoutError(Exception): pass
    assert RetryPolicy().is_retryable(ReadTimeoutError("x")) is True
    assert RetryPolicy().is_retryable(ConnectTimeoutError("x")) is True


# ---------- delays ----------


def test_delays_yield_max_attempts_minus_one():
    p = RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=1.0)
    delays = list(p.delays())
    assert len(delays) == 3
    assert all(d >= p.base_delay for d in delays)
    assert all(d <= p.max_delay for d in delays)


def test_delays_are_bounded():
    p = RetryPolicy(max_attempts=10, base_delay=0.001, max_delay=0.05)
    for d in p.delays():
        assert 0.001 <= d <= 0.05


# ---------- call() ----------


def test_call_succeeds_on_first_try(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    p = RetryPolicy(max_attempts=4, base_delay=0.0, max_delay=0.0)
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        return "ok"
    assert p.call(fn) == "ok"
    assert calls["n"] == 1


def test_call_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    p = RetryPolicy(max_attempts=4, base_delay=0.0, max_delay=0.0)
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _client_error("ThrottlingException")
        return "ok"
    assert p.call(fn) == "ok"
    assert calls["n"] == 3


def test_call_does_not_retry_validation_errors(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    p = RetryPolicy(max_attempts=4, base_delay=0.0, max_delay=0.0)
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise _client_error("ValidationException")
    with pytest.raises(Exception) as exc:
        p.call(fn)
    assert "ValidationException" in str(exc.value)
    assert calls["n"] == 1


def test_call_exhausts_attempts(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    p = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0)
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise _client_error("ThrottlingException")
    with pytest.raises(Exception):
        p.call(fn)
    assert calls["n"] == 3


def test_on_retry_callback_invoked(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    p = RetryPolicy(max_attempts=4, base_delay=0.0, max_delay=0.0)
    calls = {"n": 0}
    retries: list[int] = []
    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _client_error("ThrottlingException")
        return "ok"
    p.call(fn, on_retry=lambda exc, attempt, delay: retries.append(attempt))
    assert retries == [1, 2]


def test_model_not_ready_uses_longer_initial_delay(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda d: sleeps.append(d))
    p = RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=100.0,
                    model_not_ready_initial_delay=10.0)
    def fn():
        raise _client_error("ModelNotReadyException")
    with pytest.raises(Exception):
        p.call(fn)
    assert sleeps and sleeps[0] >= 10.0


def test_default_factory():
    p = bedrock_default()
    assert p.max_attempts == 6
    assert p.base_delay == 0.5
