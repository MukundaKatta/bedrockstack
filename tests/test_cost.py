"""Tests for the Bedrock cost ledger."""

import pytest

from bedrockstack import Ledger, ModelPrice, Usage


def test_dollars_for_uncached_call():
    led = Ledger()
    led.record(
        model="anthropic.claude-sonnet-4-5-v1:0",
        usage=Usage(input_tokens=10_000, output_tokens=2_000),
    )
    # 10k input @ $0.003/1k + 2k output @ $0.015/1k = 0.030 + 0.030 = 0.06
    assert led.dollars() == pytest.approx(0.06, abs=1e-6)


def test_dollars_for_cached_call_is_cheaper():
    led = Ledger()
    led.record(
        model="anthropic.claude-sonnet-4-5-v1:0",
        usage=Usage(input_tokens=0, output_tokens=2_000,
                    cache_read_input_tokens=10_000),
    )
    # Cache reads at 0.1x base input: 10k * 0.1 * $0.003 / 1k = 0.003 + output 0.030
    assert led.dollars() == pytest.approx(0.033, abs=1e-6)


def test_cache_creation_charged_at_125_percent():
    led = Ledger()
    led.record(
        model="anthropic.claude-sonnet-4-5-v1:0",
        usage=Usage(cache_creation_input_tokens=10_000),
    )
    assert led.dollars() == pytest.approx(0.0375, abs=1e-6)


def test_inference_profile_prefix_is_normalized():
    led = Ledger()
    led.record(
        model="us.anthropic.claude-sonnet-4-5-v1:0",
        usage=Usage(input_tokens=1_000),
    )
    assert led.dollars() == pytest.approx(0.003, abs=1e-6)


def test_litellm_prefix_is_normalized():
    led = Ledger()
    led.record(
        model="bedrock/anthropic.claude-haiku-4-5-v1:0",
        usage=Usage(input_tokens=1_000, output_tokens=1_000),
    )
    assert led.dollars() == pytest.approx(0.001 + 0.005, abs=1e-6)


def test_unknown_model_raises_with_clear_message():
    led = Ledger()
    with pytest.raises(KeyError) as exc:
        led.record(model="anthropic.notamodel-v9:0", usage=Usage(input_tokens=1))
    assert "no price for model" in str(exc.value)


def test_caller_can_supply_custom_prices():
    led = Ledger(prices={
        "custom.weird-v1": ModelPrice(0.005, 0.025, 0.0005, 0.00625)
    })
    led.record(model="custom.weird-v1", usage=Usage(input_tokens=1_000))
    assert led.dollars() == pytest.approx(0.005, abs=1e-6)


def test_scope_tagging_and_filtering():
    led = Ledger()
    with led.scope(user_id="alice", feature="rag"):
        led.record("anthropic.claude-haiku-4-5-v1:0", Usage(input_tokens=1_000))
    with led.scope(user_id="bob", feature="rag"):
        led.record("anthropic.claude-haiku-4-5-v1:0", Usage(input_tokens=2_000))
    led.record("anthropic.claude-haiku-4-5-v1:0", Usage(input_tokens=500))  # untagged

    assert led.dollars(user_id="alice") == pytest.approx(0.001)
    assert led.dollars(user_id="bob") == pytest.approx(0.002)
    assert led.dollars(feature="rag") == pytest.approx(0.003)


def test_totals_grouped():
    led = Ledger()
    led.record("anthropic.claude-sonnet-4-5-v1:0", Usage(input_tokens=1_000))
    led.record("anthropic.claude-haiku-4-5-v1:0", Usage(input_tokens=1_000))
    totals = led.totals(group_by="model")
    assert totals["anthropic.claude-sonnet-4-5-v1:0"] == pytest.approx(0.003)
    assert totals["anthropic.claude-haiku-4-5-v1:0"] == pytest.approx(0.001)


def test_records_returns_immutable_copy():
    led = Ledger()
    led.record("anthropic.claude-haiku-4-5-v1:0", Usage(input_tokens=1_000))
    snap = led.records()
    snap.append("not a record")  # type: ignore[arg-type]
    assert len(led.records()) == 1
