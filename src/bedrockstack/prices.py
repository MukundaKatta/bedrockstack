"""Hardcoded Bedrock pricing for Anthropic Claude families (May 2026).

Prices are dollars per 1,000 tokens. Cache prices follow Anthropic's standard
multipliers: cache writes 1.25x base input, cache reads 0.1x base input.

These prices change. Pass an updated `prices` dict to `Ledger(prices=...)`
when AWS rebases. Pull request welcome when the table goes stale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_1k: float
    output_per_1k: float
    cache_read_per_1k: float
    cache_creation_per_1k: float


def _claude_price(input_p: float, output_p: float) -> ModelPrice:
    return ModelPrice(
        input_per_1k=input_p,
        output_per_1k=output_p,
        cache_read_per_1k=input_p * 0.10,
        cache_creation_per_1k=input_p * 1.25,
    )


# Source: AWS Bedrock pricing page, anthropic-on-bedrock list, May 2026.
# Verify before invoicing — prices change.
BEDROCK_PRICES: dict[str, ModelPrice] = {
    "anthropic.claude-opus-4-7-v1:0": _claude_price(0.015, 0.075),
    "anthropic.claude-opus-4-6-v1:0": _claude_price(0.015, 0.075),
    "anthropic.claude-opus-4-1-v1:0": _claude_price(0.015, 0.075),
    "anthropic.claude-sonnet-4-6-v1:0": _claude_price(0.003, 0.015),
    "anthropic.claude-sonnet-4-5-v1:0": _claude_price(0.003, 0.015),
    "anthropic.claude-haiku-4-5-v1:0": _claude_price(0.001, 0.005),
    # Older Claude 3 family — still common in long-running deployments.
    "anthropic.claude-3-5-sonnet-20241022-v2:0": _claude_price(0.003, 0.015),
    "anthropic.claude-3-5-haiku-20241022-v1:0": _claude_price(0.001, 0.005),
    "anthropic.claude-3-opus-20240229-v1:0": _claude_price(0.015, 0.075),
    "anthropic.claude-3-sonnet-20240229-v1:0": _claude_price(0.003, 0.015),
    "anthropic.claude-3-haiku-20240307-v1:0": _claude_price(0.00025, 0.00125),
}
