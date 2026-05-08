"""Bedrock cost ledger.

Tracks per-call cost using a hardcoded price table that the caller can
override. The pricing data lives in `prices.py` and is intentionally simple —
$/1k input tokens and $/1k output tokens, plus a cache_read multiplier when
prompt caching applies.

This is a data primitive, not an observability platform. Callers `record()`
each invocation and aggregate via `Ledger.dollars()` / `Ledger.totals()`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Iterator

from bedrockstack.prices import BEDROCK_PRICES, ModelPrice


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class CallRecord:
    timestamp: datetime
    model: str
    region: str
    usage: Usage
    dollars: float
    scope: dict[str, str] = field(default_factory=dict)


@dataclass
class Ledger:
    """Append-only ledger of cost-bearing calls. Thread-safe."""

    prices: dict[str, ModelPrice] = field(default_factory=lambda: dict(BEDROCK_PRICES))
    _records: list[CallRecord] = field(default_factory=list)
    _scope_stack: list[dict[str, str]] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock)

    @contextmanager
    def scope(self, **tags: str) -> Iterator[None]:
        """Tag every record produced inside this block with `tags`.

            with ledger.scope(user_id="abc", feature="rag"):
                ledger.record(...)
        """
        with self._lock:
            self._scope_stack.append(dict(tags))
        try:
            yield
        finally:
            with self._lock:
                self._scope_stack.pop()

    def record(
        self,
        model: str,
        usage: Usage,
        region: str = "us-east-1",
        timestamp: datetime | None = None,
    ) -> CallRecord:
        price = self._lookup_price(model)
        dollars = _dollars(usage, price)
        scope = self._merged_scope()
        rec = CallRecord(
            timestamp=timestamp or datetime.now(),
            model=model,
            region=region,
            usage=usage,
            dollars=dollars,
            scope=scope,
        )
        with self._lock:
            self._records.append(rec)
        return rec

    def dollars(self, **filter_tags: str) -> float:
        """Sum dollars across records matching ALL of the supplied filter tags.

        `model=...`, `region=...`, or any custom scope tag.
        """
        return sum(r.dollars for r in self._matching(filter_tags))

    def totals(self, group_by: str) -> dict[str, float]:
        """Sum dollars grouped by a single field (model, region, or scope tag)."""
        out: dict[str, float] = {}
        with self._lock:
            for r in self._records:
                key = self._field(r, group_by) or "(unset)"
                out[key] = out.get(key, 0.0) + r.dollars
        return out

    def records(self) -> list[CallRecord]:
        with self._lock:
            return list(self._records)

    # ---------- internals ----------

    def _lookup_price(self, model: str) -> ModelPrice:
        if model in self.prices:
            return self.prices[model]
        # tolerate variants: "us.anthropic.claude-sonnet-4-5-v1:0",
        # "anthropic.claude-sonnet-4-5-v1:0", "bedrock/anthropic.claude-..."
        normalized = model.split("/")[-1]
        if normalized.startswith(("us.", "eu.", "apac.")):
            normalized = normalized.split(".", 1)[1]
        if normalized in self.prices:
            return self.prices[normalized]
        raise KeyError(f"no price for model {model!r}; pass via Ledger(prices={{...}})")

    def _merged_scope(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for s in self._scope_stack:
            merged.update(s)
        return merged

    def _matching(self, filters: dict[str, str]) -> list[CallRecord]:
        with self._lock:
            return [r for r in self._records if self._matches(r, filters)]

    @staticmethod
    def _matches(r: CallRecord, filters: dict[str, str]) -> bool:
        for k, v in filters.items():
            if Ledger._field(r, k) != v:
                return False
        return True

    @staticmethod
    def _field(r: CallRecord, name: str) -> str | None:
        if name == "model":
            return r.model
        if name == "region":
            return r.region
        return r.scope.get(name)


def _dollars(usage: Usage, price: ModelPrice) -> float:
    input_priced = usage.input_tokens
    cache_read = usage.cache_read_input_tokens
    cache_create = usage.cache_creation_input_tokens
    cost = (
        (input_priced / 1000) * price.input_per_1k
        + (cache_read / 1000) * price.cache_read_per_1k
        + (cache_create / 1000) * price.cache_creation_per_1k
        + (usage.output_tokens / 1000) * price.output_per_1k
    )
    return round(cost, 6)
