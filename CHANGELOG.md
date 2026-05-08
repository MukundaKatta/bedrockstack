# Changelog

## 0.1.0 — initial release

- `RetryPolicy` — Bedrock-aware exponential backoff with decorrelated jitter.
  Encodes the documented retryable codes (ThrottlingException,
  ModelNotReadyException, ServiceUnavailableException, etc.) and the
  never-retry list (ValidationException, AccessDeniedException,
  InvalidRequestException). `ModelNotReadyException` gets a longer initial
  delay since the model is warming.
- `Ledger` — append-only thread-safe cost ledger with scope tagging
  (`with led.scope(user_id="...", feature="..."):`), grouped totals, and
  filterable dollar sums. Hardcoded May-2026 prices for the Anthropic Claude
  family on Bedrock; pass `Ledger(prices={...})` to override.
- `wrap_stream()` and `normalize_exception()` — translate AnthropicBedrock and
  native Anthropic streaming errors into a single normalized exception
  hierarchy (`StreamOverloadedError`, `StreamRateLimitError`,
  `StreamModelError`, `StreamServerError`, `StreamConnectionError`).
- 31 tests across retry, cost, and streaming modules.
- Zero runtime dependencies. boto3 is not imported anywhere; this library is
  the layer above your boto client, not a replacement for it.
