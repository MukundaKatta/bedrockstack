# bedrockstack

[![ci](https://github.com/MukundaKatta/bedrockstack/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/bedrockstack/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/bedrockstack.svg)](https://pypi.org/project/bedrockstack/)
[![python](https://img.shields.io/pypi/pyversions/bedrockstack.svg)](https://pypi.org/project/bedrockstack/)

Low-level Python ergonomics for AWS Bedrock + Anthropic Claude. Three primitives every team building on Bedrock ends up rewriting from scratch:

1. **`RetryPolicy`** — Bedrock-aware backoff that knows the difference between a `ThrottlingException` (retry), a `ModelNotReadyException` (wait longer), and a `ValidationException` (never retry).
2. **`Ledger`** — thread-safe cost tracking with scope tags, grouped totals, and Anthropic-on-Bedrock pricing baked in (override per `Ledger(prices=...)`).
3. **`wrap_stream()`** — normalize the [streaming-error parity gap](https://github.com/anthropics/anthropic-sdk-python/issues/866) between `AnthropicBedrock` and the native Anthropic client into a single exception hierarchy.

This is the layer above `boto3.client('bedrock-runtime')`, not a replacement. Sibling library: [bedrockcache](https://github.com/MukundaKatta/bedrockcache) for prompt-caching audits.

## Install

```bash
pip install bedrockstack
```

Zero runtime dependencies. Works alongside whatever Bedrock client you already use (`boto3`, `AnthropicBedrock`, LiteLLM, Strands).

## Retries

```python
import boto3
from bedrockstack import bedrock_default

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
policy = bedrock_default()  # 6 attempts, 0.5s..30s exponential w/ jitter

response = policy.call(
    lambda: bedrock.converse(modelId="anthropic.claude-sonnet-4-5-v1:0", ...)
)
```

What it does and doesn't retry:

| Retries | Never retries |
|---|---|
| ThrottlingException | ValidationException |
| ServiceUnavailableException | AccessDeniedException |
| InternalServerException | ResourceNotFoundException |
| ModelNotReadyException (with longer initial delay) | InvalidRequestException |
| ModelTimeoutException | ModelErrorException |
| ModelStreamErrorException | (anything else not in the retryable list) |
| ServiceQuotaExceededException | |
| Connection / read timeouts (matched by class name) | |

## Cost ledger

```python
from bedrockstack import Ledger, Usage

led = Ledger()

with led.scope(user_id="abc", feature="rag"):
    led.record(
        model="anthropic.claude-sonnet-4-5-v1:0",
        usage=Usage(input_tokens=10_000, output_tokens=2_000,
                    cache_read_input_tokens=8_000),
    )

print(led.dollars(user_id="abc"))             # 0.027
print(led.dollars(feature="rag"))             # 0.027
print(led.totals(group_by="model"))           # {'anthropic.claude-sonnet-4-5-v1:0': 0.027}
```

Inference-profile model IDs (`us.anthropic.claude-...`) and LiteLLM-style prefixes (`bedrock/anthropic.claude-...`) are normalized automatically.

To override prices when AWS rebases:

```python
from bedrockstack import Ledger, ModelPrice
led = Ledger(prices={
    "anthropic.claude-sonnet-4-5-v1:0": ModelPrice(0.0025, 0.0125, 0.00025, 0.003125),
})
```

## Streaming-error normalization

```python
from anthropic import AnthropicBedrock
from bedrockstack import wrap_stream, StreamOverloadedError, StreamRateLimitError

client = AnthropicBedrock()
stream = client.messages.create(model="anthropic.claude-sonnet-4-5-v1:0",
                                stream=True, messages=[...])

try:
    for event in wrap_stream(stream):
        ...
except StreamOverloadedError:
    # 529 mid-stream — back off long
    ...
except StreamRateLimitError:
    # 429 mid-stream — back off short
    ...
```

The same `wrap_stream()` works against the native Anthropic client and `AnthropicBedrock`. You handle one set of exceptions instead of two.

## What it explicitly is not

- Not an agent framework. Use Strands or pydantic-ai for that.
- Not a router. LiteLLM exists.
- Not a prompt-caching auditor — that's [bedrockcache](https://github.com/MukundaKatta/bedrockcache).
- Bedrock + Anthropic-only. No OpenAI / Vertex / Azure surface.

## Roadmap

- v0.2: `ToolRunner` with AnthropicBedrock parity (matches the native client's `messages.tool_runner`).
- v0.3: `BedrockClient` wrapper that bundles retry + ledger + streaming around a boto3 client.
- v0.4: region failover for inference-profile model IDs.
- v0.5: SageMaker Async Inference helper.

## Develop

```bash
pip install -e ".[dev]"
pytest -v
```

## License

MIT
