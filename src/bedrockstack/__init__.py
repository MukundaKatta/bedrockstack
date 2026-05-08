"""bedrockstack: low-level Python ergonomics for AWS Bedrock + Anthropic."""

from bedrockstack.cost import CallRecord, Ledger, Usage
from bedrockstack.prices import BEDROCK_PRICES, ModelPrice
from bedrockstack.retry import RetryPolicy, bedrock_default
from bedrockstack.streaming import (
    StreamConnectionError,
    StreamingError,
    StreamModelError,
    StreamOverloadedError,
    StreamRateLimitError,
    StreamServerError,
    normalize_exception,
    wrap_stream,
)

__all__ = [
    # retry
    "RetryPolicy",
    "bedrock_default",
    # cost
    "Ledger",
    "Usage",
    "CallRecord",
    "ModelPrice",
    "BEDROCK_PRICES",
    # streaming
    "wrap_stream",
    "normalize_exception",
    "StreamingError",
    "StreamOverloadedError",
    "StreamRateLimitError",
    "StreamModelError",
    "StreamServerError",
    "StreamConnectionError",
]
__version__ = "0.1.0"
