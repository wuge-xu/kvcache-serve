from prometheus_client import Counter, Histogram, Gauge

LLM_REQUEST_TOTAL = Counter(
    "llm_request_total",
    "Total number of LLM API requests",
    ["endpoint", "status"],
)

LLM_REQUEST_LATENCY_SECONDS = Histogram(
    "llm_request_latency_seconds",
    "Latency of LLM API requests in seconds",
    ["endpoint"],
)

LLM_PREFILL_SECONDS = Histogram(
    "llm_prefill_seconds",
    "Prefill latency in seconds",
)

LLM_DECODE_SECONDS = Histogram(
    "llm_decode_seconds",
    "Decode latency in seconds",
)

LLM_TTFT_SECONDS = Histogram(
    "llm_ttft_seconds",
    "Time to first token in seconds",
)

LLM_ITL_SECONDS = Histogram(
    "llm_itl_seconds",
    "Average inter-token latency in seconds",
)

LLM_TOKENS_GENERATED_TOTAL = Counter(
    "llm_tokens_generated_total",
    "Total number of generated tokens",
)

LLM_KV_CACHE_TOKENS = Gauge(
    "llm_kv_cache_tokens",
    "Number of tokens stored in KV cache",
)

LLM_KV_CACHE_MEMORY_BYTES = Gauge(
    "llm_kv_cache_memory_bytes",
    "Estimated KV cache memory usage in bytes",
)

LLM_ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Number of active LLM requests",
)
