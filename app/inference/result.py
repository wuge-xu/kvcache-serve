from dataclasses import dataclass


@dataclass
class GenerationResult:
    request_id: str

    answer: str
    model: str
    backend: str
    device: str

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    latency_ms: float
    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    avg_itl_ms: float
    tokens_per_second: float

    kv_cache_tokens: int
    kv_cache_memory_bytes: int
    kv_cache_memory_mb: float

    kv_policy: str = "noop"
    kv_policy_applied_count: int = 0
    kv_policy_tokens_before: int = 0
    kv_policy_tokens_after: int = 0
    kv_policy_evicted_tokens: int = 0
