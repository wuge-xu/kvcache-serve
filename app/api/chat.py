import asyncio
import uuid
from typing import Any

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.inference.backend import GenerationRequest
from app.inference.mock_backend import mock_backend
from app.inference.transformers_backend import transformers_backend
from app.kvcache.registry import kv_cache_policy_registry
from app.kvcache.runtime import kv_cache_runtime
from app.metrics.prometheus import (
    LLM_REQUEST_TOTAL,
    LLM_REQUEST_LATENCY_SECONDS,
    LLM_PREFILL_SECONDS,
    LLM_DECODE_SECONDS,
    LLM_TTFT_SECONDS,
    LLM_ITL_SECONDS,
    LLM_TOKENS_GENERATED_TOTAL,
    LLM_KV_CACHE_TOKENS,
    LLM_KV_CACHE_MEMORY_BYTES,
    LLM_ACTIVE_REQUESTS,
)

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")
    model: str = Field(default="local-llm", description="local-llm or mock-llm")
    max_tokens: int = Field(default=64, ge=1, le=512)
    kv_policy: str = Field(default="noop", description="KV Cache policy name")
    kv_policy_config: dict[str, Any] = Field(
        default_factory=dict,
        description="KV Cache policy configuration",
    )


class ChatResponse(BaseModel):
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

    kv_policy: str
    kv_policy_applied_count: int
    kv_policy_tokens_before: int
    kv_policy_tokens_after: int
    kv_policy_evicted_tokens: int


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    try:
        policy = kv_cache_policy_registry.create(
            request.kv_policy,
            request.kv_policy_config,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    request_id = str(uuid.uuid4())

    LLM_ACTIVE_REQUESTS.inc()
    kv_cache_runtime.request_started(request_id)

    try:
        generation_request = GenerationRequest(
            request_id=request_id,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            kv_policy=policy.name,
            kv_policy_config=policy.config,
        )

        if request.model == "mock-llm":
            result = await asyncio.to_thread(
                mock_backend.generate,
                generation_request,
            )
        else:
            result = await asyncio.to_thread(
                transformers_backend.generate,
                generation_request,
            )

        LLM_TOKENS_GENERATED_TOTAL.inc(result.completion_tokens)
        LLM_REQUEST_TOTAL.labels(endpoint="/chat", status="success").inc()
        LLM_REQUEST_LATENCY_SECONDS.labels(endpoint="/chat").observe(result.latency_ms / 1000)

        if result.prefill_ms > 0:
            LLM_PREFILL_SECONDS.observe(result.prefill_ms / 1000)

        if result.decode_ms > 0:
            LLM_DECODE_SECONDS.observe(result.decode_ms / 1000)

        if result.ttft_ms > 0:
            LLM_TTFT_SECONDS.observe(result.ttft_ms / 1000)

        if result.avg_itl_ms > 0:
            LLM_ITL_SECONDS.observe(result.avg_itl_ms / 1000)

        LLM_KV_CACHE_TOKENS.set(result.kv_cache_tokens)
        LLM_KV_CACHE_MEMORY_BYTES.set(result.kv_cache_memory_bytes)

        kv_cache_runtime.request_finished(request_id)

        return ChatResponse(
            request_id=result.request_id,

            answer=result.answer,
            model=result.model,
            backend=result.backend,
            device=result.device,

            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,

            latency_ms=result.latency_ms,
            prefill_ms=result.prefill_ms,
            decode_ms=result.decode_ms,
            ttft_ms=result.ttft_ms,
            avg_itl_ms=result.avg_itl_ms,
            tokens_per_second=result.tokens_per_second,

            kv_cache_tokens=result.kv_cache_tokens,
            kv_cache_memory_bytes=result.kv_cache_memory_bytes,
            kv_cache_memory_mb=result.kv_cache_memory_mb,

            kv_policy=result.kv_policy,
            kv_policy_applied_count=result.kv_policy_applied_count,
            kv_policy_tokens_before=result.kv_policy_tokens_before,
            kv_policy_tokens_after=result.kv_policy_tokens_after,
            kv_policy_evicted_tokens=result.kv_policy_evicted_tokens,
        )

    except HTTPException:
        LLM_REQUEST_TOTAL.labels(endpoint="/chat", status="bad_request").inc()
        kv_cache_runtime.request_failed(request_id, "bad_request")
        raise

    except Exception as e:
        LLM_REQUEST_TOTAL.labels(endpoint="/chat", status="error").inc()
        kv_cache_runtime.request_failed(request_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        LLM_ACTIVE_REQUESTS.dec()
