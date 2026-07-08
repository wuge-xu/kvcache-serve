import asyncio
import time

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.inference.generator import local_generator
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
    model: str = Field(default="local-llm", description="Model name: local-llm or mock-llm")
    max_tokens: int = Field(default=64, ge=1, le=512)


class ChatResponse(BaseModel):
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cache_status: str
    device: str | None = None

    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    ttft_ms: float = 0.0
    avg_itl_ms: float = 0.0
    tokens_per_second: float = 0.0

    kv_cache_tokens: int = 0
    kv_cache_memory_bytes: int = 0
    kv_cache_memory_mb: float = 0.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 2)


async def mock_generate(request: ChatRequest) -> ChatResponse:
    start_time = time.perf_counter()

    await asyncio.sleep(0.3)

    answer = (
        "这是 KVCache-Serve 当前的 Mock LLM 回答。"
        f"你输入的问题是：{request.prompt}。"
        "当前阶段重点是先打通 API、Metrics 和服务骨架。"
    )

    prompt_tokens = estimate_tokens(request.prompt)
    completion_tokens = estimate_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return ChatResponse(
        answer=answer,
        model="mock-llm",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        cache_status="none",
        device="mock",
        tokens_per_second=round(completion_tokens / (latency_ms / 1000), 2),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    LLM_ACTIVE_REQUESTS.inc()

    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt cannot be empty")

        if request.model == "mock-llm":
            result = await mock_generate(request)
        else:
            generation = await asyncio.to_thread(
                local_generator.generate,
                request.prompt,
                request.system_prompt,
                request.max_tokens,
            )

            result = ChatResponse(
                answer=generation.answer,
                model=generation.model,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                latency_ms=generation.latency_ms,
                cache_status="full",
                device=generation.device,

                prefill_ms=generation.prefill_ms,
                decode_ms=generation.decode_ms,
                ttft_ms=generation.ttft_ms,
                avg_itl_ms=generation.avg_itl_ms,
                tokens_per_second=generation.tokens_per_second,

                kv_cache_tokens=generation.kv_cache_tokens,
                kv_cache_memory_bytes=generation.kv_cache_memory_bytes,
                kv_cache_memory_mb=generation.kv_cache_memory_mb,
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

        return result

    except HTTPException:
        LLM_REQUEST_TOTAL.labels(endpoint="/chat", status="bad_request").inc()
        raise

    except Exception as e:
        LLM_REQUEST_TOTAL.labels(endpoint="/chat", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        LLM_ACTIVE_REQUESTS.dec()
