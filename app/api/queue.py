from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.kvcache.registry import kv_cache_policy_registry
from app.scheduler.redis_queue import redis_queue


router = APIRouter(prefix="/queue", tags=["queue"])


class QueueChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    system_prompt: str | None = None
    model: str = "local-llm"
    max_tokens: int = Field(default=64, ge=1, le=512)
    kv_policy: str = Field(
        default="noop",
        description="KV Cache policy name",
    )
    kv_policy_config: dict[str, Any] = Field(
        default_factory=dict,
        description="KV Cache policy configuration",
    )


@router.post("/chat")
async def enqueue_chat(request: QueueChatRequest):
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

    job_id = redis_queue.enqueue(
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        model=request.model,
        max_tokens=request.max_tokens,
        kv_policy=policy.name,
        kv_policy_config=policy.config,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "kv_policy": policy.name,
        "kv_policy_config": policy.config,
        "queue_size": redis_queue.queue_size(),
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    status = redis_queue.get_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="job not found")

    status["queue_size"] = redis_queue.queue_size()
    return status


@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    status = redis_queue.get_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="job not found")

    current_status = status.get("status")

    if current_status == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": status.get("error", "unknown worker error"),
            "attempts": status.get("attempts", 0),
            "max_retries": status.get("max_retries", 0),
        }

    if current_status != "completed":
        return {
            "job_id": job_id,
            "status": current_status,
            "message": "result is not ready",
            "queue_size": redis_queue.queue_size(),
        }

    result = redis_queue.get_result(job_id)

    return {
        "job_id": job_id,
        "status": "completed",
        "result": result,
    }


@router.get("/stats")
async def queue_stats():
    return redis_queue.get_stats()


@router.get("/health")
async def queue_health():
    return {
        "redis": "ok" if redis_queue.ping() else "error",
        "queue_size": redis_queue.queue_size(),
    }
