from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.scheduler.redis_queue import redis_queue


router = APIRouter(prefix="/queue", tags=["queue"])


class QueueChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    system_prompt: str | None = None
    model: str = "local-llm"
    max_tokens: int = Field(default=64, ge=1, le=512)


@router.post("/chat")
async def enqueue_chat(request: QueueChatRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    job_id = redis_queue.enqueue(
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        model=request.model,
        max_tokens=request.max_tokens,
    )

    return {
        "job_id": job_id,
        "status": "queued",
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

    if status.get("status") != "finished":
        return {
            "job_id": job_id,
            "status": status.get("status"),
            "message": "result is not ready",
            "queue_size": redis_queue.queue_size(),
        }

    result = redis_queue.get_result(job_id)

    return {
        "job_id": job_id,
        "status": "finished",
        "result": result,
    }


@router.get("/health")
async def queue_health():
    return {
        "redis": "ok" if redis_queue.ping() else "error",
        "queue_size": redis_queue.queue_size(),
    }
