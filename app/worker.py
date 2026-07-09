import json
import time

from app.inference.backend import GenerationRequest
from app.inference.mock_backend import mock_backend
from app.inference.transformers_backend import transformers_backend
from app.scheduler.redis_queue import redis_queue


def result_to_dict(result):
    return {
        "request_id": result.request_id,
        "answer": result.answer,
        "model": result.model,
        "backend": result.backend,
        "device": result.device,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": result.latency_ms,
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.decode_ms,
        "ttft_ms": result.ttft_ms,
        "avg_itl_ms": result.avg_itl_ms,
        "tokens_per_second": result.tokens_per_second,
        "kv_cache_tokens": result.kv_cache_tokens,
        "kv_cache_memory_bytes": result.kv_cache_memory_bytes,
        "kv_cache_memory_mb": result.kv_cache_memory_mb,
    }


def handle_task(task):
    redis_queue.set_running(task.job_id)

    generation_request = GenerationRequest(
        request_id=task.job_id,
        prompt=task.prompt,
        system_prompt=task.system_prompt,
        max_tokens=task.max_tokens,
    )

    if task.model == "mock-llm":
        result = mock_backend.generate(generation_request)
    else:
        result = transformers_backend.generate(generation_request)

    redis_queue.set_result(task.job_id, result_to_dict(result))


def main():
    print("[Worker] KVCache-Serve inference worker started.")
    print("[Worker] Waiting for tasks from Redis...")

    while True:
        task = redis_queue.dequeue(timeout=5)

        if task is None:
            continue

        print(f"[Worker] Received job: {task.job_id}, model={task.model}")

        try:
            handle_task(task)
            print(f"[Worker] Finished job: {task.job_id}")

        except Exception as e:
            print(f"[Worker] Failed job: {task.job_id}, error={e}")
            redis_queue.set_error(task.job_id, str(e))

        time.sleep(0.1)


if __name__ == "__main__":
    main()
