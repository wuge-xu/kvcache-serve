import time

from app.inference.backend import GenerationRequest
from app.inference.mock_backend import mock_backend
from app.inference.transformers_backend import transformers_backend
from app.scheduler.redis_queue import QueueTask, redis_queue


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


def resolve_backend(model_name: str):
    if model_name == "mock-llm":
        return mock_backend

    return transformers_backend


def process_task(task: QueueTask, queue=redis_queue, backend=None) -> str:
    queue.set_running(task)

    generation_request = GenerationRequest(
        request_id=task.job_id,
        prompt=task.prompt,
        system_prompt=task.system_prompt,
        max_tokens=task.max_tokens,
    )

    selected_backend = backend or resolve_backend(task.model)

    try:
        result = selected_backend.generate(generation_request)
        queue.set_result(task.job_id, result_to_dict(result))
        return "finished"

    except Exception as error:
        error_message = str(error)

        if task.attempts < task.max_retries:
            queue.requeue(task, error_message)
            return "retrying"

        queue.set_error(task, error_message)
        return "failed"


def main():
    print("[Worker] KVCache-Serve inference worker started.")
    print("[Worker] Waiting for tasks from Redis...")

    while True:
        task = redis_queue.dequeue(timeout=5)

        if task is None:
            continue

        print(
            f"[Worker] Received job: {task.job_id}, "
            f"model={task.model}, "
            f"attempt={task.attempts}/{task.max_retries}"
        )

        outcome = process_task(task)

        if outcome == "finished":
            print(f"[Worker] Finished job: {task.job_id}")

        elif outcome == "retrying":
            print(
                f"[Worker] Retrying job: {task.job_id}, "
                f"next_attempt={task.attempts}/{task.max_retries}"
            )

        else:
            print(
                f"[Worker] Permanently failed job: {task.job_id}, "
                f"attempts={task.attempts}/{task.max_retries}"
            )

        time.sleep(0.1)


if __name__ == "__main__":
    main()
