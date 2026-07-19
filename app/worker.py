import os
import socket
import time

from app.inference.backend import GenerationRequest
from app.inference.mock_backend import mock_backend
from app.inference.transformers_backend import transformers_backend
from app.kvcache.registry import kv_cache_policy_registry
from app.scheduler.redis_queue import QueueTask, redis_queue


class ControlledFailureBackend:
    def generate(self, request):
        time.sleep(1)
        raise RuntimeError("simulated inference failure")


controlled_failure_backend = ControlledFailureBackend()


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
        "kv_policy": getattr(
            result,
            "kv_policy",
            "noop",
        ),
        "kv_policy_applied_count": getattr(
            result,
            "kv_policy_applied_count",
            0,
        ),
        "kv_policy_tokens_before": getattr(
            result,
            "kv_policy_tokens_before",
            0,
        ),
        "kv_policy_tokens_after": getattr(
            result,
            "kv_policy_tokens_after",
            0,
        ),
        "kv_policy_evicted_tokens": getattr(
            result,
            "kv_policy_evicted_tokens",
            0,
        ),
    }


def resolve_backend(model_name: str):
    if model_name == "mock-llm":
        return mock_backend

    if model_name == "fail-test":
        return controlled_failure_backend

    return transformers_backend


def process_task(task: QueueTask, queue=redis_queue, backend=None) -> str:
    queue.set_processing(task)

    selected_backend = backend or resolve_backend(task.model)
    inference_started_at = time.perf_counter()

    try:
        policy = kv_cache_policy_registry.create(
            task.kv_policy,
            task.kv_policy_config,
        )

        generation_request = GenerationRequest(
            request_id=task.job_id,
            prompt=task.prompt,
            system_prompt=task.system_prompt,
            max_tokens=task.max_tokens,
            kv_policy=policy.name,
            kv_policy_config=policy.config,
        )

        result = selected_backend.generate(generation_request)
        queue.set_result(task, result_to_dict(result))
        return "completed"

    except Exception as error:
        error_message = str(error)

        if task.attempts < task.max_retries:
            queue.requeue(task, error_message)
            return "queued"

        queue.set_error(task, error_message)
        return "failed"

    finally:
        recorder = getattr(
            queue,
            "record_inference_duration",
            None,
        )

        if callable(recorder):
            recorder(
                time.perf_counter()
                - inference_started_at
            )


def main():
    worker_id = (
        os.getenv("WORKER_ID")
        or f"{socket.gethostname()}-{os.getpid()}"
    )

    print("[Worker] KVCache-Serve inference worker started.")
    print(f"[Worker] Worker id: {worker_id}")
    print("[Worker] Waiting for tasks from Redis...")

    while True:
        task = redis_queue.dequeue(
            timeout=5,
            worker_id=worker_id,
        )

        if task is None:
            continue

        print(
            f"[Worker] Received job: {task.job_id}, "
            f"model={task.model}, "
            f"attempt={task.attempts}/{task.max_retries}"
        )

        outcome = process_task(task)

        if outcome == "completed":
            print(f"[Worker] Completed job: {task.job_id}")

        elif outcome == "queued":
            print(
                f"[Worker] Requeued job: {task.job_id}, "
                f"next_attempt={task.attempts}/{task.max_retries}"
            )

        else:
            print(
                f"[Worker] Failed job: {task.job_id}, "
                f"attempts={task.attempts}/{task.max_retries}"
            )

        time.sleep(0.1)


if __name__ == "__main__":
    main()
