import json

from app.inference.backend import GenerationRequest
from app.inference.result import GenerationResult
from app.kvcache.policy import KVCachePolicyContext
from app.kvcache.registry import (
    KVCachePolicyRegistry,
    kv_cache_policy_registry,
)
from app.kvcache.runtime import KVCacheRuntime
from app.scheduler.redis_queue import QueueTask, RedisQueue
from app.worker import process_task


class RecordingQueue:
    def __init__(self):
        self.processing = []
        self.results = []
        self.errors = []
        self.requeued = []

    def set_processing(self, task):
        self.processing.append(task.job_id)

    def set_result(self, task, result):
        self.results.append((task, result))

    def set_error(self, task, error):
        self.errors.append((task, error))

    def requeue(self, task, error):
        self.requeued.append((task, error))

    def record_inference_duration(self, duration):
        return None


class RecordingBackend:
    def __init__(self):
        self.request = None

    def generate(self, request):
        self.request = request

        return GenerationResult(
            request_id=request.request_id,
            answer="ok",
            model="test-model",
            backend="test",
            device="cpu",
            prompt_tokens=2,
            completion_tokens=1,
            total_tokens=3,
            latency_ms=1.0,
            prefill_ms=0.5,
            decode_ms=0.5,
            ttft_ms=0.5,
            avg_itl_ms=0.5,
            tokens_per_second=1000.0,
            kv_cache_tokens=3,
            kv_cache_memory_bytes=48,
            kv_cache_memory_mb=0.000046,
            kv_policy=request.kv_policy,
            kv_policy_applied_count=1,
            kv_policy_tokens_before=3,
            kv_policy_tokens_after=3,
            kv_policy_evicted_tokens=0,
        )


def test_noop_policy_preserves_cache():
    cache = object()

    context = KVCachePolicyContext(
        request_id="noop-test",
        stage="prefill",
        decode_step=0,
        prompt_tokens=8,
        generated_tokens=0,
        cached_tokens=8,
        model_name="test-model",
        device="cpu",
    )

    policy = kv_cache_policy_registry.create(
        "noop",
        {"example": True},
    )

    result = policy.apply(cache, context)

    assert result.past_key_values is cache
    assert result.request_id == "noop-test"
    assert result.policy_name == "noop"
    assert result.tokens_before == 8
    assert result.tokens_after == 8
    assert result.evicted_tokens == 0
    assert result.metadata["cache_modified"] is False


def test_registry_normalizes_name():
    policy = kv_cache_policy_registry.create(
        " NOOP ",
    )

    assert policy.name == "noop"
    assert kv_cache_policy_registry.available() == [
        "noop"
    ]


def test_registry_rejects_unknown_policy():
    registry = KVCachePolicyRegistry()

    try:
        registry.create("missing-policy")
    except ValueError as error:
        assert "unknown KV Cache policy" in str(
            error
        )
    else:
        raise AssertionError(
            "unknown policy was accepted"
        )


def test_request_config_is_not_shared():
    first = GenerationRequest(
        request_id="first",
        prompt="hello",
    )

    second = GenerationRequest(
        request_id="second",
        prompt="world",
    )

    first.kv_policy_config["budget"] = 16

    assert second.kv_policy_config == {}


def test_runtime_records_policy_event():
    runtime = KVCacheRuntime()

    context = KVCachePolicyContext(
        request_id="runtime-test",
        stage="decode",
        decode_step=2,
        prompt_tokens=5,
        generated_tokens=2,
        cached_tokens=7,
        model_name="test-model",
        device="cpu",
    )

    result = kv_cache_policy_registry.create(
        "noop"
    ).apply(
        object(),
        context,
    )

    runtime.policy_applied(result)
    event = runtime.get_status()[
        "last_policy_event"
    ]

    assert event["request_id"] == "runtime-test"
    assert event["policy_name"] == "noop"
    assert event["decode_step"] == 2
    assert event["evicted_tokens"] == 0


def test_queue_task_serializes_policy():
    task = QueueTask(
        job_id="queue-policy-test",
        prompt="hello",
        system_prompt=None,
        model="mock-llm",
        max_tokens=4,
        created_at=1.0,
        kv_policy="noop",
        kv_policy_config={
            "budget": 32,
        },
    )

    payload = RedisQueue._serialize_task(task)
    restored = QueueTask(**json.loads(payload))

    assert restored.kv_policy == "noop"
    assert restored.kv_policy_config == {
        "budget": 32,
    }


def test_worker_passes_policy_to_backend():
    task = QueueTask(
        job_id="worker-policy-test",
        prompt="hello",
        system_prompt=None,
        model="mock-llm",
        max_tokens=4,
        created_at=1.0,
        kv_policy="noop",
        kv_policy_config={
            "budget": 24,
        },
    )

    queue = RecordingQueue()
    backend = RecordingBackend()

    outcome = process_task(
        task,
        queue=queue,
        backend=backend,
    )

    assert outcome == "completed"
    assert backend.request is not None
    assert backend.request.kv_policy == "noop"
    assert backend.request.kv_policy_config == {
        "budget": 24,
    }

    result = queue.results[0][1]

    assert result["kv_policy"] == "noop"
    assert result[
        "kv_policy_applied_count"
    ] == 1


def main():
    test_noop_policy_preserves_cache()
    test_registry_normalizes_name()
    test_registry_rejects_unknown_policy()
    test_request_config_is_not_shared()
    test_runtime_records_policy_event()
    test_queue_task_serializes_policy()
    test_worker_passes_policy_to_backend()

    print(
        "[PASS] KV Cache Policy tests completed"
    )


if __name__ == "__main__":
    main()
