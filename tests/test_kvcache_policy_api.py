from fastapi.testclient import TestClient

from app.api import queue as queue_api
from app.main import app


client = TestClient(app)


def test_sync_mock_policy():
    response = client.post(
        "/chat",
        json={
            "prompt": "KV Cache Policy API test",
            "model": "mock-llm",
            "max_tokens": 8,
            "kv_policy": "noop",
            "kv_policy_config": {
                "budget": 32,
            },
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["backend"] == "mock"
    assert data["kv_policy"] == "noop"
    assert data["kv_policy_applied_count"] == 0
    assert data["kv_policy_tokens_before"] == 0
    assert data["kv_policy_tokens_after"] == 0
    assert data["kv_policy_evicted_tokens"] == 0


def test_sync_unknown_policy():
    response = client.post(
        "/chat",
        json={
            "prompt": "unknown policy test",
            "model": "mock-llm",
            "kv_policy": "missing-policy",
        },
    )

    assert response.status_code == 400
    assert (
        "unknown KV Cache policy"
        in response.json()["detail"]
    )


def test_async_queue_policy():
    captured = {}

    original_enqueue = (
        queue_api.redis_queue.enqueue
    )
    original_queue_size = (
        queue_api.redis_queue.queue_size
    )

    def fake_enqueue(**kwargs):
        captured.update(kwargs)
        return "policy-api-job"

    try:
        queue_api.redis_queue.enqueue = (
            fake_enqueue
        )

        queue_api.redis_queue.queue_size = (
            lambda: 1
        )

        response = client.post(
            "/queue/chat",
            json={
                "prompt": "queue policy test",
                "model": "mock-llm",
                "max_tokens": 8,
                "kv_policy": "noop",
                "kv_policy_config": {
                    "budget": 48,
                },
            },
        )
    finally:
        queue_api.redis_queue.enqueue = (
            original_enqueue
        )

        queue_api.redis_queue.queue_size = (
            original_queue_size
        )

    assert response.status_code == 200

    data = response.json()

    assert data["job_id"] == "policy-api-job"
    assert data["kv_policy"] == "noop"
    assert data["kv_policy_config"] == {
        "budget": 48,
    }

    assert captured["kv_policy"] == "noop"
    assert captured["kv_policy_config"] == {
        "budget": 48,
    }


def test_async_unknown_policy():
    response = client.post(
        "/queue/chat",
        json={
            "prompt": "unknown queue policy",
            "model": "mock-llm",
            "kv_policy": "not-registered",
        },
    )

    assert response.status_code == 400
    assert (
        "unknown KV Cache policy"
        in response.json()["detail"]
    )


def main():
    test_sync_mock_policy()
    test_sync_unknown_policy()
    test_async_queue_policy()
    test_async_unknown_policy()

    print(
        "[PASS] KV Cache Policy API tests completed"
    )


if __name__ == "__main__":
    main()
