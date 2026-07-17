import json
import time

from app.scheduler.redis_queue import (
    PROCESSING_CLAIMS_KEY,
    PROCESSING_QUEUE_NAME,
    QUEUE_NAME,
    RESULT_PREFIX,
    STATUS_PREFIX,
    QueueTask,
    RedisQueue,
)


class FakePipeline:
    def __init__(self, client):
        self.client = client

    def set(self, key, value, ex=None):
        self.client.values[key] = value
        return self

    def lrem(self, key, count, value):
        values = self.client.lists.setdefault(
            key,
            [],
        )

        removed = 0

        while value in values:
            values.remove(value)
            removed += 1

            if count > 0 and removed >= count:
                break

        return self

    def hdel(self, key, field):
        self.client.hashes.setdefault(
            key,
            {},
        ).pop(field, None)

        return self

    def execute(self):
        return []


class FakeRedis:
    def __init__(self):
        self.lists = {}
        self.hashes = {}
        self.values = {}

    def execute_command(self, *args):
        assert args[0] == "BLMOVE"

        source = args[1]
        destination = args[2]
        source_side = args[3]
        destination_side = args[4]

        assert source_side == "LEFT"
        assert destination_side == "RIGHT"

        pending = self.lists.setdefault(
            source,
            [],
        )

        if not pending:
            return None

        payload = pending.pop(0)

        self.lists.setdefault(
            destination,
            [],
        ).append(payload)

        return payload

    def hset(self, key, field, value):
        self.hashes.setdefault(
            key,
            {},
        )[field] = value

        return 1

    def set(self, key, value, ex=None):
        self.values[key] = value
        return True

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    def llen(self, key):
        return len(
            self.lists.get(key, [])
        )


def make_queue(client):
    queue = RedisQueue(host="unused")
    queue.client = client
    return queue


def test_single_task_has_single_worker_owner():
    client = FakeRedis()
    worker_a = make_queue(client)
    worker_b = make_queue(client)

    task = QueueTask(
        job_id="multi-worker-job",
        prompt="multi worker claim test",
        system_prompt=None,
        model="mock-llm",
        max_tokens=8,
        created_at=time.time(),
    )

    payload = worker_a._serialize_task(task)

    client.lists[QUEUE_NAME] = [
        payload,
    ]

    claimed_a = worker_a.dequeue(
        timeout=0,
        worker_id="worker-a",
    )

    claimed_b = worker_b.dequeue(
        timeout=0,
        worker_id="worker-b",
    )

    assert claimed_a is not None
    assert claimed_b is None
    assert claimed_a.worker_id == "worker-a"

    assert client.lists[QUEUE_NAME] == []
    assert client.lists[
        PROCESSING_QUEUE_NAME
    ] == [payload]

    worker_a.set_processing(claimed_a)

    claim = json.loads(
        client.hashes[
            PROCESSING_CLAIMS_KEY
        ][task.job_id]
    )

    assert claim["worker_id"] == "worker-a"
    assert claim["claimed_by"] == "worker-a"
    assert claim["claimed_at"] is not None
    assert (
        claim["processing_started_at"]
        is not None
    )

    processing_status = json.loads(
        client.values[
            STATUS_PREFIX + task.job_id
        ]
    )

    assert (
        processing_status["status"]
        == "processing"
    )
    assert (
        processing_status["claimed_by"]
        == "worker-a"
    )
    assert (
        processing_status[
            "processing_started_at"
        ]
        is not None
    )

    worker_a.set_result(
        claimed_a,
        {
            "request_id": task.job_id,
            "answer": "ok",
        },
    )

    completed_status = json.loads(
        client.values[
            STATUS_PREFIX + task.job_id
        ]
    )

    result = json.loads(
        client.values[
            RESULT_PREFIX + task.job_id
        ]
    )

    assert (
        completed_status["status"]
        == "completed"
    )
    assert (
        completed_status["last_worker_id"]
        == "worker-a"
    )
    assert result["worker_id"] == "worker-a"
    assert result["claimed_by"] == "worker-a"

    assert client.lists[
        PROCESSING_QUEUE_NAME
    ] == []

    assert (
        task.job_id
        not in client.hashes.get(
            PROCESSING_CLAIMS_KEY,
            {},
        )
    )

    print(
        "[PASS] one task had one Worker owner"
    )
    print(
        {
            "job_id": task.job_id,
            "claimed_by": "worker-a",
            "second_worker_claimed": False,
        }
    )


if __name__ == "__main__":
    test_single_task_has_single_worker_owner()
