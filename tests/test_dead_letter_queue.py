import json
import time

from app.scheduler.redis_queue import (
    DEAD_LETTER_QUEUE_NAME,
    PROCESSING_CLAIMS_KEY,
    PROCESSING_QUEUE_NAME,
    QUEUE_NAME,
    STATUS_PREFIX,
    QueueTask,
    RedisQueue,
)


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.lists = {}
        self.values = {}

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))


class TestRedisQueue(RedisQueue):
    def _dead_letter_claim_atomically(
        self,
        job_id,
        expected_claim,
        old_payload,
        dead_letter_record,
        status_data,
    ):
        current_claim = self.client.hashes.get(
            PROCESSING_CLAIMS_KEY,
            {},
        ).get(job_id)

        if current_claim != expected_claim:
            return False

        processing = self.client.lists.setdefault(
            PROCESSING_QUEUE_NAME,
            [],
        )

        try:
            processing.remove(old_payload)
        except ValueError:
            return False

        self.client.lists.setdefault(
            DEAD_LETTER_QUEUE_NAME,
            [],
        ).append(
            json.dumps(
                dead_letter_record,
                ensure_ascii=False,
            )
        )

        self.client.hashes[
            PROCESSING_CLAIMS_KEY
        ].pop(job_id, None)

        self.client.values[
            STATUS_PREFIX + job_id
        ] = json.dumps(
            status_data,
            ensure_ascii=False,
        )

        return True


def test_recovery_limit_moves_task_to_dead_letter():
    queue = TestRedisQueue(host="unused")
    queue.client = FakeRedis()

    task = QueueTask(
        job_id="dead-letter-job",
        prompt="dead letter test",
        system_prompt=None,
        model="local-llm",
        max_tokens=8,
        created_at=time.time(),
        attempts=0,
        max_retries=2,
        recoveries=0,
    )

    payload = queue._serialize_task(task)

    claim = json.dumps(
        {
            "job_id": task.job_id,
            "claimed_at": time.time() - 60,
            "worker_id": "dead-worker",
            "payload": payload,
        },
        ensure_ascii=False,
    )

    queue.client.lists[PROCESSING_QUEUE_NAME] = [
        payload
    ]

    queue.client.lists[QUEUE_NAME] = []

    queue.client.hashes[PROCESSING_CLAIMS_KEY] = {
        task.job_id: claim
    }

    result = queue.recover_stale_tasks(
        timeout_seconds=5,
        max_recoveries=0,
    )

    assert result["stale"] == 1
    assert result["requeued"] == 0
    assert result["dead_lettered"] == 1

    assert queue.client.lists[PROCESSING_QUEUE_NAME] == []
    assert queue.client.lists[QUEUE_NAME] == []

    dead_letters = queue.client.lists[
        DEAD_LETTER_QUEUE_NAME
    ]

    assert len(dead_letters) == 1

    record = json.loads(dead_letters[0])

    assert record["job_id"] == task.job_id
    assert record["recoveries"] == 1
    assert record["max_recoveries"] == 0

    status = json.loads(
        queue.client.values[
            STATUS_PREFIX + task.job_id
        ]
    )

    assert status["status"] == "failed"
    assert status["dead_letter"] is True

    print("[PASS] recovery limit moved task to DLQ")
    print(result)
    print(record)


if __name__ == "__main__":
    test_recovery_limit_moves_task_to_dead_letter()
