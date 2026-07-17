import json
import time

from app.scheduler.redis_queue import (
    PROCESSING_CLAIMS_KEY,
    PROCESSING_QUEUE_NAME,
    QUEUE_NAME,
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

    def eval(self, script, key_count, *args):
        processing_key = args[0]
        claims_key = args[1]
        pending_key = args[2]
        status_key = args[3]

        job_id = args[4]
        expected_claim = args[5]
        old_payload = args[6]
        new_payload = args[7]
        status_value = args[8]

        current_claim = self.hashes.get(
            claims_key,
            {},
        ).get(job_id)

        if current_claim != expected_claim:
            return 0

        processing = self.lists.setdefault(
            processing_key,
            [],
        )

        try:
            processing.remove(old_payload)
        except ValueError:
            self.hashes.get(claims_key, {}).pop(
                job_id,
                None,
            )
            return 0

        self.lists.setdefault(
            pending_key,
            [],
        ).append(new_payload)

        self.hashes.get(claims_key, {}).pop(
            job_id,
            None,
        )

        self.values[status_key] = status_value
        return 1


def test_stale_processing_task_is_requeued():
    queue = RedisQueue(host="unused")
    queue.client = FakeRedis()

    task = QueueTask(
        job_id="stale-job",
        prompt="stale processing test",
        system_prompt=None,
        model="mock-llm",
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
    queue.client.hashes[PROCESSING_CLAIMS_KEY] = {
        task.job_id: claim
    }

    result = queue.recover_stale_tasks(
        timeout_seconds=10,
    )

    assert result["stale"] == 1
    assert result["requeued"] == 1

    assert queue.client.lists[
        PROCESSING_QUEUE_NAME
    ] == []

    pending = queue.client.lists[QUEUE_NAME]
    assert len(pending) == 1

    recovered_task = json.loads(pending[0])
    assert recovered_task["recoveries"] == 1

    assert (
        task.job_id
        not in queue.client.hashes[
            PROCESSING_CLAIMS_KEY
        ]
    )

    status = json.loads(
        queue.client.values[
            "kvcache:status:" + task.job_id
        ]
    )

    assert status["status"] == "queued"
    assert status["recoveries"] == 1
    assert status["last_error"] == (
        "worker processing timeout"
    )

    print("[PASS] stale task was requeued")
    print(result)


if __name__ == "__main__":
    test_stale_processing_task_is_requeued()
