import time

from app.scheduler.redis_queue import (
    DEAD_LETTER_QUEUE_NAME,
    QUEUE_NAME,
    QueueTask,
    RedisQueue,
)


class FakePipeline:
    def __init__(self, client):
        self.client = client

    def set(self, key, value, ex=None):
        self.client.set(key, value, ex=ex)
        return self

    def rpush(self, key, value):
        self.client.rpush(key, value)
        return self

    def lrem(self, key, count, value):
        self.client.lrem(key, count, value)
        return self

    def hdel(self, key, field):
        self.client.hdel(key, field)
        return self

    def hincrby(self, key, field, amount):
        self.client.hincrby(key, field, amount)
        return self

    def hincrbyfloat(self, key, field, amount):
        self.client.hincrbyfloat(
            key,
            field,
            amount,
        )
        return self

    def execute(self):
        return []


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.lists = {}

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    def set(self, key, value, ex=None):
        self.values[key] = value
        return True

    def rpush(self, key, value):
        items = self.lists.setdefault(key, [])
        items.append(value)
        return len(items)

    def llen(self, key):
        return len(self.lists.get(key, []))

    def lrem(self, key, count, value):
        items = self.lists.setdefault(key, [])

        removed = 0
        remaining = []

        for item in items:
            if item == value and removed < count:
                removed += 1
            else:
                remaining.append(item)

        self.lists[key] = remaining
        return removed

    def hdel(self, key, field):
        values = self.hashes.setdefault(key, {})
        return int(values.pop(field, None) is not None)

    def hincrby(self, key, field, amount):
        values = self.hashes.setdefault(key, {})
        values[field] = int(values.get(field, 0)) + amount
        return values[field]

    def hincrbyfloat(self, key, field, amount):
        values = self.hashes.setdefault(key, {})
        values[field] = (
            float(values.get(field, 0.0))
            + float(amount)
        )
        return values[field]

    def hgetall(self, key):
        return {
            field: str(value)
            for field, value in self.hashes.get(
                key,
                {},
            ).items()
        }


def make_task(job_id, model, attempts=0):
    return QueueTask(
        job_id=job_id,
        prompt="queue statistics test",
        system_prompt=None,
        model=model,
        max_tokens=8,
        created_at=time.time() - 0.2,
        attempts=attempts,
        max_retries=2,
    )


def test_queue_statistics():
    queue = RedisQueue(host="unused")
    queue.client = FakeRedis()

    completed_job_id = queue.enqueue(
        prompt="successful task",
        system_prompt=None,
        model="mock-llm",
        max_tokens=8,
    )

    completed_task = make_task(
        completed_job_id,
        model="mock-llm",
    )

    queue.set_processing(completed_task)
    queue.requeue(
        completed_task,
        "temporary inference failure",
    )
    queue.set_processing(completed_task)
    queue.set_result(
        completed_task,
        {"answer": "ok"},
    )

    failed_job_id = queue.enqueue(
        prompt="failed task",
        system_prompt=None,
        model="fail-test",
        max_tokens=8,
    )

    failed_task = make_task(
        failed_job_id,
        model="fail-test",
        attempts=2,
    )

    queue.set_processing(failed_task)
    queue.set_error(
        failed_task,
        "simulated inference failure",
    )

    queue.record_inference_duration(0.25)

    queue.client.lists[QUEUE_NAME] = []

    stats = queue.get_stats()

    assert stats == {
        "queue_size": 0,
        "processing_size": 0,
        "dead_letter_size": 1,
        "jobs_submitted_total": 2,
        "processing_attempts_total": 3,
        "jobs_completed_total": 1,
        "jobs_failed_total": 1,
        "retries_total": 1,
        "recoveries_total": 0,
        "dead_lettered_total": 1,
    }

    snapshot = queue.get_metrics_snapshot()

    assert (
        snapshot[
            "queue_wait_histogram"
        ]["count"]
        == 3
    )

    assert (
        snapshot[
            "inference_duration_histogram"
        ]["count"]
        == 1
    )

    assert queue.dead_letter_size() == len(
        queue.client.lists[
            DEAD_LETTER_QUEUE_NAME
        ]
    )

    print("[PASS] queue statistics are correct")
    print(stats)
    print(snapshot)


if __name__ == "__main__":
    test_queue_statistics()
