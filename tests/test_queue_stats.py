from app.scheduler.redis_queue import (
    QUEUE_NAME,
    QueueTask,
    RedisQueue,
)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.lists = {}

    def set(self, key, value, ex=None):
        self.values[key] = value
        return True

    def rpush(self, key, value):
        items = self.lists.setdefault(key, [])
        items.append(value)
        return len(items)

    def llen(self, key):
        return len(self.lists.get(key, []))

    def hincrby(self, key, field, amount):
        values = self.hashes.setdefault(key, {})
        values[field] = int(values.get(field, 0)) + amount
        return values[field]

    def hgetall(self, key):
        return {
            field: str(value)
            for field, value in self.hashes.get(key, {}).items()
        }


def make_task(job_id, model, attempts=0):
    return QueueTask(
        job_id=job_id,
        prompt="queue statistics test",
        system_prompt=None,
        model=model,
        max_tokens=8,
        created_at=0.0,
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
    queue.requeue(completed_task, "temporary inference failure")
    queue.set_processing(completed_task)
    queue.set_result(completed_task, {"answer": "ok"})

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
    queue.set_error(failed_task, "simulated inference failure")

    queue.client.lists[QUEUE_NAME] = []

    stats = queue.get_stats()

    assert stats == {
        "queue_size": 0,
        "jobs_submitted_total": 2,
        "processing_attempts_total": 3,
        "jobs_completed_total": 1,
        "jobs_failed_total": 1,
        "retries_total": 1,
    }

    print("[PASS] queue statistics are correct")
    print(stats)


if __name__ == "__main__":
    test_queue_statistics()
