import time

from app.scheduler.redis_queue import QueueTask
from app.worker import process_task


class FailingBackend:
    def generate(self, request):
        raise RuntimeError("simulated inference failure")


class FakeQueue:
    def __init__(self):
        self.running = []
        self.retried = []
        self.failed = []
        self.results = []

    def set_running(self, task):
        self.running.append(task.job_id)

    def requeue(self, task, error):
        task.attempts += 1
        self.retried.append(
            {
                "job_id": task.job_id,
                "attempts": task.attempts,
                "error": error,
            }
        )

    def set_error(self, task, error):
        self.failed.append(
            {
                "job_id": task.job_id,
                "attempts": task.attempts,
                "error": error,
            }
        )

    def set_result(self, job_id, result):
        self.results.append(
            {
                "job_id": job_id,
                "result": result,
            }
        )


def make_task(attempts):
    return QueueTask(
        job_id=f"retry-test-{attempts}",
        prompt="retry test",
        system_prompt=None,
        model="local-llm",
        max_tokens=8,
        created_at=time.time(),
        attempts=attempts,
        max_retries=2,
    )


def test_first_failure_is_retried():
    task = make_task(attempts=0)
    queue = FakeQueue()

    outcome = process_task(
        task,
        queue=queue,
        backend=FailingBackend(),
    )

    assert outcome == "retrying"
    assert task.attempts == 1
    assert len(queue.retried) == 1
    assert len(queue.failed) == 0

    print("[PASS] first failure changed to retrying")
    print(queue.retried[0])


def test_final_failure_is_not_retried():
    task = make_task(attempts=2)
    queue = FakeQueue()

    outcome = process_task(
        task,
        queue=queue,
        backend=FailingBackend(),
    )

    assert outcome == "failed"
    assert task.attempts == 2
    assert len(queue.retried) == 0
    assert len(queue.failed) == 1

    print("[PASS] retry limit changed status to failed")
    print(queue.failed[0])


if __name__ == "__main__":
    test_first_failure_is_retried()
    test_final_failure_is_not_retried()
    print("[PASS] all retry tests completed")
