import time

from app.scheduler.redis_queue import QueueTask
from app.worker import process_task


class FailingBackend:
    def generate(self, request):
        raise RuntimeError("simulated inference failure")


class FakeQueue:
    def __init__(self):
        self.processing = []
        self.retried = []
        self.failed = []
        self.results = []

    def set_processing(self, task):
        self.processing.append(task.job_id)

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

    def set_result(self, task, result):
        self.results.append(
            {
                "job_id": task.job_id,
                "result": result,
            }
        )


def make_task(attempts):
    return QueueTask(
        job_id=f"failure-test-{attempts}",
        prompt="failure test",
        system_prompt=None,
        model="fail-test",
        max_tokens=8,
        created_at=time.time(),
        attempts=attempts,
        max_retries=2,
    )


def test_first_failure_is_requeued():
    task = make_task(attempts=0)
    queue = FakeQueue()

    outcome = process_task(
        task,
        queue=queue,
        backend=FailingBackend(),
    )

    assert outcome == "queued"
    assert task.attempts == 1
    assert len(queue.processing) == 1
    assert len(queue.retried) == 1
    assert len(queue.failed) == 0

    print("[PASS] first failure was requeued")


def test_final_failure_is_saved():
    task = make_task(attempts=2)
    queue = FakeQueue()

    outcome = process_task(
        task,
        queue=queue,
        backend=FailingBackend(),
    )

    assert outcome == "failed"
    assert len(queue.processing) == 1
    assert len(queue.retried) == 0
    assert len(queue.failed) == 1
    assert queue.failed[0]["error"] == "simulated inference failure"

    print("[PASS] final failure status and error were saved")


if __name__ == "__main__":
    test_first_failure_is_requeued()
    test_final_failure_is_saved()
    print("[PASS] all worker failure tests completed")
