import json
import os
import time
import uuid
from dataclasses import asdict, dataclass

import redis


QUEUE_NAME = "kvcache:tasks"
RESULT_PREFIX = "kvcache:result:"
STATUS_PREFIX = "kvcache:status:"
STATS_KEY = "kvcache:stats"

DEFAULT_MAX_RETRIES = int(os.getenv("QUEUE_MAX_RETRIES", "2"))


@dataclass
class QueueTask:
    job_id: str
    prompt: str
    system_prompt: str | None
    model: str
    max_tokens: int
    created_at: float
    attempts: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES


class RedisQueue:
    def __init__(self, host: str | None = None, port: int | None = None):
        host = host or os.getenv("REDIS_HOST", "localhost")
        port = port or int(os.getenv("REDIS_PORT", "6379"))

        self.client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
        )

    def ping(self) -> bool:
        return bool(self.client.ping())

    def enqueue(
        self,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
    ) -> str:
        job_id = str(uuid.uuid4())

        task = QueueTask(
            job_id=job_id,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            created_at=time.time(),
        )

        self._save_status(
            job_id,
            {
                "status": "queued",
                "job_id": job_id,
                "attempts": task.attempts,
                "max_retries": task.max_retries,
            },
        )

        self.client.rpush(
            QUEUE_NAME,
            json.dumps(asdict(task), ensure_ascii=False),
        )

        self._increment_stat("jobs_submitted_total")
        return job_id

    def dequeue(self, timeout: int = 5) -> QueueTask | None:
        item = self.client.blpop(QUEUE_NAME, timeout=timeout)

        if item is None:
            return None

        _, payload = item
        data = json.loads(payload)

        data.setdefault("attempts", 0)
        data.setdefault("max_retries", DEFAULT_MAX_RETRIES)

        return QueueTask(**data)

    def set_processing(self, task: QueueTask):
        self._save_status(
            task.job_id,
            {
                "status": "processing",
                "job_id": task.job_id,
                "attempts": task.attempts,
                "max_retries": task.max_retries,
            },
        )

        self._increment_stat("processing_attempts_total")

    def requeue(self, task: QueueTask, error: str):
        task.attempts += 1
        short_error = self._short_error(error)

        self._save_status(
            task.job_id,
            {
                "status": "queued",
                "job_id": task.job_id,
                "attempts": task.attempts,
                "max_retries": task.max_retries,
                "last_error": short_error,
            },
        )

        self.client.rpush(
            QUEUE_NAME,
            json.dumps(asdict(task), ensure_ascii=False),
        )

        self._increment_stat("retries_total")

    def set_result(self, task: QueueTask, result: dict):
        self.client.set(
            RESULT_PREFIX + task.job_id,
            json.dumps(result, ensure_ascii=False),
            ex=3600,
        )

        self._save_status(
            task.job_id,
            {
                "status": "completed",
                "job_id": task.job_id,
                "attempts": task.attempts,
                "max_retries": task.max_retries,
            },
        )

        self._increment_stat("jobs_completed_total")

    def set_error(self, task: QueueTask, error: str):
        short_error = self._short_error(error)

        self._save_status(
            task.job_id,
            {
                "status": "failed",
                "job_id": task.job_id,
                "attempts": task.attempts,
                "max_retries": task.max_retries,
                "error": short_error,
            },
        )

        self._increment_stat("jobs_failed_total")

    def get_status(self, job_id: str) -> dict | None:
        raw = self.client.get(STATUS_PREFIX + job_id)

        if raw is None:
            return None

        return json.loads(raw)

    def get_result(self, job_id: str) -> dict | None:
        raw = self.client.get(RESULT_PREFIX + job_id)

        if raw is None:
            return None

        return json.loads(raw)

    def queue_size(self) -> int:
        return int(self.client.llen(QUEUE_NAME))

    def get_stats(self) -> dict:
        raw_stats = self.client.hgetall(STATS_KEY)

        stat_names = (
            "jobs_submitted_total",
            "processing_attempts_total",
            "jobs_completed_total",
            "jobs_failed_total",
            "retries_total",
        )

        stats = {
            name: int(raw_stats.get(name, 0))
            for name in stat_names
        }

        return {
            "queue_size": self.queue_size(),
            **stats,
        }

    def _increment_stat(self, name: str):
        self.client.hincrby(STATS_KEY, name, 1)

    def _save_status(self, job_id: str, data: dict):
        self.client.set(
            STATUS_PREFIX + job_id,
            json.dumps(data, ensure_ascii=False),
            ex=3600,
        )

    @staticmethod
    def _short_error(error: str) -> str:
        return str(error).strip()[:300]


redis_queue = RedisQueue()
