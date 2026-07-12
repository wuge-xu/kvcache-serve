import json
import os
import time
import uuid
from dataclasses import asdict, dataclass

import redis


QUEUE_NAME = "kvcache:tasks"
RESULT_PREFIX = "kvcache:result:"
STATUS_PREFIX = "kvcache:status:"

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
            attempts=0,
            max_retries=DEFAULT_MAX_RETRIES,
        )

        self.client.set(
            STATUS_PREFIX + job_id,
            json.dumps(
                {
                    "status": "queued",
                    "job_id": job_id,
                    "attempts": task.attempts,
                    "max_retries": task.max_retries,
                },
                ensure_ascii=False,
            ),
            ex=3600,
        )

        self.client.rpush(
            QUEUE_NAME,
            json.dumps(asdict(task), ensure_ascii=False),
        )

        return job_id

    def dequeue(self, timeout: int = 5) -> QueueTask | None:
        item = self.client.blpop(QUEUE_NAME, timeout=timeout)

        if item is None:
            return None

        _, payload = item
        data = json.loads(payload)

        # 兼容升级前已经存在于 Redis 中的旧任务。
        data.setdefault("attempts", 0)
        data.setdefault("max_retries", DEFAULT_MAX_RETRIES)

        return QueueTask(**data)

    def set_running(self, task: QueueTask):
        self.client.set(
            STATUS_PREFIX + task.job_id,
            json.dumps(
                {
                    "status": "running",
                    "job_id": task.job_id,
                    "attempts": task.attempts,
                    "max_retries": task.max_retries,
                },
                ensure_ascii=False,
            ),
            ex=3600,
        )

    def requeue(self, task: QueueTask, error: str):
        task.attempts += 1

        self.client.set(
            STATUS_PREFIX + task.job_id,
            json.dumps(
                {
                    "status": "retrying",
                    "job_id": task.job_id,
                    "attempts": task.attempts,
                    "max_retries": task.max_retries,
                    "last_error": error,
                },
                ensure_ascii=False,
            ),
            ex=3600,
        )

        self.client.rpush(
            QUEUE_NAME,
            json.dumps(asdict(task), ensure_ascii=False),
        )

    def set_result(self, job_id: str, result: dict):
        self.client.set(
            RESULT_PREFIX + job_id,
            json.dumps(result, ensure_ascii=False),
            ex=3600,
        )

        self.client.set(
            STATUS_PREFIX + job_id,
            json.dumps(
                {
                    "status": "finished",
                    "job_id": job_id,
                },
                ensure_ascii=False,
            ),
            ex=3600,
        )

    def set_error(self, task: QueueTask, error: str):
        self.client.set(
            STATUS_PREFIX + task.job_id,
            json.dumps(
                {
                    "status": "failed",
                    "job_id": task.job_id,
                    "attempts": task.attempts,
                    "max_retries": task.max_retries,
                    "error": error,
                },
                ensure_ascii=False,
            ),
            ex=3600,
        )

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


redis_queue = RedisQueue()
