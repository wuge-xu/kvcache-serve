import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field

import redis


QUEUE_NAME = "kvcache:tasks"
PROCESSING_QUEUE_NAME = "kvcache:processing"
PROCESSING_CLAIMS_KEY = "kvcache:processing:claims"
DEAD_LETTER_QUEUE_NAME = "kvcache:dead_letter"

QUEUE_STATS_KEY = "kvcache:queue:stats"
QUEUE_WAIT_HISTOGRAM_KEY = "kvcache:queue:wait_histogram"
INFERENCE_DURATION_HISTOGRAM_KEY = (
    "kvcache:worker:inference_duration_histogram"
)

QUEUE_WAIT_BUCKETS = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
)

INFERENCE_DURATION_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)

RESULT_PREFIX = "kvcache:result:"
STATUS_PREFIX = "kvcache:status:"

DEFAULT_MAX_RETRIES = int(os.getenv("QUEUE_MAX_RETRIES", "2"))
DEFAULT_PROCESSING_TIMEOUT_SECONDS = float(
    os.getenv("PROCESSING_TIMEOUT_SECONDS", "30")
)
DEFAULT_MAX_RECOVERIES = int(
    os.getenv("MAX_RECOVERIES", "2")
)


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
    recoveries: int = 0

    processing_payload: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    claimed_at: float | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    worker_id: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )


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
                "recoveries": task.recoveries,
            },
        )

        self.client.rpush(
            QUEUE_NAME,
            self._serialize_task(task),
        )

        self._increment_stat("jobs_submitted_total")

        return job_id

    def dequeue(
        self,
        timeout: int = 5,
        worker_id: str | None = None,
    ) -> QueueTask | None:
        payload = self.client.execute_command(
            "BLMOVE",
            QUEUE_NAME,
            PROCESSING_QUEUE_NAME,
            "LEFT",
            "RIGHT",
            timeout,
        )

        if payload is None:
            return None

        data = json.loads(payload)
        data.setdefault("attempts", 0)
        data.setdefault("max_retries", DEFAULT_MAX_RETRIES)
        data.setdefault("recoveries", 0)

        task = QueueTask(**data)
        task.processing_payload = payload
        task.claimed_at = time.time()
        task.worker_id = worker_id or os.getenv("HOSTNAME", "worker")

        claim = {
            "job_id": task.job_id,
            "claimed_at": task.claimed_at,
            "worker_id": task.worker_id,
            "payload": payload,
        }

        self.client.hset(
            PROCESSING_CLAIMS_KEY,
            task.job_id,
            json.dumps(claim, ensure_ascii=False),
        )

        return task

    def set_processing(self, task: QueueTask):
        self._save_status(
            task.job_id,
            {
                "status": "processing",
                "job_id": task.job_id,
                "attempts": task.attempts,
                "max_retries": task.max_retries,
                "recoveries": task.recoveries,
                "claimed_at": task.claimed_at,
                "worker_id": task.worker_id,
            },
        )

        self._increment_stat("processing_attempts_total")

        if task.created_at > 0:
            wait_seconds = max(
                0.0,
                time.time() - task.created_at,
            )

            self._observe_histogram(
                QUEUE_WAIT_HISTOGRAM_KEY,
                QUEUE_WAIT_BUCKETS,
                wait_seconds,
            )

    def requeue(self, task: QueueTask, error: str):
        task.attempts += 1
        task.created_at = time.time()

        status_data = {
            "status": "queued",
            "job_id": task.job_id,
            "attempts": task.attempts,
            "max_retries": task.max_retries,
            "recoveries": task.recoveries,
            "last_error": self._short_error(error),
        }

        pipe = self.client.pipeline(transaction=True)

        pipe.set(
            STATUS_PREFIX + task.job_id,
            json.dumps(status_data, ensure_ascii=False),
            ex=3600,
        )

        pipe.rpush(
            QUEUE_NAME,
            self._serialize_task(task),
        )

        self._queue_ack(pipe, task)
        pipe.execute()
        self._clear_claim(task)
        self._increment_stat("retries_total")

    def set_result(self, task: QueueTask, result: dict):
        status_data = {
            "status": "completed",
            "job_id": task.job_id,
            "attempts": task.attempts,
            "max_retries": task.max_retries,
            "recoveries": task.recoveries,
        }

        pipe = self.client.pipeline(transaction=True)

        pipe.set(
            RESULT_PREFIX + task.job_id,
            json.dumps(result, ensure_ascii=False),
            ex=3600,
        )

        pipe.set(
            STATUS_PREFIX + task.job_id,
            json.dumps(status_data, ensure_ascii=False),
            ex=3600,
        )

        self._queue_ack(pipe, task)
        pipe.execute()
        self._clear_claim(task)
        self._increment_stat("jobs_completed_total")

    def set_error(self, task: QueueTask, error: str):
        short_error = self._short_error(error)

        status_data = {
            "status": "failed",
            "job_id": task.job_id,
            "attempts": task.attempts,
            "max_retries": task.max_retries,
            "recoveries": task.recoveries,
            "error": short_error,
            "dead_letter": True,
            "dead_letter_reason": "inference retry limit exceeded",
        }

        dead_letter_record = {
            "job_id": task.job_id,
            "reason": "inference retry limit exceeded",
            "error": short_error,
            "attempts": task.attempts,
            "max_retries": task.max_retries,
            "recoveries": task.recoveries,
            "failed_at": time.time(),
            "task": json.loads(self._serialize_task(task)),
        }

        pipe = self.client.pipeline(transaction=True)

        pipe.set(
            STATUS_PREFIX + task.job_id,
            json.dumps(status_data, ensure_ascii=False),
            ex=3600,
        )

        pipe.rpush(
            DEAD_LETTER_QUEUE_NAME,
            json.dumps(dead_letter_record, ensure_ascii=False),
        )

        self._queue_ack(pipe, task)
        pipe.execute()
        self._clear_claim(task)

        self._increment_stat("jobs_failed_total")
        self._increment_stat("dead_lettered_total")

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

    def processing_size(self) -> int:
        return int(self.client.llen(PROCESSING_QUEUE_NAME))

    def get_processing_claims(self) -> dict[str, dict]:
        raw_claims = self.client.hgetall(PROCESSING_CLAIMS_KEY)
        claims: dict[str, dict] = {}

        for job_id, raw_claim in raw_claims.items():
            try:
                claims[job_id] = json.loads(raw_claim)
            except json.JSONDecodeError:
                continue

        return claims

    def recover_stale_tasks(
        self,
        timeout_seconds: float | None = None,
        max_recoveries: int | None = None,
    ) -> dict[str, int]:
        timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else DEFAULT_PROCESSING_TIMEOUT_SECONDS
        )
        max_recoveries = (
            max_recoveries
            if max_recoveries is not None
            else DEFAULT_MAX_RECOVERIES
        )

        now = time.time()
        raw_claims = self.client.hgetall(PROCESSING_CLAIMS_KEY)

        result = {
            "scanned": len(raw_claims),
            "stale": 0,
            "requeued": 0,
            "dead_lettered": 0,
            "skipped": 0,
        }

        for job_id, raw_claim in raw_claims.items():
            try:
                claim = json.loads(raw_claim)
                claimed_at = float(claim["claimed_at"])
                payload = str(claim["payload"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                result["skipped"] += 1
                continue

            if now - claimed_at < timeout_seconds:
                continue

            result["stale"] += 1

            try:
                task_data = json.loads(payload)
                task_data.setdefault("attempts", 0)
                task_data.setdefault(
                    "max_retries",
                    DEFAULT_MAX_RETRIES,
                )
                task_data.setdefault("recoveries", 0)

                task = QueueTask(**task_data)
                task.recoveries += 1
                new_payload = self._serialize_task(task)
            except (TypeError, ValueError, json.JSONDecodeError):
                result["skipped"] += 1
                continue

            if task.recoveries > max_recoveries:
                error_message = (
                    "processing timeout recovery limit exceeded"
                )

                status_data = {
                    "status": "failed",
                    "job_id": task.job_id,
                    "attempts": task.attempts,
                    "max_retries": task.max_retries,
                    "recoveries": task.recoveries,
                    "max_recoveries": max_recoveries,
                    "error": error_message,
                    "dead_letter": True,
                    "dead_letter_reason": error_message,
                }

                dead_letter_record = {
                    "job_id": task.job_id,
                    "reason": error_message,
                    "error": "worker processing timeout",
                    "attempts": task.attempts,
                    "max_retries": task.max_retries,
                    "recoveries": task.recoveries,
                    "max_recoveries": max_recoveries,
                    "failed_at": now,
                    "task": json.loads(new_payload),
                }

                dead_lettered = (
                    self._dead_letter_claim_atomically(
                        job_id=job_id,
                        expected_claim=raw_claim,
                        old_payload=payload,
                        dead_letter_record=dead_letter_record,
                        status_data=status_data,
                    )
                )

                if dead_lettered:
                    result["dead_lettered"] += 1
                    self._increment_stat("jobs_failed_total")
                    self._increment_stat("dead_lettered_total")
                else:
                    result["skipped"] += 1

                continue

            status_data = {
                "status": "queued",
                "job_id": task.job_id,
                "attempts": task.attempts,
                "max_retries": task.max_retries,
                "recoveries": task.recoveries,
                "max_recoveries": max_recoveries,
                "last_error": "worker processing timeout",
                "recovered_at": now,
            }

            recovered = self._recover_claim_atomically(
                job_id=job_id,
                expected_claim=raw_claim,
                old_payload=payload,
                new_payload=new_payload,
                status_data=status_data,
            )

            if recovered:
                result["requeued"] += 1
                self._increment_stat("recoveries_total")
            else:
                result["skipped"] += 1

        return result

    def _recover_claim_atomically(
        self,
        job_id: str,
        expected_claim: str,
        old_payload: str,
        new_payload: str,
        status_data: dict,
    ) -> bool:
        script = """
        local current_claim = redis.call(
            'HGET',
            KEYS[2],
            ARGV[1]
        )

        if not current_claim or current_claim ~= ARGV[2] then
            return 0
        end

        local removed = redis.call(
            'LREM',
            KEYS[1],
            1,
            ARGV[3]
        )

        if removed == 0 then
            redis.call('HDEL', KEYS[2], ARGV[1])
            return 0
        end

        redis.call('RPUSH', KEYS[3], ARGV[4])
        redis.call('HDEL', KEYS[2], ARGV[1])
        redis.call(
            'SET',
            KEYS[4],
            ARGV[5],
            'EX',
            3600
        )

        return 1
        """

        recovered = self.client.eval(
            script,
            4,
            PROCESSING_QUEUE_NAME,
            PROCESSING_CLAIMS_KEY,
            QUEUE_NAME,
            STATUS_PREFIX + job_id,
            job_id,
            expected_claim,
            old_payload,
            new_payload,
            json.dumps(status_data, ensure_ascii=False),
        )

        return int(recovered) == 1

    def _dead_letter_claim_atomically(
        self,
        job_id: str,
        expected_claim: str,
        old_payload: str,
        dead_letter_record: dict,
        status_data: dict,
    ) -> bool:
        script = """
        local current_claim = redis.call(
            'HGET',
            KEYS[2],
            ARGV[1]
        )

        if not current_claim or current_claim ~= ARGV[2] then
            return 0
        end

        local removed = redis.call(
            'LREM',
            KEYS[1],
            1,
            ARGV[3]
        )

        if removed == 0 then
            redis.call('HDEL', KEYS[2], ARGV[1])
            return 0
        end

        redis.call(
            'RPUSH',
            KEYS[3],
            ARGV[4]
        )

        redis.call('HDEL', KEYS[2], ARGV[1])

        redis.call(
            'SET',
            KEYS[4],
            ARGV[5],
            'EX',
            3600
        )

        return 1
        """

        moved = self.client.eval(
            script,
            4,
            PROCESSING_QUEUE_NAME,
            PROCESSING_CLAIMS_KEY,
            DEAD_LETTER_QUEUE_NAME,
            STATUS_PREFIX + job_id,
            job_id,
            expected_claim,
            old_payload,
            json.dumps(
                dead_letter_record,
                ensure_ascii=False,
            ),
            json.dumps(
                status_data,
                ensure_ascii=False,
            ),
        )

        return int(moved) == 1

    def dead_letter_size(self) -> int:
        return int(
            self.client.llen(DEAD_LETTER_QUEUE_NAME)
        )

    def get_stats(self) -> dict[str, int]:
        raw_stats = self.client.hgetall(QUEUE_STATS_KEY)

        return {
            "queue_size": self.queue_size(),
            "processing_size": self.processing_size(),
            "dead_letter_size": self.dead_letter_size(),
            "jobs_submitted_total": self._stat_value(
                raw_stats,
                "jobs_submitted_total",
            ),
            "processing_attempts_total": self._stat_value(
                raw_stats,
                "processing_attempts_total",
            ),
            "jobs_completed_total": self._stat_value(
                raw_stats,
                "jobs_completed_total",
            ),
            "jobs_failed_total": self._stat_value(
                raw_stats,
                "jobs_failed_total",
            ),
            "retries_total": self._stat_value(
                raw_stats,
                "retries_total",
            ),
            "recoveries_total": self._stat_value(
                raw_stats,
                "recoveries_total",
            ),
            "dead_lettered_total": self._stat_value(
                raw_stats,
                "dead_lettered_total",
            ),
        }

    def get_metrics_snapshot(self) -> dict:
        stats = self.get_stats()

        stats["queue_wait_histogram"] = (
            self._histogram_snapshot(
                QUEUE_WAIT_HISTOGRAM_KEY,
                QUEUE_WAIT_BUCKETS,
            )
        )

        stats["inference_duration_histogram"] = (
            self._histogram_snapshot(
                INFERENCE_DURATION_HISTOGRAM_KEY,
                INFERENCE_DURATION_BUCKETS,
            )
        )

        return stats

    def record_inference_duration(
        self,
        duration_seconds: float,
    ):
        self._observe_histogram(
            INFERENCE_DURATION_HISTOGRAM_KEY,
            INFERENCE_DURATION_BUCKETS,
            max(0.0, float(duration_seconds)),
        )

    def _increment_stat(
        self,
        field: str,
        amount: int = 1,
    ) -> int:
        increment = getattr(self.client, "hincrby", None)

        if not callable(increment):
            return 0

        return int(
            increment(
                QUEUE_STATS_KEY,
                field,
                amount,
            )
        )

    def _observe_histogram(
        self,
        key: str,
        buckets: tuple[float, ...],
        value: float,
    ):
        pipeline_factory = getattr(
            self.client,
            "pipeline",
            None,
        )

        if not callable(pipeline_factory):
            return

        pipe = pipeline_factory(transaction=True)

        if not hasattr(pipe, "hincrbyfloat"):
            return

        pipe.hincrby(key, "count", 1)
        pipe.hincrbyfloat(key, "sum", value)

        for bucket in buckets:
            if value <= bucket:
                pipe.hincrby(
                    key,
                    self._bucket_field(bucket),
                    1,
                )

        pipe.execute()

    def _histogram_snapshot(
        self,
        key: str,
        buckets: tuple[float, ...],
    ) -> dict:
        raw = self.client.hgetall(key)

        count = self._stat_value(raw, "count")
        total = float(raw.get("sum", 0.0))

        bucket_values = []

        for bucket in buckets:
            bucket_values.append(
                {
                    "le": str(bucket),
                    "count": self._stat_value(
                        raw,
                        self._bucket_field(bucket),
                    ),
                }
            )

        bucket_values.append(
            {
                "le": "+Inf",
                "count": count,
            }
        )

        return {
            "count": count,
            "sum": total,
            "buckets": bucket_values,
        }

    @staticmethod
    def _bucket_field(bucket: float) -> str:
        return f"le:{bucket}"

    @staticmethod
    def _stat_value(
        raw_stats: dict,
        field: str,
    ) -> int:
        try:
            return int(raw_stats.get(field, 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _serialize_task(task: QueueTask) -> str:
        data = asdict(task)
        data.pop("processing_payload", None)
        data.pop("claimed_at", None)
        data.pop("worker_id", None)
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _queue_ack(pipe, task: QueueTask):
        if task.processing_payload is not None:
            pipe.lrem(
                PROCESSING_QUEUE_NAME,
                1,
                task.processing_payload,
            )

        pipe.hdel(PROCESSING_CLAIMS_KEY, task.job_id)

    @staticmethod
    def _clear_claim(task: QueueTask):
        task.processing_payload = None
        task.claimed_at = None
        task.worker_id = None

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
